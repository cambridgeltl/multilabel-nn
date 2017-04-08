from sys import stdout
from logging import info
from datetime import datetime

from abc import ABCMeta, abstractmethod

from keras.callbacks import Callback

from evaluation import evaluate_classification, summarize_classification
from util import unique
from defaults import defaults

class WeightStore(Callback):
    """Stores results of model.get_weights() on epoch end."""
    def __init__(self, weights=None):
        super(WeightStore, self).__init__()
        if weights is None:
            weights = []
        self.weights = weights

    def on_epoch_end(self, epoch, logs={}):
        self.weights.append(self.model.get_weights())

class LtlCallback(Callback):
    """Adds after_epoch_end() to Callback.

    after_epoch_end() is invoked after all calls to on_epoch_end() and
    is intended to work around the fixed callback ordering in Keras,
    which can cause output from callbacks to mess up the progress bar
    (related: https://github.com/fchollet/keras/issues/2521).
    """

    def __init__(self):
        super(LtlCallback, self).__init__()
        self.epoch = 0

    def after_epoch_end(self, epoch):
        pass

    def on_epoch_begin(self, epoch, logs={}):
        if epoch > 0:
            self.after_epoch_end(self.epoch)
        self.epoch += 1

    def on_train_end(self, logs={}):
        self.after_epoch_end(self.epoch)

class CallbackChain(Callback):
    """Chain of callbacks."""

    def __init__(self, callbacks):
        super(CallbackChain, self).__init__()
        self._callbacks = callbacks

    def _set_params(self, params):
        for callback in self._callbacks:
            callback._set_params(params)

    def _set_model(self, model):
        for callback in self._callbacks:
            callback._set_model(model)

    def on_epoch_begin(self, *args, **kwargs):
        for callback in self._callbacks:
            callback.on_epoch_begin(*args, **kwargs)

    def on_epoch_end(self, *args, **kwargs):
        for callback in self._callbacks:
            callback.on_epoch_end(*args, **kwargs)

    def on_batch_begin(self, *args, **kwargs):
        for callback in self._callbacks:
            callback.on_batch_begin(*args, **kwargs)

    def on_batch_end(self, *args, **kwargs):
        for callback in self._callbacks:
            callback.on_batch_end(*args, **kwargs)

    def on_train_begin(self, *args, **kwargs):
        for callback in self._callbacks:
            callback.on_train_begin(*args, **kwargs)

    def on_train_end(self, *args, **kwargs):
        for callback in self._callbacks:
            callback.on_train_end(*args, **kwargs)

class EvaluatorCallback(LtlCallback):
    """Abstract base class for evaluator callbacks."""

    __metaclass__ = ABCMeta

    def __init__(self, dataset, label=None, writer=None, results=None):
        super(EvaluatorCallback, self).__init__()
        if label is None:
            label = dataset.name
        if writer is None:
            writer = info
        if results is None:
            results = {}
        self.dataset = dataset
        self.label = label
        self.writer = writer
        self.summaries = []
        self.results = results

    @abstractmethod
    def evaluation_results(self):
        """Evaluate and return dict mapping metric names to results."""
        pass

    @abstractmethod
    def evaluation_summary(self, results):
        """Return string summarizing evaluation results."""
        pass

    def after_epoch_end(self, epoch):
        results = self.evaluation_results()
        for k, v in results.items():
            k = '{}/{}'.format(self.label, k)
            if k not in self.results:
                self.results[k] = []
            self.results[k].append(v)
        summary = self.evaluation_summary(results)
        self.summaries.append(summary)
        for s in summary.split('\n'):
            self.writer('{} Ep: {} {}'.format(self.label, epoch, s))

class EpochTimer(LtlCallback):
    """Callback that logs timing information."""

    def __init__(self, label='', writer=info):
        super(EpochTimer, self).__init__()
        self.label = '' if not label else label + ' '
        self.writer = writer

    def on_epoch_begin(self, epoch, logs={}):
        super(EpochTimer, self).on_epoch_begin(epoch, logs)
        self.start_time = datetime.now()

    def after_epoch_end(self, epoch):
        end_time = datetime.now()
        delta = end_time - self.start_time
        start = str(self.start_time).split('.')[0]
        end = str(end_time).split('.')[0]
        self.writer('{}Ep: {} {}s (start {}, end {})'.format(
                self.label, epoch, delta.seconds, start, end
                ))

class Predictor(LtlCallback):
    """Makes and stores predictions for data item sequence."""

    def __init__(self, dataitems):
        super(Predictor, self).__init__()
        self.dataitems = dataitems

    def after_epoch_end(self, epoch):
        predictions = self.model.predict(self.dataitems.inputs)
        self.dataitems.set_predictions(predictions)

class PredictionMapper(LtlCallback):
    """Maps predictions to strings for data item sequence."""

    def __init__(self, dataitems, mapper):
        super(PredictionMapper, self).__init__()
        self.dataitems = dataitems
        self.mapper = mapper

    def after_epoch_end(self, epoch):
        self.dataitems.map_predictions(self.mapper)
        # TODO check if summary() is defined
        info(self.mapper.summary())

class DocumentEvaluator(EvaluatorCallback):
    """Evaluates performance using document-level metrics."""

    def __init__(self, dataset, label=None, writer=None, results=None):
        super(DocumentEvaluator, self).__init__(dataset, label, writer,
                                                results)

    def evaluation_results(self):
        return evaluate_classification(self.dataset.documents)

    def evaluation_summary(self, results):
        return summarize_classification(results)

def document_evaluator(dataset, label=None, writer=None, results=None):
    """Return appropriate evaluator callback for dataset."""
    callbacks = []
    callbacks.append(Predictor(dataset.documents))
    callbacks.append(DocumentEvaluator(dataset, label=label, writer=writer,
                                       results=results))
    return CallbackChain(callbacks)
