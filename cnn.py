#!/usr/bin/env python

from __future__ import print_function

import sys
#import numpy as np
import utility

#from ltlib import filelog

from ltlib.settings import cli_settings
from ltlib.docdata import load_dir
from ltlib.features import NormEmbeddingFeature, FixedWidthInput
from ltlib.layers import FixedEmbedding, concat
from ltlib.optimizers import get_optimizer
from ltlib.callbacks import WeightStore, EpochTimer, document_evaluator, EvaluatorCallback, Predictor, CallbackChain
from ltlib.evaluation import evaluate_classification, summarize_classification
from ltlib.metrics import f1, prec, rec
import ltlib.util
from keras.models import Model
from keras.layers import Input, Embedding, Flatten, Reshape, Dense, Dropout
from keras.layers import Convolution2D, MaxPooling2D
from keras.regularizers import l2
from multiLabelDataReader import MultiLabelDataReader

from config import Defaults

#import logging


class Ind_document_evaluator(EvaluatorCallback):
    """Evaluates performance using document-level metrics."""

    def __init__(self, dataset, label=None, writer=None, results=None):
        super(Ind_document_evaluator, self).__init__(dataset, label, writer,
                                                results)

        self.bestRes = None

    def evaluation_results(self):
       # print("Evaluation----------->::  " + self.dataset.name + " "+str(self.dataset.eval()))
        print ("evaluating dataset:" + self.dataset.name)
        print ("with: " +str(len(self.dataset.children)))
        res = self.dataset.eval()#evaluate_classification(self.dataset.documents)
        print (str(res))
        if self.bestRes == None or self.bestRes["fscore"] < res["fscore"]:
            print ("new best F-score: " + str(res["fscore"]))
            res["best_epoch"] = self.epoch
            self.bestRes = res
            utility.writeDictAsStringFile(res, Defaults.output_path+ str(Defaults.index)+".txt")
            ltlib.util.save_keras(self.model,Defaults.saved_mod_path+str(Defaults.index))
        return res

    def evaluation_summary(self, results):
        print("eval summary")
        return summarize_classification(results)

def evaluator(dataset, label=None, writer=None, results=None):
    """Return appropriate evaluator callback for dataset."""
    callbacks = []
    print ("evaluating: " + str(dataset.name))
    callbacks.append(Predictor(dataset.documents))
    callbacks.append(Ind_document_evaluator(dataset, label=label, writer=writer, results=results))

    return CallbackChain(callbacks)

def W_regularizer(config):
    if config.l2_lambda is not None:
        return l2(config.l2_lambda)
    else:
        return None

def inputs_and_embeddings(features, config):
    inputs, embeddings = [], []
    for f in features:
        E = Embedding if not config.fixed_embedding else FixedEmbedding
        i = Input(shape=(config.doc_size,), dtype='int32', name=f.name)
        e = E(f.input_dim, f.output_dim, weights=[f.weights],
              input_length=config.doc_size)(i)
        inputs.append(i)
        embeddings.append(e)
    return inputs, embeddings

# def get_best_epoch(results, label, config):
#     """Get index of best epoch based on metric identified in config."""
#     key = '{}/{}'.format(label, config.target_metric)
#     epoch = np.argmax(results[key])
#     value = results[key][epoch]
#     logging.info('best epoch for {}: {} ({})'.format(key, epoch+1, value))
#     return epoch

# def set_best_weights(model, weights, results, label, config):
#     """Set best epoch weights based on metric identified in config."""
#     key = '{}/{}'.format(label, config.target_metric)
#     epoch = np.argmax(results[key])
#     value = results[key][epoch]
#     logging.info('best epoch for {}: {} ({})'.format(key, epoch+1, value))
#     model.set_weights(weights[epoch])

# def evaluation_summary(model, dataset, threshold, config):
#     predictions = model.predict(
#         dataset.documents.inputs,
#         batch_size=config.batch_size
#     )
#     mapper = None #if not threshold else make_thresholded_mapper(threshold)
#     dataset.documents.set_predictions(predictions, mapper=mapper)
#     results = dataset.eval()#evaluate_classification(dataset.documents)
#     return summarize_classification(results)

# def make_thresholded_mapper(threshold):
#     from ltlib.data import default_prediction_mapper
#     def thresholded_mapper(item):
#         item.prediction[0] += threshold
#         default_prediction_mapper(item)
#     return thresholded_mapper

def main(argv):
    global data
    config = cli_settings(['datadir', 'wordvecs', 'index'], Defaults)

    print ("STARTING CLASIFCATION FOR INDEX = " + str(config.index))



    force_oov = set(l.strip() for l in open(config.oov)) if config.oov else None
    w2v = NormEmbeddingFeature.from_file(config.wordvecs,
                                         max_rank=config.max_vocab_size,
                                         vocabulary=data.vocabulary,
                                         force_oov=force_oov,
                                         name='text')
    # Add word vector features to tokens
    print ("finished reading embeddings")
    features = [w2v]
    data.tokens.add_features(features)
    # Summarize word vector featurizer statistics (OOV etc.)
#    logging.info(features[0].summary())
    # Create inputs at document level
    data.documents.add_inputs([
        FixedWidthInput(config.doc_size, f['<PADDING>'], f.name)
        for f in features
    ])

    # Create keras input and embedding for each feature
    inputs, embeddings = inputs_and_embeddings(features, config)

    # Combine and reshape for convolution
    seq = concat(embeddings)
    cshape = (config.doc_size, sum(f.output_dim for f in features))   #calculating the size of documents and all features.
    seq = Reshape((1,)+cshape)(seq)
    #seq = Reshape((1, config.doc_size, w2v.output_dim))(embeddings) #old way of doing the above

    # Convolution(s)
    convLayers = []
    for filter_size, filter_num in zip(config.filter_sizes, config.filter_nums):
        seq2 = Convolution2D(
            filter_num,
            filter_size,
            cshape[1],
            border_mode='valid',
            activation='relu',
            dim_ordering='th'
        )(seq)
        seq2 = MaxPooling2D(
            pool_size=(config.doc_size-filter_size+1, 1),
            dim_ordering='th'
        )(seq2)
        seq2 = Flatten()(seq2)
        convLayers.append(seq2)

    seq = concat(convLayers)
    if config.drop_prob:
        seq = Dropout(config.drop_prob)(seq)
    for s in config.hidden_sizes:
        seq = Dense(s, activation='relu')(seq)
    out = Dense(
        data.documents.target_dim,
        W_regularizer=W_regularizer(config),
        activation='softmax'
        )(seq)
    model = Model(input=inputs, output=out)

    # if config.verbosity != 0:
    #     logging.info(model.summary())

    optimizer = get_optimizer(config)
    model.compile(
        loss='categorical_crossentropy',
        optimizer=optimizer
        #metrics=['accuracy', f1, prec, rec]
    )

    weights, results = [], {}
    callbacks = [
        EpochTimer(),
        WeightStore(weights),
        #document_evaluator(data.train, label='train', results=results),
        evaluator(data.devel, label='devel', results=results),

    ]
    # if config.test:
    #     callbacks.append(evaluator(data.test, label='test', results=results))

    hist = model.fit(
        data.train.documents.inputs,
        data.train.documents.targets,
        validation_data=(
            data.devel.documents.inputs,
            data.devel.documents.targets,
        ),
        batch_size=config.batch_size,
        nb_epoch=config.epochs,
        verbose=config.verbosity,
        callbacks=callbacks
    )
    #logging.info(history.history)

    # for k, values in results.items():
    #     s = lambda v: str(v) if not isinstance(v, float) else '{:.4f}'.format(v)
    #     logging.info('\t'.join(s(i) for i in [k] + values))

    #evalsets = [data.devel] + ([data.test] if config.test else [])
    # for s in evalsets:
    #     logging.info('last epoch, {}: {}'.format(
    #         s.name, evaluation_summary(model, s, 0, config))
    #     )
    # epoch = get_best_epoch(results, 'devel', config)
    #model.set_weights(weights[epoch])
    # if config.threshold:
    #     threshold = results['devel/maxf-threshold'][epoch]
    # else:
    #     threshold = 0.0
    # for s in evalsets:
    #     logging.info('best devel epoch th {} ({}), {}: {}'.format(
    #         threshold, config.target_metric, s.name, evaluation_summary(model, s, threshold, config))
    #     )
def eval_test(modelPath,index):
    global data
    #data = MultiLabelDataReader(Defaults.input_path).load(index)
    model = ltlib.util.load_keras(modelPath+str(index))
    optimizer = get_optimizer(Defaults)

    print("STARTING TEST FOR INDEX = " + str(index))


    force_oov = set(l.strip() for l in open(Defaults.oov)) if Defaults.oov else None
    w2v = NormEmbeddingFeature.from_file(Defaults.embedding_path,
                                         max_rank=Defaults.max_vocab_size,
                                         vocabulary=data.vocabulary,
                                         force_oov=force_oov,
                                         name='text')
    # Add word vector features to tokens

    features = [w2v]
    data.tokens.add_features(features)
    # Summarize word vector featurizer statistics (OOV etc.)
    #    logging.info(features[0].summary())
    # Create inputs at document level
    data.documents.add_inputs([
                                  FixedWidthInput(Defaults.doc_size, f['<PADDING>'], f.name)
                                  for f in features
                                  ])

    # Create keras input and embedding for each feature
    inputs, embeddings = inputs_and_embeddings(features, Defaults)

    model.compile(
        loss='categorical_crossentropy',
        optimizer=optimizer,
        metrics=['accuracy', f1, prec, rec]
    )
    predictions = model.predict(data.test.documents.inputs, batch_size=Defaults.batch_size)
    #print(str(predictions))
    data.test.documents.set_predictions(predictions)
    print ("TEST RESULTS for: " + str(len(predictions)))
    res = data.test.eval()
    print(str(res))
    utility.writeDictAsStringFile(res,Defaults.results_path + str(index)+".txt")







if __name__ == '__main__':

    print ("number of classes = " + str(Defaults.number_classes))
    home = "/home/sb/"
    sys.argv.append(Defaults.input_path)  # path to data
    sys.argv.append(Defaults.embedding_path)
    sys.argv.append("0")
    #sys.argv.append(Defaults.results_path)

    utility.createDirIfNotExist(Defaults.results_path)
      # load_dir(config.datadir, config)
    for i in range(int(Defaults.number_classes)):
        print("starting index = " + str(i))
        Defaults.index = i
        utility.createDirIfNotExist(Defaults.saved_mod_path)
        utility.createDirIfNotExist(Defaults.output_path)
        data = MultiLabelDataReader(Defaults.input_path).load(Defaults.index)
        print("finished reading data")
        sys.argv[-1]= str(i)
        print(str(sys.argv))
        main(sys.argv)
        eval_test(Defaults.saved_mod_path,i)





