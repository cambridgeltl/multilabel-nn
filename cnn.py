#!/usr/bin/env python

from __future__ import print_function

import sys
import numpy as np

from ltlib import filelog

from ltlib.settings import cli_settings
from ltlib.docdata import load_dir
from ltlib.features import NormEmbeddingFeature, FixedWidthInput
from ltlib.layers import FixedEmbedding, concat
from ltlib.optimizers import get_optimizer
from ltlib.callbacks import WeightStore, EpochTimer, document_evaluator
from ltlib.evaluation import evaluate_classification, summarize_classification
from ltlib.metrics import f1, prec, rec

from keras.models import Model
from keras.layers import Input, Embedding, Flatten, Reshape, Dense, Dropout
from keras.layers import Convolution2D, MaxPooling2D
from keras.regularizers import l2

from config import Defaults

import logging

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

def get_best_epoch(results, label, config):
    """Get index of best epoch based on metric identified in config."""
    key = '{}/{}'.format(label, config.target_metric)
    epoch = np.argmax(results[key])
    value = results[key][epoch]
    logging.info('best epoch for {}: {} ({})'.format(key, epoch+1, value))
    return epoch

def set_best_weights(model, weights, results, label, config):
    """Set best epoch weights based on metric identified in config."""
    key = '{}/{}'.format(label, config.target_metric)
    epoch = np.argmax(results[key])
    value = results[key][epoch]
    logging.info('best epoch for {}: {} ({})'.format(key, epoch+1, value))
    model.set_weights(weights[epoch])

def evaluation_summary(model, dataset, threshold, config):
    predictions = model.predict(
        dataset.documents.inputs,
        batch_size=config.batch_size
    )
    mapper = None if not threshold else make_thresholded_mapper(threshold)
    dataset.documents.set_predictions(predictions, mapper=mapper)
    results = evaluate_classification(dataset.documents)
    return summarize_classification(results)

def make_thresholded_mapper(threshold):
    from ltlib.data import default_prediction_mapper
    def thresholded_mapper(item):
        item.prediction[0] += threshold
        default_prediction_mapper(item)
    return thresholded_mapper

def main(argv):
    config = cli_settings(['datadir', 'wordvecs'], Defaults)
    data = load_dir(config.datadir, config)

    force_oov = set(l.strip() for l in open(config.oov)) if config.oov else None
    w2v = NormEmbeddingFeature.from_file(config.wordvecs,
                                         max_rank=config.max_vocab_size,
                                         vocabulary=data.vocabulary,
                                         force_oov=force_oov,
                                         name='text')
    # Add word vector features to tokens
    features = [w2v]
    data.tokens.add_features(features)
    # Summarize word vector featurizer statistics (OOV etc.)
    logging.info(features[0].summary())
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

    if config.verbosity != 0:
        logging.info(model.summary())

    optimizer = get_optimizer(config)
    model.compile(
        loss='categorical_crossentropy',
        optimizer=optimizer,
        metrics=['accuracy', f1, prec, rec]
    )

    weights, results = [], {}
    callbacks = [
        EpochTimer(),
        WeightStore(weights),
        document_evaluator(data.train, label='train', results=results),
        document_evaluator(data.devel, label='devel', results=results),
    ]
    if config.test:
        callbacks.append(document_evaluator(data.test, label='test',
                                            results=results))

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
    # logging.info(history.history)

    for k, values in results.items():
        s = lambda v: str(v) if not isinstance(v, float) else '{:.4f}'.format(v)
        logging.info('\t'.join(s(i) for i in [k] + values))

    evalsets = [data.devel] + ([data.test] if config.test else [])
    for s in evalsets:
        logging.info('last epoch, {}: {}'.format(
            s.name, evaluation_summary(model, s, 0, config))
        )
    epoch = get_best_epoch(results, 'devel', config)
    model.set_weights(weights[epoch])
    if config.threshold:
        threshold = results['devel/maxf-threshold'][epoch]
    else:
        threshold = 0.0
    for s in evalsets:
        logging.info('best devel epoch th {} ({}), {}: {}'.format(
            threshold, config.target_metric, s.name, evaluation_summary(model, s, threshold, config))
        )

if __name__ == '__main__':
    sys.exit(main(sys.argv))
