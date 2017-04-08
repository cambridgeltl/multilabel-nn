import numpy as np    # TODO remove dependency

from collections import namedtuple
from itertools import chain

from sklearn import metrics as skmetrics

from util import unique

from logging import warn

BinaryClassificationCounts = namedtuple('BinaryClassificationCounts',
                                        'tp tn fp fn')
BinaryClassificationMetrics = namedtuple('BinaryClassificationMetrics',
                                         'tp tn fp fn acc prec rec fscore')
PrCurvePoint = namedtuple('PrCurvePoint', 'prec rec fscore threshold')

def accuracy(gold, pred):
    if len(gold) != len(pred):
        raise ValueError('count mismatch')
    correct = sum(int(g == p) for g, p in zip(gold, pred))
    return 1.*correct/len(gold)

def tp_tn_fp_fn(gold, pred):
    """Return (TP, FN, FP, FN) counts for gold and prediced values.

    Assumes that 0 is negative and all others positive.
    """
    tp, tn, fp, fn = 0, 0, 0, 0
    for g, p in zip(gold, pred):
        if g == p:
            if g == 0:
                tn += 1
            else:
                tp += 1
        else:
            if g == 0:
                fp += 1
            else:
                fn += 1
    return BinaryClassificationCounts(tp, tn, fp, fn)

def precision_recall_fscore(tp, fp, fn):
    """Return (precision, recall, f-score) for given counts."""
    prec = 0.0 if tp + fp == 0 else 1.*tp / (tp + fp)
    rec = 0.0 if tp + fn == 0 else 1.*tp / (tp + fn)
    f = 0.0 if prec + rec == 0.0 else 2 * prec * rec / (prec + rec)
    return prec, rec, f

def evaluate_binary_classification(gold, pred, positive):
    """Evaluate binary classification performance.

    Map labels in positive to 1 and others to 0.

    Return BinaryClassificationMetrics.
    """
    if len(gold) != len(pred):
        raise ValueError('count mismatch')

    gold = _binarize(gold, positive)
    pred = _binarize(pred, positive)

    if not any(i for i in gold):
        warn('no positive gold labels for %s' % str(positive))

    acc = accuracy(gold, pred)
    tp, tn, fp, fn = tp_tn_fp_fn(gold, pred)
    prec, rec, f = precision_recall_fscore(tp, fp, fn)

    return BinaryClassificationMetrics(tp, tn, fp, fn, acc, prec, rec, f)

def _binarize(a, positive):
    """Return values mapped to 1 or 0.

    Map values in positive to 1 and others to 0.
    """
    return [1 if i in positive else 0 for i in a]

def average_precision_recall_fscore(results, micro=True):
    """Return average precision, recall and f-score for list of
    BinaryClassificationMetrics.
    """
    if micro:
        total = BinaryClassificationMetrics(*tuple(np.sum(results, axis=0)))
        return precision_recall_fscore(total.tp, total.fp, total.fn)
    else:
        avg = BinaryClassificationMetrics(*tuple(np.average(results, axis=0)))
        return avg.prec, avg.rec, avg.fscore

def _positive_label(labels):
    """Return label representing the positive class or None if ambiguous."""
    if set(labels) == set(['positive', 'negative']):
        return 'positive'
    elif set(labels) == set(['pos', 'neg']):
        return 'pos'
    else:
        return None    # TODO other alternatives

def is_binary_labeling(labels):
    """Return True iff given labels represent binary classification."""
    return len(labels) == 2 and _positive_label(labels) is not None

def _binary_labels(dataitems):
    gold = dataitems.target_strs
    pred = dataitems.prediction_strs
    labels = unique(chain(gold, pred))
    return is_binary_labeling(labels)

def f1_score(prec, rec):
    from math import isnan
    if isnan(prec) or isnan(rec) or prec+rec == 0.0:
        return float('nan')
    else:
        return 2*prec*rec/(prec+rec)

def max_f_point(dataitems):
    """Return PrCurvePoint with maximal f1 score."""
    import logging
    from sklearn.metrics import precision_recall_curve
    y_true = np.argmax(dataitems.targets, axis=-1)
    prob_neg = dataitems.predictions[:,0]    # 1st column
    prob_pos = dataitems.predictions[:,1]    # 2nd column
    pos_score = prob_pos - prob_neg
    precs, recs, tholds = precision_recall_curve(y_true, pos_score)
    max_f, max_point = float('-inf'), PrCurvePoint(None, None, None, None)
    for p, r, t in zip(precs, recs, tholds):
        f = f1_score(p, r)
        if f > max_f:
            max_f, max_point = f, PrCurvePoint(p, r, f, t)
    return max_point

def evaluate_binary_labeling(dataitems):
    gold = dataitems.target_strs
    pred = dataitems.prediction_strs
    labels = unique(chain(gold, pred))
    pos =  _positive_label(labels)
    res = {}
    res['acc'] = accuracy(gold, pred)
    bcm = evaluate_binary_classification(gold, pred, pos)
    res.update(bcm._asdict())
    res['auc'] = skmetrics.roc_auc_score(dataitems.targets,
                                         dataitems.predictions)
    res['ap'] = skmetrics.average_precision_score(dataitems.targets,
                                                  dataitems.predictions)
    maxfp = max_f_point(dataitems)
    res.update({ 'maxf-{}'.format(k): v for k, v in maxfp._asdict().items() })
    return res

def summarize_classification(results):
    return (
        'acc: {acc:.2%} auc: {auc:.2%} ap: {ap:.2%} ' +
        'f: {fscore:.2%} (p:{prec:.1%} r:{rec:.1%} ' +
        'tp:{tp} fp:{fp} fn:{fn}) ' +
        'maxf: {maxf-fscore:.2%} (p:{maxf-prec:.1%} r:{maxf-rec:.1%} ' +
        'th:{maxf-threshold:.2})'
    ).format(**results)

def evaluate_classification(dataitems):
    if _binary_labels(dataitems):
        return evaluate_binary_labeling(dataitems)
    else:
        raise NotImplementedError()
