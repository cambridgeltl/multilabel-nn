import io
import re
import random

from os import path

from copy import deepcopy
from itertools import cycle

from data import Datasets, Dataset, Document, Sentence, Token
from common import FormatError

from defaults import defaults

def balance_examples(by_label, oversample=True):
    if oversample:
        max_len = max(len(e) for e in by_label)
        balanced_by_label = []
        for examples in by_label:
            balanced = examples[:]
            for i in cycle(range(len(examples))):
                if len(balanced) >= max_len:
                    break
                balanced.append(deepcopy(examples[i]))
            balanced_by_label.append(balanced)
        return balanced_by_label
    else:
        raise NotImplementedError()

def load_dir(directory, config=defaults):
    """Load train, devel and test data from directory, return Datasets.

    Expects each of the training, development and test datasets to be
    contained in two files with basenames 'train', 'devel' and 'test'
    (resp.) in the given directory and suffixes '.pos' for positive
    and '.neg' for negative examples.
    """
    if config.random_seed is not None:
        random.seed(config.random_seed)
    datasets = []
    for dset in ('train', 'devel', 'test'):
        fname = lambda l: path.join(directory, '{}.{}'.format(dset, l))
        pos_doc = load_documents(fname('pos'), 'pos', config)
        neg_doc = load_documents(fname('neg'), 'neg', config)
        if config.oversample and dset == 'train':
            pos_doc, neg_doc = balance_examples([pos_doc, neg_doc])
        documents = pos_doc + neg_doc
        # Shuffle list of document to avoid training batches consisting
        # of only positive or negative example.
        random.shuffle(documents)
        dname = '{}-{}'.format(path.basename(directory.rstrip('/')), dset)
        datasets.append(Dataset(documents=documents, name=dname))
    return Datasets(*datasets)

def tokenize(text, config=defaults):
    token_regex = re.compile(config.token_regex, re.UNICODE)
    tokens = [t for t in token_regex.split(text) if t]
    if ''.join(tokens) != text:
        raise ValueError('tokenized does not match text (token-regex error?)')
    tokens = [t for t in tokens if t and not t.isspace()]
    return tokens

def load_documents(filename, label, config=defaults):
    """Load documents from file, return list of Document objects."""
    with io.open(filename, encoding=config.encoding) as f:
        return read_documents(f, label, config)

def read_documents(flo, label, config=defaults):
    """Load documents from file-like object, return list of Document objects."""
    documents = []
    for ln, line in enumerate(flo, start=1):
        line = line.strip()
        if not line:
            raise FormatError('empty line {} in {}'.format(ln, flo.name))
        token_texts = tokenize(line, config)
        document = make_document(token_texts, label)
        documents.append(document)
    return documents

def make_document(token_texts, label):
    """Return Document object initialized with given token texts."""
    tokens = [Token(t) for t in token_texts]
    # We don't have sentence splitting, but the data structure expects
    # Documents to contain Sentences which in turn contain Tokens.
    # Create a dummy sentence containing all document tokens to work
    # around this constraint.
    sentences = [Sentence(tokens=tokens)]
    return Document(target_str=label, sentences=sentences)
