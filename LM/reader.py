import logging
import argparse
import string
import sys
import numpy as np
import mxnet as mx
from mxnet import nd


class Pipeline(object):
    def __init__(self, description="NULL"):
        self.description = description
        self.pipe = []
        self.name_book = []

    def register(self, func):
        self.pipe.append(func)
        self.name_book.append(func.__name__)

    def show(self):
        for i, func in enumerate(self.pipe):
            logging.info("%2d: %s" % (i, func.__name__))

    def remove(self, name):
        idx = self.name_book.index(name)
        self.pipe.remove(idx)
        self.name_book.remove(idx)

    def run(self, data):
        for func in self.pipe:
            logging.info("Running " + func.__name__ +
                         "() ...")
            data = func(data)

        return data


def process_raw_data(data):
    data = list(map(lambda x: x.strip().lower(), data))

    table = str.maketrans({
        key: '' for key in string.punctuation
    })
    data = list(map(lambda x: x.translate(table), data))

    # remove space
    for i in range(data.count("")):
        data.remove("")

    return data


def relax(data):
    dataset = []
    for item in data:
        dataset += item.split()
    return dataset


def create_vocab(word_list, vocab_size, unk_idx=0):
    from collections import Counter
    word_freq = Counter(word_list)
    selected_words = word_freq.most_common()[:vocab_size]
    vocab = {
        '<unk>': unk_idx
    }
    for idx, word in enumerate(selected_words):
        vocab[word[0]] = idx + 1

    return vocab


def word2idx(word_list, vocab):
    idx_list = []
    for word in word_list:
        try:
            idx_list.append(vocab[word])
        except KeyError:
            idx_list.append(vocab['<unk>'])

    return idx_list


def idx2word(idx, vocab):
    idx_vocab = {v: k for k, v in vocab.items()}
    return word2idx(idx, idx_vocab)


def generate_lm_date(corpus, n_gram=4):
    data = []
    label = []
    num_data = len(corpus) // (n_gram + 1)
    for i in range(num_data):
        start_idx = (n_gram + 1) * i
        data.append(corpus[start_idx:start_idx + n_gram])
        label.append(corpus[start_idx + n_gram + 1])

    return nd.array(data), nd.array(label, dtype=np.int32)


def generate_dataset(filename, args):
    pipeline = Pipeline()

    with open(filename) as f:
        data = f.readlines()
        pipeline.register(process_raw_data)
        data = pipeline.run(data)
        data = relax(data)

    logging.info("total words: %d" % len(data))
    vocab = create_vocab(data, args.vocab)
    raw_data = word2idx(data, vocab)

    X, y = generate_lm_date(raw_data, n_gram=args.ngram)
    logging.info("Data sanity check:")
    logging.info("shape of training data: %s" % str(X.shape))
    logging.info("shape of label: %s" % str(y.shape))

    return X, y


def save(X, y, args):
    logging.info("saving dataset as file: %s", args.output)
    nd.save(args.output, [X, y])


def main(filename, args):
    X, y = generate_dataset(filename, args)
    save(X, y, args)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=None)

    parser.add_argument("file", help="target file")
    parser.add_argument("-l", "--level", default='info',
                        choices=['debug', 'info',
                                 'warning', 'error', 'critical'],
                        help="output steam level")
    parser.add_argument("-o", "--output", default='lm.dat',
                        help="output filename")

    parser.add_argument("-n", "--ngram", default=5, type=int,
                        help="n-gram dataset for generating dataset")
    parser.add_argument("-v", "--vocab", default=10000, type=int,
                        help="vocabulary size for generating dataset")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s]: %(message)s",
                        datefmt="%m-%d %H:%M:%S")

    main(args.file, args)
