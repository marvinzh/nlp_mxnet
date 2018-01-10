import string
import numpy as np
import mxnet as mx
from mxnet import nd, autograd, gluon
from time import time
import argparse
import logging

from MLP import *
from reader import *


def get_data(args):
    X, y = generate_dataset(args.file, args)
    train = gluon.data.DataLoader(gluon.data.ArrayDataset(
        X, y), batch_size=args.n_batch, shuffle=True)
    return train


def train_model(data, args):
    if args.use_cpu:
        model_ctx = mx.cpu()
    elif args.use_gpu:
        model_ctx = mx.gpu()
    else:
        logging.error("No explicit specified training device, use cpu as default")
        model_ctx = mx.cpu()

    mlp_lm = NeuralLanguageModel(
        args.n_input, args.n_embed, args.n_hidden, args.n_out)
    mlp_lm.initialize(mx.init.Normal(sigma=0.1), ctx=model_ctx)
    softmax_loss = gluon.loss.SoftmaxCrossEntropyLoss()
    optimizer = gluon.Trainer(mlp_lm.collect_params(), 'adam')

    logging.info("======training Neual Language Model=======")
    for e in range(args.n_epoch):
        cum_logprobs = 0.
        for i, (data, label) in enumerate(data):
            data = data.as_in_context(model_ctx).reshape([-1, args.ngram])
            label = label.as_in_context(model_ctx)
            with autograd.record():
                output = mlp_lm.forward(data)
                loss = softmax_loss(output, label)

            loss.backward()
            optimizer.step(args.n_batch)
            idx_mask = nd.one_hot(label, args.n_out)
            cum_logprobs += nd.sum((output * idx_mask)).asscalar()

        logging.info("Epoch: %s,  Perplexity: %s" %
                     (e + 1, np.exp(-cum_logprobs / len(data))))


if __name__ == "__main__":

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s]: %(message)s",
                        datefmt="%m-%d %H:%M:%S")
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

    parser.add_argument("--n_input", type=int, default=5,
                        help="dimension of input data point.")
    parser.add_argument("--n_embed", type=int, default=128,
                        help="dimension of embedding layer")
    parser.add_argument("--n_hidden", type=int, default=256,
                        help="dimension of hidden layer")
    parser.add_argument("--n_out", type=int, default=10000,
                        help="dimension of output layer")
    parser.add_argument("--n_batch", type=int, default=32, help="batch size")
    parser.add_argument("--n_epoch", type=int, default=50,
                        help="Epoches for training process")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("-use-cpu", action="store_true",
                       help="use CPU for training process")
    group.add_argument("--use-gpu", action="store_true",
                       help="use GPU for training process")

    args = parser.parse_args()

    data = get_data(args)
    train_model(data, args)
