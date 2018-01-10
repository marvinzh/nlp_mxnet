import mxnet as mx
from mxnet import nd, gluon


class NeuralLanguageModel(gluon.Block):
    def __init__(self, num_input, num_embed, num_hidden, out_size, **kwargs):
        super(NeuralLanguageModel, self,).__init__(**kwargs)
        with self.name_scope():
            self.embed = gluon.nn.Embedding(num_input, num_embed)
            self.hidden = gluon.nn.Dense(num_hidden)
            self.out = gluon.nn.Dense(out_size)

    def forward(self, x):
        x = self.embed(x)
        x = x.reshape([x.shape[0], -1])
        x = nd.relu(self.hidden(x))
        out = nd.log_softmax(self.out(x))
        return out
