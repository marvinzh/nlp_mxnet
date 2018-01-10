import mxnet
from mxnet import nd

def predict_next_word(data, model, vocab, best_n=1, ctx=mx.cpu()):
    data = word2idx(data.split(), vocab)
    data = nd.array(data).as_in_context(ctx).reshape([-1, n_gram])

    output = model.forward(data).reshape([-1])

    preds_idx = output.argsort(is_ascend=False).asnumpy()[:best_n]
    
    preds_words = idx2word(preds_idx, vocab)
    out = []
    for pred_idx, word in zip(preds_idx, preds_words):
        out.append(
            (word, np.exp(output[int(pred_idx)].asscalar()))
        )

    return out