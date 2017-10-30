import numpy
import chainer
from chainer import Variable, serializers, Chain, cuda
from sentence_data import EOS_ID
import chainer.functions as F
import chainer.links as L

PADDING = -1

class LanguageModelLSTM(chainer.Chain):
    def __init__(self, source_vocabulary_size, embed_size=100):
        super(LanguageModelLSTM, self).__init__()
        with self.init_scope():
            self.W_x_hi=L.EmbedID(source_vocabulary_size, embed_size, ignore_label=PADDING)
            self.W_lstm=L.LSTM(embed_size, embed_size)
            self.W_hr_y=L.Linear(embed_size, source_vocabulary_size)
        self.reset_state()


    def reset_state(self):
        self.W_lstm.reset_state()

        
    def __call__(self, source_words, target_words=None):
        if chainer.config.train:

            source_words = source_words.transpose((1, 0, 2))
            target_words = target_words.transpose((1, 0, 2))
            loss = 0
            for src, tgt in zip(source_words, target_words):
                y = self.run(src)
                loss += F.softmax_cross_entropy(y, tgt.flatten())
            chainer.report({'loss': loss / len(source_words)}, self)
            return loss
        else:
            return self.xp.array(
                [self.run(source_word) for source_word in source_words])

    def run(self, word):
        hi = self.W_x_hi(word)
        hr = self.W_lstm(hi)
        y = self.W_hr_y(hr)
        return y
