import numpy
import chainer
from chainer import Variable, serializers, Chain, cuda
import chainer.functions as F
import chainer.links as L

# GPU上で計算を行う場合は，この変数を非Noneの整数にする
gpu_id = None

if gpu_id is not None:
    xp = cuda.cupy
else:
    xp = numpy


# 言語モデル用ニューラルネットワークの定義
class LanguageModelRNN(chainer.Chain):
    def __init__(self, source_vocabulary_size, embed_size=100):
        # パラメータを chainer.Chain に渡す
        super(LanguageModelRNN, self).__init__(
            W_x_hi=L.EmbedID(source_vocabulary_size, embed_size),
            W_hi_hr=L.Linear(embed_size, embed_size),
            W_hr_hr=L.Linear(embed_size, embed_size),
            W_hr_y=L.Linear(embed_size, source_vocabulary_size),
        )
        # 隠れ層の次元数を保存
        self.embed_size = embed_size
        self.reset_state()

        if gpu_id is not None:
            cuda.get_device(gpu_id).use()
            self.to_gpu(gpu_id)

    def reset_state(self):
        # 隠れ層の状態をリセットする
        self.hr = xp.zeros((1, self.embed_size), dtype=xp.float32)

    def __call__(self, word):
        hi = self.W_x_hi(Variable(xp.array([word], dtype=xp.int32)))
        self.hr =  F.tanh(self.W_hi_hr(hi) + self.W_hr_hr(self.hr))
        y = self.W_hr_y(self.hr)
        return y

    def loss(self, source_word, target_word):
        # self() で__call__が呼ばれる
        y = self(source_word)
        target = Variable(xp.array([target_word], dtype=xp.int32))
        # 正解の単語とのクロスエントロピーを取ってlossとする
        return F.softmax_cross_entropy(y, target)

    # モデルを読み込む
    def load_model(self, file_name):
        serializers.load_npz(file_name, self)

    # モデルを書き出す
    def save_model(self, file_name):
        serializers.save_npz(file_name, self)
