import numpy
import chainer
from chainer import Variable, serializers, Chain, cuda
import chainer.functions as F
import chainer.links as L

# GPU上で計算を行う場合は，この変数を非Noneの整数にする
gpu_id = 0

if gpu_id is not None:
    xp = cuda.cupy
else:
    xp = numpy


# 言語モデル用ニューラルネットワークの定義
class LanguageModelLSTM(chainer.Chain):
    def __init__(self, source_vocabulary_size, embed_size=100):
        # パラメータを chainer.Chain に渡す
        super(LanguageModelLSTM, self).__init__(
            W_x_hi=L.EmbedID(source_vocabulary_size, embed_size),
            W_lstm=L.LSTM(embed_size, embed_size),
            W_hr_y=L.Linear(embed_size, source_vocabulary_size),
        )
        self.reset_state()

        if gpu_id is not None:
            cuda.get_device(gpu_id).use()
            self.to_gpu(gpu_id)

    def reset_state(self):
        # 隠れ層の状態をリセットする
        self.W_lstm.reset_state()

    def __call__(self, word):
        # ここを実装する
        hi = self.W_x_hi(Variable(xp.array([word], dtype=xp.int32)))
        hr = self.W_lstm(hi) 
        y = self.W_hr_y(hr)
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
