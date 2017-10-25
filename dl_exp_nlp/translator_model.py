import numpy
import chainer
from chainer import Variable, serializers, Chain, cuda
import chainer.functions as F
import chainer.links as L

from sentence_data import EOS_ID

# GPU上で計算を行う場合は，この変数を非Noneの整数にする
gpu_id = None

if gpu_id is not None:
    xp = cuda.cupy
else:
    xp = numpy


# Encoder-Decoderモデルを用いた翻訳モデルの定義
class TranslatorModel(chainer.Chain):
    def __init__(self, source_vocabulary_size, target_vocabulary_size, embed_size=100):
        super(TranslatorModel, self).__init__(
            W_x_hi=L.EmbedID(source_vocabulary_size, embed_size),
            W_y_hi=L.EmbedID(target_vocabulary_size, embed_size),
            W_lstm_enc=L.StatelessLSTM(embed_size, embed_size),
            W_lstm_dec=L.StatelessLSTM(embed_size, embed_size),
            W_hr_y=L.Linear(embed_size, target_vocabulary_size),
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
        # メモリーセルの状態も初期化する
        self.c = xp.zeros((1, self.embed_size), dtype=xp.float32)

    def encode(self, word):
        hi = self.W_x_hi(Variable(xp.array([word], dtype=xp.int32)))
        self.c, self.hr = self.W_lstm_enc(self.c, self.hr, hi)

    def decode(self, hi):
        self.c, self.hr = self.W_lstm_dec(self.c, self.hr, hi)
        y = self.W_hr_y(self.hr)
        return y

    # 入力データと正解データから，lossを計算する
    def loss(self, source_words, target_words):
        for word in source_words:
            self.encode(word)
        # メモリーセルの状態は引き継がない
        self.c = xp.zeros((1, self.embed_size), dtype=xp.float32)
        hi = self.W_x_hi(Variable(xp.array([EOS_ID], dtype=xp.int32)))
        accum_loss = None
        for target_word in target_words + [EOS_ID]:
            y = self.decode(hi)
            target = Variable(xp.array([target_word], dtype=xp.int32))
            # 正解の単語とのクロスエントロピーを取ってlossとする
            loss = F.softmax_cross_entropy(y, target)
            accum_loss = loss if accum_loss is None else accum_loss + loss
            # 正解データをLSTMの入力にする
            hi = self.W_y_hi(target)
        return accum_loss

    # 現在のモデルを用いて，入力単語列から，翻訳結果の出力単語列を返す
    def test(self, source_words):
        result = []
        for word in source_words:
            self.encode(word)
        # メモリーセルの状態は引き継がない
        self.c = xp.zeros((1, self.embed_size), dtype=xp.float32)
        hi = self.W_x_hi(Variable(xp.array([EOS_ID], dtype=xp.int32)))
        accum_loss = None
        # 最大30単語で打ち切る
        for i in range(30):
            y = self.decode(hi)
            # もっともそれらしい単語を得る
            id = numpy.argmax(cuda.to_cpu(y.data))
            # EOSが出力されたら打ち切る
            if id == EOS_ID:
                break
            result.append(id)
            # 推定した単語をLSTMの入力にする
            hi = self.W_y_hi(Variable(xp.array([id], dtype=xp.int32)))
        return result

    # モデルを読み込む
    def load_model(self, file_name):
        serializers.load_npz(file_name, self)

    # モデルを書き出す
    def save_model(self, file_name):
        serializers.save_npz(file_name, self)
