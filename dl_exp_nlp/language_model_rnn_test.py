#!/usr/bin/env python3

import sys
import numpy
import chainer
from chainer import cuda

import sentence_data
from sentence_data import EOS_ID
from language_model_rnn import LanguageModelRNN

dataset = sentence_data.SentenceData("dataset/data_1000.txt")

model = LanguageModelRNN(dataset.japanese_word_size())

model.load_model("trained_model/langage_model_rnn_10.model")

beginning_words = input("input beginning words : ").split(' ')

next_y = None
for word in beginning_words:
    if not word:
        # 単語が空だったら飛ばす
        continue
    print(word)
    word_id = dataset.japanese_word_id(word)
    if word_id is None:
        sys.stderr.write("Error : Unknown word " + word + "\n")
        sys.exit()
    # 単語をRNNに入力する
    next_y = model(word_id)

if next_y is None:
    sys.stderr.write("Error : Empty input\n")
    sys.exit()

# 最大30単語で打ち切る
for i in range(30):
    # もっともそれらしい単語を取ってくる
    word_id = numpy.argmax(cuda.to_cpu(next_y.data))
    # EOSが出力されたら終了
    if word_id == EOS_ID:
        break
    print(dataset.japanese_word(word_id))
    next_y = model(word_id)
