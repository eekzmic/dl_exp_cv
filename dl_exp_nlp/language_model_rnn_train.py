#!/usr/bin/env python3

import sys
import chainer
from chainer import optimizers, cuda

import sentence_data
from sentence_data import EOS_ID
from language_model_rnn import LanguageModelRNN

dataset = sentence_data.SentenceData("dataset/data_1000.txt")

model = LanguageModelRNN(dataset.japanese_word_size())

optimizer = optimizers.Adam()
optimizer.setup(model)

epoch_num = 10
for epoch in range(epoch_num):
    print("{0} / {1} epoch start.".format(epoch + 1, epoch_num))

    sum_loss = 0.0
    for i, sentence in enumerate(dataset.japanese_sentences()):
        model.reset_state()
        model.zerograds()
        accum_loss = None
        # 文の1単語目を入力して出力された2単語目，
        # 文の1単語目と2単語目を入力して出力された3単語目，のように，
        # 文の1～n-1単語目を入力して出力されたn単語目を全て確認して，
        # accum_lossに加算する
        # 最後の単語を入力した後，EOSが正しく出力されるかどうかも確認する
        for cur_word, next_word in zip(sentence, sentence[1:] + [EOS_ID]):
            loss = model.loss(cur_word, next_word)
            accum_loss = loss if accum_loss is None else accum_loss + loss
        accum_loss.backward()
        accum_loss.unchain_backward()
        optimizer.update()
        sum_loss += float(cuda.to_cpu(accum_loss.data))

        if (i + 1) % 100 == 0:
            print("{0} / {1} sentences finished.".format(
                i + 1, dataset.sentences_size()))

    print("mean loss = {0}.".format(sum_loss / dataset.sentences_size()))

    # 1 epoch 毎にファイルに書き出す
    model_file = "trained_model/langage_model_rnn_" + str(epoch + 1) + ".model"
    model.save_model(model_file)
