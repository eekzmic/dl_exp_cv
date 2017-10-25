#!/usr/bin/env python3

import sys
import chainer
from chainer import optimizers, cuda

import sentence_data
from sentence_data import EOS_ID
from translator_model import TranslatorModel

dataset = sentence_data.SentenceData("dataset/data_1000.txt")

model = TranslatorModel(dataset.english_word_size(),
                        dataset.japanese_word_size())

optimizer = optimizers.Adam()
optimizer.setup(model)

epoch_num = 10
for epoch in range(epoch_num):
    print("{0} / {1} epoch start.".format(epoch + 1, epoch_num))

    sum_loss = 0.0
    for i, (english_sentence, japanese_sentence) in enumerate(
            zip(dataset.english_sentences(), dataset.japanese_sentences())):

        model.reset_state()
        model.zerograds()
        loss = model.loss(english_sentence, japanese_sentence)
        loss.backward()
        loss.unchain_backward()
        optimizer.update()
        sum_loss += float(cuda.to_cpu(loss.data))

        if (i + 1) % 100 == 0:
            print("{0} / {1} sentences finished.".format(
                i + 1, dataset.sentences_size()))

    print("mean loss = {0}.".format(sum_loss / dataset.sentences_size()))

    # 1 epoch 毎にファイルに書き出す
    modelfile = "trained_model/translator_" + str(epoch + 1) + ".model"
    model.save_model(modelfile)
