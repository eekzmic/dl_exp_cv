#!/usr/bin/env python3
import matplotlib
matplotlib.use('Agg')
import sys
import chainer
from chainer import optimizers, cuda
from chainer import training
from chainer.training import extensions
import numpy as np

import sentence_data
from sentence_data import EOS_ID
from language_model_lstm import LanguageModelLSTM
from language_model_lstm import PADDING

gpu_id = 0
epoch = 10
batchsize = 10
out = 'result'
resume_npz = ''
model_npz = ''

class LangDataset(chainer.dataset.DatasetMixin):
    def __init__(self, sentences):
        super(LangDataset, self).__init__()
        self.sentences = sentences
        self.max_sentence_words = self.max_sentence_size()

    def __len__(self):
        return len(self.sentences)

    def max_sentence_size(self):
        return max([len(s) for s in self.sentences])

    def get_example(self, i):
        source_sentence = self.sentences[i]
        target_sentence = source_sentence[1:]
        source_sentence.extend(
            [PADDING] * (self.max_sentence_words - len(source_sentence)))
        target_sentence.extend(
            [PADDING] * (self.max_sentence_words - len(target_sentence)))
        return np.array(source_sentence, np.int32).reshape((-1, 1)),\
               np.array(target_sentence, np.int32).reshape((-1, 1))

if __name__ == '__main__':
    dataset = sentence_data.SentenceData("dataset/data_1000.txt")
    japanese_dataset = LangDataset(dataset.japanese_sentences())
    print('{} japanese sentences found'.format(len(japanese_dataset)))

    model = LanguageModelLSTM(dataset.japanese_word_size())
    if gpu_id >= 0:
        cuda.get_device_from_id(gpu_id).use()
        model.to_gpu()

    optimizer = optimizers.Adam()
    optimizer.setup(model)

    train_iter = chainer.iterators.SerialIterator(
        japanese_dataset, batchsize, repeat=True, shuffle=True)


    updater = training.StandardUpdater(train_iter, optimizer, device=gpu_id)
    trainer = training.Trainer(updater, (10, 'epoch'), out='result')

        # Evaluate the model with the test mini_cifar for each epoch
    trainer.extend(extensions.dump_graph('main/loss'))

    trainer.extend(extensions.snapshot(filename='snapshot_{.updater.epoch}'), trigger=(1, 'epoch'))
    trainer.extend(extensions.snapshot_object(model, 'model_{.updater.epoch}'), trigger=(1, 'epoch'))


    trainer.extend(extensions.LogReport())
    if extensions.PlotReport.available():
        trainer.extend(
            extensions.PlotReport(['main/loss'], 'epoch', file_name='loss.png'))

    trainer.extend(extensions.PrintReport(
        ['epoch', 'iteration', 'main/loss', 'elapsed_time']))

    trainer.extend(extensions.ProgressBar())

    if resume_npz:
        chainer.serializers.load_npz(resume_npz, trainer)

    with chainer.using_config('train', True):
        trainer.run()