import argparse

import matplotlib

matplotlib.use('Agg')
import chainer
from net import CifarCNN
from dataset import MyCifarDataset

def main():
    parser = argparse.ArgumentParser(description='Practice: Cifar10')
    parser.add_argument('--batchsize', '-b', type=int, default=100,
                        help='Number of images in each mini-batch')
    parser.add_argument('--gpu', '-g', type=int, default=-1,
                        help='GPU ID (negative value indicates CPU)')
    parser.add_argument('--model', '-m', default='result/model_20',
                        help='Path to the model')
    parser.add_argument('--dataset', '-d', default='mini_cifar/test',
                        help='Directory for train mini_cifar')
    args = parser.parse_args()

    print('GPU: {}'.format(args.gpu))
    print('')

    model = CifarCNN(10)
    chainer.serializers.load_npz(args.model, model)

    if args.gpu >= 0:
        # Make a specified GPU current
        chainer.cuda.get_device_from_id(args.gpu).use()
        model.to_gpu()  # Copy the model to the GPU

    # Load the Cifar-10 mini_cifar
    # trainとvalに分ける
    test = MyCifarDataset(args.dataset)
    print('test data : {}'.format(len(test)))

    test_iter = chainer.iterators.SerialIterator(test, args.batchsize,
                                                  repeat=False, shuffle=False)

    correct_cnt = 0
    while True:
        try:
            batch = test_iter.next()
        except StopIteration:
            break
        images = model.xp.array([image for image, _ in batch])
        labels = model.xp.array([label for _, label in batch])
        with chainer.using_config('train', False), chainer.no_backprop_mode():
            predicts = model.predict(images)
            for l, p in zip(labels, predicts):
                if l == p:
                    correct_cnt += 1

    print('accuracy : {}'.format(correct_cnt/len(test)))


if __name__ == '__main__':
    main()