import chainer
from chainer import links as L
from chainer import functions as F


class MLP(chainer.Chain):
    def __init__(self, n_units, n_out):
        super(MLP, self).__init__()
        with self.init_scope():
            self.l1 = L.Linear(None, n_units)  # n_in -> n_units
            self.l2 = L.Linear(None, n_units)  # n_units -> n_units
            self.l3 = L.Linear(None, n_out)  # n_units -> n_out

    def __call__(self, x, t=None):
        h = F.relu(self.l1(x))
        h = F.relu(self.l2(h))
        h = self.l3(h)

        loss = F.softmax_cross_entropy(h, t)
        accuracy = F.accuracy(h, t)
        chainer.report({'loss': loss}, self)
        chainer.report({'accuracy': accuracy}, self)

        if chainer.config.train:
            return loss
        else:
            return h

    def predict(self, x):
        h = F.relu(self.l1(x))
        h = F.relu(self.l2(h))
        h = self.l3(h)
        return h


class MnistCNN(chainer.Chain):
    def __init__(self, n_out):
        super(MnistCNN, self).__init__()
        with self.init_scope():
            self.conv1 = ('畳み込み層を定義してね')
            self.conv2 = ('畳み込み層を定義してね')
            self.l_out = L.Linear(10 * 7 * 7, n_out)

    def __call__(self, x, t):
        h = 'xのshapeを(len(x), 784)から(len(x), 1, 28, 28)に変形してね'
        h = F.relu(self.conv1(h))
        h = F.relu(self.conv2(h))
        h = self.l_out(h)

        loss = F.softmax_cross_entropy(h, t)
        accuracy = F.accuracy(h, t)
        chainer.report({'loss': loss}, self)
        chainer.report({'accuracy': accuracy}, self)

        if chainer.config.train:
            return loss
        else:
            return h


class CifarCNN(chainer.Chain):
    def __init__(self, n_out):
        super(CifarCNN, self).__init__()
        with self.init_scope():
            self.model = L.VGG16Layers()
            self.l_out = L.Linear(None, n_out)

    def __call__(self, x, t):
        with chainer.using_config('train', False), chainer.no_backprop_mode():
            h = self.model(x, layers=['pool5'])['pool5']
        h = self.l_out(h)

        t = self.xp.asarray(t, self.xp.int32)
        loss = F.softmax_cross_entropy(h, t)
        accuracy = F.accuracy(h, t)
        chainer.report({'loss': loss}, self)
        chainer.report({'accuracy': accuracy}, self)

        if chainer.config.train:
            return loss
        else:
            return h

    def predict(self, x):
        h = self.model(x, layers=['pool5'])['pool5']
        h = self.l_out(h)
        predicts = F.argmax(h, axis=1)
        return predicts.data
