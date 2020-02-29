'''
    Implementing of Logistic Regression
'''
import numpy as np
from datas import WatermelonDatas


class LogisticRegression(object):
    """docstring for LogisticRegression"""
    def __init__(self, data_x, data_y):
        super(LogisticRegression, self).__init__()
        self.data_x = data_x
        self.data_y = data_y
        self.sample_num, self.feature_num = self.data_x.shape

    def _weights(self, size):
        return np.random.normal(loc=0.0, scale=0.025, size=size)

    def sigmoid(self, z):
        H = 1 / (np.exp(-z) + 1)
        return H

    def loss(self, y_true, y_pred):
        cost = - y_true * np.log(y_pred) -\
             (1 - y_true)*np.log(1 - y_pred)
        return cost

    def calGradient(self, x, y_pred, y_true):
        grad = np.matmul(x.T, (y_pred - y_true))
        return grad

    def accuracy(self, y_true, y_pred):
        res = (np.reshape(y_true, (-1, 1)) == np.reshape(y_pred, (-1, 1)))
        return float(sum(res)/len(res))

    def train(self, epochs, lr):

        # Initialize Weight Matrix
        self.W = self._weights(size=(self.feature_num, 1))

        for epoch in range(1, epochs+1):
            loss = []
            preds = []
            for _,(x, y) in enumerate(zip(self.data_x, self.data_y)):
                # Reshape the data X and Y
                x = np.reshape(x, (1, self.feature_num))
                y = np.reshape(y, (1, 1))
                # cal Z = XW
                z = np.matmul(x, self.W)
                # cal h = sigmoid(z) = 1 / (exp(-z) + 1)
                h = self.sigmoid(z)
                preds.append(np.rint(h))
                # cal loss and gradient
                loss.append(self.loss(y, h))
                grad = self.calGradient(x, h, y)
                # updata Weight Matrix
                self.W -= lr * grad
            acc = self.accuracy(self.data_y, preds)
            showmsg = 'Epoch:{:>3d} - Loss:{:>.3f} - acc:{:>.3f}'.format(epoch, np.mean(loss), acc)
            print(showmsg)

    def predict(self, inputs):
        inputs = np.reshape(np.array(inputs), (1, self.feature_num))
        res = np.rint(self.sigmoid(np.matmul(inputs, self.W)))
        if res == 1:
            print('好瓜！')
        else:
            print('坏瓜！')

data = WatermelonDatas()
Model = LogisticRegression(data.wl3a_x, data.wl3a_y)
Model.train(epochs=500, lr=0.1)
Model.predict(inputs=[0.75, 0.26])