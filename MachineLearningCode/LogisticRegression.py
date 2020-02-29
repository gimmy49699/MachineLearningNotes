'''
    Implementing of Logistic Regression
'''
import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from datas import WatermelonDatas


class MyLogisticRegression(object):
    """docstring for MyLogisticRegression"""
    def __init__(self, data_x, data_y):
        super(MyLogisticRegression, self).__init__()
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
        res = []
        for x in inputs:
            x = np.reshape(x, (1, self.feature_num))
            res.append(np.rint(self.sigmoid(np.matmul(x, self.W))))
        return np.array(res)


if __name__ == "__main__":
    # Load Watermelon Datasets
    data = WatermelonDatas()

    # Bulid My Logistic Regression Model
    Model = MyLogisticRegression(data.wl3a_x, data.wl3a_y)
    # Training model with watermelon datasets
    Model.train(epochs=1000, lr=0.01)
    print('My Model: ', Model.predict(inputs=[[0.481, 0.150],]))

    # Sci-kit Logisitic Regression Model
    sciModel = LogisticRegression(max_iter=1000)
    sciModel.fit(data.wl3a_x, data.wl3a_y)
    # 
    print('Sci-kit Model: ', sciModel.predict(X=([[0.481, 0.150],])))

    # Visualization of decision boundary
    x0, x1 = np.meshgrid(
            np.linspace(0, 1, 100).reshape(-1, 1),
            np.linspace(0, 1, 100).reshape(-1, 1))
    X_new = np.c_[x0.ravel(), x1.ravel()]

    # My Logistic Regression
    p1 = plt.figure(num="MyLogisticRegression")
    plt.title("My Logistic Regression")
    my_pred = Model.predict(X_new)
    mzz = my_pred.reshape(x0.shape)
    plt.contourf(x0, x1, mzz)
    plt.scatter(data.wl3a_x[:8,0], data.wl3a_x[:8,1], c='b')
    plt.scatter(data.wl3a_x[8:,0], data.wl3a_x[8:,1], c='r')
    plt.show()

    # Sci-Kit Logistic Regression
    p2 = plt.figure(num="ScikitLogisticRegression")
    plt.title("Scikit Logistic Regression")
    y_pred = sciModel.predict(X_new)
    zz = y_pred.reshape(x0.shape)
    plt.contourf(x0, x1, zz)
    plt.scatter(data.wl3a_x[:8,0], data.wl3a_x[:8,1], c='b')
    plt.scatter(data.wl3a_x[8:,0], data.wl3a_x[8:,1], c='r')
    plt.show()