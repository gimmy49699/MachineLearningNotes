'''
    Implementing of Neural Network.
    Basic neural network, one input layer, one hidden layer and one output layer.
    Loss function ∑-ylog(y')-(1-y)log(1-y'), activation function g(x) = 1/(1+exp(-x)); 
'''
import sklearn
import keras
import numpy as np
np.random.seed(0)
import matplotlib.pyplot as plt

from sklearn.datasets import make_moons


class MyNeuralNetwork(object):
    """docstring for MyNeuralNetwork"""
    def __init__(self, data, output_num, hidden_layer, hidden_nums, epoch, batch=1, lr=0.01):
        super(MyNeuralNetwork, self).__init__()
        self.data_x, self.data_y = data
        self.output_num = output_num
        self.hidden_layer = hidden_layer
        self.hidden_nums = hidden_nums
        self.epoch = epoch
        self.batch = batch
        self.num_batch = self.data_x.shape[0] // self.batch
        self.lr = lr

    def _weights(self, size):
        return np.random.normal(loc=0.0, scale=0.025, size=size)

    def _bias(self, size):
        return np.zeros(size)

    def sigmoid(self, z):
        H = 1 / (np.exp(-z) + 1)
        return H

    def loss(self, y_true, y_pred):
        cost = - y_true * np.log(y_pred) -\
             (1 - y_true)*np.log(1 - y_pred)
        return np.mean(cost)

    def _caldiff(self, y_pred, y_true):
        return y_pred - y_true

    def accuracy(self, y_true, y_pred):
        res = (np.reshape(y_true, (-1, 1)) == np.reshape(y_pred, (-1, 1)))
        return float(sum(res)/len(res))

    def _build(self):
        self.model = {}
        n_all, n_f = self.data_x.shape
        self.model["Wx"] = self._weights(size=[self.batch, n_f, self.hidden_nums[0]])
        self.model["bx"] = self._bias(size=[self.batch, 1, self.hidden_nums[0]])
        for l in range(1, self.hidden_layer):
            self.model["W_h"+str(l)] = self._weights(size=[self.batch, self.hidden_nums[l-1], self.hidden_nums[l]])
            self.model["b_h"+str(l)] = self._bias(size=[self.batch, 1, self.hidden_nums[l]])
        self.model["W_h"+str(self.hidden_layer)] = self._weights(size=[self.batch, self.hidden_nums[-1], self.output_num])
        self.model["b_h"+str(self.hidden_layer)] = self._bias(size=[self.batch, 1, self.output_num])

    def _forward(self, data_x, data_y):
        n_out = data_y.shape[-1]
        self.forward = {}
        self.forward["h1"] = np.tanh(np.matmul(data_x, self.model["Wx"]) + self.model["bx"])
        # self.forward["h1"] = self.sigmoid(np.matmul(data_x, self.model["Wx"]) + self.model["bx"])
        for i in range(1, self.hidden_layer):
            self.forward["h"+str(i+1)] = np.tanh(np.matmul(self.forward["h"+str(i)], self.model["W_h"+str(i)]) + self.model["b_h"+str(i)])
            # self.forward["h"+str(i+1)] = self.sigmoid(np.matmul(self.forward["h"+str(i)], self.model["W_h"+str(i)]) + self.model["b_h"+str(i)])
        self.forward["O"] = self.sigmoid(np.reshape(np.matmul(self.forward["h"+str(self.hidden_layer)], \
            self.model["W_h"+str(self.hidden_layer)]) + self.model["b_h"+str(self.hidden_layer)], (-1, self.output_num)))
        if self.forward["O"].shape[-1] != n_out:
            raise ValueError("Output dim must be the same with input label!")

    def _backward(self, data, data_y):
        self.backward = {}
        self.backward["diff_"+str(self.hidden_layer)] = self._caldiff(self.forward["O"], data_y).reshape(-1, 1, self.output_num)
        self.backward["delta_W_h"+str(self.hidden_layer)] = np.matmul(np.transpose(self.forward["h"+str(self.hidden_layer)], [0, 2, 1]), self.backward["diff_"+str(self.hidden_layer)])
        self.backward["delta_b_h"+str(self.hidden_layer)] = self.backward["diff_"+str(self.hidden_layer)]
        for i in range(self.hidden_layer-1, 0, -1):
            tmp_1 = np.matmul(self.backward["diff_"+str(i+1)], np.transpose(self.model["W_h"+str(i+1)], [0, 2, 1]))
            tmp_2 = 1 - np.tanh(self.forward["h"+str(i+1)])**2
            # tmp_2 = np.multiply(self.forward["h"+str(i+1)], (1-self.forward["h"+str(i+1)]))
            self.backward["diff_"+str(i)] = np.multiply(tmp_1, tmp_2)
            self.backward["delta_W_h"+str(i)] = np.matmul(np.transpose(self.forward["h"+str(i)], [0, 2, 1]), self.backward["diff_"+str(i)])
            self.backward["delta_b_h"+str(i)] = self.backward["diff_"+str(i)]
        self.backward["diff_0"] = np.multiply(np.matmul(self.backward["diff_1"], np.transpose(self.model["W_h1"], [0, 2, 1])),
                                              1 - np.tanh(self.forward["h1"])**2)
                                              # np.multiply(self.forward["h1"], (1-self.forward["h1"])))

        self.backward["delta_Wx"] = np.matmul(np.transpose(data, [0, 2, 1]), self.backward["diff_0"])
        self.backward["delta_bx"] = self.backward["diff_0"]
    
    def train(self, data_x, data_y):
        data_x = data_x[:, np.newaxis, :]
        data_y = data_y[:, np.newaxis]
        # Building fully connected neural network, initialize weight matrix.
        self._build()
        # Starting training.
        for epoch in range(1, self.epoch+1):
            _loss, _acc = [], []
            for batch in range(1, self.num_batch+1):
                x = data_x[(batch-1)*self.batch: batch*self.batch]
                y = data_y[(batch-1)*self.batch: batch*self.batch]
                # Forward calculating
                self._forward(x, y)
                # Backward calculating
                self._backward(x, y)
                # Updating weigth matrix
                for k, v in self.backward.items():
                    if k.startswith("delta"):
                        self.model[k[6:]] -= self.lr * v
                # show training process
                _acc.append(self.accuracy(y, np.rint(self.forward["O"])))
                _loss.append(self.loss(y, self.forward["O"]))
                showmsg = "Epoch:{:>2d} - Loss:{:>.3f} - Acc:{:>.3f} \r".format(epoch, np.mean(_loss), np.mean(_acc))
                # print(showmsg, end="")
            showmsg = "Epoch:{:>2d} - Loss:{:>.3f} - Acc:{:>.3f}".format(epoch, np.mean(_loss), np.mean(_acc))
            print(showmsg)
        # print(self.model)

    def predict(self, data_x):
        self.predict = {}
        self.predict["h1"] = np.tanh(np.matmul(data_x, self.model["Wx"]) + self.model["bx"])
        # self.predict["h1"] = self.sigmoid(np.matmul(data_x, self.model["Wx"]) + self.model["bx"])
        for i in range(1, self.hidden_layer):
            self.predict["h"+str(i+1)] = np.tanh(np.matmul(self.predict["h"+str(i)], self.model["W_h"+str(i)]) + self.model["b_h"+str(i)])
            # self.predict["h"+str(i+1)] = self.sigmoid(np.matmul(self.predict["h"+str(i)], self.model["W_h"+str(i)]) + self.model["b_h"+str(i)])
        self.predict["O"] = self.sigmoid(np.reshape(np.matmul(self.predict["h"+str(self.hidden_layer)], \
            self.model["W_h"+str(self.hidden_layer)]) + self.model["b_h"+str(self.hidden_layer)], (-1, self.output_num)))
        return np.rint(self.predict["O"])


if __name__ == "__main__":
    # Create Datsets.
    dX, dY = make_moons(200, noise=0.20)

    # Build My Neural Network Model
    # 2 hidden layers with untis of 8 and units of 4;
    # activation function is Sigmoid;
    # training 200 epochs with batch size 1.
    MyNN = MyNeuralNetwork(data=(dX, dY), output_num=1, hidden_layer=2, hidden_nums=[8, 4], epoch=200, batch=1)
    # Start training
    MyNN.train(MyNN.data_x, MyNN.data_y)

    # Visualization of my model decision boundary
    x0, x1 = np.meshgrid(
            np.linspace(-1.5, 2.5, 100).reshape(-1, 1),
            np.linspace(-1.5, 2.5, 100).reshape(-1, 1))
    X_new = np.c_[x0.ravel(), x1.ravel()]
    p1 = plt.figure(num="MyNeuralNetwork")
    plt.title("My Neural Network")
    my_pred = MyNN.predict(X_new)
    mzz = my_pred.reshape(x0.shape)
    plt.contourf(x0, x1, mzz)
    plt.scatter(dX[:, 0], dX[:, 1], s=40, c=dY, cmap=plt.cm.binary)
    plt.show()

    # Keras basic fully connected Neural Netwrok.
    # Set the hyper-parameters as the same as my NN model
    model = keras.models.Sequential()
    model.add(keras.layers.Dense(units=8, activation='tanh', input_dim=2))
    model.add(keras.layers.Dense(units=4, activation='tanh'))
    model.add(keras.layers.Dense(units=1, activation='sigmoid'))
    sgd = keras.optimizers.SGD(lr=0.1)
    model.compile(loss='binary_crossentropy', optimizer=sgd, metrics=['acc'])
    model.fit(dX, dY, epochs=200)

    # Visualization of decision boundary
    p2 = plt.figure(num="KerasNeuralNetwork")
    plt.title("Keras Neural Network")
    pred = np.rint(model.predict(X_new))
    zz = pred.reshape(x0.shape)
    plt.contourf(x0, x1, zz)
    plt.scatter(dX[:, 0], dX[:, 1], s=40, c=dY, cmap=plt.cm.binary)
    plt.show()
