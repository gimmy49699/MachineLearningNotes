'''
    Implementing of K-Nearest Neighbor.
'''
import numpy as np

from datas import WatermelonDatas
from sklearn.neighbors import KNeighborsClassifier

label = {0: "坏瓜", 1: "好瓜"}


class MyKNN(object):
    """docstring for MyKNN"""
    def __init__(self, datax, datay, k, labelsize):
        super(MyKNN, self).__init__()
        self.k = k
        self.data = np.concatenate((datax, datay.reshape(-1, 1)), axis=1)
        self.n_sample = datax.shape[0]
        self.n_feature = datax.shape[1]
        self.labelsize = labelsize

    def _cal_dist(self, x1, x2, fn='p-norm', p=2):
        '''
            Calculate the distance between two vectors
            Input:
                x1: [1, n] - array.
                x2: [1, n] - array.
                fn: method of calculating the distance.
                    default 'p-norm', can be 'p-norm' and 'cos'.
                d: p of 'p-norm', default 2, i.e., 2-norm.
        '''
        x1, x2 = np.array(x1), np.array(x2)
        if fn == 'p-norm':
            return np.linalg.norm((x1-x2), ord=p)
        elif fn == 'cos':
            molecular = np.matmul(x1, x2.T)
            x1_norm = np.linalg.norm(x1, ord=2)
            x2_norm = np.linalg.norm(x2, ord=2)
            return molecular / (x1_norm + x2_norm)

    def _cal_dist_all(self, x1, fn='p-norm'):
        dist_all = []
        for idx in range(self.n_sample):
            dist_all.append(self._cal_dist(x1, self.data[idx][:self.n_feature], fn=fn))
        return dist_all

    def _cal_neighbor(self, dist_all):
        res = [int(np.argwhere(np.argsort(dist_all)==i)) for i in range(self.k)]
        return res

    def _vote(self, neighbor, labelsize):
        voteres = [0]*labelsize
        for idx in neighbor:
            voteres[int(self.data[idx][-1])] += 1
        return np.argmax(voteres)

    def _predict(self, x1):
        dist_all = self._cal_dist_all(x1)
        neighbors = self._cal_neighbor(dist_all)
        res = self._vote(neighbors, self.labelsize)
        msg = 'Test Sample Feature: < ' + ' , '.join([str(s) for s in x1]) + ' >'
        print(msg)
        print('My KNN predicted label: '+ label[res])


if __name__ == '__main__':
    # define k neighbors
    n_neighbors = 5
    # test sample
    test_data = [0.225, 0.665]
    # Load watermelon dataset
    Data = WatermelonDatas()
    # Build my K-Nearest Neighbors
    MyModel = MyKNN(Data.wl3a_x, Data.wl3_y, k=n_neighbors, labelsize=2)
    MyModel._predict(test_data)
    # Sci-kit K-Nearest Neighbors
    skKNN = KNeighborsClassifier(n_neighbors=n_neighbors)
    skKNN.fit(Data.wl3a_x, Data.wl3_y)
    skKNNres = skKNN.predict(np.array(test_data).reshape(1, -1))
    print('Sci-kit KNN predicted label: '+ label[skKNNres[0]])