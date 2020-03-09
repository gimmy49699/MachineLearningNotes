'''
    Implementing of K-means clustering.
'''
import numpy as np
import matplotlib.pyplot as plt

from datas import WatermelonDatas
from sklearn.cluster import KMeans


clr = {'c1': 'r', 'c2': 'b', 'c3': 'g', 'c4': 'k', 'c5': 'y'}


class MyKmeans(object):
    """docstring for MyKmeans"""
    def __init__(self, data, k):
        super(MyKmeans, self).__init__()
        self.data = data
        self.n_sample = self.data.shape[0]
        self.n_feature = self.data.shape[1]
        self.k = k

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
        if fn == 'p-norm':
            return np.linalg.norm((x1-x2), ord=p)
        elif fn == 'cos':
            molecular = np.matmul(x1, x2.T)
            x1_norm = np.linalg.norm(x1, ord=2)
            x2_norm = np.linalg.norm(x2, ord=2)
            return molecular / (x1_norm + x2_norm)
        else:
            raise NameError("Invalid fn! 'p-norm' or 'cos' only!!! ")

    def _cal_centroids(self, c):
        '''
            Calculate the centroid of class c by x_avg = (1/n)∑x_i.
        '''
        if not c:
            return np.zeros(shape=(self.n_feature))
        return np.mean(c, axis=0)

    def _fit(self, maxiter, data):
        # randomly choose k cluster centroids
        self.centroids = {}
        init = np.random.choice(self.n_sample, size=self.k, replace=False)
        for idx in range(self.k):
            self.centroids["c"+str(idx+1)] = data[init[idx]].tolist()
        # K-means algorithm
        itertimes = 0
        while itertimes < maxiter:
            print('itertimes: ', itertimes + 1)
            # initialize k-clusters
            self.classes = {}
            for idx in range(self.k):
                self.classes["c"+str(idx+1)] = []
            # calculate distance between (d, ci), i = 1, 2, ..., k;
            for d in data:
                tmp = [0]*self.k
                for c, avg in self.centroids.items():
                    tmp[int(c[-1])-1] = self._cal_dist(d, avg)
                b2c = np.argmin(tmp)
                self.classes["c"+str(b2c+1)].append(d.tolist())
            # update centroids of k class;
            tmp_centroids = [0]*self.k
            for c, cd in self.classes.items():
                tmp_centroids[int(c[-1])-1] = self._cal_centroids(cd).tolist()
            # stop condition
            stop_condition = [0]*self.k
            for c, avg in self.centroids.items():
                if tmp_centroids[int(c[-1])-1] == avg:
                    stop_condition[int(c[-1])-1] = 1
            if sum(stop_condition) == self.k:
                break
            # update centroids
            for c, avg in self.centroids.items():
                self.centroids[c] = tmp_centroids[int(c[-1])-1]
            itertimes += 1

    def _plot_k_class(self):
        pic = plt.figure(num="K-means classes")
        plt.title(str(self.k) + " classes by K-means")
        for c, cd in self.classes.items():
            cd = np.array(cd)
            plt.scatter(cd[:, 0], cd[:, 1], c=clr[c], label='Class'+c[-1])
        cen = np.array([cavg for c, cavg in self.centroids.items()])
        plt.scatter(cen[:, 0], cen[:, 1], c='k', marker='X', label='centroids')
        plt.legend()
        plt.show()


if __name__ == "__main__":
    # Load water melon dataset
    Data = WatermelonDatas()
    # Build My K-means model
    MyModel = MyKmeans(Data.wl3a_x, k=3)
    # Clustering process
    MyModel._fit(maxiter=100, data=MyModel.data)
    # Visualization
    MyModel._plot_k_class()

    # Sci-kit K-means model
    n_clusters = 3
    skKmeans = KMeans(n_clusters=n_clusters)
    skKmeans.fit(Data.wl3a_x)
    # clustering result
    y = skKmeans.fit_predict(Data.wl3a_x)
    out = [[] for _ in range(n_clusters)]
    for idx, (d, l) in enumerate(zip(Data.wl3a_x, y)):
        out[int(l)].append(d.tolist())
    # Visualization
    pic = plt.figure(num="Sci-kit K-means classes")
    plt.title(str(n_clusters) + " classes by Sci-kit K-means")
    for idx in range(n_clusters):
        outd = np.array(out[idx])
        plt.scatter(outd[:, 0], outd[:, 1], c=clr["c"+str(idx+1)], label='Class'+str(idx+1))
    plt.scatter(skKmeans.cluster_centers_[:, 0], skKmeans.cluster_centers_[:, 1], c='k', marker='X', label='centroids')
    plt.legend()
    plt.show()