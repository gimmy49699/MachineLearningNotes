'''
    Implementing of Linear Dscriminant Analysis(LDA).
    LDA is a method of reducing the feature dimentions.
    The main idea is to project each feature to a plain
     which can maximize difference among categories and
     minimize the difference in categories.
'''
import numpy as np
import matplotlib.pyplot as plt

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from datas import WatermelonDatas


class MyLinearDiscriminantAnalysis(object):
    """docstring for LinearDiscriminantAnalysis"""
    def __init__(self):
        super(MyLinearDiscriminantAnalysis, self).__init__()

    def _data(self, data_x, data_y):
        self.c1_data_x, self.c2_data_x = data_x[:8], data_x[8:]
        self.c1_num, self.n_feature = self.c1_data_x.shape
        self.c2_num, _ = self.c2_data_x.shape

    def _cal_average(self, data):
        return np.reshape(np.mean(data, axis=0), (-1, 1))

    def _cal_covmat(self, data, miu):
        n = len(data)
        cov = data - miu.T
        return np.matmul(cov.T, cov)/n

    def _cal_Sw(self, *args):
        Sw = np.empty(args[0].shape)
        for s in args:
            Sw += s
        return Sw

    def _cal_Sb(self, miu1, miu2):
        avg_miu = miu1-miu2
        return np.matmul(avg_miu, avg_miu.T)

    def _cal_inv(self, data):
        return np.linalg.inv(data)

    def _cal_projection_matrix(self):
        c1_miu, c2_miu = self._cal_average(self.c1_data_x), self._cal_average(self.c2_data_x)
        c1_cov, c2_cov = self._cal_covmat(self.c1_data_x, c1_miu), self._cal_covmat(self.c2_data_x, c2_miu)
        Sb = self._cal_Sb(c1_miu, c2_miu)
        Sw = self._cal_Sw(c1_cov, c2_cov)
        inv_Sw = self._cal_inv(Sw)
        self.best_w = np.matmul(inv_Sw, (c1_miu-c2_miu))
        self.eigen, self.projection_matrix = np.linalg.eig(np.matmul(inv_Sw, Sb))

    def _cal_1dprojection(self, data):
        return np.matmul(data, self.best_w)

    def _plot_original(self):
        p1 = plt.figure(num="Original Distribution")
        plt.title("Original Distribution")
        plt.scatter(self.c1_data_x[:, 0], self.c1_data_x[:, 1], c='b')
        plt.scatter(self.c2_data_x[:, 0], self.c2_data_x[:, 1], c='r')
        plt.show()

    def _plot_2dprojection(self):
        new_c1 = np.matmul(self.c1_data_x, self.projection_matrix)
        new_c2 = np.matmul(self.c2_data_x, self.projection_matrix)
        p2 = plt.figure(num="New Distribution")
        plt.title("New Distribution")
        plt.scatter(new_c1[:, 0], new_c1[:, 1], c='b')
        plt.scatter(new_c2[:, 0], new_c2[:, 1], c='r')
        plt.show()

    def _plot_1dprojection(self):
        new_c1 = np.matmul(self.c1_data_x, self.best_w)
        new_c2 = np.matmul(self.c2_data_x, self.best_w)
        p2 = plt.figure(num="New Distribution")
        plt.title("New Distribution")
        plt.scatter(new_c1, np.zeros(len(new_c1)), c='b')
        plt.scatter(new_c2, np.zeros(len(new_c2)), c='r')
        plt.show()


if __name__ == '__main__':
    # Load Watermelon Datasets
    Data = WatermelonDatas()

    # Bulid My Linear Discriminant Analysis Model
    Model = MyLinearDiscriminantAnalysis()
    # process datas
    Model._data(Data.wl3a_x, Data.wl3a_y)
    # calculate the projection matrix and best projection vector
    Model._cal_projection_matrix()
    # plot the original data distribution
    Model._plot_original()
    # polt the new projections in 2 dimensions
    # Model._plot_2dprojection()
    # plot the new projections in 1 dimensions
    Model._plot_1dprojection()

    # Sci-kit Linear Discriminant Analysis Model
    sciLDA = LinearDiscriminantAnalysis(n_components=1)
    new_data = sciLDA.fit_transform(Data.wl3a_x, Data.wl3a_y)
    new = plt.figure("Sci-kit LDA")
    plt.title("Sci-kit LDA")
    plt.scatter(new_data[:8], np.zeros(8), c='b')
    plt.scatter(new_data[8:], np.zeros(9), c='r')
    plt.show()