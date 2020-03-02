'''
    Implementing of Principal Component Analysis(PCA).
    PCA is an unsuperwised method of reducing dimensions.
    The main idea is to find the most valueable features of the samples.
    By decomposing the covariance matrix, PCA give a projection matrix
    to reduce the feature dimensions of the datas.
'''
import numpy as np
import matplotlib.pyplot as plt

from datas import WatermelonDatas
from sklearn.decomposition import PCA


class MyPrincipalComponentAnalysis(object):
    """docstring for MyPrincipalComponentAnalysis"""
    def __init__(self, data):
        super(MyPrincipalComponentAnalysis, self).__init__()
        self.data = data

    def _centralized(self, data):
        n_sample, n_feature = data.shape
        data = data.T
        cen_data = data - np.mean(data, axis=1).reshape(-1, 1)
        return cen_data

    def _cal_covmat(self, data):
        _, n_sample = data.shape
        return np.matmul(data, data.T)/n_sample

    def _decompose_by_eigen(self, k):
        centeralized_data = self._centralized(self.data)
        covmat = self._cal_covmat(centeralized_data)
        eigen, matrix = np.linalg.eig(covmat)
        sort_index = np.argsort(eigen)
        return eigen[sort_index[:-k-1:-1]], matrix[sort_index[:-k-1:-1]]

    def _decompose_by_svd(self, k):
        centeralized_data = self._centralized(self.data)
        covmat = self._cal_covmat(centeralized_data)
        u, s, vh = np.linalg.svd(covmat)
        sort_index = np.argsort(s)
        return s[sort_index[:-k-1:-1]], u[sort_index[:-k-1:-1]]

    def _new_data(self, matrix, data):
        return np.matmul(matrix, data.T)

    def _plot_1dprojection(self, matrix, data, name):
        data = self._new_data(matrix, data).T
        pic = plt.figure(name)
        plt.title(name)
        plt.scatter(data[:8], np.zeros(len(data[:8])), c='b')
        plt.scatter(data[8:], np.zeros(len(data[8:])), c='r')
        plt.show()


if __name__ == '__main__':
    # Load Watermelon Datasets
    Data = WatermelonDatas()

    # Bulid Principal Component Analysis Model
    Model = MyPrincipalComponentAnalysis(Data.wl3a_x)
    # PCA by Eigenvalue Decomposition
    eigin, eigin_projmat = Model._decompose_by_eigen(1)
    # PCA by Singular Value Decomposition
    s, svd_projmat = Model._decompose_by_svd(1)

    # Visualization - original
    new = plt.figure("Original Distribution")
    plt.title("Original Distribution")
    plt.scatter(Data.wl3a_x[:8, 0], Data.wl3a_x[:8, 1], c='b')
    plt.scatter(Data.wl3a_x[8:, 0], Data.wl3a_x[8:, 1], c='r')
    plt.show()
    
    # Visualization - PCA by Eigenvalue decomposition
    Model._plot_1dprojection(eigin_projmat, Data.wl3a_x, "PCA by ED")
    # Visualization - PCA by Singular Value Decomposition
    Model._plot_1dprojection(svd_projmat, Data.wl3a_x, "PCA by SVD")

    # Sci-kit Principal Component Analysis
    sciPCA = PCA(n_components=1)
    newdata = sciPCA.fit_transform(Data.wl3a_x)
    p1 = plt.figure("Sci-kit PCA")
    plt.title("Sci-kit PCA")
    plt.scatter(newdata[:8], np.zeros(len(newdata[:8])), c='b')
    plt.scatter(newdata[8:], np.zeros(len(newdata[8:])), c='r')
    plt.show()