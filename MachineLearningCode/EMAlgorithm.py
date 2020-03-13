'''
    EM-Algorithm is a method to infer parameters with hidden varible. Applications like K-means and GMM, et, al.
    EM-Algorithm:
        Independent variable: x; Hidden variable: z; parameter: θ, Tolerance: ε.
        Maximum likelihood function: L(θ) = ∑logP(x|θ); P(x|θ) = P(x,z|θ)
            L(θ) = ∑logP(x|θ)
                 = ∑logP(x,z|θ)
                 = ∑log∑P(z|θ)P(x|z,θ)
            L(θ) - L(θ`) = ∑log∑P(z|θ)P(x|z,θ) - ∑logP(x|θ`)                            suppose L(θ`) is known
                         = ∑log∑P(z|x,θ`)*{(P(z|θ)P(x|z,θ))/P(z|x,θ`)} - ∑logP(x|θ`)
                         ≥ ∑∑P(z|x,θ`)*log{(P(z|θ)P(x|z,θ))/P(z|x,θ`)} - ∑logP(x|θ`)    jensen inequation
                         = ∑∑P(z|x,θ`)*log{(P(z|θ)P(x|z,θ))/P(z|x,θ`)P(x|θ`)}           ∑logP(x|θ`) = ∑∑P(z|x,θ`)logP(x|θ`), ∑P(z|x,θ`) = 1
            ∴ L(θ) ≥ L(θ`) + Q(θ|θ`).
              where Q(θ|θ`) = ∑∑P(z|x,θ`)*log{(P(z|θ)P(x|z,θ))/P(z|x,θ`)P(x|θ`)}.
        EM:
            1). initialize parameter θ.
            2). E-step: calculate P(z|x,θ)
            3). M-step: maximize Q(θ|θ`) based on P(z|x,θ) calculated in 2).
            4). stop if |θ - θ`| ≤ ε, else go to 2).

    GMM(Gaussian Mixture Model)
'''
import numpy as np
import matplotlib.pyplot as plt

from math import *
from sklearn.datasets.samples_generator import make_blobs
from sklearn.mixture import GaussianMixture
from matplotlib.patches import Ellipse

epsilon = 1e-9

class MyGMM(object):
    """docstring for MyGMM"""
    def __init__(self, data_x, data_y, n_clusters, tolerance, maxiter):
        super(MyGMM, self).__init__()
        self.data_x = data_x
        self.data_y = data_y
        self.n_clusters = n_clusters
        self.n_samples = data_x.shape[0]
        self.n_feature = data_x.shape[1]
        self.tolerance = tolerance
        self.maxiter = maxiter

    def _init_paras(self):
        # self.miu = np.random.normal(loc=1.0, scale=0.5, size=(self.n_clusters, self.n_feature))
        self.miu = np.array([np.mean(self.data_x, axis=0) for _ in range(self.n_clusters)])
        self.covmat = np.array([np.diag(x) for x in np.random.normal(loc=1.0, scale=0.2, size=(self.n_clusters, self.n_feature))])
        # print(self.miu)
        # print(self.covmat)
        self.wk = np.ones([self.n_clusters,]) / self.n_clusters

    def _gx(self):
        gx = []
        for kdx in range(self.n_clusters):
            p1 = self.data_x - self.miu[kdx]
            p2 = np.linalg.inv(self.covmat[kdx])[np.newaxis, :, :]
            expnum = np.matmul(np.matmul(p1[:, np.newaxis, :], p2), p1[:, :, np.newaxis]).reshape(self.n_samples, 1)
            part1 = sqrt(pow(2*np.pi, self.n_feature))
            detcovmat = sqrt(np.linalg.det(self.covmat[kdx]))
            pp1 = pow(part1*detcovmat, -1)
            gx.append(pp1*np.exp(-0.5 * expnum))
        return np.array(gx)

    def _Rij(self):
        sumRij = []
        gxs = self._gx()
        Rij = (self.wk[np.newaxis, :, np.newaxis] * np.transpose(gxs, [1, 0, 2]))
        Rij = Rij / np.sum(Rij, axis=1, keepdims=True)
        return Rij.reshape(self.n_samples, self.n_clusters)

    def _EM(self):
        self._init_paras()
        itertimes = 0
        Rij = self._Rij()
        # print(Rij)
        while itertimes < self.maxiter:
            print("Itertimes: ", itertimes+1)
            # E-step
            # cal new wk
            new_wk = np.sum(Rij, axis=0)/self.n_samples
            # check if stop
            stop_cond = [abs(nwk) <= self.tolerance for nwk in new_wk - self.wk]
            # print(stop_cond)
            if sum(stop_cond) == self.n_clusters:
                break
            # update new wk
            self.wk = new_wk
            # M-step
            # cal new Rij
            Rij = self._Rij()
            sum_rij = np.sum(Rij, axis=0) + epsilon
            # update miu and var
            new_miu = np.matmul(Rij[:, :, np.newaxis],self.data_x[:, np.newaxis, :])
            self.miu = np.sum(new_miu, axis=0) / sum_rij[:, np.newaxis]
            tmpdiff = self.data_x[:, np.newaxis, :] - self.miu[np.newaxis, :, :]
            b = np.matmul(tmpdiff[:, :, :, np.newaxis], tmpdiff[:, :, np.newaxis, :])
            self.covmat = np.sum(np.multiply(Rij[:, :, np.newaxis, np.newaxis], b), axis=0) / sum_rij[:, np.newaxis, np.newaxis]
            # updata itertimes
            itertimes += 1


if __name__ == "__main__":
    # Generate dataset
    data_x, data_y = make_blobs(n_samples=50, centers=3, cluster_std=0.6, random_state=0)
    # Build my gmm model
    mymodel = MyGMM(data_x, data_y, 3, 0.001, 100)
    mymodel._EM()
    # Sci-kit gmm model
    skgmm = GaussianMixture(n_components=3)
    skgmm.fit(data_x)
    miu = skgmm.means_
    cov = skgmm.covariances_
    # show the mean vectors
    print("The mean vectors of my gmm model\n", mymodel.miu)
    print("The mean vectors of sklearn gmm model\n", miu)
    # Visualization
    plt.figure('Picutre')
    plt.title('Data Distribution')
    plt.scatter(data_x[:, 0], data_x[:, 1],  c=data_y, s=20, cmap='viridis')
    plt.scatter(mymodel.miu[:, 0], mymodel.miu[:, 1], marker='X', c='r')
    plt.scatter(miu[:, 0], miu[:, 1], marker='X', c='b')
    ax = plt.gca()
    for i in range(3):
        plot_args1 = {'fc': 'None', 'lw': 2, 'edgecolor': 'b'}
        plot_args2 = {'fc': 'None', 'lw': 2, 'edgecolor': 'r'}
        ellipse1 = Ellipse(miu[i], 4*cov[i][0][0], 4*cov[i][1][1], **plot_args1)
        ellipse2 = Ellipse(mymodel.miu[i], 4*mymodel.covmat[i][0][0], 4*mymodel.covmat[i][1][1], **plot_args2)
        ax.add_patch(ellipse1)
        ax.add_patch(ellipse2)
    plt.show()