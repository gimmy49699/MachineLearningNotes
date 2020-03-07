'''
    Implementing of Support Vector Machine with soft margin and kernel function.
    min{a} (1/2)∑i∑j[aiajyiyj*K<xi,xj>] - ∑i[ai]
      s.t.  ∑i[aiyi] = 0
            0 ≤ ai ≤ C , i = 1, 2, ..., N

    SMO algorithm:
        1) Initialize alpha, beta = [0, ..., 0], 0
        2) Find alpha_1 by:
            if 0 < alpha_i < C and y_i*g(x_i) != 1
            or alpha_i = 0 and y_i*g(x_i) < 1
            or alpha_i = C and y_i*g(x_i) > 1
        3) Find alpha_2 by:
            max{alpha_2} |E1 - E2|
            where E_i = g(x_i) - y_i
        4) Update unconstraint alpha_2 by:
            new_unc_alpha_2 = alpha_2 + y_2(E1-E2) / (K11 + K22- 2*K12)
            where Kij = kernel<x_i, x_j>
        5) Calculate lower bound and upper bound and update alpha_2 by:
            / L = max(0, alpha_2 + alpha_1 - C), H = min(C, alpha_2 + alpha_1) if y_1 == y_2;
            \ L = max(0, alpha_2 - alpha_1), H = min(C, C + alpha_2 - alpha_1) if y_1 != y_2.

                           /       H            if new_unc_alpha_2 > H;
            new_alpha_2 = -  new_unc_alpha_2    if l <= new_unc_alpha_2 <= H;
                           \       L            if new_unc_alpha_2 < L.
        6) Update alpha_1 by:
            new_alpha_1 = alpha_1 + y_1*y_2*(alpha_2 - new_alpha_2)
        7) Calculate bias beta and update E by:
            new_b1 = -y_1*K11*(new_alpha_1 - alpha_1) - y_2*K22*(new_alpha_2 - alpha_2) - E1
            new_b2 = -y_1*K11*(new_alpha_1 - alpha_1) - y_2*K22*(new_alpha_2 - alpha_2) - E2
                        /      new_b1            if 0 < new_alpha_1 < C;
            new_beta = -       new_b2            if 0 < new_alpha_2 < C;
                        \(new_b1 + new_b2)/2            otherwise.
            new_E1 = g(x_1) - y_1, new_E2 = g(x_2) - y_2
        8) Stop conditions:
            in the range of precision ε:
            1). ∑i[aiyi] = 0;
            2). 0 <= alpha_i <= C, i = 1, 2, ... , N;
            3). new_alpha == 0 and y_i*g(x_i) >= 1;
            4). 0 <= new_alpha <= C and y_i*g(x_i) == 1;
            5). new_alpha == C and y_i*g(x_i) <= 1;
            6). Max iteration times.
        9) If satisfied 8), return new_alpha, or go to 2).
'''
import numpy as np
import matplotlib.pyplot as plt

from datas import WatermelonDatas
from sklearn.svm import SVC


class MySupportVectorMachine(object):
    """docstring for MySupportVectorMachine"""
    def __init__(self, datax, datay, C, kernel="linear", maxiter=100):
        super(MySupportVectorMachine, self).__init__()
        self.datax = datax
        self.datay = np.array([1 if a==1 else -1 for a in datay])
        self.n_sample = self.datax.shape[0]
        self.n_feature = self.datax.shape[1]
        self.alpha = np.zeros((self.n_sample, 1))
        self.Ecache = -np.copy(self.datay)
        self.beta = np.zeros(1)
        self.C = C
        self.kernel = self._linearKernel if kernel == "linear" else self._RBFKernel
        self.maxiter = maxiter

    def _linearKernel(self, x1, x2):
        return np.matmul(x1, x2.T)

    def _RBFKernel(self, x1, x2):
        g = 1/self.n_feature
        return np.exp(-np.linalg.norm(x1-x2)/g**2)

    def _f(self, x):
        res = 0
        for idx in range(self.n_sample):
            res += self.alpha[idx] * self.datay[idx] * self.kernel(self.datax[idx], x.T)
        return res + self.beta

    def _g(self, xs):
        res = []
        for x in xs:
            res.append(self._f(x))
        return np.array(res)

    def _calW(self):
        self.W = np.zeros((1, self.n_feature))
        for idx in range(self.n_sample):
            self.W += self.alpha[idx]*self.datay[idx]*self.datax[idx]

    def _calE(self, idx):
        return self._f(self.datax[idx]) - self.datay[idx]

    def _calEcache(self):
        for idx in range(self.n_sample):
            self.Ecache[idx] = self._calE(idx)

    def _calAlpha1idx(self):
        # can be optimized
        for idx in range(self.n_sample):
            if 0 < self.alpha[idx] < self.C:
                if self.datay[idx]*self._f(self.datax[idx]) != 1:
                    return idx
        for idx in range(self.n_sample):
            if self.alpha[idx] == 0 and self.datay[idx]*self._f(self.datax[idx]) < 1:
                return idx
            if self.alpha[idx] == self.C and self.datay[idx]*self._f(self.datax[idx]) > 1:
                return idx

    def _calAlpha2idx(self, idx1):
        tmp = [abs(self.Ecache[idx1] - a) for a in np.copy(self.Ecache)]
        return np.argmax(tmp)

    def _calLH(self, y1, y2, alpha1old, alpha2old):
        if y1 == y2:
            return max(0, alpha2old + alpha1old - self.C), min(self.C, alpha2old + alpha1old)
        else:
            return max(0, alpha2old - alpha1old), min(self.C, self.C + alpha2old - alpha1old)

    def _calAlphaNew(self, L, H, alpha):
        if alpha > H:
            return H
        elif alpha < L:
            return L
        else:
            return alpha

    def _cheak(self):
            if np.matmul(self.alpha.T, self.datay) != 0:
                return False
            for idx in range(self.n_sample):
                if self.alpha[idx] < 0 or self.alpha[idx] > self.C:
                    return False
                elif self.alpha[idx] == 0 and self.datay[idx]*self._f(self.datax[idx]) < 1:
                    return False
                elif self.alpha[idx] == self.C and self.datay[idx]*self._f(self.datax[idx]) > 1:
                    return False
                elif self.datay[idx]*self._f(self.datax[idx]) != 1:
                    return False
            return True

    def _SMO(self):
        itertimes = 0
        while itertimes < self.maxiter:
            print("itertime: ", itertimes + 1)
            itertimes += 1
            a1 = self._calAlpha1idx()
            a2 = self._calAlpha2idx(a1)
            K11 = self.kernel(self.datax[a1], self.datax[a1])
            K22 = self.kernel(self.datax[a2], self.datax[a2])
            K12 = self.kernel(self.datax[a1], self.datax[a2])
            K = K11 + K22 - 2*K12
            a2new = self.alpha[a2] + ((self.datay[a2]*(self.Ecache[a1] - self.Ecache[a2])))/K
            L, H = self._calLH(self.datay[a1], self.datay[a2], self.alpha[a1], self.alpha[a2])
            a2new = self._calAlphaNew(L, H, a2new)
            a1new = self.alpha[a1] + self.datay[a1]*self.datay[a2]*(self.alpha[a2]-a2new)
            b1new = self.beta + (self.alpha[a1]-a1new)*self.datay[a1]*K11 \
                + (self.alpha[a2] - a2new)*self.datay[a2]*K22 - self.Ecache[a1]
            b2new = self.beta + (self.alpha[a1]-a1new)*self.datay[a1]*K11 \
                + (self.alpha[a2] - a2new)*self.datay[a2]*K22 - self.Ecache[a2]
            if 0 < a1new < self.C:
                self.beta = b1new
            elif 0 < a1new < self.C:
                self.beta = b2new
            else:
                self.beta = (b1new + b2new) / 2
            self.alpha[a1], self.alpha[a2] = a1new, a2new
            self.Ecache[a1], self.Ecache[a2] = self._calE(a1), self._calE(a2)
            if self._cheak == True:
                break


if __name__ == "__main__":
    # Load Watermelon Datasets
    Data = WatermelonDatas()
    # Build My support Vector Machine
    SVM = MySupportVectorMachine(Data.wl3a_x, Data.wl3a_y, 1)
    # Iteratively calculatting and updating coef alpha and bias beta
    SVM._SMO()
    # Calculating weight matrix
    SVM._calW()
    # Visualization of My SVM decision boundary
    p1 = plt.figure(num="MySupportVectorMachine")
    plt.title("My Support Vector Machine")
    x = np.arange(0.0, 1.0, 0.01)
    y = (-SVM.beta-SVM.W[0][0]*x)/SVM.W[0][1]
    plt.plot(x, y, 'k')
    plt.scatter(Data.wl3a_x[:8,0], Data.wl3a_x[:8,1], c='b')
    plt.scatter(Data.wl3a_x[8:,0], Data.wl3a_x[8:,1], c='r')
    plt.show()

    # Sci-kit SVC
    skSVC = SVC(kernel='linear', max_iter=100)
    skSVC.fit(Data.wl3a_x, Data.wl3a_y)
    # Visualization of Sci-kit SVC decision boundary
    p2 = plt.figure(num="ScikitSupportVectorMachine")
    plt.title("Scikit Support Vector Machine")
    y1 = (-skSVC.intercept_-skSVC.coef_[0][0]*x)/skSVC.coef_[0][1]
    plt.plot(x, y1, 'k')
    plt.scatter(Data.wl3a_x[:8,0], Data.wl3a_x[:8,1], c='b')
    plt.scatter(Data.wl3a_x[8:,0], Data.wl3a_x[8:,1], c='r')
    plt.show()
