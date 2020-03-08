'''
    Implementing of Naive Bayes Classifier
    NBC: h*(x) = argmax{c∈y} P(c) * ∏P(x|c)
'''
import numpy as np

from math import *
from datas import WatermelonDatas

sqrt2pi = sqrt(2 * pi)
# Feature - Subfeature
f0 = {"0":"乌黑", "1":"青绿", "2":"浅白"}
f1 = {"0":"蜷缩", "1":"稍蜷", "2":"硬挺"}
f2 = {"0":"浊响", "1":"沉闷", "2":"清脆"}
f3 = {"0":"清晰", "1":"稍糊", "2":"模糊"}
f4 = {"0":"凹陷", "1":"稍凹", "2":"平坦"}
f5 = {"0":"硬滑", "1":"软粘"}


class MyNaiveBayesClassifier(object):
    """docstring for MyNaiveBayesClassifier"""
    def __init__(self, data_x, data_y):
        super(MyNaiveBayesClassifier, self).__init__()
        self.data = np.concatenate((data_x, data_y.reshape(-1, 1)), axis=1)
        self.n_sample = self.data.shape[0]
        self.n_feature = self.data.shape[1] - 1

    def _countPosNeg(self):
        self.labelCounts = {}
        for idx in range(self.n_sample):
            if self.data[idx][-1] not in self.labelCounts.keys():
                self.labelCounts[self.data[idx][-1]] = 1
            else:
                self.labelCounts[self.data[idx][-1]] += 1
        self.labelsize = len(self.labelCounts.keys())

    def _calPosNegProba(self):
        self.labelProba = {}
        for k, v in self.labelCounts.items():
            # Laplacian correction
            # self.labelProba[k] = v / self.n_sample
            self.labelProba[k] = (v + 1) / (self.n_sample + self.labelsize)

    def _countDiscreteFeature(self):
        self.discreteFeatureCounts = {}
        for idx in range(self.n_sample):
            for fidx in range(self.n_feature - 2):
                if fidx not in self.discreteFeatureCounts.keys():
                    tmp = [0]*self.labelsize
                    tmp[int(self.data[idx][-1])] = 1
                    self.discreteFeatureCounts[fidx] = {self.data[idx][fidx]: tmp}
                else:
                    if self.data[idx][fidx] not in self.discreteFeatureCounts[fidx].keys():
                        tmp = [0]*self.labelsize
                        tmp[int(self.data[idx][-1])] = 1
                        self.discreteFeatureCounts[fidx][self.data[idx][fidx]] = tmp
                    else:
                        self.discreteFeatureCounts[fidx][self.data[idx][fidx]][int(self.data[idx][-1])] += 1

    def _calDiscreteFeatureProba(self):
        self.discreteFeatureCountsProba = {}
        for f, c in self.discreteFeatureCounts.items():
            self.discreteFeatureCountsProba[f] = {}
            for k, v in c.items():
                tmp = [0]*self.labelsize
                for label, count in self.labelCounts.items():
                    # Laplacian correction
                    # tmp[int(label)] = v[int(label)] / count
                    tmp[int(label)] = (v[int(label)] + 1) / (count + len(c.keys()))
                self.discreteFeatureCountsProba[f][k] = tmp

    def _calContinuousFeature(self):
        # calculate mean values
        self.contFeatAvg = {}
        for idx in range (self.n_sample):
            for fidx in range(self.n_feature-2, self.n_feature):
                if fidx not in self.contFeatAvg.keys():
                    tmp = [0]*self.labelsize
                    tmp[int(self.data[idx][-1])] = self.data[idx][fidx]
                    self.contFeatAvg[fidx] = tmp
                else:
                    self.contFeatAvg[fidx][int(self.data[idx][-1])] += self.data[idx][fidx]
        for k, v in self.contFeatAvg.items():
            tmp = [0]*self.labelsize
            for label, count in self.labelCounts.items():
                tmp[int(label)] = v[int(label)] / count
            self.contFeatAvg[k] = tmp
        # calculate variance
        self.contFeatSTVar = {}
        for idx in range (self.n_sample):
            for fidx in range(self.n_feature-2, self.n_feature):
                if fidx not in self.contFeatSTVar.keys():
                    tmp = [0]*self.labelsize
                    tmp[int(self.data[idx][-1])] = (self.data[idx][fidx] - self.contFeatAvg[fidx][int(self.data[idx][-1])])**2
                    self.contFeatSTVar[fidx] = tmp
                else:
                    self.contFeatSTVar[fidx][int(self.data[idx][-1])] += (self.data[idx][fidx] - self.contFeatAvg[fidx][int(self.data[idx][-1])])**2
        for k, v in self.contFeatSTVar.items():
            tmp = [0]*self.labelsize
            for label, count in self.labelCounts.items():
                tmp[int(label)] = sqrt(v[int(label)] / (count-1))
            self.contFeatSTVar[k] = tmp

    def _count(self):
        self._countPosNeg()
        self._calPosNegProba()
        self._countDiscreteFeature()
        self._calDiscreteFeatureProba()
        self._calContinuousFeature()

    def _predict(self, x):
        # cal P(c)
        res = [1]*self.labelsize
        for k, v in self.labelProba.items():
            res[int(k)] *= v
        for fidx in range(self.n_feature-2):
            for k, v in self.labelProba.items():
                res[int(k)] *= self.discreteFeatureCountsProba[fidx][x[fidx]][int(k)]
        for fidx in range(self.n_feature-2, self.n_feature):
            for k, v in self.labelProba.items():
                tmp1 = (x[fidx] - self.contFeatAvg[fidx][int(k)])**2
                tmp2 = 2*self.contFeatSTVar[fidx][int(k)]**2
                res[int(k)] *= exp(-tmp1/tmp2)/(sqrt2pi*self.contFeatSTVar[fidx][int(k)])
        outstr = "好瓜" if np.argmax(res) == 1 else "坏瓜"
        msg = "色泽:{:>2} - 根蒂:{:>2} - 敲声:{:>2} - 纹理:{:>2} - 脐部:{:>2} - 触感:{:>2} - 密度:{:>.3f} - 甜度:{:>.3f}".format(
                f0[str(x[0])],
                f1[str(x[1])],
                f2[str(x[2])],
                f3[str(x[3])],
                f4[str(x[4])],
                f5[str(x[5])],
                x[6],
                x[7])
        print("输入样本特征 = " + msg + "\n" + "分类结果: " + outstr)
        return np.argmax(res)


if __name__ == "__main__":
    # Test sample
    test_data = [0, 1, 2, 2, 1, 0, .345, .876 ]
    # Load Watermelon Datasets
    Data = WatermelonDatas()
    # Build My Naive Bayes Classifier with laplacian correction
    MyNBC = MyNaiveBayesClassifier(Data.wl3_x, Data.wl3_y)
    # Count statistics
    MyNBC._count()
    # Predict given test sample
    MyNBC._predict(test_data)