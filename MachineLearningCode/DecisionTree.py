'''
    Implementing of Decision Tree.
    notes:
        Information entropy: H(y) = -∑P(y)log(P(y))
        Conditional entropy: H(y|x) = -∑∑p(y,x)log(p(y|x))
        Information gain: Gain(y, x) = H(y) - H(y|x)
'''
import math
import numpy as np

from datas import WatermelonDatas
from sklearn.tree import DecisionTreeClassifier, export_graphviz

# Feature - Subfeature  - Label
f0 = {"0.0":"乌黑", "1.0":"青绿", "2.0":"浅白"}
f1 = {"0.0":"蜷缩", "1.0":"稍蜷", "2.0":"硬挺"}
f2 = {"0.0":"浊响", "1.0":"沉闷", "2.0":"清脆"}
f3 = {"0.0":"清晰", "1.0":"稍糊", "2.0":"模糊"}
f4 = {"0.0":"凹陷", "1.0":"稍凹", "2.0":"平坦"}
f5 = {"0.0":"硬滑", "1.0":"软粘"}
features = {"色泽": f0, "根蒂": f1, "敲声": f2,
            "纹理": f3, "脐部": f4, "触感": f5}
indicator = np.array(["色泽", "根蒂", "敲声", "纹理", "脐部", "触感", "Label"]).reshape(1, -1)


class MyDecisionTree(object):
    """docstring for MyDecisionTree"""
    def __init__(self, data_x, data_y):
        super(MyDecisionTree, self).__init__()
        self.data = np.concatenate((data_x[:,:-2], data_y.reshape(-1, 1)), axis=1)
        self.data = np.concatenate((indicator, self.data), axis=0)

    def _cal_informationentropy(self, data):
        '''
            Calculate infromation entropy, ent = -∑P(y_i)log(P(y_i)),
            where P(y_i) means probability of i-th class, i ∈ n-class
        '''
        n_all, n_f = data.shape
        labelCounts = {}
        for label in range(1, n_all):
            if data[label][-1] not in labelCounts: labelCounts[data[label][-1]] = 1
            else: labelCounts[data[label][-1]] += 1
        Ent = 0.0
        for key, value in labelCounts.items():
            prob = float(value) / (n_all-1)
            Ent = Ent - prob * math.log(prob)
        return Ent

    def _cal_conditionalentropy(self, data):
        '''
            Calculate conditional entropy, H(y|x) = -∑∑p(y,x)log(p(y|x))
            where x is feature label, y is corresponding label.
            e.g. x can be "乌黑", "青绿" and "浅白" in feature "色泽", y can be "好瓜" or "坏瓜".
        '''
        n_all, n_f = data.shape
        featureCounts = []
        for f in range(n_f-1):
            feature_f = {}
            for idx in range(1, n_all):
                if data[idx][f] not in feature_f:
                    if data[idx,-1] == "1.0":
                        feature_f[data[idx][f]] = [1, 0]
                    else:
                        feature_f[data[idx][f]] = [0, 1]
                else:
                    if data[idx,-1] == "1.0":
                        feature_f[data[idx][f]][0] += 1
                    else:
                        feature_f[data[idx][f]][1] += 1
            featureCounts.append(feature_f)
        con_ent = []
        for fea in featureCounts:
            fea_ent = 0.0
            for key, value in fea.items():
                res = 0.0
                f_sum = sum(value)
                for x in value:
                    prob = (x / f_sum)
                    res -= prob*math.log(prob) if prob != 0 else 0
                fea_ent += (f_sum/(n_all-1))*res
            con_ent.append(fea_ent)
        return con_ent, featureCounts

    def splitdata(self, idx, dic, data):
        '''
            Delete the choosen feature set.
        '''
        msg = np.delete((data[0].reshape(1, -1)), idx, axis=1)
        split_data = {}
        for key, value in dic.items():
            tmp = []
            for d in data[1:]:
                if d[idx] == key:
                    tmp.append(d.tolist())
            tmp = np.delete(tmp, idx, axis=1)
            tmp = np.concatenate((msg, tmp), axis=0)
            split_data[key] = tmp
        return split_data

    def BuildID3Tree(self, data):
        '''
            main process of building the id3 tree.
        '''
        ent = self._cal_informationentropy(data)
        con_ent, dic = self._cal_conditionalentropy(data)
        choseidx = np.argmax([ent - ce for ce in con_ent])
        subdic = dic[choseidx]
        splitd = self.splitdata(choseidx, subdic, data)
        return choseidx, subdic, splitd

    def ID3(self, data):
        flag, l = True, data[1][-1]
        for d in data[1:]:
            if d[-1] != l: flag = False
        if flag == True:
            if l == '1.0': return "好瓜"
            else: return "坏瓜"
        if len(data) == 2:
            if data[1][0] == 1: return "好瓜"
            else: return "坏瓜"

        id3tree = {}
        while True:
            chosefeat, subdic, splitd = self.BuildID3Tree(data)
            id3tree[data[0][chosefeat]] = {}
            for k, v in subdic.items():
                id3tree[data[0][chosefeat]][features[data[0][chosefeat]][k]] = self.ID3(splitd[k])
            break
        return id3tree


if __name__ == "__main__":
    # Load Watermelon Datasets
    Data = WatermelonDatas()

    # Bulid Decision Tree Model.
    DT = MyDecisionTree(Data.wl3_x, Data.wl3_y)
    # Recursively building ID3 tree.
    print(DT.ID3(DT.data))

    # Sci-kit Decision Tree
    scidt = DecisionTreeClassifier(criterion='entropy')
    scidt.fit(Data.wl3_x[:, :-2], Data.wl3_y)
    export_graphviz(scidt)