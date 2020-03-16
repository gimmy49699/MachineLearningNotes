'''
   Chinese Part of speech tagging
   Dataset: The People's Daily Corpus.
   Algorithm: Hidden Markov Model.
'''
import jieba   # [pip install jieba] before use!
import numpy as np
import jieba.posseg as pseg

from HMM import MyHMM
# Data Pre-processing
class dataProcessor(object):
    """docstring for data"""
    def __init__(self):
        super(dataProcessor, self).__init__()

    def _count(self, filepath):
        corpus = open(filepath, encoding='utf8')
        self.vocabCount = {}
        self.posCount = {}
        self.clean_pos_trans = {}
        self.pos2word = {}
        self.startprob = {}
        for sen in corpus.readlines():
            tmp_pos = []
            for wordpos in sen.split(' ')[1:-1]:
                if wordpos == "":
                    continue
                # get wort with pos.
                word, pos = wordpos.split('/')
                if '[' in word: word = word.replace('[', '')
                if ']' in pos: pos = pos.split(']')[0]
                tmp_pos.append(pos)
                # build pos to word map.
                if pos not in self.pos2word.keys():
                    self.pos2word[pos] = {word:1}
                else:
                    if word not in self.pos2word[pos].keys():
                        self.pos2word[pos][word] = 1
                    else:
                        self.pos2word[pos][word] += 1
                # counting vocabularies.
                if word not in self.vocabCount.keys():
                    self.vocabCount[word] = 1
                else:
                    self.vocabCount[word] += 1
                # counting poses
                if pos not in self.posCount.keys():
                    self.posCount[pos] = 1
                else:
                    self.posCount[pos] += 1
            # counting start pos and calculate start probabilities.
            if tmp_pos == []:
                continue
            if tmp_pos[0] not in self.startprob.keys():
                self.startprob[tmp_pos[0]] = 1
            else:
                self.startprob[tmp_pos[0]] += 1
            # calculate pos transition probabilities.
            for idx in range(1, len(tmp_pos)):
                if tmp_pos[idx-1] not in self.clean_pos_trans.keys():
                    self.clean_pos_trans[tmp_pos[idx-1]] = {tmp_pos[idx]:1}
                else:
                    if tmp_pos[idx] not in self.clean_pos_trans[tmp_pos[idx-1]]:
                        self.clean_pos_trans[tmp_pos[idx-1]][tmp_pos[idx]] = 1
                    else:
                        self.clean_pos_trans[tmp_pos[idx-1]][tmp_pos[idx]] += 1

    def _buildVocabs(self):
        self.vocab = {0: 'unk'}
        self.vocab2id = {'unk':0}
        sortvocab = sorted([[v, k] for i,(k, v) in enumerate(self.vocabCount.items())])
        sortvocab.reverse()
        totalvocab = 0
        for idx in range(1, len(sortvocab)+1):
            totalvocab += sortvocab[idx-1][0]
            self.vocab[idx] = sortvocab[idx-1][1]
            self.vocab2id[sortvocab[idx-1][1]] = idx

    def _buildPos(self):
        self.pos = {}
        self.pos2id = {}
        sortpos = sorted([[v, k] for i,(k, v) in enumerate(self.posCount.items())])
        sortpos.reverse()
        totalpos = 0
        for idx in range(0, len(sortpos)):
            totalpos += sortpos[idx][0]
            self.pos[idx] = sortpos[idx][1]
            self.pos2id[sortpos[idx][1]] = idx

    def _buildDatas(self):
        self.transMat = np.ones((len(self.pos), len(self.pos)))
        for k, v in self.clean_pos_trans.items():
            for subk, subv in v.items():
                self.transMat[self.pos2id[k], self.pos2id[subk]] = subv
            self.transMat[self.pos2id[k]] = self.transMat[self.pos2id[k]]/np.sum(self.transMat[self.pos2id[k]])

        self.emissionMat = np.ones((len(self.pos), len(self.vocab)))
        for k, v in self.pos2word.items():
            for subk, subv in v.items():
                self.emissionMat[self.pos2id[k], self.vocab2id[subk]] = subv
            self.emissionMat[self.pos2id[k]] = self.emissionMat[self.pos2id[k]]/np.sum(self.emissionMat[self.pos2id[k]])

        self.startProbs = np.ones((len(self.pos)))
        for k, v in self.startprob.items():
            self.startProbs[self.pos2id[k]] = v
        self.startProbs = self.startProbs/np.sum(self.startProbs)

    def _fit(self, filepath):
        self._count(filepath)
        self._buildVocabs()
        self._buildPos()
        self._buildDatas()


if __name__ == '__main__':
    # loading training corpus
    dataloader = dataProcessor()
    filepath = r'.\data\词性标注@人民日报199801.txt'
    dataloader._fit(filepath)
    # test data
    test = "岁月对人们的赐予是没有条件的。"
    segSen = [x for x in jieba.cut(test, cut_all=False)]
    segSen2id = [dataloader.vocab2id[x] if x in dataloader.vocab2id.keys() else dataloader.vocab2id['unk'] for x in segSen]
    # build hmm model
    myhmm = MyHMM(startProbs=dataloader.startProbs, hiddenStates=dataloader.pos, obserStates=dataloader.vocab, transMat=dataloader.transMat, emissionMat=dataloader.emissionMat)
    res, prob = myhmm._predict(segSen2id)
    # jieba pos segment
    jiebaRes = pseg.cut(test)
    # show result.
    print('输入语句(已分词)： '+' / '.join(segSen))
    print('HMM词性标注结果： '+' '.join([w+'/'+dataloader.pos[x] for i,(w, x) in enumerate(zip(segSen, res))]))
    print('Jieba词性标注结果： '+' '.join([w.word+'/'+w.flag for w in jiebaRes]))