'''
    word2vec:
    dataset: "词性标注@人民日报199801.txt"
'''
import os
import pickle, time
import numpy as np


def softmax(x):
    ex = np.exp(x)
    return ex/np.sum(ex)


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


class word2vec(object):
    """docstring for word2vec"""
    def __init__(self, type='skipgram'):
        super(word2vec, self).__init__()
        self.type = type

    def _process_text(self, filepath, minfreq):
        # pre-processing
        with open(filepath, encoding='utf8') as corpus:
            self.countwords = {}
            self.rawdata, self.data = [], []
            for sen in corpus.readlines():
                tmp = []
                for wordpos in sen.split(' ')[1:-1]:
                    if wordpos == "":
                        continue
                    word, pos = wordpos.split('/')
                    if '[' in word: word = word.replace('[', '')
                    if word not in self.countwords.keys():
                        self.countwords[word] = 1
                    else:
                        self.countwords[word] += 1
                    tmp.append(word)
                if tmp == []: continue
                self.rawdata.append(tmp)
        # build id2vocab and vocab2id
        self.vocab2id, self.id2vocab = {'unk':0}, {0:'unk'}
        sortvocab = sorted([[v, k] for k, v in self.countwords.items()])
        sortvocab = [[v, k] for v, k in sortvocab if v > minfreq]
        idx = 1
        for v, k in sortvocab:
            self.vocab2id[k] = idx
            self.id2vocab[idx] = k
            idx += 1
        self.vocabsize = len(self.vocab2id)
        for datas in self.rawdata:
            self.data.append([self.vocab2id[w] if w in self.vocab2id.keys() else 0 for w in datas])

    def _train_tgt(self, windowsize):
        if self.type == 'skipgram':
            self.traindata = []
            for intdata in self.data:
                for idx in range(len(intdata)):
                    srtidx = idx - windowsize if (idx - windowsize) > 0 else 0
                    endidx = idx + windowsize if (idx + windowsize) < len(intdata) - 2 else len(intdata) - 2
                    tgtwords = intdata[srtidx:idx] + intdata[idx+1:endidx+1]
                    for tgt in tgtwords:
                        self.traindata.append([intdata[idx], tgt])
            self.traindata = np.array(self.traindata)
        elif self.type == 'cbow':
            self.traindata = []
            for intdata in self.data:
                for idx in range(len(intdata)):
                    srtidx = idx - windowsize if (idx - windowsize) > 0 else 0
                    endidx = idx + windowsize if (idx + windowsize) < len(intdata) - 2 else len(intdata) - 2
                    inputwords = intdata[srtidx:idx] + intdata[idx+1:endidx+1]
                    if len(inputwords) < 2*windowsize:
                        inputwords = inputwords + [0]*(2*windowsize - len(inputwords))
                    self.traindata.append([inputwords, intdata[idx]])
            self.traindata = np.array(self.traindata)
        else:
            raise TypeError("incorrect type of model!")

    def _init_vec(self, vocabsize, vecsize):
        self.lookuptable = np.random.normal(loc=0.0, scale=0.25, size=(vocabsize, vecsize))
        self.weight = np.random.normal(loc=0.0, scale=0.25, size=(vocabsize, vecsize))

    def _skipgram_forward_backward(self, x, y, lr, K):
        def _negsample(y, K):
            res = [None]*K
            for idx in range(K):
                tmpidx = np.random.randint(0, self.vocabsize)
                while tmpidx == y:
                    tmpidx = np.random.randint(0, self.vocabsize)
                res[idx] = tmpidx
            return res
        # forward
        inputx = self.lookuptable[x]
        targety = self.weight[y]
        outputprobs = sigmoid(np.dot(inputx, targety))

        # backward
        samples = _negsample(y, K)
        deltal = (outputprobs - 1.0) * targety
        self.weight[y] -= lr * (outputprobs - 1.0) * inputx
        loss = -np.log(outputprobs + 1e-5)
        for idx in samples:
            tmpoutput = sigmoid(np.dot(inputx, self.weight[idx]))
            loss -= np.log(tmpoutput + 1e-5)
            deltal += (tmpoutput - 1.0) * self.weight[idx]
            self.weight[idx] -= lr*(tmpoutput - 1.0)*inputx
        self.lookuptable[x] -= lr*deltal

        return loss

    def _train_skipgram(self, traindata, epochs, lr, K):
        print('Start training...!')
        # training
        for epoch in range(1, epochs+1):
            if epoch > 2:
                lr = lr/2
            loss, itertimes = 0, 1
            strtime = time.time()
            for x, y in traindata:
                loss += self._skipgram_forward_backward(x, y, lr, K)
                if itertimes % 1000 == 0:
                    stptime = time.time()
                    showmsg = 'Epoch {:>2d}/{:>2d} - itertimes: {:>7d} - loss:{:>.4f} - {:>.4f} sec/1000'.format(
                        epoch, epochs, itertimes, loss/1000, stptime-strtime)
                    print(showmsg)
                    loss = 0
                    strtime = time.time()
                # save
                if itertimes % 500000 == 0:
                    pickle.dump(self.lookuptable, open(r'.\data\skipgramwordvectors.pkl', 'wb'))
                itertimes += 1

    def _cbow_forward_backword(self, x, y, lr, K):
        def _negsample(y, K):
            res = [None]*K
            for idx in range(K):
                tmpidx = np.random.randint(0, self.vocabsize)
                while tmpidx == y:
                    tmpidx = np.random.randint(0, self.vocabsize)
                res[idx] = tmpidx
            return res
        # forward
        inputx = sum([self.weight[w] for w in x])/len(x)
        targety = self.lookuptable[y]
        outputprobs = sigmoid(np.dot(inputx, targety))

        # backward
        samples = _negsample(y, K)
        deltaw = (outputprobs - 1.0)*targety
        self.lookuptable[y] -= lr*(outputprobs - 1.0)*inputx
        for idx in x:
            self.weight[idx] -= lr*(deltaw/len(x))
        loss = -np.log(outputprobs + 1e-5)
        for idx in samples:
            tmpoutput = sigmoid(np.dot(inputx, self.lookuptable[idx]))
            loss -= np.log(tmpoutput + 1e-5)
            deltal = (tmpoutput - 1.0)*inputx
            self.lookuptable[idx] -= lr*deltal
        return loss

    def _train_cbow(self, traindata, epochs, lr, K):
        print('Start training...!')
        # training
        for epoch in range(1, epochs+1):
            if epoch > 2:
                lr = lr/2
            loss, itertimes = 0, 1
            strtime = time.time()
            for x, y in traindata:
                loss += self._cbow_forward_backword(x, y, lr, K)
                if itertimes % 500 == 0:
                    stptime = time.time()
                    showmsg = 'Epoch {:>2d}/{:>2d} - itertimes: {:>7d} - loss:{:>.4f} - {:>.4f} sec/500'.format(
                        epoch, epochs, itertimes, loss/500, stptime-strtime)
                    print(showmsg)
                    loss = 0
                    strtime = time.time()
                # save
                if itertimes % 100000 == 0:
                    pickle.dump(self.lookuptable, open(r'.\data\cbowwordvectors.pkl', 'wb'))
                itertimes += 1

    def _train(self, epochs, lr, vecsize, K=10):
        self._init_vec(self.vocabsize, vecsize)
        if self.type == 'skipgram':
            self._train_skipgram(traindata=self.traindata, epochs=epochs, lr=lr, K=K)
        elif self.type == 'cbow':
            self._train_cbow(traindata=self.traindata, epochs=epochs, lr=lr, K=K)
        else:
            raise TypeError("incorrect type of model!")

    def _cheak(self, shownum, k, dist='cosine'):
        if not (os.path.exists(r'.\data\skipgramwordvectors.pkl') or os.path.exists(r'.\data\cbowwordvectors.pkl')):
            raise ValueError('No word vectors found!')
        if self.type == 'skipgram':
            self.lookuptable = pickle.load(open(r'.\data\skipgramwordvectors.pkl', 'rb'))
        elif self.type == 'cbow':
            self.lookuptable = pickle.load(open(r'.\data\cbowwordvectors.pkl', 'rb'))
        else:
            raise TypeError("incorrect type of model!")
        random_chose = [np.random.randint(0, self.vocabsize) for _ in range(shownum)]
        vecnorm = [np.linalg.norm(vec, ord=2) for vec in self.lookuptable]
        print('Word Vectors of ' + self.type)
        for choice in random_chose:
            tmp = []
            for idx in range(self.vocabsize):
                if dist == 'cosine':
                    tmp.append(np.dot(self.lookuptable[idx], self.lookuptable[choice]) / (vecnorm[idx]*vecnorm[choice]))
                elif dist == '2-norm':
                    tmp.append(np.linalg.norm((self.lookuptable[idx] - self.lookuptable[choice]), ord=2))
                else:
                    raise TypeError("Invalid method of calculating distance!")
            topk = [int(np.argwhere(np.argsort(tmp)==i)) for i in range(k)]
            topkneighbors = [self.id2vocab[idx] for idx in topk]
            showmsg = 'Top '+str(k) + ' words nearest to [{:^6}] : '.format(self.id2vocab[choice])
            res = ', '.join(topkneighbors)
            print(showmsg+res)


if __name__ == "__main__":
    # loading dataset
    filepath = r'.\data\词性标注@人民日报199801.txt'
    # building word2vec model
    w2v = word2vec(type='skipgram')
    # pre-processing data
    w2v._process_text(filepath=filepath, minfreq=10)
    # generating training data
    w2v._train_tgt(windowsize=2)
    # training word vectors
    # w2v._train(epochs=5, lr=0.001, vecsize=100)
    # show res
    w2v._cheak(10, 5)