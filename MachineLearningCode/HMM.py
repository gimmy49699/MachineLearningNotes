'''
    Implementing of Hidden markov model with viterbi algorithm.
    Hidden Markov Model P(X|Y) = ∏P(yi|xi)P(xi|xi-1):
        ◍ -> ◍ -> ◍ -> ... -> ◍
        ↓     ↓    ↓      ↓     ↓
        ◉    ◉    ◉    ...    ◉
        where ◍ means hidden states, ◉ means observation states. k-th hidden
    state is only depended on k-1-th hidden state. k-th observation state is only
    depended on k-th hidden state.(1st-ord markov hypothesis)
        we have 3 kinds of problems:
            1). given a set of observation states, to calculating the most possible
                hidden states which generated the observation states.
            2). given a set of observation states, to calculating the transition
                probability of each hidden states.
            3). given a set of observation states, to calculating the probability
                of the observation states.
        methods: 1).Viterbi Algorithm; 2).EM Algorithm; 3).Bayes Formula.
    We implement the HMM with problem 3).
'''
import numpy as np

# [pip install hmmlearn] before use!
from hmmlearn import hmm

class MyHMM(object):
    """docstring for MyHMM"""
    def __init__(self, startProbs, hiddenStates, obserStates, transMat, emissionMat):
        super(MyHMM, self).__init__()
        self.startProbs = startProbs
        self.hiddenStates = hiddenStates
        self.obserStates = obserStates
        self.transMat = transMat
        self.emissionMat = emissionMat
        self.n_hs = len(self.hiddenStates)
        self.n_os = len(self.obserStates)

    def _predict(self, data):
        start = [[[x], self.startProbs[x]*emissionMat[x][data[0]]] for x in range(self.n_hs)]
        for oidx in range(1, len(data)):
            for hidx in range(self.n_hs):
                tmp = [start[hidx][1]*transMat[hidx][x]*emissionMat[x][data[oidx]] for x in range(self.n_hs)]
                maxobs = tmp.index(max(tmp))
                start[hidx][0].append(maxobs)
                start[hidx][1] = tmp[maxobs]
        maxprob = 0.0
        for hidx in range(self.n_hs):
            if start[hidx][1] >= maxprob:
                res = start[hidx][0]
                maxprob = start[hidx][1]
        showmsg = "The corresponding hidden states are: " + "->".join([self.hiddenStates[x] for x in res]) + " with probability: {:>.6f}".format(maxprob)
        print("The observation states are: " + '-'.join([self.obserStates[x] for x in data]))
        print(showmsg)
        print('The probability of the observation states are: {:>.6f}'.format(np.sum(np.array(start)[:, 1])))

if __name__ == "__main__":
    # set-ups
    hiddenStates = {0: 'Sunny', 1: 'Rainy'}
    obserStates = {0: 'Walking', 1: 'Shopping', 2: 'Cleaning'}
    transMat = [[0.7, 0.3],
                [0.4, 0.6]]
    emissionMat = [[0.4, 0.5, 0.1],
                   [0.1, 0.3, 0.6]]
    test_data = [0, 1, 2, 1]
    startProbs = [0.6, 0.4]
    # Build my HMM model
    myhmm = MyHMM(startProbs=startProbs, hiddenStates=hiddenStates, obserStates=obserStates, transMat=transMat, emissionMat=emissionMat)
    myhmm._predict(test_data)
    # Sci-kit HMM model
    skhmm = hmm.MultinomialHMM(n_components=2)
    skhmm.startprob_ = startProbs
    skhmm.transmat_ = transMat
    skhmm.emissionprob_ = emissionMat
    # result
    prob, h = skhmm.decode(np.array(test_data).reshape(-1, 1), algorithm='viterbi')
    showmsg = "The corresponding hidden states are: " + "->".join([hiddenStates[x] for x in h]) + " with log-probability: {:>.6f}".format(prob)
    print(showmsg)