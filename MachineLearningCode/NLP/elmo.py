'''
    Implementing ELMo with tensorflow
'''
import pickle, os
import tensorflow as tf
import numpy as np

from keras.preprocessing.sequence import pad_sequences


class dataload(object):
    """docstring for dataload"""
    def __init__(self, filepath, minfreq):
        super(dataload, self).__init__()
        self.filepath = filepath
        self.minfreq = minfreq
        self._loaddata()
        np.random.shuffle(self.intdata)

    def _loaddata(self):
        rawdata = []
        charcount = {'<num>': 0, '<eng>': 0}
        texts = open(self.filepath, encoding='utf8')
        for text in texts.readlines():
            label, txt = text.split('    ')
            txt.replace('-_-', '')
            tmp_sen = []
            for idx in range(len(txt)):
                if txt[idx] == '\n':
                    continue
                if 'a' <= txt[idx] <= 'z' or 'A' <= txt[idx] <= 'Z':
                    charcount['<eng>'] += 1
                    if tmp_sen and tmp_sen[-1] == '<eng>':
                        continue
                    tmp_sen.append('<eng>')
                elif '0' <= txt[idx] <= '9':
                    charcount['<num>'] += 1
                    if tmp_sen and tmp_sen[-1] == '<num>':
                        continue
                    tmp_sen.append('<num>')
                else:
                    if txt[idx] in charcount.keys():
                        charcount[txt[idx]] += 1
                    else:
                        charcount[txt[idx]] = 1
                    tmp_sen.append(txt[idx])
            rawdata.append([label, tmp_sen[:-1]])

        sortchar = sorted([[v, k] for k, v in charcount.items()])
        self.id2char = {0: '<pad>', 1: '<S>', 2: '</S>', 3: '<unk>'}
        self.char2id = {'<pad>': 0, '<S>': 1, '</S>': 2, '<unk>': 3}
        idx = 4
        for v, k in sortchar:
            if v < self.minfreq:
                continue
            self.id2char[idx] = k
            self.char2id[k] = idx
            idx += 1

        self.intdata = []
        length = []
        for label, raw in rawdata:
            tmp = [self.char2id[char] if char in self.char2id.keys() else 1 for char in raw]
            length.append(len(tmp))
            self.intdata.append([[1] + tmp, tmp + [2]])
        self.intdata = np.array(self.intdata)
        self.avglen = sum(length)/len(length)


class bidirectionalLanguageModel(object):
    """docstring for bidirectionalLanguageModel"""
    def __init__(self, paras):
        super(bidirectionalLanguageModel, self).__init__()
        self.paras = paras

    def _initial(self):
        self.inputx = tf.placeholder(tf.int32, [self.paras['batch_size'], self.paras['sentence_len']], name='inputx')
        self.targety = tf.placeholder(tf.int32, [self.paras['batch_size'], self.paras['sentence_len']], name='targety')
        self.embeddings = tf.Variable(tf.random_normal(self.paras['embedding_setups']), name='embeddings')
        self.embedinput = tf.nn.embedding_lookup(self.embeddings, self.inputx)

    def _arch(self):
        n_layers = self.paras['n_layers']
        lstm_dim = self.paras['lstm']['lstm_dim']
        proj_dim = self.paras['lstm']['proj_dim']
        cell_clip = self.paras['lstm']['cell_clip']
        proj_clip = self.paras['lstm']['proj_clip']
        batch_size = self.paras['batch_size']

        mask = self.inputx > 0
        seq_len = tf.reduce_sum(tf.cast(mask, tf.int32), axis=1)
        self.forward_output, self.backward_output = [], []
        for i in range(2):
            if i == 1:
                _input = self.embedinput
            else:
                _input = tf.reverse_sequence(
                    self.embedinput,
                    seq_len,
                    seq_axis=1,
                    batch_axis=0)
            for lyr in range(1, n_layers+1):

                lstm_cell = tf.nn.rnn_cell.LSTMCell(
                    lstm_dim, num_proj=proj_dim,
                    cell_clip=cell_clip, proj_clip=proj_clip)
                lstm_cell = tf.nn.rnn_cell.ResidualWrapper(lstm_cell)

                init_state = lstm_cell.zero_state(batch_size, dtype=tf.float32)

                if i == 0:
                    name = 'Layer_{}/forward'.format(lyr)
                else:
                    name = 'Layer_{}/backward'.format(lyr)
                with tf.variable_scope(name):
                    output, output_states = tf.nn.dynamic_rnn(
                        lstm_cell,
                        _input,
                        sequence_length=seq_len,
                        initial_state=init_state,
                        dtype=tf.float32)
                    if i == 0:
                        self.forward_output.append(output)
                    else:
                        output = tf.reverse_sequence(
                            output,
                            seq_len,
                            seq_axis=1,
                            batch_axis=0)
                        self.backward_output.append(output)
                _input = output
        print(self.forward_output, self.backward_output)

    def _loss(self):
        softmax_w = tf.tile(tf.expand_dims(tf.transpose(self.embeddings, [1, 0]), axis=0), [self.forward_output[-1].shape[0], 1, 1])
        softmax_b = tf.get_variable(
            name='bias', shape=self.embeddings.shape[0],
            dtype=tf.float32, initializer=tf.constant_initializer(0.0))
        forward_softmax = tf.matmul(self.forward_output[-1], softmax_w) + softmax_b
        backward_softmax = tf.matmul(self.backward_output[-1], softmax_w) + softmax_b
        forward_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
            logits=forward_softmax,
            labels=self.targety)
        backward_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
            logits=backward_softmax,
            labels=self.targety)
        self.loss = 0.5 * (tf.reduce_mean(forward_loss) + tf.reduce_mean(backward_loss))

    def _graph(self):
        graph = tf.Graph()
        with graph.as_default():
            self._initial()
            self._arch()
            self._loss()
            self.optimizer = tf.train.AdamOptimizer(self.paras['lr'])
            self.train_op = self.optimizer.minimize(self.loss, name='train_op')
            print(tf.trainable_variables())
        return graph

    def _train(self, datas):
        graph_cfg = tf.ConfigProto()
        graph_cfg.gpu_options.allow_growth = True
        graph = self._graph()
        batch_size = self.paras['batch_size']
        seq_lens = self.paras['sentence_len']
        with tf.Session(graph=graph, config=graph_cfg) as sess:
            sess.run(tf.global_variables_initializer())
            saver = tf.train.Saver(tf.global_variables())
            for epoch in range(1, 1+self.paras['epochs']):
                if epoch > 2:
                    self.paras['lr'] /= 2
                num_batchs = len(datas)//batch_size
                losses = 0.0
                for batch in range(1, 1+num_batchs):
                    x = pad_sequences(datas[(batch-1)*batch_size: batch*batch_size, 0], maxlen=seq_lens, padding='post', truncating='post')
                    y = pad_sequences(datas[(batch-1)*batch_size: batch*batch_size, 1], maxlen=seq_lens, padding='post', truncating='post')
                    _, loss = sess.run(
                        fetches=(self.train_op, self.loss),
                        feed_dict={
                            self.inputx: x,
                            self.targety: y
                        })
                    losses += loss
                    if batch % 5 == 0:
                        showmsg = 'Epoch:{:>3}/{:>3} - batch:{:>4}/{:>4} - loss:{:>.4f}'.format(
                            epoch, self.paras['epochs'], batch, num_batchs, losses/5)
                        losses = 0.0
                        print(showmsg)
                    if batch % 100 == 0:
                        saver.save(sess, r'..\data\elmo\model.ckpt')

    def _extract(self, tgt, length):
        graph_cfg = tf.ConfigProto()
        graph_cfg.gpu_options.allow_growth = True
        output_vecs = ['embeddings:0',
                       ['Layer_1/forward/rnn/transpose_1:0', 'Layer_2/forward/rnn/transpose_1:0'],
                       ['Layer_1/backward/ReverseSequence:0', 'Layer_2/backward/ReverseSequence:0']]
        with tf.Session(config=graph_cfg) as sess:
            saver = tf.train.import_meta_graph(r'..\data\elmo\model.ckpt.meta')
            saver.restore(sess, r'..\data\elmo\model.ckpt')
            res = sess.run(
                fetches=output_vecs,
                feed_dict={'inputx:0': tgt})
            # embeddings = res[0]
            # forward_1, forword_2 = res[1][0], res[1][1]
            # backward_1, backword_2 = res[2][0], res[2][1]
            [embeddings, [forward_1, forward_2], [backward_1, backward_2]] = res
            sen_embed = np.array([embeddings[x] for x in tgt[1, :length]])
            print('embedding:', sen_embed)
            print('Forward 1:', forward_1[0, 0:length, :])
            print('Forward 2:', forward_2[0, 0:length, :])
            print('Backward 1:', backward_1[0, 0:length, :])
            print('Backward 2:', backward_2[0, 0:length, :])



def str2batchdata(x, char2id, batchsize, seq_length):
    data = np.zeros((batchsize, seq_length), dtype=np.int32)
    inputx = [[1] + [char2id[ch] if ch in char2id.keys() else char2id['<unk>'] for ch in x]]
    length = len(inputx[0])
    inputx = pad_sequences(inputx, maxlen=seq_length, padding='post', truncating='post')
    data[0] = inputx
    return data, length


data = dataload(filepath=r'..\data\tsb_hotel.txt', minfreq=5)
paras = {
    'epochs': 5,
    'batch_size': 8,
    'sentence_len': 150,
    'embedding_setups': [len(data.char2id), 64],
    'n_layers': 2,
    'lstm': {
        'lstm_dim': 256,
        'proj_dim': 64,
        'cell_clip': 5,
        'proj_clip': 5
    },
    'lr': 0.01
}
elmo = bidirectionalLanguageModel(paras)
# elmo._train(data.intdata)
test = '距离川沙公路较近,但是公交指示不对,如果是"蔡陆线"的话,会非常麻烦.建议用别的路线.房间较为简单.'
tdata, length = str2batchdata(test, data.char2id, 8, 150)
elmo._extract(tdata, length)