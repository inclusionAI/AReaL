import torch
import math
import numpy as np
import h5py
import torch.nn.functional as F


def compute_mask(x):
    mask = torch.ne(x, 0).float()
    if x.is_cuda:
        mask = mask.cuda()
    return mask


def to_one_hot(y_true, n_classes):
    # y_true: batch x time
    batch_size, length = y_true.size(0), y_true.size(1)
    y_onehot = torch.FloatTensor(batch_size, length, n_classes)  # batch x time x n_class
    if y_true.is_cuda:
        y_onehot = y_onehot.cuda()
    y_onehot.zero_()
    y_onehot = y_onehot.view(-1, y_onehot.size(-1))  # batch*time x n_class
    y_true = y_true.view(-1, 1)  # batch*time x 1
    y_onehot.scatter_(1, y_true, 1)  # batch*time x n_class
    return y_onehot.view(batch_size, length, -1)


def NegativeLogLoss(y_pred, y_true, mask=None, smoothing_eps=0.0):
    """
    Shape:
        - y_pred:    batch x time x vocab
        - y_true:    batch x time
        - mask:      batch x time
    """
    y_true_onehot = to_one_hot(y_true, y_pred.size(-1))  # batch x time x vocab
    if smoothing_eps > 0.0:
        y_true_onehot = y_true_onehot * (1.0 - smoothing_eps) + (1.0 - y_true_onehot) * smoothing_eps / (y_pred.size(-1) - 1)
    P = y_true_onehot * y_pred  # batch x time x vocab
    P = torch.sum(P, dim=-1)  # batch x time
    gt_zero = torch.gt(P, 0.0).float()  # batch x time
    epsilon = torch.le(P, 0.0).float() * 1e-8  # batch x time
    log_P = torch.log(P + epsilon) * gt_zero  # batch x time
    if mask is not None:
        log_P = log_P * mask
    output = -torch.sum(log_P, dim=1)  # batch
    return output


def GetGenerationQValue(y_pred, y_true, mask):
    """
    Shape:
        - y_pred:    batch x time x vocab
        - y_true:    batch x time
        - mask:      batch x time
    """
    y_true_onehot = to_one_hot(y_true, y_pred.size(-1))  # batch x time x vocab
    P = y_true_onehot * y_pred  # batch x time x vocab
    P = torch.sum(P, dim=-1)  # batch x time
    
    P = P * mask
    mask_sum = torch.sum(mask, -1)  # batch
    tmp = torch.eq(mask_sum, 0).float()
    if P.is_cuda:
        tmp = tmp.cuda()
    mask_sum = mask_sum + tmp
    res = torch.sum(P, -1)  # batch
    res = res / mask_sum  # batch
    return res

    
def NLL(y_pred, y_true_onehot, mask=None):
    """
    Shape:
        - y_pred:    batch x n_class
        - y_true:    batch x n_class
        - mask:      batch x n_class
    """
    y_true_onehot = y_true_onehot.float()
    P = y_true_onehot * y_pred  # batch x n_class
    gt_zero = torch.gt(P, 0.0).float()  # batch x n_class
    epsilon = torch.le(P, 0.0).float() * 1e-8  # batch x n_class
    log_P = torch.log(P + epsilon) * gt_zero  # batch x n_class
    if mask is not None:
        log_P = log_P * mask
    output = -torch.sum(log_P, dim=1)  # batch
    return torch.mean(output)


def masked_softmax(x, m=None, axis=-1):
    '''
    Softmax with mask (optional)
    '''
    x = torch.clamp(x, min=-15.0, max=15.0)
    if m is not None:
        m = m.float()
        x = x * m
    e_x = torch.exp(x - torch.max(x, dim=axis, keepdim=True)[0])
    if m is not None:
        e_x = e_x * m
    softmax = e_x / (torch.sum(e_x, dim=axis, keepdim=True) + 1e-6)
    return softmax


def masked_mean(x, m=None, dim=1):
    """
        mean pooling when there're paddings
        input:  tensor: batch x time x h
                mask:   batch x time
        output: tensor: batch x h
    """
    if m is None:
        return torch.mean(x, dim=dim)
    x = x * m.unsqueeze(-1)
    mask_sum = torch.sum(m, dim=-1)  # batch
    tmp = torch.eq(mask_sum, 0).float()
    if x.is_cuda:
        tmp = tmp.cuda()
    mask_sum = mask_sum + tmp
    res = torch.sum(x, dim=dim)  # batch x h
    res = res / mask_sum.unsqueeze(-1)
    return res


def masked_ave_aggregator(x, mask):
    """
    ave aggregator of the node embedding
    input: tensor: batch x num_nodes x dim
    mask: tensor: batch x num_nodes

    """
    mask_sum = torch.sum(mask, -1).unsqueeze(-1) # batch x 1
    mask = mask.unsqueeze(-1) # batch x num_nodes x 1
    mask = mask.expand(-1, -1, x.shape[-1]) # batch x num_nodes x dim
    masked_x = x * mask # batch x num_nodes x dim
    sum_masked_x = torch.sum(masked_x, 1)
    ave_masked_x = sum_masked_x / mask_sum
    return ave_masked_x


class LayerNorm(torch.nn.Module):

    def __init__(self, input_dim):
        super(LayerNorm, self).__init__()
        self.gamma = torch.nn.Parameter(torch.ones(input_dim))
        self.beta = torch.nn.Parameter(torch.zeros(input_dim))
        self.eps = 1e-6

    def forward(self, x, mask=None):
        # x:        nbatch x hidden
        # mask:     nbatch
        mean = x.mean(-1, keepdim=True)
        std = torch.sqrt(x.var(dim=-1, keepdim=True) + self.eps)
        output = self.gamma * (x - mean) / (std + self.eps) + self.beta
        if mask is not None:
            return output * mask.unsqueeze(-1)
        else:
            return output


class H5EmbeddingManager(object):
    def __init__(self, h5_path):
        f = h5py.File(h5_path, 'r')
        self.W = np.array(f['embedding'])
        print("embedding data type=%s, shape=%s" % (type(self.W), self.W.shape))
        self.id2word = f['words_flatten'][0].split('\n')
        self.word2id = dict(zip(self.id2word, range(len(self.id2word))))

    def __getitem__(self, item):
        item_type = type(item)
        if item_type is str:
            index = self.word2id[item]
            embs = self.W[index]
            return embs
        else:
            raise RuntimeError("don't support type: %s" % type(item))

    def word_embedding_initialize(self, words_list, dim_size=300, scale=0.1, oov_init='random'):
        shape = (len(words_list), dim_size)
        self.rng = np.random.RandomState(42)
        if 'zero' == oov_init:
            W2V = np.zeros(shape, dtype='float32')
        elif 'one' == oov_init:
            W2V = np.ones(shape, dtype='float32')
        else:
            W2V = self.rng.uniform(low=-scale, high=scale, size=shape).astype('float32')
        W2V[0, :] = 0
        in_vocab = np.ones(shape[0], dtype=np.bool)
        word_ids = []
        for i, word in enumerate(words_list):
            if word in self.word2id:
                word_ids.append(self.word2id[word])
            else:
                in_vocab[i] = False
        W2V[in_vocab] = self.W[np.array(word_ids, dtype='int32')][:, :dim_size]
        return W2V


class Embedding(torch.nn.Module):
    '''
    inputs: x:          batch x ...
    outputs:embedding:  batch x ... x emb
            mask:       batch x ...
    '''

    def __init__(self, embedding_size, vocab_size, dropout_rate=0.0, trainable=True, id2word=None,
                 embedding_oov_init='random', load_pretrained=False, pretrained_embedding_path=None):
        super(Embedding, self).__init__()
        self.embedding_size = embedding_size
        self.vocab_size = vocab_size
        self.id2word = id2word
        self.dropout_rate = dropout_rate
        self.load_pretrained = load_pretrained
        self.embedding_oov_init = embedding_oov_init
        self.pretrained_embedding_path = pretrained_embedding_path
        self.trainable = trainable
        self.embedding_layer = torch.nn.Embedding(self.vocab_size, self.embedding_size, padding_idx=0)
        self.init_weights()

    def init_weights(self):
        init_embedding_matrix = self.embedding_init()
        if self.embedding_layer.weight.is_cuda:
            init_embedding_matrix = init_embedding_matrix.cuda()
        self.embedding_layer.weight = torch.nn.Parameter(init_embedding_matrix)
        if not self.trainable:
            self.embedding_layer.weight.requires_grad = False

    def embedding_init(self):
        # Embeddings
        if self.load_pretrained is False:
            word_embedding_init = np.random.uniform(low=-0.05, high=0.05, size=(self.vocab_size, self.embedding_size))
            word_embedding_init[0, :] = 0
        else:
            embedding_initr = H5EmbeddingManager(self.pretrained_embedding_path)
            word_embedding_init = embedding_initr.word_embedding_initialize(self.id2word,
                                                                            dim_size=self.embedding_size,
                                                                            oov_init=self.embedding_oov_init)
            del embedding_initr
        word_embedding_init = torch.from_numpy(word_embedding_init).float()
        return word_embedding_init

    def compute_mask(self, x):
        mask = torch.ne(x, 0).float()
        if x.is_cuda:
            mask = mask.cuda()
        return mask

    def forward(self, x):
        embeddings = self.embedding_layer(x)  # batch x time x emb
        embeddings = F.dropout(embeddings, p=self.dropout_rate, training=self.training)
        mask = self.compute_mask(x)  # batch x time
        return embeddings, mask


class FastUniLSTM(torch.nn.Module):
    """
    Adapted from https://github.com/facebookresearch/DrQA/
    now supports:   different rnn size for each layer
                    all zero rows in batch (from time distributed layer, by reshaping certain dimension)
    """

    def __init__(self, ninp, nhids, dropout_between_rnn_layers=0.):
        super(FastUniLSTM, self).__init__()
        self.ninp = ninp
        self.nhids = nhids
        self.nlayers = len(self.nhids)
        self.dropout_between_rnn_layers = dropout_between_rnn_layers
        self.stack_rnns()

    def stack_rnns(self):
        rnns = [torch.nn.LSTM(self.ninp if i == 0 else self.nhids[i - 1],
                              self.nhids[i],
                              num_layers=1,
                              bidirectional=False) for i in range(self.nlayers)]
        self.rnns = torch.nn.ModuleList(rnns)

    def forward(self, x, mask):

        def pad_(tensor, n):
            if n > 0:
                zero_pad = torch.autograd.Variable(torch.zeros((n,) + tensor.size()[1:]))
                if x.is_cuda:
                    zero_pad = zero_pad.cuda()
                tensor = torch.cat([tensor, zero_pad])
            return tensor

        """
        inputs: x:          batch x time x inp
                mask:       batch x time
        output: encoding:   batch x time x hidden[-1]
        """
        # Compute sorted sequence lengths
        batch_size = x.size(0)
        lengths = mask.data.eq(1).long().sum(1)  # .squeeze()
        _, idx_sort = torch.sort(lengths, dim=0, descending=True)
        _, idx_unsort = torch.sort(idx_sort, dim=0)

        lengths = list(lengths[idx_sort])
        idx_sort = torch.autograd.Variable(idx_sort)
        idx_unsort = torch.autograd.Variable(idx_unsort)

        # Sort x
        x = x.index_select(0, idx_sort)

        # remove non-zero rows, and remember how many zeros
        n_nonzero = np.count_nonzero(lengths)
        n_zero = batch_size - n_nonzero
        if n_zero != 0:
            lengths = lengths[:n_nonzero]
            x = x[:n_nonzero]

        # Transpose batch and sequence dims
        x = x.transpose(0, 1)

        # Pack it up
        rnn_input = torch.nn.utils.rnn.pack_padded_sequence(x, lengths)

        # Encode all layers
        outputs = [rnn_input]
        for i in range(self.nlayers):
            rnn_input = outputs[-1]

            # dropout between rnn layers
            if self.dropout_between_rnn_layers > 0:
                dropout_input = F.dropout(rnn_input.data,
                                          p=self.dropout_between_rnn_layers,
                                          training=self.training)
                rnn_input = torch.nn.utils.rnn.PackedSequence(dropout_input,
                                                              rnn_input.batch_sizes)
            seq, last = self.rnns[i](rnn_input)
            outputs.append(seq)
            if i == self.nlayers - 1:
                # last layer
                last_state = last[0]  # (num_layers * num_directions, batch, hidden_size)
                last_state = last_state[0]  # batch x hidden_size

        # Unpack everything
        for i, o in enumerate(outputs[1:], 1):
            outputs[i] = torch.nn.utils.rnn.pad_packed_sequence(o)[0]
        output = outputs[-1]

        # Transpose and unsort
        output = output.transpose(0, 1)  # batch x time x enc

        # re-padding
        output = pad_(output, n_zero)
        last_state = pad_(last_state, n_zero)

        output = output.index_select(0, idx_unsort)
        last_state = last_state.index_select(0, idx_unsort)

        # Pad up to original batch sequence length
        if output.size(1) != mask.size(1):
            padding = torch.zeros(output.size(0),
                                  mask.size(1) - output.size(1),
                                  output.size(2)).type(output.data.type())
            output = torch.cat([output, torch.autograd.Variable(padding)], 1)

        output = output.contiguous() * mask.unsqueeze(-1)
        return output, last_state, mask


class FastBiLSTM(torch.nn.Module):
    """
    Adapted from https://github.com/facebookresearch/DrQA/
    now supports:   different rnn size for each layer
                    all zero rows in batch (from time distributed layer, by reshaping certain dimension)
    """

    def __init__(self, ninp, nhids, dropout_between_rnn_layers=0.):
        super(FastBiLSTM, self).__init__()
        self.ninp = ninp
        self.nhids = [h // 2 for h in nhids]
        self.nlayers = len(self.nhids)
        self.dropout_between_rnn_layers = dropout_between_rnn_layers
        self.stack_rnns()

    def stack_rnns(self):
        rnns = [torch.nn.LSTM(self.ninp if i == 0 else self.nhids[i - 1] * 2,
                              self.nhids[i],
                              num_layers=1,
                              bidirectional=True) for i in range(self.nlayers)]
        self.rnns = torch.nn.ModuleList(rnns)

    def forward(self, x, mask):

        def pad_(tensor, n):
            if n > 0:
                zero_pad = torch.autograd.Variable(torch.zeros((n,) + tensor.size()[1:]))
                if x.is_cuda:
                    zero_pad = zero_pad.cuda()
                tensor = torch.cat([tensor, zero_pad])
            return tensor

        """
        inputs: x:          batch x time x inp
                mask:       batch x time
        output: encoding:   batch x time x hidden[-1]
        """
        # Compute sorted sequence lengths
        batch_size = x.size(0)
        lengths = mask.data.eq(1).long().sum(1)  # .squeeze()
        _, idx_sort = torch.sort(lengths, dim=0, descending=True)
        _, idx_unsort = torch.sort(idx_sort, dim=0)

        lengths = list(lengths[idx_sort])
        idx_sort = torch.autograd.Variable(idx_sort)
        idx_unsort = torch.autograd.Variable(idx_unsort)

        # Sort x
        x = x.index_select(0, idx_sort)

        # remove non-zero rows, and remember how many zeros
        n_nonzero = np.count_nonzero(lengths)
        n_zero = batch_size - n_nonzero
        if n_zero != 0:
            lengths = lengths[:n_nonzero]
            x = x[:n_nonzero]

        # Transpose batch and sequence dims
        x = x.transpose(0, 1)

        # Pack it up
        rnn_input = torch.nn.utils.rnn.pack_padded_sequence(x, lengths)

        # Encode all layers
        outputs = [rnn_input]
        for i in range(self.nlayers):
            rnn_input = outputs[-1]

            # dropout between rnn layers
            if self.dropout_between_rnn_layers > 0:
                dropout_input = F.dropout(rnn_input.data,
                                          p=self.dropout_between_rnn_layers,
                                          training=self.training)
                rnn_input = torch.nn.utils.rnn.PackedSequence(dropout_input,
                                                              rnn_input.batch_sizes)
            seq, last = self.rnns[i](rnn_input)
            outputs.append(seq)
            if i == self.nlayers - 1:
                # last layer
                last_state = last[0]  # (num_layers * num_directions, batch, hidden_size)
                last_state = torch.cat([last_state[0], last_state[1]], 1)  # batch x hid_f+hid_b

        # Unpack everything
        for i, o in enumerate(outputs[1:], 1):
            outputs[i] = torch.nn.utils.rnn.pad_packed_sequence(o)[0]
        output = outputs[-1]

        # Transpose and unsort
        output = output.transpose(0, 1)  # batch x time x enc

        # re-padding
        output = pad_(output, n_zero)
        last_state = pad_(last_state, n_zero)

        output = output.index_select(0, idx_unsort)
        last_state = last_state.index_select(0, idx_unsort)

        # Pad up to original batch sequence length
        if output.size(1) != mask.size(1):
            padding = torch.zeros(output.size(0),
                                  mask.size(1) - output.size(1),
                                  output.size(2)).type(output.data.type())
            output = torch.cat([output, torch.autograd.Variable(padding)], 1)

        output = output.contiguous() * mask.unsqueeze(-1)
        return output, last_state, mask


class LSTMCell(torch.nn.Module):

    """A basic LSTM cell."""

    def __init__(self, input_size, hidden_size, use_bias=True):
        """
        Most parts are copied from torch.nn.LSTMCell.
        """

        super(LSTMCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.use_bias = use_bias
        self.pre_act_linear = torch.nn.Linear(input_size + hidden_size, 4 * hidden_size, bias=False)
        if use_bias:
            self.bias_f = torch.nn.Parameter(torch.FloatTensor(hidden_size))
            self.bias_iog = torch.nn.Parameter(torch.FloatTensor(3 * hidden_size))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.pre_act_linear.weight.data)
        if self.use_bias:
            self.bias_f.data.fill_(1.0)
            self.bias_iog.data.fill_(0.0)

    def get_init_hidden(self, bsz, use_cuda):
        h_0 = torch.autograd.Variable(torch.FloatTensor(bsz, self.hidden_size).zero_())
        c_0 = torch.autograd.Variable(torch.FloatTensor(bsz, self.hidden_size).zero_())
        if use_cuda:
            h_0, c_0 = h_0.cuda(), c_0.cuda()
        return h_0, c_0

    def forward(self, input_, mask_=None, h_0=None, c_0=None):
        """
        Args:
            input_:     A (batch, input_size) tensor containing input features.
            mask_:      (batch)
            hx:         A tuple (h_0, c_0), which contains the initial hidden
                        and cell state, where the size of both states is
                        (batch, hidden_size).
        Returns:
            h_1, c_1: Tensors containing the next hidden and cell state.
        """
        if h_0 is None or c_0 is None:
            h_init, c_init = self.get_init_hidden(input_.size(0), use_cuda=input_.is_cuda)
            if h_0 is None:
                h_0 = h_init
            if c_0 is None:
                c_0 = c_init
        
        if mask_ is None:
            mask_ = torch.ones_like(torch.sum(input_, -1))
            if input_.is_cuda:
                mask_ = mask_.cuda()

        pre_act = self.pre_act_linear(torch.cat([input_, h_0], -1))  # batch x 4*hid
        if self.use_bias:
            pre_act = pre_act + torch.cat([self.bias_f, self.bias_iog]).unsqueeze(0)
        f, i, o, g = torch.split(pre_act, split_size_or_sections=self.hidden_size, dim=1)
        expand_mask_ = mask_.unsqueeze(1)  # batch x 1
        c_1 = torch.sigmoid(f) * c_0 + torch.sigmoid(i) * torch.tanh(g)
        c_1 = c_1 * expand_mask_ + c_0 * (1 - expand_mask_)
        h_1 = torch.sigmoid(o) * torch.tanh(c_1)
        h_1 = h_1 * expand_mask_ + h_0 * (1 - expand_mask_)
        return h_1, c_1

    def __repr__(self):
        s = '{name}({input_size}, {hidden_size})'
        return s.format(name=self.__class__.__name__, **self.__dict__)


class MatchLSTMAttention(torch.nn.Module):
    '''
        input:  p:          batch x inp_p
                p_mask:     batch
                q:          batch x time x inp_q
                q_mask:     batch x time
                h_tm1:      batch x out
                depth:      int
        output: z:          batch x inp_p+inp_q
    '''

    def __init__(self, input_p_dim, input_q_dim, output_dim):
        super(MatchLSTMAttention, self).__init__()
        self.input_p_dim = input_p_dim
        self.input_q_dim = input_q_dim
        self.output_dim = output_dim
        self.nlayers = len(self.output_dim)

        W_p_r = [torch.nn.Linear(self.output_dim[i] + (self.input_p_dim if i == 0 else self.output_dim[i - 1]), self.output_dim[i]) for i in range(self.nlayers)]
        W_q = [torch.nn.Linear(self.input_q_dim, self.output_dim[i]) for i in range(self.nlayers)]
        w = [torch.nn.Parameter(torch.FloatTensor(self.output_dim[i])) for i in range(self.nlayers)]
        match_b = [torch.nn.Parameter(torch.FloatTensor(1)) for i in range(self.nlayers)]

        self.W_p_r = torch.nn.ModuleList(W_p_r)
        self.W_q = torch.nn.ModuleList(W_q)
        self.w = torch.nn.ParameterList(w)
        self.match_b = torch.nn.ParameterList(match_b)
        self.init_weights()

    def init_weights(self):
        for i in range(self.nlayers):
            torch.nn.init.xavier_uniform_(self.W_p_r[i].weight.data)
            torch.nn.init.xavier_uniform_(self.W_q[i].weight.data)
            self.W_p_r[i].bias.data.fill_(0)
            self.W_q[i].bias.data.fill_(0)
            torch.nn.init.normal_(self.w[i].data, mean=0, std=0.05)
            self.match_b[i].data.fill_(1.0)

    def forward(self, input_p, mask_p, input_q, mask_q, h_tm1, depth):
        G_p_r = self.W_p_r[depth](torch.cat([input_p, h_tm1], -1)).unsqueeze(1)  # batch x None x out
        G_q = self.W_q[depth](input_q)  # batch x time x out
        G = torch.tanh(G_p_r + G_q)  # batch x time x out
        alpha = torch.matmul(G, self.w[depth])  # batch x time
        alpha = alpha + self.match_b[depth].unsqueeze(0)  # batch x time
        alpha = masked_softmax(alpha, mask_q, axis=-1)  # batch x time
        alpha = alpha.unsqueeze(1)  # batch x 1 x time
        # batch x time x input_q, batch x 1 x time
        z = torch.bmm(alpha, input_q)  # batch x 1 x input_q
        z = z.squeeze(1)  # batch x input_q
        z = torch.cat([input_p, z], 1)  # batch x input_p+input_q
        return z


class StackedMatchLSTM(torch.nn.Module):
    '''
    inputs: p:          batch x time x inp_p
            mask_p:     batch x time
            q:          batch x time x inp_q
            mask_q:     batch x time
    outputs:
            encoding:   batch x time x h
    Dropout types:
        dropout_between_rnn_layers -- if multi layer rnns
        dropout_in_rnn_weights -- rnn weight dropout
    '''

    def __init__(self, input_p_dim, input_q_dim, nhids, attention_layer,
                 dropout_between_rnn_layers=0.):
        super(StackedMatchLSTM, self).__init__()
        self.nhids = nhids
        self.input_p_dim = input_p_dim
        self.input_q_dim = input_q_dim
        self.attention_layer = attention_layer
        self.dropout_between_rnn_layers = dropout_between_rnn_layers
        self.nlayers = len(self.nhids)
        self.stack_rnns()

    def stack_rnns(self):
        rnns = [LSTMCell((self.input_p_dim + self.input_q_dim) if i == 0 else (self.nhids[i - 1] + self.input_q_dim), self.nhids[i], use_bias=True)
                for i in range(self.nlayers)]
        self.rnns = torch.nn.ModuleList(rnns)

    def get_init_hidden(self, bsz, use_cuda):
        weight = next(self.parameters()).data
        if use_cuda:
            return [[(torch.autograd.Variable(weight.new(bsz, self.nhids[i]).zero_()).cuda(),
                      torch.autograd.Variable(weight.new(bsz, self.nhids[i]).zero_()).cuda())]
                    for i in range(self.nlayers)]
        else:
            return [[(torch.autograd.Variable(weight.new(bsz, self.nhids[i]).zero_()),
                      torch.autograd.Variable(weight.new(bsz, self.nhids[i]).zero_()))]
                    for i in range(self.nlayers)]

    def forward(self, input_p, mask_p, input_q, mask_q):
        batch_size = input_p.size(0)
        state_stp = self.get_init_hidden(batch_size, use_cuda=input_p.is_cuda)

        for d, rnn in enumerate(self.rnns):
            for t in range(input_p.size(1)):
                input_mask = mask_p[:, t]
                if d == 0:
                    # 0th layer
                    curr_input = input_p[:, t]
                else:
                    curr_input = state_stp[d - 1][t][0]
                # apply dropout layer-to-layer
                drop_input = F.dropout(curr_input, p=self.dropout_between_rnn_layers, training=self.training) if d > 0 else curr_input
                previous_h, previous_c = state_stp[d][t]
                drop_input = self.attention_layer(drop_input, input_mask, input_q, mask_q, h_tm1=previous_h, depth=d)
                new_h, new_c = rnn(drop_input, input_mask, previous_h, previous_c)
                state_stp[d].append((new_h, new_c))

        states = [h[0] for h in state_stp[-1][1:]]  # list of batch x hid
        states = torch.stack(states, 1)  # batch x time x hid
        return states


class BiMatchLSTM(torch.nn.Module):
    '''
    inputs: p:          batch x time_p x emb
            mask_p:     batch x time_p
            q:          batch x time_q x emb
            mask_q:     batch x time_q
    outputs:encoding:   batch x time_p x hid
            last state: batch x hid
    Dropout types:
        dropouth -- dropout on hidden-to-hidden connections
    '''

    def __init__(self, input_p_dim, input_q_dim, nhids, dropout_between_rnn_layers=0.):
        super(BiMatchLSTM, self).__init__()
        self.nhids = nhids
        self.input_p_dim = input_p_dim
        self.input_q_dim = input_q_dim
        self.attention_layer = MatchLSTMAttention(input_p_dim, input_q_dim,
                                                  output_dim=[hid // 2 for hid in self.nhids])

        self.forward_rnn = StackedMatchLSTM(input_p_dim=self.input_p_dim,
                                            input_q_dim=self.input_q_dim,
                                            nhids=[hid // 2 for hid in self.nhids],
                                            attention_layer=self.attention_layer,
                                            dropout_between_rnn_layers=dropout_between_rnn_layers)

        self.backward_rnn = StackedMatchLSTM(input_p_dim=self.input_p_dim,
                                             input_q_dim=self.input_q_dim,
                                             nhids=[hid // 2 for hid in self.nhids],
                                             attention_layer=self.attention_layer,
                                             dropout_between_rnn_layers=dropout_between_rnn_layers)

    def forward(self, input_p, mask_p, input_q, mask_q):
        # forward pass
        forward_states = self.forward_rnn(input_p, mask_p, input_q, mask_q)
        forward_last_state = forward_states[:, -1]  # batch x hid/2

        # backward pass
        input_p_inverted = torch.flip(input_p, dims=[1])  # batch x time x p_dim (backward)
        mask_p_inverted = torch.flip(mask_p, dims=[1])  # batch x time (backward)
        backward_states = self.backward_rnn(input_p_inverted, mask_p_inverted, input_q, mask_q)
        backward_last_state = backward_states[:, -1]  # batch x hid/2
        backward_states = torch.flip(backward_states, dims=[1])  # batch x time x hid/2

        concat_states = torch.cat([forward_states, backward_states], -1)  # batch x time x hid
        concat_states = concat_states * mask_p.unsqueeze(-1)  # batch x time x hid
        concat_last_state = torch.cat([forward_last_state, backward_last_state], -1)  # batch x hid

        return concat_states, concat_last_state


class ActionScorerAttention(torch.nn.Module):

    def __init__(self, input_q_dim, output_dim, noisy_net=False):
        super(ActionScorerAttention, self).__init__()
        self.input_q_dim = input_q_dim
        self.output_dim = output_dim
        self.noisy_net = noisy_net

        if self.noisy_net:
            self.W_a = NoisyLinear(self.input_q_dim, self.output_dim)
        else:
            self.W_a = torch.nn.Linear(self.input_q_dim, self.output_dim)
        self.init_weights()

    def init_weights(self):
        if not self.noisy_net:
            torch.nn.init.xavier_uniform_(self.W_a.weight.data, gain=1)
            self.W_a.bias.data.fill_(0)

    def forward(self, H_q):
        # H_q: batch x inp_q
        Fk_prime = self.W_a(H_q)  # batch x out
        Fk_prime = torch.sigmoid(Fk_prime)  # batch x out
        return Fk_prime

    def reset_noise(self):
        if self.noisy_net:
            self.W_a.reset_noise()


class ActionScorerAttentionAdvantage(torch.nn.Module):

    def __init__(self, input_seq_dim, input_q_dim, output_dim, noisy_net=False):
        super(ActionScorerAttentionAdvantage, self).__init__()
        self.input_seq_dim = input_seq_dim
        self.input_q_dim = input_q_dim
        self.output_dim = output_dim
        self.noisy_net = noisy_net

        if self.noisy_net:
            self.V = NoisyLinear(self.input_seq_dim, self.output_dim)
            self.W_a = NoisyLinear(self.input_q_dim, self.output_dim)
        else:
            self.V = torch.nn.Linear(self.input_seq_dim, self.output_dim)
            self.W_a = torch.nn.Linear(self.input_q_dim, self.output_dim)
        self.v = torch.nn.Parameter(torch.FloatTensor(self.output_dim))
        self.init_weights()

    def init_weights(self):
        if not self.noisy_net:
            torch.nn.init.xavier_uniform_(self.V.weight.data, gain=1)
            torch.nn.init.xavier_uniform_(self.W_a.weight.data, gain=1)
            self.V.bias.data.fill_(0)
            self.W_a.bias.data.fill_(0)
        torch.nn.init.normal_(self.v.data, mean=0, std=0.05)

    def forward(self, H_r, mask_r, H_q):
        # H_r: batch x time x input_seq_dim
        # mask_r: batch x time
        # H_q: batch x inp_q
        batch_size, time = H_r.size(0), H_r.size(1)
        Fk = self.V(H_r.view(-1, H_r.size(2)))  # batch*time x out
        Fk_prime = self.W_a(H_q)  # batch x out
        Fk = Fk.view(batch_size, time, -1)  # batch x time x out
        Fk = torch.sigmoid(Fk + Fk_prime.unsqueeze(1))  # batch x time x out
        Fk = Fk * mask_r.unsqueeze(-1)  # batch x time x out
        return Fk

    def reset_noise(self):
        if self.noisy_net:
            self.V.reset_noise()
            self.W_a.reset_noise()


class BoundaryDecoderAttention(torch.nn.Module):

    def __init__(self, input_dim, output_dim, noisy_net=False):
        super(BoundaryDecoderAttention, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.noisy_net = noisy_net

        if self.noisy_net:
            self.V = NoisyLinear(self.input_dim, self.output_dim)
            self.W_a = NoisyLinear(self.output_dim, self.output_dim)
        else:
            self.V = torch.nn.Linear(self.input_dim, self.output_dim)
            self.W_a = torch.nn.Linear(self.output_dim, self.output_dim)
        self.v = torch.nn.Parameter(torch.FloatTensor(self.output_dim))
        self.c = torch.nn.Parameter(torch.FloatTensor(1))
        self.init_weights()

    def init_weights(self):
        if not self.noisy_net:
            torch.nn.init.xavier_uniform_(self.V.weight.data)
            torch.nn.init.xavier_uniform_(self.W_a.weight.data)
            self.V.bias.data.fill_(0)
            self.W_a.bias.data.fill_(0)
        torch.nn.init.normal_(self.v.data, mean=0, std=0.05)
        self.c.data.fill_(1.0)

    def forward(self, H_r, mask_r, h_tm1):
        # H_r: batch x time x inp
        # mask_r: batch x time
        # h_tm1: batch x out
        batch_size, time = H_r.size(0), H_r.size(1)
        Fk = self.V(H_r.view(-1, H_r.size(2)))  # batch*time x out
        Fk_prime = self.W_a(h_tm1)  # batch x out
        Fk = Fk.view(batch_size, time, -1)  # batch x time x out
        Fk = torch.tanh(Fk + Fk_prime.unsqueeze(1))  # batch x time x out

        beta = torch.matmul(Fk, self.v)  # batch x time
        beta = beta + self.c.unsqueeze(0)  # batch x time
        beta = masked_softmax(beta, mask_r, axis=-1)  # batch x time
        z = torch.bmm(beta.view(beta.size(0), 1, beta.size(1)), H_r)  # batch x 1 x inp
        z = z.view(z.size(0), -1)  # batch x inp
        return z, beta

    def reset_noise(self):
        if self.noisy_net:
            self.V.reset_noise()
            self.W_a.reset_noise()


class BoundaryDecoder(torch.nn.Module):
    '''
    input:  encoded stories:    batch x time x input_dim
            story mask:         batch x time
            init states:        batch x hid
            encoded question:   batch x input_dim_q
    '''

    def __init__(self, input_dim, hidden_dim, noisy_net=False):
        super(BoundaryDecoder, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.noisy_net = noisy_net
        self.attention_layer = BoundaryDecoderAttention(input_dim=input_dim,
                                                        output_dim=hidden_dim,
                                                        noisy_net=self.noisy_net)

        self.rnn = LSTMCell(self.input_dim, self.hidden_dim, use_bias=True)

    def forward(self, x, x_mask, h_0):

        state_stp = [(h_0, h_0)]
        beta_list = []
        mask = torch.autograd.Variable(torch.ones(x.size(0)))  # fake mask
        mask = mask.cuda() if x.is_cuda else mask
        for t in range(2):
            previous_h, previous_c = state_stp[t]
            curr_input, beta = self.attention_layer(x, x_mask, h_tm1=previous_h)
            new_h, new_c = self.rnn(curr_input, mask, previous_h, previous_c)
            state_stp.append((new_h, new_c))
            beta_list.append(beta)

        # beta list: list of batch x time
        res = torch.stack(beta_list, 2)  # batch x time x 2
        res = res * x_mask.unsqueeze(2)  # batch x time x 2
        return res

    def reset_noise(self):
        if self.noisy_net:
            self.attention_layer.reset_noise()


class NoisyLinear(torch.nn.Module):
    # Factorised NoisyLinear layer with bias
    def __init__(self, in_features, out_features, std_init=0.5):
        super(NoisyLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.std_init = std_init
        self.weight_mu = torch.nn.Parameter(torch.empty(out_features, in_features))
        self.weight_sigma = torch.nn.Parameter(torch.empty(out_features, in_features))
        self.register_buffer('weight_epsilon', torch.empty(out_features, in_features))
        self.bias_mu = torch.nn.Parameter(torch.empty(out_features))
        self.bias_sigma = torch.nn.Parameter(torch.empty(out_features))
        self.register_buffer('bias_epsilon', torch.empty(out_features))
        self.reset_parameters()
        self.reset_noise()

    def reset_parameters(self):
        mu_range = 1 / math.sqrt(self.in_features)
        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.weight_sigma.data.fill_(self.std_init / math.sqrt(self.in_features))
        self.bias_mu.data.uniform_(-mu_range, mu_range)
        self.bias_sigma.data.fill_(self.std_init / math.sqrt(self.out_features))

    def _scale_noise(self, size):
        x = torch.randn(size)
        return x.sign().mul_(x.abs().sqrt_())

    def reset_noise(self):
        epsilon_in = self._scale_noise(self.in_features)
        epsilon_out = self._scale_noise(self.out_features)
        self.weight_epsilon.copy_(epsilon_out.ger(epsilon_in))
        self.bias_epsilon.copy_(epsilon_out)

    def forward(self, input):
        if self.training:
            return F.linear(input, self.weight_mu + self.weight_sigma * self.weight_epsilon, self.bias_mu + self.bias_sigma * self.bias_epsilon)
        else:
            return F.linear(input, self.weight_mu, self.bias_mu)


################################# qanet

def PosEncoder(x, min_timescale=1.0, max_timescale=1.0e4):
    length = x.size(1)
    channels = x.size(2)
    signal = get_timing_signal(length, channels, min_timescale, max_timescale)
    return x + (signal.cuda() if x.is_cuda else signal)


def get_timing_signal(length, channels, min_timescale=1.0, max_timescale=1.0e4):
    position = torch.arange(length).type(torch.float32)
    num_timescales = channels // 2
    log_timescale_increment = (math.log(float(max_timescale) / float(min_timescale)) / (float(num_timescales)-1))
    inv_timescales = min_timescale * torch.exp(
            torch.arange(num_timescales).type(torch.float32) * -log_timescale_increment)
    scaled_time = position.unsqueeze(1) * inv_timescales.unsqueeze(0)
    signal = torch.cat([torch.sin(scaled_time), torch.cos(scaled_time)], dim = 1)
    m = torch.nn.ZeroPad2d((0, (channels % 2), 0, 0))
    signal = m(signal)
    signal = signal.view(1, length, channels)
    return signal


class DepthwiseSeparableConv(torch.nn.Module):
    def __init__(self, in_ch, out_ch, k, bias=True):
        super().__init__()
        self.depthwise_conv = torch.nn.Conv1d(in_channels=in_ch, out_channels=in_ch, kernel_size=k, groups=in_ch, padding=k // 2, bias=False)
        self.pointwise_conv = torch.nn.Conv1d(in_channels=in_ch, out_channels=out_ch, kernel_size=1, padding=0, bias=bias)

    def forward(self, x, mask):
        # x: batch x length x h
        # mask: batch x length
        x = x.transpose(1, 2)
        res = self.depthwise_conv(x) * mask.unsqueeze(1)
        res = torch.relu(self.pointwise_conv(res)) * mask.unsqueeze(1)
        res = res.transpose(1, 2)
        return res


class SimpleSelfAttention(torch.nn.Module):
    def __init__(self, hidden_size):
        super(SimpleSelfAttention, self).__init__()
        self.hidden_size = hidden_size
        self.att_weights = torch.nn.Parameter(torch.Tensor(hidden_size, 1), requires_grad=True)
        torch.nn.init.xavier_uniform(self.att_weights.data)

    def forward(self, input, mask):
        batch_size = input.size(0)
        # apply attention layer
        weights = torch.bmm(input, self.att_weights.unsqueeze(0).repeat(batch_size, 1, 1)).squeeze(-1)
        attentions = masked_softmax(weights, mask)
        # apply attention weights
        weighted = torch.mul(input, attentions.unsqueeze(-1).expand_as(input))
        return weighted, attentions


class ScaledDotProductAttention(torch.nn.Module):
    ''' Scaled Dot-Product Attention '''

    def __init__(self, temperature, attn_dropout=0.1):
        super().__init__()
        self.temperature = temperature
        self.dropout = torch.nn.Dropout(attn_dropout)

    def forward(self, q, k, v, mask):
        attn = torch.bmm(q, k.transpose(1, 2))
        attn = attn / self.temperature
        attn = masked_softmax(attn, mask, 2)
        attn = self.dropout(attn)
        output = torch.bmm(attn, v)
        return output, attn


class SelfAttention(torch.nn.Module):
    ''' From Multi-Head Attention module 
    https://github.com/jadore801120/attention-is-all-you-need-pytorch'''

    def __init__(self, block_hidden_dim, n_head, dropout=0.1):
        super().__init__()

        self.n_head = n_head
        self.block_hidden_dim = block_hidden_dim
        self.w_qs = torch.nn.Linear(block_hidden_dim, n_head * block_hidden_dim, bias=False)
        self.w_ks = torch.nn.Linear(block_hidden_dim, n_head * block_hidden_dim, bias=False)
        self.w_vs = torch.nn.Linear(block_hidden_dim, n_head * block_hidden_dim, bias=False)
        torch.nn.init.normal_(self.w_qs.weight, mean=0, std=np.sqrt(2.0 / (block_hidden_dim * 2)))
        torch.nn.init.normal_(self.w_ks.weight, mean=0, std=np.sqrt(2.0 / (block_hidden_dim * 2)))
        torch.nn.init.normal_(self.w_vs.weight, mean=0, std=np.sqrt(2.0 / (block_hidden_dim * 2)))
        self.attention = ScaledDotProductAttention(temperature=np.power(block_hidden_dim, 0.5))
        self.fc = torch.nn.Linear(n_head * block_hidden_dim, block_hidden_dim)
        self.layer_norm = torch.nn.LayerNorm(self.block_hidden_dim)
        torch.nn.init.xavier_normal_(self.fc.weight)
        self.dropout = torch.nn.Dropout(dropout)

    def forward(self, q, mask, k, v):
        # q: batch x len_q x hid
        # k: batch x len_k x hid
        # v: batch x len_v x hid
        # mask: batch x len_q x len_k
        batch_size, len_q = q.size(0), q.size(1)
        len_k, len_v = k.size(1), v.size(1)
        assert mask.size(1) == len_q
        assert mask.size(2) == len_k
        residual = q

        q = self.w_qs(q).view(batch_size, len_q, self.n_head, self.block_hidden_dim)
        k = self.w_ks(k).view(batch_size, len_k, self.n_head, self.block_hidden_dim)
        v = self.w_vs(v).view(batch_size, len_v, self.n_head, self.block_hidden_dim)

        q = q.permute(2, 0, 1, 3).contiguous().view(-1, len_q, self.block_hidden_dim) # (n*b) x lq x dk
        k = k.permute(2, 0, 1, 3).contiguous().view(-1, len_k, self.block_hidden_dim) # (n*b) x lk x dk
        v = v.permute(2, 0, 1, 3).contiguous().view(-1, len_v, self.block_hidden_dim) # (n*b) x lv x dv

        mask = mask.repeat(self.n_head, 1, 1) # (n*b) x .. x ..
        output, attn = self.attention(q, k, v, mask=mask)

        output = output.view(self.n_head, batch_size, len_q, self.block_hidden_dim)
        output = output.permute(1, 2, 0, 3).contiguous().view(batch_size, len_q, -1) # b x lq x (n*dv)
        output = self.dropout(self.fc(output))
        output = self.layer_norm(output + residual)

        return output, attn


class EncoderBlock(torch.nn.Module):
    def __init__(self, conv_num, ch_num, k, block_hidden_dim, n_head, dropout):
        super().__init__()
        self.dropout = dropout
        self.convs = torch.nn.ModuleList([DepthwiseSeparableConv(ch_num, ch_num, k) for _ in range(conv_num)])
        self.self_att = SelfAttention(block_hidden_dim, n_head, dropout)
        self.FFN_1 = torch.nn.Linear(ch_num, ch_num)
        self.FFN_2 = torch.nn.Linear(ch_num, ch_num)
        self.norm_C = torch.nn.ModuleList([torch.nn.LayerNorm(block_hidden_dim) for _ in range(conv_num)])
        self.norm_1 = torch.nn.LayerNorm(block_hidden_dim)
        self.norm_2 = torch.nn.LayerNorm(block_hidden_dim)
        self.conv_num = conv_num

    def forward(self, x, mask, squared_mask, l, blks):
        total_layers = (self.conv_num + 2) * blks
        # conv layers
        out = PosEncoder(x)
        out = out * mask.unsqueeze(-1)
        for i, conv in enumerate(self.convs):
            res = out
            out = self.norm_C[i](out)
            if (i) % 2 == 0:
                out = F.dropout(out, p=self.dropout, training=self.training)
            out = conv(out, mask)
            out = self.layer_dropout(out, res, self.dropout * float(l) / total_layers)
            out = out * mask.unsqueeze(-1)
            l += 1
        res = out
        out = self.norm_1(out)
        out = F.dropout(out, p=self.dropout, training=self.training)
        # self attention
        out, _ = self.self_att(out, squared_mask, out, out)
        out = self.layer_dropout(out, res, self.dropout * float(l) / total_layers)
        out = out * mask.unsqueeze(-1)
        l += 1
        res = out
        out = self.norm_2(out)
        out = F.dropout(out, p=self.dropout, training=self.training)
        # fully connected layers
        out = self.FFN_1(out)
        out = torch.relu(out)
        out = self.FFN_2(out)
        out = self.layer_dropout(out, res, self.dropout * float(l) / total_layers)
        out = out * mask.unsqueeze(-1)
        l += 1
        return out

    def layer_dropout(self, inputs, residual, dropout):
        if self.training == True:
            pred = torch.empty(1).uniform_(0, 1) < dropout
            if pred:
                return residual
            else:
                return F.dropout(inputs, dropout, training=self.training) + residual
        else:
            return inputs + residual


class DecoderBlock(torch.nn.Module):
    def __init__(self, ch_num, k, block_hidden_dim, n_head, dropout):
        super().__init__()
        self.dropout = dropout
        self.self_att = SelfAttention(block_hidden_dim, n_head, dropout)
        self.obs_att = SelfAttention(block_hidden_dim, n_head, dropout)
        self.FFN_0 = torch.nn.Linear(block_hidden_dim, block_hidden_dim)
        self.FFN_1 = torch.nn.Linear(ch_num, ch_num)
        self.FFN_2 = torch.nn.Linear(ch_num, ch_num)
        self.norm_1 = torch.nn.LayerNorm(block_hidden_dim)
        self.norm_2 = torch.nn.LayerNorm(block_hidden_dim)

    def forward(self, x, mask, self_att_mask, obs_enc_representations, obs_mask, l, blks):
        total_layers = blks * 3
        # conv layers
        out = PosEncoder(x)
        res = out
        # self attention
        out, _ = self.self_att(out, self_att_mask, out, out)
        out_self = out * mask.unsqueeze(-1)
        out = self.layer_dropout(out_self, res, self.dropout * float(l) / total_layers)
        l += 1
        res = out
        out = self.norm_1(out)
        out = F.dropout(out, p=self.dropout, training=self.training)
        # attention with encoder outputs
        out_obs, obs_attention = self.obs_att(out, obs_mask, obs_enc_representations, obs_enc_representations)

        out = self.FFN_0(out_obs)
        out = torch.relu(out)
        out = out * mask.unsqueeze(-1)

        out = self.layer_dropout(out, res, self.dropout * float(l) / total_layers)
        l += 1
        res = out
        out = self.norm_2(out)
        out = F.dropout(out, p=self.dropout, training=self.training)
        # Fully connected layers
        out = self.FFN_1(out)
        out = torch.relu(out)
        out = self.FFN_2(out)
        out = out * mask.unsqueeze(-1)
        out = self.layer_dropout(out, res, self.dropout * float(l) / total_layers)
        l += 1
        return out, out_self, out_obs, obs_attention

    def layer_dropout(self, inputs, residual, dropout):
        if self.training == True:
            pred = torch.empty(1).uniform_(0, 1) < dropout
            if pred:
                return residual
            else:
                return F.dropout(inputs, dropout, training=self.training) + residual
        else:
            return inputs + residual


class DecoderBlockForObsGen(torch.nn.Module):
    def __init__(self, ch_num, k, block_hidden_dim, n_head, dropout):
        super().__init__()
        self.dropout = dropout
        self.self_att = SelfAttention(block_hidden_dim, n_head, dropout)
        self.obs_att = SelfAttention(block_hidden_dim, n_head, dropout)
        self.node_att = SelfAttention(block_hidden_dim, n_head, dropout)
        self.FFN_0 = torch.nn.Linear(block_hidden_dim * 2, block_hidden_dim)
        self.FFN_1 = torch.nn.Linear(ch_num, ch_num)
        self.FFN_2 = torch.nn.Linear(ch_num, ch_num)
        self.norm_1 = torch.nn.LayerNorm(block_hidden_dim)
        self.norm_2 = torch.nn.LayerNorm(block_hidden_dim)

    def forward(self, x, mask, self_att_mask, prev_action_enc_representations, prev_action_mask, node_enc_representations, node_mask, l, blks):
        total_layers = blks * 3
        # conv layers
        out = PosEncoder(x)
        res = out
        # self attention
        out, _ = self.self_att(out, self_att_mask, out, out)
        out_self = out * mask.unsqueeze(-1)
        out = self.layer_dropout(out_self, res, self.dropout * float(l) / total_layers)
        l += 1
        res = out
        out = self.norm_1(out)
        out = F.dropout(out, p=self.dropout, training=self.training)
        # attention with encoder outputs
        out_obs, obs_attention = self.obs_att(out, prev_action_mask, prev_action_enc_representations, prev_action_enc_representations)
        out_node, _ = self.node_att(out, node_mask, node_enc_representations, node_enc_representations)

        out = torch.cat([out_obs, out_node], -1)
        out = self.FFN_0(out)
        out = torch.relu(out)
        out = out * mask.unsqueeze(-1)

        out = self.layer_dropout(out, res, self.dropout * float(l) / total_layers)
        l += 1
        res = out
        out = self.norm_2(out)
        out = F.dropout(out, p=self.dropout, training=self.training)
        # Fully connected layers
        out = self.FFN_1(out)
        out = torch.relu(out)
        out = self.FFN_2(out)
        out = out * mask.unsqueeze(-1)
        out = self.layer_dropout(out, res, self.dropout * float(l) / total_layers)
        l += 1
        return out, out_self #, out_obs, obs_attention

    def layer_dropout(self, inputs, residual, dropout):
        if self.training == True:
            pred = torch.empty(1).uniform_(0, 1) < dropout
            if pred:
                return residual
            else:
                return F.dropout(inputs, dropout, training=self.training) + residual
        else:
            return inputs + residual


class CQAttention(torch.nn.Module):
    def __init__(self, block_hidden_dim, dropout=0):
        super().__init__()
        self.dropout = dropout
        w4C = torch.empty(block_hidden_dim, 1)
        w4Q = torch.empty(block_hidden_dim, 1)
        w4mlu = torch.empty(1, 1, block_hidden_dim)
        torch.nn.init.xavier_uniform_(w4C)
        torch.nn.init.xavier_uniform_(w4Q)
        torch.nn.init.xavier_uniform_(w4mlu)
        self.w4C = torch.nn.Parameter(w4C)
        self.w4Q = torch.nn.Parameter(w4Q)
        self.w4mlu = torch.nn.Parameter(w4mlu)

        bias = torch.empty(1)
        torch.nn.init.constant_(bias, 0)
        self.bias = torch.nn.Parameter(bias)

    def forward(self, C, Q, Cmask, Qmask):
        S = self.trilinear_for_attention(C, Q, Cmask, Qmask)
        Cmask = Cmask.unsqueeze(-1)
        Qmask = Qmask.unsqueeze(1)
        S1 = masked_softmax(S, Qmask, axis=2)
        S2 = masked_softmax(S, Cmask, axis=1)
        S1 = S1 * Qmask * Cmask
        S2 = S2 * Qmask * Cmask
        A = torch.bmm(S1, Q)
        B = torch.bmm(torch.bmm(S1, S2.transpose(1, 2)), C)
        out = torch.cat([C, A, torch.mul(C, A), torch.mul(C, B)], dim=2)
        return out

    def trilinear_for_attention(self, C, Q, Cmask, Qmask):
        C = F.dropout(C, p=self.dropout, training=self.training)
        Q = F.dropout(Q, p=self.dropout, training=self.training)
        max_q_len = Q.size(-2)
        max_context_len = C.size(-2)
        subres0 = torch.matmul(C, self.w4C).expand([-1, -1, max_q_len])
        subres1 = torch.matmul(Q, self.w4Q).transpose(1, 2).expand([-1, max_context_len, -1])
        subres2 = torch.matmul(C * self.w4mlu, Q.transpose(1, 2))
        res = subres0 + subres1 + subres2
        res += self.bias
        res = res * Qmask.unsqueeze(1) * Cmask.unsqueeze(-1)
        return res


class AnswerPointer(torch.nn.Module):
    def __init__(self, block_hidden_dim, noisy_net=False):
        super().__init__()
        self.noisy_net = noisy_net
        if self.noisy_net:
            self.w_head_1 = NoisyLinear(block_hidden_dim * 2, block_hidden_dim)
            self.w_tail_1 = NoisyLinear(block_hidden_dim * 2, block_hidden_dim)
            self.w_head_2 = NoisyLinear(block_hidden_dim, 1)
            self.w_tail_2 = NoisyLinear(block_hidden_dim, 1)
            # self.w_head_2 = NoisyLinear(block_hidden_dim * 2, 1)
            # self.w_tail_2 = NoisyLinear(block_hidden_dim * 2, 1)
        else:
            self.w_head_1 = torch.nn.Linear(block_hidden_dim * 2, block_hidden_dim)
            self.w_tail_1 = torch.nn.Linear(block_hidden_dim * 2, block_hidden_dim)
            self.w_head_2 = torch.nn.Linear(block_hidden_dim, 1)
            self.w_tail_2 = torch.nn.Linear(block_hidden_dim, 1)
            # self.w_head_2 = torch.nn.Linear(block_hidden_dim * 2, 1)
            # self.w_tail_2 = torch.nn.Linear(block_hidden_dim * 2, 1)

    def forward(self, M1, M2, M3, mask):
        X_head = torch.cat([M1, M2], dim=-1)
        X_tail = torch.cat([M1, M3], dim=-1)
        X_head = torch.relu(self.w_head_1(X_head))
        X_tail = torch.relu(self.w_tail_1(X_tail))
        Y_head = masked_softmax(self.w_head_2(X_head).squeeze(), mask, axis=1)
        Y_tail = masked_softmax(self.w_tail_2(X_tail).squeeze(), mask, axis=1)
        return torch.stack([Y_head, Y_tail], -1)

    def reset_noise(self):
        if self.noisy_net:
            self.w_head_1.reset_noise()
            self.w_tail_1.reset_noise()
            self.w_head_2.reset_noise()
            self.w_tail_2.reset_noise()


class Highway(torch.nn.Module):
    def __init__(self, layer_num, size, dropout=0):
        super().__init__()
        self.n = layer_num
        self.dropout = dropout
        self.linear = torch.nn.ModuleList([torch.nn.Linear(size, size) for _ in range(self.n)])
        self.gate = torch.nn.ModuleList([torch.nn.Linear(size, size) for _ in range(self.n)])

    def forward(self, x):
        #x: shape [batch_size, length, hidden_size]
        for i in range(self.n):
            gate = torch.sigmoid(self.gate[i](x))
            nonlinear = self.linear[i](x)
            nonlinear = F.dropout(nonlinear, p=self.dropout, training=self.training)
            x = gate * nonlinear + (1 - gate) * x
        return x


class MergeEmbeddings(torch.nn.Module):
    def __init__(self, block_hidden_dim, word_emb_dim, char_emb_dim, dropout=0):
        super().__init__()
        self.conv2d = torch.nn.Conv2d(char_emb_dim, block_hidden_dim, kernel_size = (1, 5), padding=0, bias=True)
        torch.nn.init.kaiming_normal_(self.conv2d.weight, nonlinearity='relu')

        self.linear = torch.nn.Linear(word_emb_dim + block_hidden_dim, block_hidden_dim, bias=False)
        self.high = Highway(2, size=block_hidden_dim, dropout=dropout)

    def forward(self, word_emb, char_emb, mask=None):
        char_emb = char_emb.permute(0, 3, 1, 2)  # batch x emb x time x nchar
        char_emb = self.conv2d(char_emb)  # batch x block_hidden_dim x time x nchar-5+1
        if mask is not None:
            char_emb = char_emb * mask.unsqueeze(1).unsqueeze(-1)
        char_emb = F.relu(char_emb)  # batch x block_hidden_dim x time x nchar-5+1
        char_emb, _ = torch.max(char_emb, dim=3)  # batch x emb x time
        char_emb = char_emb.permute(0, 2, 1)  # batch x time x emb
        emb = torch.cat([char_emb, word_emb], dim=2)
        emb = self.linear(emb)
        emb = self.high(emb)
        if mask is not None:
            emb = emb * mask.unsqueeze(-1)
        return emb


class GraphConvolution(torch.nn.Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = torch.nn.Linear(self.in_features, self.out_features, bias=False)
        if bias:
            self.bias = torch.nn.Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.weight.weight.data)
        if self.bias is not None:
            self.bias.data.fill_(0)

    def forward(self, input, adj):
        # input: batch x num_entity x in_dim
        # adj:   batch x num_entity x num_entity
        support = self.weight(input)  # batch x num_entity x out_dim
        output = torch.bmm(adj, support)  # batch x num_entity x out_dim
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'


class StackedGraphConvolution(torch.nn.Module):
    '''
    input:  entity features:    batch x num_entity x input_dim
            adjacency matrix:   batch x num_entity x num_entity
    '''

    def __init__(self, input_dim, hidden_dims, dropout_rate=0.0):
        super(StackedGraphConvolution, self).__init__()
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.dropout_rate = dropout_rate
        self.nlayers = len(self.hidden_dims)
        self.stack_rnns()

    def stack_rnns(self):
        gcns = [GraphConvolution(self.input_dim if i == 0 else self.hidden_dims[i - 1], self.hidden_dims[i])
                for i in range(self.nlayers)]
        self.gcns = torch.nn.ModuleList(gcns)

    def forward(self, x, adj):
        for i in range(self.nlayers):
            res = self.gcns[i](x, adj)  # batch x num_nodes x hid
            x = F.relu(res)
            x = F.dropout(x, self.dropout_rate, training=self.training)

        return res


class PointerSoftmax(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.pointer_softmax_context = torch.nn.Linear(input_dim, hidden_dim)
        self.pointer_softmax_target = torch.nn.Linear(input_dim, hidden_dim)
        self.pointer_softmax_squash = torch.nn.Linear(hidden_dim, 1)

    def forward(self, target_target_representations, target_source_representations, trg_decoder_output, target_mask, target_source_attention, source_mask, input_source):
        # target_target_representations:    batch x target_len x hid
        # target_source_representations:    batch x target_len x hid
        # trg_decoder_output:               batch x target len x vocab
        # target mask:                      batch x target len
        # target_source_attention:          batch x target len x source len
        # source mask:                      batch x source len
        # input source:                     batch x source len
        batch_size = target_source_attention.size(0)
        target_len = target_source_attention.size(1)
        source_len = target_source_attention.size(2)

        switch = self.pointer_softmax_context(target_source_representations)  # batch x trg_len x hid
        switch = switch  + self.pointer_softmax_target(target_target_representations)  # batch x trg_len x hid
        switch = torch.tanh(switch)
        switch = switch * target_mask.unsqueeze(-1)
        switch = self.pointer_softmax_squash(switch).squeeze(-1)  # batch x trg_len
        switch = torch.sigmoid(switch)
        switch = switch * target_mask  # batch x target len
        switch = switch.unsqueeze(-1)  # batch x target len x 1

        target_source_attention = target_source_attention * source_mask.unsqueeze(1)
        from_vocab = trg_decoder_output  # batch x target len x vocab
        from_source = torch.autograd.Variable(torch.zeros(batch_size * target_len, from_vocab.size(-1)))  # batch x target len x vocab
        if from_vocab.is_cuda:
            from_source = from_source.cuda()
        input_source = input_source.unsqueeze(1).expand(batch_size, target_len, source_len)
        input_source = input_source.contiguous().view(batch_size * target_len, -1)  # batch*target_len x source_len
        from_source = from_source.scatter_add_(1, input_source, target_source_attention.view(batch_size * target_len, -1))
        from_source = from_source.view(batch_size, target_len, -1)  # batch x target_len x vocab
        merged = switch * from_vocab + (1.0 - switch) * from_source  # batch x target_len x vocab
        merged = merged * target_mask.unsqueeze(-1)
        return merged


class RelationalGraphConvolution(torch.nn.Module):
    """
    Simple R-GCN layer, modified from theano/keras implementation from https://github.com/tkipf/relational-gcn
    We also consider relation representation here (relation labels matter)
    """

    def __init__(self, entity_input_dim, relation_input_dim, num_relations, out_dim, bias=True, num_bases=0):
        super(RelationalGraphConvolution, self).__init__()
        self.entity_input_dim = entity_input_dim
        self.relation_input_dim = relation_input_dim
        self.out_dim = out_dim
        self.num_relations = num_relations
        self.num_bases = num_bases

        if self.num_bases > 0:
            self.bottleneck_layer = torch.nn.Linear((self.entity_input_dim + self.relation_input_dim) * self.num_relations, self.num_bases, bias = False)
            self.weight = torch.nn.Linear(self.num_bases, self.out_dim, bias=False)
        else:
            self.weight = torch.nn.Linear((self.entity_input_dim + self.relation_input_dim) * self.num_relations, self.out_dim, bias=False)
        if bias:
            self.bias = torch.nn.Parameter(torch.FloatTensor(self.out_dim))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.weight.weight.data)
        if self.bias is not None:
            self.bias.data.fill_(0)

    def forward(self, node_features, relation_features, adj):
        # node_features: batch x num_entity x in_dim
        # relation_features: batch x num_relation x in_dim
        # adj:   batch x num_relations x num_entity x num_entity
        supports = []
        for relation_idx in range(self.num_relations):
            _r_features = relation_features[:, relation_idx: relation_idx + 1]  # batch x 1 x in_dim
            _r_features = _r_features.repeat(1, node_features.size(1), 1)  # batch x num_entity x in_dim
            supports.append(torch.bmm(adj[:, relation_idx], torch.cat([node_features, _r_features], dim=-1)))  # batch x num_entity x in_dim+in_dim
        supports = torch.cat(supports, dim=-1)  # batch x num_entity x (in_dim+in_dim)*num_relations
        if self.num_bases > 0:
            supports = self.bottleneck_layer(supports)
        output = self.weight(supports)  # batch x num_entity x out_dim

        if self.bias is not None:
            return output + self.bias
        else:
            return output


class StackedRelationalGraphConvolution(torch.nn.Module):
    '''
    input:  entity features:    batch x num_entity x input_dim
            relation features:  batch x num_relations x input_dim
            adjacency matrix:   batch x num_relations x num_entity x num_entity
    '''

    def __init__(self, entity_input_dim, relation_input_dim, num_relations, hidden_dims, num_bases, use_highway_connections=False, dropout_rate=0.0, real_valued_graph=False):
        super(StackedRelationalGraphConvolution, self).__init__()
        self.entity_input_dim = entity_input_dim
        self.relation_input_dim = relation_input_dim
        self.hidden_dims = hidden_dims
        self.num_relations = num_relations
        self.dropout_rate = dropout_rate
        self.num_bases = num_bases
        self.real_valued_graph = real_valued_graph
        self.nlayers = len(self.hidden_dims)
        self.stack_gcns()
        self.use_highway_connections = use_highway_connections
        if self.use_highway_connections:
            self.stack_highway_connections()

    def stack_highway_connections(self):
        highways = [torch.nn.Linear(self.hidden_dims[i], self.hidden_dims[i]) for i in range(self.nlayers)]
        self.highways = torch.nn.ModuleList(highways)
        self.input_linear = torch.nn.Linear(self.entity_input_dim, self.hidden_dims[0])

    def stack_gcns(self):
        gcns = [RelationalGraphConvolution(self.entity_input_dim if i == 0 else self.hidden_dims[i - 1], self.relation_input_dim, self.num_relations, self.hidden_dims[i], num_bases=self.num_bases)
                for i in range(self.nlayers)]
        self.gcns = torch.nn.ModuleList(gcns)

    def forward(self, node_features, relation_features, adj):
        x = node_features
        for i in range(self.nlayers):
            if self.use_highway_connections:
                if i == 0:
                    prev = self.input_linear(x)
                else:
                    prev = x.clone()
            x = self.gcns[i](x, relation_features, adj)  # batch x num_nodes x hid
            if self.real_valued_graph:
                x = torch.sigmoid(x)
            else:
                x = F.relu(x)
            x = F.dropout(x, self.dropout_rate, training=self.training)
            if self.use_highway_connections:
                gate = torch.sigmoid(self.highways[i](x))
                x = gate * x + (1 - gate) * prev
        return x


class DGIDiscriminator(torch.nn.Module):
    def __init__(self, n_h):
        super(DGIDiscriminator, self).__init__()
        self.f_k = torch.nn.Bilinear(n_h, n_h, 1)
        for m in self.modules():
            self.weights_init(m)

    def weights_init(self, m):
        if isinstance(m, torch.nn.Bilinear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def forward(self, c, h_pl, h_mi, s_bias1=None, s_bias2=None):
        c_x = torch.unsqueeze(c, 1)
        c_x = c_x.expand_as(h_pl)
        c_x = c_x.contiguous()

        sc_1 = torch.squeeze(self.f_k(h_pl, c_x), 2)
        sc_2 = torch.squeeze(self.f_k(h_mi, c_x), 2)

        if s_bias1 is not None:
            sc_1 += s_bias1
        if s_bias2 is not None:
            sc_2 += s_bias2

        logits = torch.cat((sc_1, sc_2), 1)

        return logits


class ObservationDiscriminator(torch.nn.Module):
    def __init__(self, n_h):
        super(ObservationDiscriminator, self).__init__()
        self.f_k = torch.nn.Bilinear(2 * n_h, n_h, 1)
        for m in self.modules():
            self.weights_init(m)

    def weights_init(self, m):
        if isinstance(m, torch.nn.Bilinear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def forward(self, c, h_p, p_mask, h_n, n_mask, s_bias1=None, s_bias2=None):
        masked_ave_hp = masked_mean(h_p, p_mask)
        masked_ave_hn = masked_mean(h_n, n_mask)

        sc_1 = self.f_k(c, masked_ave_hp)
        sc_2 = self.f_k(c, masked_ave_hn)

        logits = torch.cat([sc_1, sc_2], dim=0)
        return logits


class BoxFeaturesFC(torch.nn.Module):
    def __init__(self, in_features=1024, out_features=64):
        super(BoxFeaturesFC, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.linear1 = torch.nn.Linear(in_features=self.in_features, out_features=256)
        self.linear2 = torch.nn.Linear(in_features=256, out_features=self.out_features)

    def pad_and_stack(self, obs, max_feat_len=None):
        if not max_feat_len:
            max_feat_len = max([o.shape[0] for o in obs])
        padded = [F.pad(input=o, pad=(0, 0, 0, max_feat_len-o.shape[0]), mode='constant', value=0) for o in obs]
        return torch.stack(padded, dim=0)

    def forward(self, x):
        x = self.pad_and_stack(x)
        x = self.linear1(x)
        x = self.linear2(x)
        return x