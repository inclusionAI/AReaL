import torch
import argparse
import os
import yaml
import numpy as np
import string
missing_words = set()


def to_np(x):
    if isinstance(x, np.ndarray):
        return x
    return x.data.cpu().numpy()


def to_pt(np_matrix, enable_cuda=False, type='long'):
    if type == 'long':
        if enable_cuda:
            return torch.autograd.Variable(torch.from_numpy(np_matrix).type(torch.LongTensor).cuda())
        else:
            return torch.autograd.Variable(torch.from_numpy(np_matrix).type(torch.LongTensor))
    elif type == 'float':
        if enable_cuda:
            return torch.autograd.Variable(torch.from_numpy(np_matrix).type(torch.FloatTensor).cuda())
        else:
            return torch.autograd.Variable(torch.from_numpy(np_matrix).type(torch.FloatTensor))


def apply_f(inp, filter):
    # inp is a list, after applying filter, shoudl return another list
    res = filter(inp)
    if not isinstance(res, tuple):
        res = [res]
    else:
        res = list(res)
    return res


def _words_to_ids(words, word2id):
    ids = []
    for word in words:
        ids.append(_word_to_id(word, word2id))
    return ids


def _word_to_id(word, word2id):
    try:
        return word2id[word]
    except KeyError:
        key = word + "_" + str(len(word2id))
        if key not in missing_words:
            print("Warning... %s is not in vocab, vocab size is %d..." % (word, len(word2id)))
            missing_words.add(key)
            with open("missing_words.txt", 'a+') as outfile:
                outfile.write(key + '\n')
                outfile.flush()
        return 1


def max_len(list_of_list):
    if len(list_of_list) == 0:
        return 0
    return max(map(len, list_of_list))


def pad_sequences(sequences, maxlen=None, dtype='int32', value=0.):
    '''
    Partially borrowed from Keras
    # Arguments
        sequences: list of lists where each element is a sequence
        maxlen: int, maximum length
        dtype: type to cast the resulting sequence.
        value: float, value to pad the sequences to the desired value.
    # Returns
        x: numpy array with dimensions (number_of_sequences, maxlen)
    '''
    if isinstance(sequences, np.ndarray):
        return sequences
    lengths = [len(s) for s in sequences]
    nb_samples = len(sequences)
    if maxlen is None:
        maxlen = np.max(lengths)
    # take the sample shape from the first non empty sequence
    # checking for consistency in the main loop below.
    sample_shape = tuple()
    for s in sequences:
        if len(s) > 0:
            sample_shape = np.asarray(s).shape[1:]
            break
    x = (np.ones((nb_samples, maxlen) + sample_shape) * value).astype(dtype)
    for idx, s in enumerate(sequences):
        if len(s) == 0:
            continue  # empty list was found
        # pre truncating
        trunc = s[-maxlen:]
        # check `trunc` has expected shape
        trunc = np.asarray(trunc, dtype=dtype)
        if trunc.shape[1:] != sample_shape:
            raise ValueError('Shape of sample %s of sequence at position %s is different from expected shape %s' %
                             (trunc.shape[1:], idx, sample_shape))
        # post padding
        x[idx, :len(trunc)] = trunc
    return x


def ez_gather_dim_1(input, index):
    if len(input.size()) == len(index.size()):
        return input.gather(1, index)
    res = []
    for i in range(input.size(0)):
        res.append(input[i][index[i][0]])
    return torch.stack(res, 0)


def preproc(s):
    s = s.replace("\n", ' ')
    # s = s.lower()
    while(True):
        if "  " in s:
            s = s.replace("  ", " ")
        else:
            break
    s = s.strip()
    if len(s) == 0:
        return "nothing"
    return s


class HistoryScoreCache:

    def __init__(self, capacity=1):
        self.capacity = capacity
        self.reset()

    def push(self, stuff):
        """stuff is float."""
        if len(self.memory) < self.capacity:
            self.memory.append(stuff)
        else:
            self.memory = self.memory[1:] + [stuff]

    def get_avg(self):
        return np.mean(np.array(self.memory))

    def reset(self):
        self.memory = []

    def __len__(self):
        return len(self.memory)


class LinearSchedule(object):
    """
    Linear interpolation between initial_p and final_p over
    schedule_timesteps. After this many timesteps pass final_p is
    returned.
    :param schedule_timesteps: (int) Number of timesteps for which to linearly anneal initial_p to final_p
    :param initial_p: (float) initial output value
    :param final_p: (float) final output value
    """

    def __init__(self, schedule_timesteps, final_p, initial_p=1.0):
        self.schedule_timesteps = schedule_timesteps
        self.final_p = final_p
        self.initial_p = initial_p
        if schedule_timesteps < 0:
            self.fixed = True
        else:
            self.schedule = np.linspace(initial_p, final_p, schedule_timesteps)
            self.fixed = False

    def value(self, step):
        if self.fixed:
            return self.initial_p
        if step < 0:
            return self.initial_p
        if step >= self.schedule_timesteps:
            return self.final_p
        else:
            return self.schedule[step]


def load_config():
    parser = argparse.ArgumentParser()
    parser.add_argument("config_file", help="path to config file")
    parser.add_argument("-p", "--params", nargs="+", metavar="my.setting=value", default=[],
                        help="override params of the config file,"
                             " e.g. -p 'training.gamma=0.95'")
    args = parser.parse_args()
    assert os.path.exists(args.config_file), "Invalid config file"
    with open(args.config_file) as reader:
        config = yaml.safe_load(reader)
    # Parse overriden params.
    for param in args.params:
        fqn_key, value = param.split("=")
        entry_to_change = config
        keys = fqn_key.split(".")
        for k in keys[:-1]:
            entry_to_change = entry_to_change[k]
        entry_to_change[keys[-1]] = value
    # print(config)
    return config


class EpisodicCountingMemory:

    def __init__(self):
        self.reset()

    def push(self, stuff):
        assert len(stuff) > 0  # batch size should be greater than 0
        if len(self.memory) == 0:
            for _ in range(len(stuff)):
                self.memory.append(set())

        for b in range(len(stuff)):
            key = stuff[b]
            self.memory[b].add(key)

    def is_a_new_state(self, stuff):
        assert len(stuff) > 0  # batch size should be greater than 0
        res = []
        for b in range(len(stuff)):
            key = stuff[b]
            res.append(float(key not in self.memory[b]))
        return res

    def reset(self):
        self.memory = []

    def __len__(self):
        return len(self.memory)


class ObjCentricEpisodicMemory:

    def __init__(self):
        self.reset()

    def get_objects(self, str):
        # TODO: replace with a more generic noun-phrase extractor
        str_no_punc = str.translate(str.maketrans('', '', string.punctuation))
        words = str_no_punc.split()
        objects = []
        for i,w in enumerate(words):
            if w.isdigit() and i > 0:
                object = words[i-1] + " " + w
                objects.append(object)
        return objects

    def push(self, stuff):
        assert len(stuff) > 0
        if len(self.memory) == 0:
            for _ in range(len(stuff)):
                self.memory.append(set())

        for b in range(len(stuff)):
            key = stuff[b]
            objects = self.get_objects(key)
            for obj in objects:
                self.memory[b].add(obj)

    def get_object_novelty_reward(self, stuff):
        assert len(stuff) > 0
        res = []
        for b in range(len(stuff)):
            key = stuff[b]
            objects = self.get_objects(key)
            if len(objects) > 0:
                num_unseen_objects = len([obj for obj in objects if obj not in self.memory[b]])
                res.append(float(num_unseen_objects) / len(objects))
            else:
                res.append(0.0)
        return res

    def reset(self):
        self.memory = []


class BeamSearchNode(object):
    def __init__(self, previous_node, input_target, log_prob, length):

        self.previous_node = previous_node
        self.input_target = input_target
        self.log_prob = log_prob
        self.length = length
        self.val = -self.log_prob / float(self.length - 1 + 1e-6)

    # defining comparators less_than and equals
    def __lt__(self, other):
        return self.val < other.val

    def __eq__(self, other):
        if other is None:
            return False
        if not isinstance(other, BeamSearchNode):
            return False
        return self.val == other.val
