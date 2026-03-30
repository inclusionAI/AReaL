import random
import re
import time
import os
import string


def get_time_str():
    tt = time.localtime()
    time_str = ('%04d_%02d_%02d_%02d_%02d_%02d' %
                (tt.tm_year, tt.tm_mon, tt.tm_mday, tt.tm_hour, tt.tm_min, tt.tm_sec))
    return time_str


def encode(string, encoding='utf-8'):
    return string.encode(encoding)


def decode(string, encoding='utf-8'):
    return string.decode(encoding)


def multireplace(string, replacements):
    """
    Given a string and a replacement map, it returns the replaced string.
    :param str string: string to execute replacements on
    :param dict replacements: replacement dictionary {value to find: value to replace}
    :rtype: str
    Source https://gist.github.com/bgusach/a967e0587d6e01e889fd1d776c5f3729
    """
    # Place longer ones first to keep shorter substrings from matching where the longer ones should take place
    # For instance given the replacements {'ab': 'AB', 'abc': 'ABC'} against the string 'hey abc', it should produce
    # 'hey ABC' and not 'hey ABc'
    substrs = sorted(replacements, key=len, reverse=True)

    # Create a big OR regex that matches any of the substrings to replace
    regexp = re.compile('|'.join(map(re.escape, substrs)))

    # For each match, look up the new string in the replacements
    return regexp.sub(lambda match: replacements[match.group(0)], string)


class SetWithGet(set):
    def get_any(self):
        return random.sample(self, 1)[0]

    def __getitem__(self, item):
        return self.get_any()


class Noop(object):
    def noop(*args, **kw):
        pass

    def __getattr__(self, _):
        return self.noop


def walklevel(some_dir, level=1):
    some_dir = some_dir.rstrip(os.path.sep)
    assert os.path.isdir(some_dir)
    num_sep = some_dir.count(os.path.sep)
    for root, dirs, files in os.walk(some_dir):
        yield root, dirs, files
        num_sep_this = root.count(os.path.sep)
        if num_sep + level <= num_sep_this:
            del dirs[:]


def remove_spaces(s):
    cs = ' '.join(s.split())
    return cs


def remove_spaces_and_lower(s):
    cs = remove_spaces(s)
    cs = cs.lower()
    return cs


def remove_punctuation(s):
    cs = s.translate(str.maketrans('', '', string.punctuation))
    cs = remove_spaces_and_lower(cs)
    return cs