import argparse
import functools
import os
import unicodedata

import numpy as np
from typing import Optional

class NGram(object):
    def __init__(self, fname : Optional[str]):
        self.n = 26
        self.computed = False
        self._unigram = None # type: Optional[np.ndarray]
        self._bigram = None # type: Optional[np.ndarray]
        self._trigram = None # type: Optional[np.ndarray]
        if fname:
            self.computed = True
            data = np.load(fname)
            self._unigram = data['unigram']
            self._bigram = data['bigram']
            self._trigram = data['trigram']


    def _compute(self) -> None:
        # TODO: compute frequencies
        self._unigram = np.ones((self.n,), dtype=np.float)
        self._bigram = np.ones((self.n,)*2, dtype=np.float)
        self._trigram = np.ones((self.n,)*3, dtype=np.float)

        self._unigram /= self._unigram.sum()
        self._bigram /= self._bigram.sum()
        self._trigram /= self._trigram.sum()

        self.computed = True

    @property
    def unigram(self) -> np.ndarray:
        if not self.computed:
            self._compute()
        return self._unigram

    @property
    def bigram(self) -> np.ndarray:
        if not self.computed:
            self._compute()
        return self._bigram

    @property
    def trigram(self) -> np.ndarray:
        if not self.computed:
            self._compute()
        return self._trigram

root = os.path.dirname(os.path.dirname(__file__))
fname = os.path.join(root, 'data/ngrams.npz')
ngram = NGram(fname if os.path.exists(fname) else None)

def make_data(wordlist_file : str):
    from board import to_uint
    n = 26
    unigram = np.ones((n,), dtype=np.float)
    bigram = np.ones((n,)*2, dtype=np.float)
    trigram = np.ones((n,)*3, dtype=np.float)
    with open(wordlist_file) as f:
        for line in f:
            answers = line.split('\t')[0]
            for answer in answers.split('/'):
                answer = unicodedata.normalize('NFD', answer)
                if not answer.isalpha():
                    continue
                values = to_uint(answer)
                for i in range(values.size):
                    unigram[values[i]] += 1
                    if i + 1 < values.size:
                        bigram[tuple(values[i:i+1])] += 1
                    if i + 2 < values.size:
                        trigram[tuple(values[i:i+2])] += 1

    unigram /= unigram.sum()
    bigram /= bigram.sum()
    trigram /= trigram.sum()
    np.savez(fname, unigram=unigram, bigram=bigram, trigram=trigram)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('wordlist')
    args = parser.parse_args()

    make_data(args.wordlist)
