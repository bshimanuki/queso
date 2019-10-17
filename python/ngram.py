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
            data = np.load(fname)
            self._unigram_counts = data['unigram_counts'].astype(np.float)
            assert self._unigram_counts.shape == (self.n,)
            self._bigram_counts = data['bigram_counts'].astype(np.float)
            assert self._bigram_counts.shape == (self.n, self.n)
            self._trigram_counts = data['trigram_counts'].astype(np.float)
            assert self._trigram_counts.shape == (self.n, self.n, self.n)
        else:
            self._unigram_counts = np.ones((self.n,), dtype=np.float)
            self._bigram_counts = np.ones((self.n,)*2, dtype=np.float)
            self._trigram_counts = np.ones((self.n,)*3, dtype=np.float)
        self.uniform = np.ones_like(self._unigram_counts)
        self.uniform /= self.uniform.sum()

    def _compute(self, smooth=0.2) -> None:
        # TODO: compute frequencies
        self._unigram = self._unigram_counts / self._unigram_counts.sum()
        self._bigram = self._bigram_counts / self._bigram_counts.sum()
        self._trigram = self._trigram_counts / self._trigram_counts.sum()

        self._unigram = smooth * self.uniform + (1 - smooth) * self._unigram
        self._unigram /= self._unigram.sum()
        self._bigram = smooth * np.outer(self._unigram, self._unigram) + (1 - smooth) * self._bigram
        self._bigram /= self._bigram.sum()
        self._trigram = smooth * np.einsum('ij,j,jk->ijk', self._bigram, 1 / self._unigram, self._bigram) + (1 - smooth) * self._trigram
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
    unigram = np.ones((n,), dtype=np.int)
    bigram = np.ones((n,)*2, dtype=np.int)
    trigram = np.ones((n,)*3, dtype=np.int)
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

    np.savez(fname, unigram_counts=unigram, bigram_counts=bigram, trigram_counts=trigram)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('wordlist')
    args = parser.parse_args()

    make_data(args.wordlist)
