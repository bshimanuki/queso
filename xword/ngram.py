import argparse
import functools
import os
from typing import Optional

import numpy as np

from .utils import normalize_unicode, to_uint

class NGram(object):
	# 0-25 are A-Z
	# 26 is ^
	# 27 is $
	n_alpha = 26
	n_extra = 2
	n = n_alpha + n_extra
	ONE_HOT_START = np.eye(1, n, 26, dtype=np.float)[0]
	ONE_HOT_END = np.eye(1, n, 27, dtype=np.float)[0]

	def __init__(self, fname : Optional[str]):
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
		self._uniform = np.ones_like(self._unigram_counts)
		self._uniform /= self._uniform.sum()

	def _compute(self, smooth=0.2) -> None:
		# TODO: compute frequencies
		self._unigram = self._unigram_counts / self._unigram_counts.sum()
		self._bigram = self._bigram_counts / self._bigram_counts.sum()
		self._trigram = self._trigram_counts / self._trigram_counts.sum()

		self._unigram = smooth * self._uniform + (1 - smooth) * self._unigram
		self._unigram /= self._unigram.sum()
		self._bigram = smooth * np.outer(self._unigram, self._unigram) + (1 - smooth) * self._bigram
		self._bigram /= self._bigram.sum()
		self._trigram = smooth * np.einsum('ij,j,jk->ijk', self._bigram, 1 / self._unigram, self._bigram) + (1 - smooth) * self._trigram
		self._trigram /= self._trigram.sum()

		self.computed = True

	def uniform(self, boundaries : bool = False) -> np.ndarray:
		if boundaries:
			return self._uniform
		else:
			p = self._uniform[:self.n_alpha]
			return p / p.sum()

	def unigram(self, boundaries : bool = False) -> np.ndarray:
		if not self.computed:
			self._compute()
		assert self._unigram is not None
		if boundaries:
			return self._unigram
		else:
			p = self._unigram[:self.n_alpha]
			return p / p.sum()

	def bigram(self, boundaries : bool = False) -> np.ndarray:
		if not self.computed:
			self._compute()
		assert self._bigram is not None
		if boundaries:
			return self._bigram
		else:
			p = self._bigram[:self.n_alpha, :self.n_alpha]
			return p / p.sum()

	def trigram(self, boundaries : bool = False) -> np.ndarray:
		if not self.computed:
			self._compute()
		assert self._trigram is not None
		if boundaries:
			return self._trigram
		else:
			p = self._trigram[:self.n_alpha, :self.n_alpha, :self.n_alpha]
			return p / p.sum()

root = os.path.dirname(os.path.dirname(__file__))
fname = os.path.join(root, 'data/ngrams.npz')
ngram = NGram(fname if os.path.exists(fname) else None)

def make_data(wordlist_file : str):
	unigram = np.ones((NGram.n,), dtype=np.int)
	bigram = np.ones((NGram.n,)*2, dtype=np.int)
	trigram = np.ones((NGram.n,)*3, dtype=np.int)
	with open(wordlist_file) as f:
		for line in f:
			answers = line.split('\t')[0]
			for answer in answers.split('/'):
				answer = normalize_unicode(answer)
				if not answer.isalpha():
					continue
				values = to_uint(answer, boundaries=True)
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
