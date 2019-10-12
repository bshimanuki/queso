import functools

import numpy as np

class NGram(object):
    def __init__(self):
        self.computed = False
        self._unigram = None # type: Optional[np.ndarray]
        self._bigram = None # type: Optional[np.ndarray]
        self._trigram = None # type: Optional[np.ndarray]
        self.n = 26

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

ngram = NGram()
