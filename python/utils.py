import unicodedata

import numpy as np


def answerize(answer: str) -> str:
	answer = normalize_unicode(answer)
	answer = answer.upper()
	return ''.join(c for c in answer if c.isalpha())

def to_uint(answer: str, boundaries : bool = False) -> np.ndarray:
	ret = np.fromstring(answer, dtype=np.uint8) - ord('A')
	if boundaries:
		ret = np.concatenate(([26], ret, [27]), axis=-1)
	return ret

def to_str(answer: np.ndarray) -> str:
	ret = (answer + ord('A')).tostring().decode('ascii')
	return ret

def normalize_unicode(s : str) -> str:
	return unicodedata.normalize('NFKD', s).encode('ascii', 'ignore').decode('ascii')
