import enum
import functools
import math

import numpy as np
import skimage
from typing import cast, List, Optional, Sequence, Tuple, Union

from ngram import ngram

'''
Represent a crossword board and fill it in with belief propagagrion using a markov random field model.
'''


def answerize(answer: str) -> str:
	return ''.join(c for c in answer.upper() if c.isalpha())
def to_uint(answer: str) -> np.ndarray:
	ret = np.fromstring(answer, dtype=np.uint8) - ord('A')
	return ret


class Direction(enum.IntEnum):
	DOWN = 0
	ACROSS = 1


class Color(object):
	def __init__(self, c: np.ndarray):
		if c.size == 1:
			self.rgb = np.full(3, c.item())
		elif c.size == 3:
			assert c.ndim == 1
			self.rgb = c
		else:
			raise ValueError()
		self.rgb = skimage.img_as_float(self.rgb)

	def format(self, **kwargs) -> str:
		output = kwargs.get('output', 'html')
		assert output in ('plain', 'html')
		strings = []
		if output == 'html':
			strings.append('#')
			for v_float in self.rgb:
				v_uint8 = min(255, math.floor(v_float * 256))
				strings.append('{:02x}'.format(v_uint8))
		return ''.join(strings)


class Square(object):
	def __init__(self, is_cell: bool, color: np.ndarray, bar_below: bool, bar_right: bool, p: np.ndarray):
		# board
		self.is_cell = is_cell
		self.color = Color(color)
		self.bar_below = bar_below
		self.bar_right = bar_right
		self.number = None # type: Optional[int]

		# p is shared memory with a Board for the probability distribution going down and across.
		self.p = p

		self.up = None # type: Optional[Square]
		self.left = None # type: Optional[Square]
		self.down = None # type: Optional[Square]
		self.right = None # type: Optional[Square]

		# entries
		self.entries = [None] * len(Direction) # type: List[Optional[Entry]]
		self.entry_indices = [None] * len(Direction) # type: List[Optional[int]]

	def get_contents(self, **kwargs) -> Tuple[str, float]:
		'''
		Get most probable value and probability.

		reduce_op is optionally function to combine down and across probabilities.
		'''
		reduce_op = kwargs.get('reduce_op', functools.partial(np.prod, axis=0))
		p = reduce_op(self.p)
		p /= p.sum()
		idx = np.argmax(p)
		prob = p[idx]
		c = chr(idx + ord('A'))
		return c, prob

	def format(self, **kwargs) -> str:
		output = kwargs.get('output', 'html')
		fill = kwargs.get('fill', False)
		number = kwargs.get('number', True)
		probabilities = kwargs.get('probabilities', False)
		assert output in ('plain', 'html')
		strings = []
		if output == 'html':
			border_style = '2px solid #000000'
			styles = []
			styles.append('background-color:{};'.format(self.color.format(**kwargs)))
			if self.up is None or self.up.bar_below:
				styles.append('border-top:{};'.format(border_style))
			if self.left is None or self.left.bar_right:
				styles.append('border-left:{};'.format(border_style))
			if self.down is None or self.bar_below:
				styles.append('border-bottom:{};'.format(border_style))
			if self.right is None or self.bar_right:
				styles.append('border-right:{};'.format(border_style))
			style = ''.join(styles)
			strings.append('<td style="{}">'.format(style))
		if number and self.number is not None:
			strings.append(str(self.number))
		if self.is_cell:
			if fill:
				c, prob = self.get_contents(**kwargs)
				strings.append(c)
			elif probabilities and not number:
				c, prob = self.get_contents(**kwargs)
				strings.append(str(prob))
		if output == 'html':
			strings.append('</td>')
		elif output == 'plain':
			strings.append(' ')
		return ''.join(strings)

	def set_entry(self, entry: 'Entry') -> None:
		direction = entry.direction # type: ignore forward reference
		self.entries[direction] = entry
		self.entry_indices[direction] = len(entry)

	def update_p(self) -> None:
		if self.is_cell:
			for p, entry, index in zip(self.p, self.entries, self.entry_indices):
				if entry is not None:
					assert index is not None
					p[...] = 0
					for answer, answer_p in zip(entry.answers, entry.p):
						if answer is None:
							# TODO: trigram factors
							p += answer_p * ngram.unigram
						else:
							p[answer[index]] += answer_p


class Entry(object):
	def __init__(self, board: 'Board', start: Tuple[int, int], direction: Direction):
		# board
		self.board = board
		start_y, start_x = start

		self.number = self.board.grid[start].number
		self.direction = direction
		self.length = 0

		cell = self.board.grid[start]
		while cell is not None and cell.is_cell:
			cell.set_entry(self)
			if direction == Direction.DOWN:
				if cell.bar_below:
					cell = None
				else:
					cell = cell.down
			else:
				if cell.bar_right:
					cell = None
				else:
					cell = cell.right
			self.length += 1

		if direction == Direction.DOWN:
			self.slice = (slice(start_y, start_y + self.length), start_x) # type: Union[Tuple[int, slice], Tuple[slice, int]]
		else:
			self.slice = (start_y, slice(start_x, start_x + self.length))
		self.cells = self.board.grid[self.slice]

		# probability
		# shared memory going in the crossed direction
		self.p_cells = self.board.p_cells[self.slice + (1 - self.direction,)]
		self.set_answers([None], [1])

	def __len__(self):
		return self.length

	def set_answers(self, answers: Sequence[Optional[str]], scores: Sequence[float]) -> None:
		'''Initialize possible answer probabilities. None is used for unknown.'''
		assert len(answers) == len(scores)
		for answer in answers:
			assert answer is None or len(answer) == len(self) and answer == answerize(answer)
		self.answers = [to_uint(answer) if answer is not None else None for answer in answers]
		self.scores = np.asarray(scores, dtype=np.float)
		self.p = self.scores / self.scores.sum()

	def update_p(self) -> None:
		self.p[...] = self.scores
		for i, answer in enumerate(self.answers):
			if answer is None:
				# TODO: trigrams?
				self.p[i] *= np.prod(self.p_cells @ ngram.unigram)
			else:
				self.p[i] *= np.prod(np.choose(answer, self.p_cells.T))
		self.p /= self.p.sum()


class Board(object):
	'''
	Class to Represent a board.
	'''
	def __init__(
			self,
			cells: np.ndarray,
			background: Optional[np.ndarray],
			numbered_cells: Optional[np.ndarray],
			bar_below: Optional[np.ndarray],
			bar_right: Optional[np.ndarray],
	):
		assert len(cells.shape) == 2
		if background is None:
			background = np.where(cells, np.ones(cells.shape + (3,)), np.zeros(cells.shape + (3,)))
		assert background.shape[:2] == cells.shape
		if numbered_cells is not None:
			assert numbered_cells.shape == cells.shape
		if bar_below is None:
			bar_below = np.zeros_like(cells)
		assert bar_below.shape == cells.shape
		if bar_right is None:
			bar_right = np.zeros_like(cells)
		assert bar_right.shape == cells.shape

		self.shape = cells.shape
		self.p_cells = np.tile(ngram.unigram[np.newaxis, np.newaxis, np.newaxis, :], self.shape + (2, 1)) # y, x, direction, letter

		self.grid = np.array([[Square(*args) for args in zip(*row)] for row in zip(cells, background, bar_below, bar_right, self.p_cells)])
		self.entries = [[], []] # type: List[List[Entry]]

		possibly_numbered_cells = np.zeros_like(cells)
		possibly_numbered_cells |= np.insert(np.logical_not(cells), 0, True, axis=0)[:-1]
		possibly_numbered_cells |= np.insert(np.logical_not(cells), 0, True, axis=1)[:,:-1]
		possibly_numbered_cells |= np.insert(np.logical_not(bar_below), 0, False, axis=0)[:-1]
		possibly_numbered_cells |= np.insert(np.logical_not(bar_right), 0, False, axis=1)[:,:-1]
		possibly_numbered_cells &= cells
		if numbered_cells is None:
			numbered_cells = possibly_numbered_cells
		if numbered_cells.any() and (numbered_cells != possibly_numbered_cells).all():
			print('Warning: numbered cells don\'t match board shape')

		# cell global properties
		n = 0
		for y in range(self.shape[0]):
			entry = None
			for x in range(self.shape[1]):
				square = self.grid[y, x]
				if numbered_cells[y, x]:
					n += 1
					square.number = n
				if y - 1 >= 0:
					square.up = self.grid[y-1, x];
				if x - 1 >= 0:
					square.left = self.grid[y, x-1];
				if y + 1 < self.grid.shape[0]:
					square.down = self.grid[y+1, x];
				if x + 1 < self.grid.shape[1]:
					square.right = self.grid[y, x+1];

		# make entries
		for y in range(self.shape[0]):
			entry = None
			for x in range(self.shape[1]):
				square = self.grid[y, x]
				if square.number:
					if x == 0 or not self.grid[y, x-1].is_cell or self.grid[y, x-1].bar_right:
						entry = Entry(self, (y, x), Direction.ACROSS)
						self.entries[Direction.ACROSS].append(entry)
		for x in range(self.shape[1]):
			entry = None
			for y in range(self.shape[0]):
				square = self.grid[y, x]
				if square.number:
					if y == 0 or not self.grid[y-1, x].is_cell or self.grid[y-1, x].bar_below:
						entry = Entry(self, (y, x), Direction.DOWN)
						self.entries[Direction.DOWN].append(entry)

	def format(self, **kwargs) -> str:
		output = kwargs.get('output', 'html')
		assert output in ('plain', 'html')
		strings = []
		if output == 'html':
			strings.append('<table><tbody>')
		for row in self.grid:
			if output == 'html':
				strings.append('<tr>')
			for square in row:
				strings.append(square.format(**kwargs))
			if output == 'html':
				strings.append('</tr>')
			elif output == 'plain':
				strings.append('\n')
		if output == 'html':
			strings.append('</tbody></table>')
		return ''.join(strings)

	def update_cells(self) -> None:
		for row in self.grid:
			for square in row:
				square.update_p()

	def update_entries(self) -> None:
		for entries in self.entries:
			for entry in entries:
				entry.update_p()
