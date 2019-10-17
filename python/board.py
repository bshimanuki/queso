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
def to_str(answer: np.ndarray) -> str:
	ret = (answer + ord('A')).tostring().decode('ascii')
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


def marginalize_trigram(
		prepre: Optional[np.ndarray],
		pre: Optional[np.ndarray],
		post: Optional[np.ndarray],
		postpost: Optional[np.ndarray]
) -> np.ndarray:
	num = sum(1 for x in (prepre, pre, post, postpost) if x is not None)
	if num == 0:
		p = ngram.unigram
	elif num == 1:
		p = ngram.bigram
	elif num == 2:
		p = ngram.trigram
	else:
		raise ValueError('too many parameters for marginalizing a trigram')
	if prepre is not None:
		p = prepre @ p
	if pre is not None:
		p = pre @ p
	if postpost is not None:
		p = p @ postpost
	if post is not None:
		p = p @ post
	return p


def marginalize_trigram_smoothed(
		prepre: Optional[np.ndarray],
		pre: Optional[np.ndarray],
		post: Optional[np.ndarray],
		postpost: Optional[np.ndarray]
) -> np.ndarray:
	before = marginalize_trigram(prepre, pre, None, None)
	mid = marginalize_trigram(None, pre, post, None)
	after = marginalize_trigram(None, None, post, postpost)
	ret =  (before * mid * after) ** (1 / 3)
	return ret


class Square(object):
	@property
	def is_cell(self) -> bool:
		return self.board.cells[self.y, self.x]

	@property
	def bar_below(self) -> bool:
		return self.board.bar_below[self.y, self.x]

	@property
	def bar_right(self) -> bool:
		return self.board.bar_right[self.y, self.x]

	@property
	def p(self) -> np.ndarray: # (direction, letter_dist)
		return self.board.p_cells[self.y, self.x]

	@property
	def p_next(self) -> np.ndarray: # (letter_dist,)
		return self.board.p_cells_next[self.y, self.x]

	def __init__(self, board: 'Board', y: int, x: int):
		# board
		self.board = board
		self.y = y
		self.x = x
		self.color = Color(self.board.background[self.y, self.x])
		self.number = None # type: Optional[int]

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
		direction = entry.direction # type: ignore # forward reference
		self.entries[direction] = entry
		self.entry_indices[direction] = len(entry)

	def update_p(self) -> None:
		if self.is_cell:
			for direction in Direction:
				entry = self.entries[direction]
				index = self.entry_indices[direction]
				if entry is not None:
					assert index is not None

					surrounding = [
						entry.cells[i].p[1 - direction] # type: ignore # forward reference
						if i >= 0 and i < len(entry)
						else None
						for i in range(index - 2, index + 3)
						if i != index
					]

					# remove self from aggregated factor in factor graph
					answer_ps = np.array(entry.p)
					for i, answer in enumerate(entry.answers):
						if answer is None:
							answer_ps[i] /= np.dot(self.p[1 - direction], ngram.uniform)
						else:
							answer_ps[i] /= self.p[1 - direction, answer[index]]
					answer_ps /= answer_ps.sum()

					self.p_next[direction] = 0
					for answer, answer_p in zip(entry.answers, answer_ps):
						if answer is None:
							# TODO: trigram factors
							self.p_next[direction] += answer_p * marginalize_trigram_smoothed(*surrounding)
						else:
							self.p_next[direction, answer[index]] += answer_p
					self.p_next[direction] /= self.p_next[direction].sum()


class Entry(object):
	@property
	def p_cells(self) -> np.ndarray: # (cell, letter_dist)
		# probability
		# shared memory going in the crossed direction
		return self.board.p_cells[self.slice + (1 - self.direction,)] # type: ignore # forward reference

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
			if self.direction == Direction.DOWN:
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

		if self.direction == Direction.DOWN:
			self.slice = (slice(start_y, start_y + self.length), start_x) # type: Union[Tuple[int, slice], Tuple[slice, int]]
		else:
			self.slice = (start_y, slice(start_x, start_x + self.length))
		self.cells = self.board.grid[self.slice]

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
				# self.p[i] *= np.prod(self.p_cells @ ngram.unigram)
				self.p[i] *= np.prod(self.p_cells @ ngram.uniform)
				# if len(self.p_cells) == 1:
					# self.p[i] *= ngram.unigram @ self.p_cells[0]
				# elif len(self.p_cells) > 1:
					# self.p[i] *= (ngram.bigram @ self.p_cells[1] @ self.p_cells[0]) ** (1 / 2)
					# self.p[i] *= (ngram.bigram @ self.p_cells[-1] @ self.p_cells[-2]) ** (1 / 2)
					# for j in range(len(self.p_cells) - 2):
						# self.p[i] *= (ngram.trigram @ self.p_cells[j+2] @ self.p_cells[j+1] @ self.p_cells[j]) ** (1 / 3)
			else:
				self.p[i] *= np.prod(np.choose(answer, self.p_cells.T))
		self.p /= self.p.sum()
		# if self.direction == Direction.ACROSS and self.number == 4:
			# print([None if answer is None else to_str(answer) for answer in self.answers])
			# print(self.p)


class Board(object):
	'''
	Class to Represent a board.
	'''
	@property
	def shape(self):
		return self.cells.shape

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

		self.cells = cells
		self.p_cells = np.tile(ngram.unigram[np.newaxis, np.newaxis, np.newaxis, :], self.shape + (2, 1)) # y, x, direction, letter
		self.p_cells_next = np.tile(ngram.unigram[np.newaxis, np.newaxis, np.newaxis, :], self.shape + (2, 1)) # y, x, direction, letter
		self.background = background
		self.bar_below = bar_below
		self.bar_right = bar_right

		self.grid = np.array([[Square(self, y, x) for x in range(self.shape[1])] for y in range(self.shape[0])])
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
		# self.p_cells, self.p_cells_next = self.p_cells_next, self.p_cells
		self.p_cells = np.average((self.p_cells, self.p_cells_next), axis=0)

	def update_entries(self) -> None:
		for entries in self.entries:
			for entry in entries:
				entry.update_p()
