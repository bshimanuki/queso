import asyncio
from collections import defaultdict
import copy
import enum
import functools
import math
import operator
import re
from typing import cast, Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union
import warnings

import aiohttp
import numpy as np
import skimage

from clues import Tracker
from ngram import ngram
from utils import answerize, to_str, to_uint

'''
Represent a crossword board and fill it in with belief propagagrion using a markov random field model.
'''


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
	ret =  (before * mid**2 * after) ** (1 / 4)
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
		display = kwargs.get('display', True)
		fill = kwargs.get('fill', False)
		number = kwargs.get('number', True)
		probabilities = kwargs.get('probabilities', False)
		cell_order = kwargs.get('cell_order', None)
		num_cells = kwargs.get('num_cells', None)
		color_probability = kwargs.get('color_probability', not number)
		assert output in ('plain', 'html')
		strings = []
		if output == 'html':
			border_style = '2px solid #000000'
			styles = []
			if display:
				if self.is_cell and color_probability and cell_order is not None:
					assert num_cells is not None
					v = cell_order[self.y, self.x]
					percentile = v / (num_cells - 1) if num_cells > 1 else 0.5
					white = np.array([1, 1, 1], dtype=np.float)
					red = np.array([230, 124, 115], dtype=np.float) / 255
					green = np.array([87, 187, 138], dtype=np.float) / 255
					colors = np.stack((red, white, green))
					rgb = np.array([np.interp(percentile, [0, 0.5, 1], _color) for _color in colors.T])
					color = Color(rgb)
				else:
					color = self.color
				styles.append('background-color:{};'.format(color.format(**kwargs)))
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
		if display:
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


@functools.total_ordering
class Entry(object):
	def __init__(self, board: 'Board', start: Tuple[int, int], direction: Direction):
		# board
		self.board = board
		start_y, start_x = start

		self.number = self.board.grid[start].number
		self.direction = direction
		self.length = 0
		self.clue = None # type: Optional[str]

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

	@property
	def p_cells(self) -> np.ndarray: # (cell, letter_dist)
		# probability
		# shared memory going in the crossed direction
		return self.board.p_cells[self.slice + (1 - self.direction,)] # type: ignore # forward reference

	@property
	def name(self) -> str:
		return '{}-{}'.format(self.number, self.direction.name)

	def __len__(self):
		return self.length

	def __lt__(self, other):
		return self.number < other.number

	def set_answers(self, answers: Sequence[Optional[str]], scores: Sequence[float], clue=Optional[str]) -> None:
		'''Initialize possible answer probabilities. None is used for unknown.'''
		assert len(answers) == len(scores)
		# sort by score descending then answer alphabetical
		answers, scores = zip(*sorted(zip(answers, scores), key=lambda pair: (np.inf, np.inf) if pair[0] is None else (-pair[1], pair[0])))
		for answer in answers:
			assert answer is None or len(answer) == len(self) and answer == answerize(answer)
		if clue is not None:
			self.clue = clue
		self.answers = [to_uint(answer) if answer is not None else None for answer in answers]
		self.scores = np.asarray(scores, dtype=np.float)
		self.p = self.scores / self.scores.sum()

	async def use_clue(self, clue : str, session : aiohttp.ClientSession, weight_for_unknown : float) -> None:
		answer_scores = await Tracker.aggregate_scores(clue, session, length_guess=len(self))
		answers = [None] # type: List[Optional[str]]
		weights = [weight_for_unknown]
		for answer, score in answer_scores.items():
			# TODO: weight function, partial answer, rebus
			answer = answerize(answer)
			weight = score
			if len(answer) == len(self):
				answers.append(answer)
				weights.append(weight)
		self.set_answers(answers, weights)
		self.clue = clue

	def update_p(self) -> None:
		self.p[...] = self.scores
		for i, answer in enumerate(self.answers):
			if answer is None:
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
		self.entries[Direction.DOWN] = sorted(self.entries[Direction.DOWN])

	def format(self, **kwargs) -> str:
		return self.format_multiple(board_args=(({},),), **kwargs)

	def format_multiple(self, board_kwargs : Optional[Iterable[Iterable[Optional[Dict[str, Any]]]]] = None, padding=2, **kwargs):
		'''
		board_args should be a 2d grid of keyword arguments to pass to squares or None for empty space.
		padding is the number of width inserted between boards.
		'''
		output = kwargs.get('output', 'html')
		assert output in ('plain', 'html')
		if board_kwargs is None:
			board_kwargs = (
				(
					{'fill': False, 'number': True, 'probabilities': False},
					{'fill': True, 'number': False, 'probabilities': False},
				),
				(
					{'fill': False, 'number': False, 'probabilities': True, 'reduce_op': operator.itemgetter(0)},
					{'fill': False, 'number': False, 'probabilities': True, 'reduce_op': operator.itemgetter(1)},
				),
				(
					{'fill': True, 'number': False, 'probabilities': False, 'reduce_op': operator.itemgetter(0)},
					{'fill': True, 'number': False, 'probabilities': False, 'reduce_op': operator.itemgetter(1)},
				),
				(
					{'fill': False, 'number': False, 'probabilities': True},
					None,
				),
			)
		else:
			board_kwargs = copy.deepcopy(board_kwargs)
		for row_board_kwargs in board_kwargs:
			for square_board_kwargs in row_board_kwargs:
				if square_board_kwargs is not None:
					for key, value in kwargs.items():
						if key not in square_board_kwargs:
							square_board_kwargs[key] = value
					p_cells = np.array(
						[
							[
								square.get_contents(**square_board_kwargs)[1] if square.is_cell else np.inf
								for square in row
							]
							for row in self.grid
						],
						dtype=np.float)
					sort_order_list = np.unravel_index(np.argsort(p_cells, axis=None), p_cells.shape)
					sort_order = np.zeros_like(p_cells, dtype=np.int)
					sort_order[sort_order_list] = np.arange(sort_order.size)
					square_board_kwargs['cell_order'] = sort_order
					square_board_kwargs['num_cells'] = self.cells.sum()
		strings = []
		if output == 'html':
			strings.append('<table><tbody>')
		for board_y, row_board_kwargs in enumerate(board_kwargs):
			if board_y != 0:
				if output == 'html':
					strings.append('<tr></tr>' * padding)
				elif output == 'plain':
					strings.append('\n' * padding)
			for row in self.grid:
				if output == 'html':
					strings.append('<tr>')
				for board_x, square_board_kwargs in enumerate(row_board_kwargs):
					_kwargs = square_board_kwargs or {'display': False}
					if board_x != 0:
						strings.append('<td></td>' * padding)
					elif output == 'plain':
						strings.append(' ' * padding)
					for square in row:
						strings.append(square.format(**_kwargs))
				if output == 'html':
					strings.append('</tr>')
				elif output == 'plain':
					strings.append('\n')
		if output == 'html':
			strings.append('</tbody></table>')
		return ''.join(strings)

	def dump_entries(self, dump_clues=True, dump_scores=True) -> str:
		'''Dump the list of possible answers for each entry.'''
		lines = []
		for entries in self.entries:
			for entry in entries:
				line = '{}:'.format(entry.name)
				if dump_clues and entry.clue is not None:
					line += ' ' + entry.clue
				lines.append(line)
				for answer, score in zip(entry.answers, entry.scores):
					if answer is not None:
						line = to_str(answer)
						if dump_scores:
							line += ' ' + str(score)
						lines.append(line)
		return '\n'.join(lines)

	def load_entries(self, data : str, weight_for_unknown : float) -> None:
		entry_answers = defaultdict(list) # type: Dict[str, List[Tuple[str, float]]]
		entry_name = None # type: Optional[str]
		clues = {} # type: Dict[str, Optional[str]]
		for line in data.split('\n'):
			if ':' in line:
				entry_name, clue = line.split(':')
				clue = clue.strip()
				clues[entry_name] = clue if clue else None
			else:
				assert entry_name is not None
				tokens = line.split()
				answer = tokens[0]
				score = float(tokens[1]) if len(tokens) > 1 else 1
				entry_answers[entry_name].append((answer, score))
		for entries in self.entries:
			for entry in entries:
				if entry.name not in entry_answers:
					raise ValueError('{} not in loaded data'.format(entry.name))
		for entries in self.entries:
			for entry in entries:
				answers, scores = zip(*entry_answers[entry.name]) # type: Sequence, Sequence
				# TODO: standardize None order
				answers = [None] + list(answers)
				scores = [weight_for_unknown] + list(scores)
				entry.set_answers(answers, scores, clue=clue)


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

	def parse_clues(self, clues : str) -> List[List[str]]:
		assert min(map(len, self.entries)) > 0
		lines = clues.split('\n')
		missing_entries = {} # type: Dict[Tuple[Direction, ...], Entry]
		for direction_order in (tuple(Direction), tuple(reversed(tuple(Direction)))):
			clues_lists = [[], []] # type: List[List[str]]

			direction_iter = iter(direction_order)
			direction = None # type: Optional[Direction]
			next_direction = next(direction_iter) # type: Optional[Direction]
			next_direction_first = cast(Direction, next_direction)
			next_entry = self.entries[next_direction_first][0] # type: Optional[Entry]
			last_entry = None # type: Optional[Entry]
			for line in lines:
				line = line.strip()
				if not line:
					continue
				if next_direction is not None and next_direction != direction:
					# eat empty lines and DOWN/ACROSS section
					if line.upper() == next_direction.name:
						direction = next_direction
						continue
				if next_direction is None:
					next_entry = None
				else:
					next_entry = self.entries[next_direction][len(clues_lists[next_direction])]
				if next_entry is not None:
					assert next_direction is not None
					match = re.match(r'(\d+)\b[\s\.:]*(.*)', line)
					if match:
						number_str, clue = match.groups()
						number = int(number_str)
						if number == next_entry.number:
							# start next clue
							direction = next_direction
							last_entry = self.entries[direction][len(clues_lists[direction])]
							clues_lists[direction].append(clue)
							if len(clues_lists[next_direction]) == len(self.entries[next_direction]):
								try:
									next_direction = next(direction_iter)
								except StopIteration:
									next_direction = None
							continue
				# next clue identifier not found, continue previous clue
				if direction is not None and clues_lists[direction]:
					clues_lists[direction][-1] += ' ' + line
			# return the clues if we found them, otherwise indicate what we didn't find
			if tuple(map(len, clues_lists)) == tuple(map(len, self.entries)):
				return clues_lists
			else:
				assert last_entry is not None
				missing_entries[direction_order] = last_entry

		missing = ('last{}: {}'.format(tuple(direction.name for direction in direction_order), entry.name) for direction_order, entry in missing_entries.items())
		raise ValueError('could not find all clues: {}'.format(' '.join(missing)))

	def use_clues(self, clues : str, weight_for_unknown : float, session : Optional[aiohttp.ClientSession] = None) -> None:
		print('Fetching clue answers...')
		owns_session = session is None
		clues_lists = self.parse_clues(clues)
		if owns_session:
			headers = {
				'User-Agent': 'queso AppleWebKit/9000 Chrome/9000',
			}
			connector = aiohttp.TCPConnector(
				limit=100, # defualt is 100 simultaneous connections
				limit_per_host=50,
			)
			session = aiohttp.ClientSession(headers=headers, connector=connector)
		assert session is not None
		tasks = []
		for entries, clues_list in zip(self.entries, clues_lists):
			for entry, clue in zip(entries, clues_list):
				tasks.append(entry.use_clue(clue, session, weight_for_unknown))
		loop = asyncio.get_event_loop()
		answer_scores = loop.run_until_complete(asyncio.gather(*tasks))
		if owns_session:
			loop.run_until_complete(session.close())
		loop.close()
		print('Fetched clue answers!')
		for tracker in Tracker:
			if tracker.value.expected_answers and not tracker.value.site_gave_answers:
				warnings.warn('Did not get any answers from {}'.format(tracker.__name__))
