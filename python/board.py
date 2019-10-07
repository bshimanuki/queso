from collections import defaultdict
import enum
import math

import numpy as np
import skimage


class Color(object):
	def __init__(self, c):
		if c.size == 1:
			self.rgb = np.full(3, c.item())
		elif c.size == 3:
			assert c.ndim == 1
			self.rgb = c
		else:
			raise ValueError()
		self.rgb = skimage.img_as_float(self.rgb)

	def format(self, output):
		assert output in ('plain', 'html')
		strings = []
		if output == 'html':
			strings.append('#')
			for v_float in self.rgb:
				v_uint8 = min(255, math.floor(v_float * 256))
				strings.append('{:02x}'.format(v_uint8))
		return ''.join(strings)


class Square(object):
	def __init__(self, is_cell, color, bar_below, bar_right):
		self.is_cell = is_cell
		self.color = Color(color)
		self.bar_below = bar_below
		self.bar_right = bar_right
		self.number = None
		self.entries = {direction: None for direction in Direction}
		self.entry_indices = {direction: None for direction in Direction}

	def format(self, output):
		assert output in ('plain', 'html')
		strings = []
		if output == 'html':
			strings.append('<td style="background-color:{};">'.format(self.color.format(output)))
		if self.number:
			strings.append(str(self.number))
		if output == 'html':
			strings.append('</td>')
		elif output == 'plain':
			strings.append(' ')
		return ''.join(strings)


class Direction(enum.Enum):
	DOWN = enum.auto()
	ACROSS = enum.auto()


class Entry(object):
	def __init__(self, number, direction):
		self.cells = []
		self.number = number
		self.direction = direction

	def __len__(self):
		return len(self.cells)

	def append(self, cell):
		cell.entries[self.direction] = self
		cell.entry_indices[self.direction] = len(self)
		self.cells.append(cell)


class Board(object):
	'''
	Class to Represent a board.
	'''
	def __init__(
			self,
			cells,
			background,
			numbered_cells,
			bar_below,
			bar_right,
	):
		assert len(cells.shape) == 2
		assert background.shape[:2] == cells.shape
		assert numbered_cells.shape == cells.shape
		assert bar_below.shape == cells.shape
		assert bar_right.shape == cells.shape
		self.grid = np.array([[Square(*args) for args in zip(*row)] for row in zip(cells, background, bar_below, bar_right)])
		self.entries = defaultdict(list)

		n = 0
		for y in range(self.shape[0]):
			entry = None
			for x in range(self.shape[1]):
				square = self.grid[y, x]
				if numbered_cells[y, x]:
					n += 1
					square.number = n
					if x == 0 or not self.grid[y, x-1].is_cell or self.grid[y, x-1].bar_right:
						entry = Entry(square.number, Direction.ACROSS)
						self.entries[Direction.ACROSS].append(entry)
				if not square.is_cell:
					entry = None
				if entry is not None:
					entry.append(square)
				if square.bar_right:
					entry = None
		for x in range(self.shape[1]):
			entry = None
			for y in range(self.shape[0]):
				square = self.grid[y, x]
				if square.number:
					if y == 0 or not self.grid[y-1, x].is_cell or self.grid[y-1, x].bar_below:
						entry = Entry(square.number, Direction.DOWN)
						self.entries[Direction.DOWN].append(entry)
				if not square.is_cell:
					entry = None
				if entry is not None:
					entry.append(square)
				if square.bar_below:
					entry = None

	@property
	def shape(self):
		return self.grid.shape

	def format(self, output='html'):
		assert output in ('plain', 'html')
		strings = []
		if output == 'html':
			strings.append('<table><tbody>')
		for row in self.grid:
			if output == 'html':
				strings.append('<tr>')
			for square in row:
				strings.append(square.format(output))
			if output == 'html':
				strings.append('</tr>')
			elif output == 'plain':
				strings.append('\n')
		if output == 'html':
			strings.append('</tbody></table>')
		return ''.join(strings)
