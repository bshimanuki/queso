import argparse
import os
import signal
import sys
from typing import Optional

import imageio
import numpy as np

from board import Board
from board_extract import make_board
from clipboard_qt import get_application, get_clipboard, set_clipboard


class Session(object):
	ROOT_DIR = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
	IMAGE_FILE = os.path.join(ROOT_DIR, 'last-image.png')
	CLUES_FILE = os.path.join(ROOT_DIR, 'last-clues.txt')
	ENTRIES_FILE = os.path.join(ROOT_DIR, 'last-entries.txt')
	OUTPUT_FILE = os.path.join(ROOT_DIR, 'last-output.html')

	def __init__(
			self,
			image_file : Optional[str] = None,
			clues_file : Optional[str] = None,
			entries_file : Optional[str] = None,
			output_file : Optional[str] = None,
			use_entries : bool = True,
	):
		# state variables
		self.board = None
		self.clues = None

		# file name variables
		self._image_file = image_file
		self._clues_file = clues_file
		self._entries_file = entries_file
		self._output_file = output_file
		self.use_entries = use_entries

		# board solving variables
		self.iterations = 30
		self.weight_for_unknown = 100
		self.weight_func = lambda x: x ** 2

		# clipboard variables
		self.app = get_application()
		self.clip = self.app.clipboard()
		self.clip.dataChanged.connect(self.check_clipboard)

	@property
	def image_file(self) -> str:
		return self._image_file or self.IMAGE_FILE

	@property
	def clues_file(self) -> str:
		return self._clues_file or self.CLUES_FILE

	@property
	def entries_file(self) -> str:
		return self._entries_file or self.ENTRIES_FILE

	@property
	def output_file(self) -> str:
		return self._output_file or self.OUTPUT_FILE

	def load_image(self, raise_not_exist : bool = False) -> None:
		fname = self.image_file
		try:
			img = imageio.imread(fname)
		except FileNotFoundError:
			if self._image_file is not None or raise_not_exist:
				raise
		else:
			sys.stderr.write('Loaded {}\n'.format(fname))
			self.board = make_board(img)

	def load_clues(self, raise_not_exist : bool = False) -> None:
		fname = self.clues_file
		try:
			with open(fname, 'rb') as f:
				clues = f.read().decode('utf-8')
		except FileNotFoundError:
			if self._clues_file is not None or raise_not_exist:
				raise
		else:
			sys.stderr.write('Loaded {}\n'.format(fname))
			self.clues = clues

	def load_entries(self, raise_not_exist : bool = False) -> None:
		fname = self.entries_file
		try:
			with open(fname, 'rb') as f:
				entries = f.read().decode('utf-8')
		except FileNotFoundError:
			if self._entries_file is not None or raise_not_exist:
				raise
		else:
			sys.stderr.write('Loaded {}\n'.format(fname))
			self.entries = entries

	def load(self) -> None:
		self.load_image()
		self.load_clues()
		self.load_entries()
		if self.board is not None:
			if self.use_entries and self.entries is not None:
				self.board.load_entries(self.entries, weight_for_unknown=self.weight_for_unknown)
			elif self.clues is not None:
				self.board.use_clues(self.clues, weight_for_unknown=self.weight_for_unknown, weight_func=self.weight_func)
				with open(self.entries_file, 'wb') as f:
					f.write(self.board.dump_entries().encode('utf-8'))

	def run(self) -> None:
		if self.board is not None and self.board.has_clues:
			for _ in range(self.iterations):
				self.board.update_cells()
				self.board.update_entries()
			self.board.update_cells()

	def set_output(self) -> None:
		output = None # type: Optional[str]
		if self.board is not None:
			if self.board.has_clues:
				output = self.board.format_multiple()
				sys.stderr.write('Copying filled grids\n')
			else:
				output = self.board.format(show_entries=False)
				sys.stderr.write('Copying empty grid\n')

		if output is not None:
			with open(self.output_file, 'wb') as f:
				f.write(output.encode('utf-8'))
			set_clipboard(html=output)

	def check_clipboard(self) -> None:
		application = get_clipboard('text/application')
		if application == b'queso':
			# don't respond to ourselves
			return
		img_data = get_clipboard('image/png')
		text_data = get_clipboard('text/plain')
		if img_data:
			with open(self.image_file, 'wb') as f:
				f.write(img_data)
			img = imageio.imread(img_data)
			self.board = make_board(img)
		elif text_data:
			with open(self.clues_file, 'wb') as f:
				f.write(text_data)
			self.clues = text_data.decode('utf-8')
		else:
			# nothing recognized
			return

		output = None # type: Optional[str]
		if self.board is not None and self.clues is not None:
			self.board.use_clues(self.clues, weight_for_unknown=self.weight_for_unknown, weight_func=self.weight_func)
			with open(self.entries_file, 'wb') as f:
				f.write(self.board.dump_entries().encode('utf-8'))
			self.run()
		self.set_output()


def main():
	signal.signal(signal.SIGINT, signal.SIG_DFL)

	parser = argparse.ArgumentParser()
	parser.add_argument('--image', '-i')
	parser.add_argument('--clues', '-c')
	parser.add_argument('--entries', '-e')
	parser.add_argument('--output', '-o')
	parser.add_argument('--use_entries', dest='use_entries', action='store_true')
	parser.add_argument('--no_use_entries', dest='use_entries', action='store_false')
	parser.add_argument('--clip', action='store_true')
	args = parser.parse_args()

	session = Session(
		image_file=args.image,
		clues_file=args.clues,
		entries_file=args.entries,
		output_file=args.output,
		use_entries=args.use_entries,
	)
	if args.clip:
		session.check_clipboard()
	else:
		session.load()
	session.run()
	session.set_output()
	session.app.exec_()


if __name__ == '__main__':
	main()
