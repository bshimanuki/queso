'''
Server for automated crossword solving.

This tracks changes on the clipboard to update source data. After an image is copied, the server will generate the board as html, which can be pasted into Google Sheets. After an image and a set of clues are copied, the server will generate its best guess at filling out the board along with board and clue information.

The server will keep running in the background, taking whatever is put on the clipboard, solving crosswords, and putting the results back on the clipboard. In particular, it does not matter whether the image of the board or the text of the clues is copied first. Once finished, it is recommended to stop the server so that it will not change new clipboard contents.

On repeated uses, all crossword answer queries are cached. Additionally, the last image, clues, generated entry answers, and output are all stored to last-image.png, etc, under the queso project root. The last inputs are reused by default. To ignore a file, use the options to set the input file to /dev/null.

implementation details:
This program has two parts, board extraction and crossword solving. A board is extracted from an image by performing autocorrelation with itself and then cross-correlation with a constructed grid of the detected size. Crossword solving is done by scraping a series of online crossword clue databases for potential answers (via proxies so that three are no rate limits). The answer scores are aggreated and used as priors for Bayesian inference (using Markov Random Fields). Belief propagation is performed using the sum-product algorithm where the values of cells in the crossword are the variables and the probability distributions over the answer candidates are the factors. Unknown answers are accounted for by a special answer candidate that uses a smoothed trigram model over past NYT answers.
'''

import argparse
import asyncio
import logging
import multiprocessing
import os
import queue
import signal
import sys
import textwrap
import threading
import traceback
from typing import Optional

import imageio
import numpy as np
from PyQt5.Qt import pyqtSignal, QObject, QThread
import tqdm

from .board import Board
from .board_extract import make_board
from .clipboard_qt import get_application, get_clipboard, set_clipboard
from .utils import BoardError


class Signal(QObject):
	closeApp = pyqtSignal()


class QueueItem(object):
	def __init__(
			self,
			img : Optional[np.ndarray] = None,
			clues : Optional[str] = None,
			entries : Optional[str] = None,
			loading_mode : bool = False,
	):
		self.img = img
		self.clues = clues
		self.entries = entries
		self.loading_mode = loading_mode

	@staticmethod
	def equal(a, b):
		if a.__class__ != b.__class__:
			return False
		if isinstance(a, np.ndarray) and isinstance(b, np.ndarray):
			return a.shape == b.shape and (a == b).all()
		return a == b

	def __eq__(self, other):
		if self.__class__ != other.__class__:
			return False
		if self.__dict__.keys() != other.__dict__.keys():
			return False
		for key in self.__dict__:
			if not self.equal(self.__dict__[key], other.__dict__[key]):
				return False
		return True


class Server(object):
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
	):
		# state variables
		self.board = None # type: Optional[Board]
		self.clues = None # type: Optional[str]
		self.entries = None # type: Optional[str]

		# file name variables
		self._image_file = image_file
		self._clues_file = clues_file
		self._entries_file = entries_file
		self._output_file = output_file

		# board solving variables
		self.iterations = 30
		self.weight_for_unknown = 100
		self.weight_func = lambda x: x ** 2

		# clipboard variables
		self.app = get_application()
		self.clip = self.app.clipboard()
		self.clip.dataChanged.connect(self.check_clipboard)
		logging.info('Started clipboard event handler. Copy an image or text to start.')
		self.queue = queue.Queue() # type: queue.Queue[QueueItem]
		self.thread = QThread()
		self.signal = Signal()
		self.signal.closeApp.connect(self.exit)
		self.thread.run = self.handle
		self.thread.start()
		logging.info('Started crossword solver worker thread.')

	def exit(self):
		self.thread.quit()
		self.thread.wait()
		self.app.quit()

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

	def set_image(self, img : np.ndarray, loading_mode : bool = False) -> None:
		self.board = make_board(img)
		self.set_output()
		self.solve_board(loading_mode=loading_mode)

	def set_clues(self, clues : str, loading_mode : bool = False) -> None:
		self.clues = clues
		self.solve_board(loading_mode=loading_mode)

	def set_entries(self, entries : str, loading_mode : bool = False) -> None:
		self.entries = entries
		with open(self.entries_file, 'wb') as f:
			f.write(self.entries.encode('utf-8'))
		if self.clues is None:
			self.solve_board(loading_mode=loading_mode)

	def solve_board(self, loading_mode=False) -> None:
		updated = False
		if self.board is not None:
			if self.entries is not None:
				if loading_mode and not self.board.has_clues:
					try:
						self.board.load_entries(self.entries, weight_for_unknown=self.weight_for_unknown)
						updated = True
					except Exception:
						self.entries = None
						raise
			if self.clues is not None:
				if not loading_mode or not self.board.has_clues:
					self.board.use_clues(self.clues, weight_for_unknown=self.weight_for_unknown, weight_func=self.weight_func)
					self.set_entries(self.board.dump_entries(), loading_mode=loading_mode)
					updated = True
			if updated and self.board.has_clues:
				self.run()
				self.set_output()

	def load_image(self, raise_not_exist : bool = False) -> None:
		fname = self.image_file
		try:
			img = imageio.imread(fname)
		except FileNotFoundError:
			if self._image_file is not None or raise_not_exist:
				raise
		else:
			logging.info('Loaded image: {}'.format(fname))
			self.queue.put(QueueItem(img=img, loading_mode=True))

	def load_clues(self, raise_not_exist : bool = False) -> None:
		fname = self.clues_file
		try:
			with open(fname, 'rb') as f:
				clues = f.read().decode('utf-8')
		except FileNotFoundError:
			if self._clues_file is not None or raise_not_exist:
				raise
		else:
			logging.info('Loaded clues: {}'.format(fname))
			self.queue.put(QueueItem(clues=clues, loading_mode=True))

	def load_entries(self, raise_not_exist : bool = False) -> None:
		fname = self.entries_file
		try:
			with open(fname, 'rb') as f:
				entries = f.read().decode('utf-8')
		except FileNotFoundError:
			if self._entries_file is not None or raise_not_exist:
				raise
		else:
			logging.info('Loaded entries: {}'.format(fname))
			self.queue.put(QueueItem(entries=entries, loading_mode=True))

	def load(self) -> None:
		self.load_image()
		self.load_entries()
		self.load_clues()

	def run(self) -> None:
		if self.board is not None and self.board.has_clues:
			logging.info('Running bayesian model...')
			for _ in tqdm.trange(self.iterations):
				self.board.update_cells()
				self.board.update_entries()
			self.board.update_cells()

	def set_output(self) -> None:
		output = None # type: Optional[str]
		if self.board is not None:
			if self.board.has_clues:
				output = self.board.format_multiple()
				logging.info('Copied filled grids to clipboard!')
			else:
				output = self.board.format(show_entries=False)
				logging.info('Copied empty grid to clipboard!')

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
		img = None
		if img_data:
			with open(self.image_file, 'wb') as f:
				f.write(img_data)
			img = imageio.imread(img_data)
			logging.info('Enqueued clipboard image: {} bytes'.format(len(img_data)))
		elif text_data:
			with open(self.clues_file, 'wb') as f:
				f.write(text_data)
			logging.info('Enqueued clipboard text: {} bytes'.format(len(text_data)))
		self.queue.put(QueueItem(img=img, clues=text_data.decode('utf-8')))

	def handle(self) -> None:
		try:
			asyncio.set_event_loop(asyncio.new_event_loop())
			last_qi = None
			while True:
				qi = self.queue.get()
				if qi == last_qi:
					# don't run if nothing has changed
					logging.info('Clipboard contents have not changed, skipping...')
					continue
				try:
					if qi.img is not None:
						self.set_image(qi.img, loading_mode=qi.loading_mode)
					elif qi.clues:
						self.set_clues(qi.clues, loading_mode=qi.loading_mode)
					elif qi.entries:
						self.set_entries(qi.entries, loading_mode=qi.loading_mode)
				except BoardError:
					traceback.print_exc()
				last_qi = qi
		finally:
			self.signal.closeApp.emit()


def main():
	signal.signal(signal.SIGINT, signal.SIG_DFL)
	logging.basicConfig(format='[%(levelname)s:%(module)s] %(message)s', level=logging.INFO)

	class LineWrapDescriptionHelpFormatter(argparse.RawDescriptionHelpFormatter):
		def _fill_text(self, text, width, indent):
			return '\n'.join('\n'.join(textwrap.wrap(paragraph.strip(), width=width, initial_indent=indent)) for paragraph in text.split('\n'))
	parser = argparse.ArgumentParser()
	parser.prog = os.path.basename(os.path.dirname(os.path.realpath(__file__)))
	parser.formatter_class = LineWrapDescriptionHelpFormatter
	# parser.formatter_class = argparse.RawDescriptionHelpFormatter
	parser.description = __doc__
	parser.add_argument('--image', '-i', help='Image file to read and extract a board from.')
	parser.add_argument('--clues', '-c', help='Text file to read clues from.')
	parser.add_argument('--entries', '-e', help='Text file to read entry scores from.')
	parser.add_argument('--output', '-o', help='Output file to write html (same data that is copied to the clipboard).')
	parser.add_argument('--clip', action='store_true', help='Use the clipboard contents on startup instead of loading from files. (Changes in clipboard contents are always used.)')
	args = parser.parse_args()

	server = Server(
		image_file=args.image,
		clues_file=args.clues,
		entries_file=args.entries,
		output_file=args.output,
	)
	if args.clip:
		server.check_clipboard()
	else:
		server.load()
	server.app.exec_()


if __name__ == '__main__':
	main()
