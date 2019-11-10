import argparse
import asyncio
import logging
import multiprocessing
import os
import queue
import signal
import sys
import threading
import traceback
from typing import Optional

import imageio
import numpy as np
from PyQt5.Qt import pyqtSignal, QObject, QThread
import tqdm

from board import Board
from board_extract import make_board
from clipboard_qt import get_application, get_clipboard, set_clipboard
from utils import BoardError


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
		if isinstance(a, np.ndarray):
			return (a == b).all()
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
		self.entries = None

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
		self.queue = queue.Queue()
		self.thread = QThread()
		self.signal = Signal()
		self.signal.closeApp.connect(self.exit)
		self.thread.run = self.handle
		self.thread.start()

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

	def set_image(self, img, loading_mode=False) -> None:
		self.board = make_board(img)
		self.set_output()
		self.solve_board(loading_mode=loading_mode)

	def set_clues(self, clues, loading_mode=False) -> None:
		self.clues = clues
		self.solve_board(loading_mode=loading_mode)

	def set_entries(self, entries, loading_mode=False) -> None:
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
					except:
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
			logging.info('Loaded {}'.format(fname))
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
			logging.info('Loaded {}'.format(fname))
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
			logging.info('Loaded {}'.format(fname))
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
	logging.basicConfig(format='%(levelname)s: %(message)s', level=logging.INFO)

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
	session.app.exec_()


if __name__ == '__main__':
	main()
