import functools
import os

from PyQt5.Qt import QApplication, QClipboard, QMimeData

def get_clipboard(target='STRING'):
	app = QApplication([])
	clip = app.clipboard()
	mime = clip.mimeData()
	return mime.data(target).data()

def set_clipboard(text=None, html=None):
	if text is not None and html is not None:
		raise ValueError()

	if not os.fork():
		mime = QMimeData()
		if text is not None:
			mime.setText(text)
		if html is not None:
			mime.setHtml(html)
		app = QApplication([])
		clip = app.clipboard()
		clip.setMimeData(mime)
		clip.dataChanged.connect(app.exit)
		app.exec_()
		exit()
