import functools
import os

from PyQt5.Qt import QApplication, QClipboard, QMimeData

def get_clipboard(target='STRING', app=None):
	if app is None:
		app = QApplication([])
	clip = app.clipboard()
	mime = clip.mimeData()
	return mime.data(target).data()

def set_clipboard(text=None, html=None):
	if text is not None and html is not None:
		raise ValueError()

	print('forking')
	if not os.fork():
		print('forked child')
		mime = QMimeData()
		mime.setData('text/application', b'queso')
		if text is not None:
			mime.setText(text)
		if html is not None:
			mime.setHtml(html)
		print('setting up app')
		app = QApplication([])
		clip = app.clipboard()
		print('setting clipboard')
		clip.setMimeData(mime)
		clip.dataChanged.connect(app.exit)
		print('waiting')
		app.exec_()
		exit()
	print('forked parent')
