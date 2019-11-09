'''
Clipboard utilities using PyQt5.

This must be imported before PyQt5 or any other module that uses multiprocessing.
'''
import functools
import multiprocessing
import os
import sys

from PyQt5.Qt import QApplication, QMimeData

_app = None
def get_application():
	global _app
	if _app is None:
		_app = QApplication([])
	return _app

def get_clipboard(target='STRING'):
	mime = get_application().clipboard().mimeData()
	return mime.data(target).data()

def set_clipboard(**kwargs):
	text = kwargs.get('text', None)
	html = kwargs.get('html', None)
	internal = kwargs.get('internal', False)
	kwargs['internal'] = True # set internal for nested invocations
	if text is not None and html is not None:
		raise ValueError()
	if internal:
		os.close(sys.stdin.fileno())
		os.close(sys.stdout.fileno())
		os.close(sys.stderr.fileno())
		mime = QMimeData()
		mime.setData('text/application', b'queso')
		if text is not None:
			mime.setText(text)
		if html is not None:
			mime.setHtml(html)
		clip = get_application().clipboard()
		clip.setMimeData(mime)
		clip.dataChanged.connect(get_application().exit)
		get_application().exec_()
		exit()
	else:
		p = multiprocessing.get_context('spawn').Process(target=set_clipboard, kwargs=kwargs, daemon=False)
		p.start()
