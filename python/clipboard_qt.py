from PyQt5.Qt import QApplication, QClipboard, QMimeData

mime = QMimeData()
mime.setText('test')
mime.setHtml('<meta http-equiv="content-type" content="text/html; charset=utf-8"><style type="text/css"><!--td {border: 1px solid #ccc;}br {mso-data-placement:same-cell;}--></style><span style="font-size:10pt;font-family:Arial;font-style:normal;text-decoration:line-through;color:#ff0000;" data-sheets-value="{&quot;1&quot;:2,&quot;2&quot;:&quot;c&quot;}" data-sheets-userformat="{&quot;2&quot;:330241,&quot;3&quot;:{&quot;1&quot;:0},&quot;12&quot;:0,&quot;14&quot;:{&quot;1&quot;:2,&quot;2&quot;:16711680},&quot;19&quot;:1,&quot;21&quot;:0}">c</span>')
# mime.data()
# mime.data('STRING')
# mime.data('STRING').data()
app = QApplication([])
clip = app.clipboard()
clip.setMimeData(mime)

def kill_app():
	app.exit()
clip.dataChanged.connect(kill_app)

app.exec_()
