import struct
import time
import tkinter

def print_clipboard():
	tk = tkinter.Tk()
	tk.withdraw()
	print(tk.selection_get(selection='CLIPBOARD', type='text/plain'))
	print(tk.selection_get(selection='CLIPBOARD', type='text/html'))

def format(s):
	# this is broken because tkinter formats as 32bit and then puts into 64bit
	s = bytes(s, 'utf-8')
	ii = []
	while len(s) % 4:
		s += b'\0'
	for i in range(0, len(s), 4):
		ii.append(struct.unpack_from('<L', s, i)[0])
	print(' '.join(hex(i) for i in ii))
	return ' '.join(hex(i) for i in ii)

def copy():
	tk = tkinter.Tk()
	tk.withdraw()
	# tk.selection_handle(lambda off, maxchars: 'string', selection='CLIPBOARD', type='STRING')
	tk.selection_handle(lambda off, maxchars: format('testing'), format='text/plain', selection='CLIPBOARD', type='text/plain')
	html = r'<meta http-equiv="content-type" content="text/html; charset=utf-8"><style type="text/css"><!--td {border: 1px solid #ccc;}br {mso-data-placement:same-cell;}--></style><span style="font-size:10pt;font-family:Arial;font-style:normal;text-decoration:line-through;color:#ff0000;" data-sheets-value="{&quot;1&quot;:2,&quot;2&quot;:&quot;c&quot;}" data-sheets-userformat="{&quot;2&quot;:330241,&quot;3&quot;:{&quot;1&quot;:0},&quot;12&quot;:0,&quot;14&quot;:{&quot;1&quot;:2,&quot;2&quot;:16711680},&quot;19&quot;:1,&quot;21&quot;:0}">c</span>'
	# tk.selection_handle(lambda off, maxchars: html, selection='CLIPBOARD', type='text/html')
	tk.selection_own(command=lambda: tk.destroy(), selection='CLIPBOARD')
	# tk.clipboard_clear()
	# tk.clipboard_append("string", type="STRING")
	# tk.clipboard_append("plain", type="text/plain")
	# tk.clipboard_append("html", type="text/html")
	# tk.after(10000, lambda: tk.destroy())
	tk.mainloop()

if __name__ == '__main__':
	copy()
	# print_clipboard()
