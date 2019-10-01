from fractions import Fraction

import cv2
import imageio
import numpy as np
import scipy.interpolate
import scipy.signal
import skimage

def autocorrelate(x):
	ret = scipy.signal.fftconvolve(x, np.flip(x), mode='full')
	assert ret.shape == tuple(2*s-1 for s in x.shape)
	for i, s in enumerate(x.shape):
		m = np.minimum(np.arange(1, 2*s), np.arange(2*s-1, 0, -1))
		m = m.reshape(m.shape + (1,) * (x.ndim - i - 1))
		# m = np.maximum(m, s // 2)
		ret *= s / m
		cutoff = s // 2
		ret = np.moveaxis(np.moveaxis(ret, i, 0)[cutoff:m.size-cutoff], 0, i)
	return ret

def get_mountain_slice(x, mid):
	low = mid
	while low-1 > 0 and x[low-1] < x[low]:
		low -= 1
	high = mid
	while high+1 < len(x) and x[high+1] < x[high]:
		high += 1
	return slice(low, high)

def is_far_from_edge(im, loc, thresh=1):
	assert im.ndim == len(loc)
	for s, x in zip(im.shape, loc):
		if x - thresh < 0 or x + thresh >= s:
			return False
	return True

def get_interpolated_peak(im, dis_loc, delta=1, order=2):
	assert is_far_from_edge(im, dis_loc, delta)
	y, x = dis_loc
	im_partial = im[y-delta:y+delta+1, x-delta:x+delta+1]
	spline = scipy.interpolate.RectBivariateSpline(
		np.arange(y-delta, y+delta+1),
		np.arange(x-delta, x+delta+1),
		-im_partial, # negative because optimize finds the min
		kx=order,
		ky=order,
	)
	result = scipy.optimize.minimize(
		lambda x: spline(*x),
		dis_loc,
		bounds=((y-delta, y+delta), (x-delta, x+delta)),
	)
	return result.x

def save_color(fname, im):
	out = np.array(im)
	out = np.interp(out, (out.min(), out.max()), (0, 1))
	out = skimage.img_as_ubyte(out)
	out = cv2.cvtColor(cv2.applyColorMap(out, cv2.COLORMAP_INFERNO), cv2.COLOR_BGR2RGB)
	imageio.imwrite(fname, out)

def get_cell_size(im):
	bw = im.astype(np.float)
	while bw.ndim > 2:
		bw = bw[...,0]
	bw -= np.mean(bw)
	# f = np.fft.fft2(bw)
	# f = scipy.signal.fftconvolve(bw, bw[::-1,::-1])
	f = autocorrelate(bw)
	f_mag = np.abs(f)
	g = np.array(f_mag)

	# remove zero frequencies
	g_y = g.mean(axis=1)
	y_mid = g_y.size // 2
	y_mid_slice = get_mountain_slice(g_y, y_mid)
	g_x = g.mean(axis=0)
	x_mid = g_x.size // 2
	x_mid_slice = get_mountain_slice(g_x, x_mid)
	g[y_mid_slice] = 0
	g[:,x_mid_slice] = 0

	h = np.zeros_like(g)

	save_color('test_pre.png', g**3)

	peaks = []
	for i in range(20): #TODO:edit number of tries
		dis_loc = np.unravel_index(g.argmax(), g.shape)
		if is_far_from_edge(g, dis_loc):
			cont_loc = get_interpolated_peak(g, dis_loc)
			peaks.append(cont_loc)
			print(dis_loc, cont_loc)
		y_peak_slice = get_mountain_slice(g[:,dis_loc[1]], dis_loc[0])
		x_peak_slice = get_mountain_slice(g[dis_loc[0]], dis_loc[1])
		h[y_peak_slice, x_peak_slice] = g[y_peak_slice, x_peak_slice]
		g[y_peak_slice, x_peak_slice] = 0

	save_color('test_keep.png', h**3)
	save_color('test.png', g**3)

	#TODO: offset peak loc by origin
	# find gcd iteratively
	# increase gcd threshold each iteration until peak doesn't match
	peak_y, peak_x = np.stack(peaks).transpose()
	peak_y = np.sort(peak_y)
	peak_x = np.sort(peak_x)

	def condense(peaks):
		THRESH = 3 # pixels
		xs = []
		i = 0
		last_i = 0
		while i < peaks.size:
			if peaks[i] - peaks[last_i] < THRESH:
				i += 1
			else:
				xs.append(np.mean(peaks[last_i:i]))
				last_i = i
		xs.append(np.mean(peaks[last_i:i]))
		xs = np.array(xs)
		return xs
	peak_y = condense(peak_y)
	peak_x = condense(peak_x)
	print(peak_y)
	print(peak_x)

	y_diff = np.diff(peak_y)
	x_diff = np.diff(peak_x)
	np.set_printoptions(suppress=True)
	print(y_diff)
	print(x_diff)

	def get_gcd(xs):
		#TODO: outliers
		def is_approx(a, b):
			THRESH = 1.1
			return a * THRESH > b and a < b * THRESH
		weight = 0
		val = None
		for x in xs:
			if val is None:
				val = x
				weight = 1
			else:
				f = Fraction(x / val).limit_denominator(8)
				if is_approx(float(f) * val, x):
					x_weight = f.numerator
					x_val = x / x_weight
					weight *= f.denominator
					val /= f.denominator
					total = val * weight + x_val * x_weight
					weight += x_weight
					val = total / weight
		return val

	y = get_gcd(y_diff)
	x = get_gcd(x_diff)
	return y, x


if __name__ == '__main__':
	import os
	import sys
	root = os.path.dirname(os.path.dirname(__file__))
	# impath = 'wild.png'
	# impath = 'smogon.png'
	impath = 'fill_in_blanks.png'
	# impath = 'smogon_70.png'
	# impath = 'smogon_33.png'
	im = imageio.imread(os.path.join(root, 'img_test', impath))

	cell_size = get_cell_size(im)
	print(cell_size)

