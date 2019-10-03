from fractions import Fraction
import math
import warnings

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
	out = np.asarray(im)
	out = np.interp(out, (out.min(), out.max()), (0, 1))
	with warnings.catch_warnings():
		warnings.simplefilter('ignore') # suppress precision loss warning
		out = skimage.img_as_ubyte(out)
	out = cv2.cvtColor(cv2.applyColorMap(out, cv2.COLORMAP_INFERNO), cv2.COLOR_BGR2RGB)
	imageio.imwrite(fname, out)

def save(fname, im, resize=None):
	out = np.asarray(im)
	with warnings.catch_warnings():
		warnings.simplefilter('ignore') # suppress precision loss warning
		out = skimage.img_as_ubyte(out)
	if resize is not None:
		out = cv2.resize(out, resize.shape[1::-1], interpolation=cv2.INTER_NEAREST)
	imageio.imwrite(fname, out)

def make_normalized_mono(im):
	bw = skimage.img_as_float(im)
	while bw.ndim > 2:
		bw = bw.min(axis=-1)
	bw -= np.mean(bw)
	return bw

def get_cell_size(im):
	bw = make_normalized_mono(im)
	ac = autocorrelate(bw)
	ac_mag = np.abs(ac)

	# use g as workspace to find peaks
	g = np.array(ac_mag)
	# remove zero frequencies
	g_y = g.mean(axis=1)
	y_mid = g_y.size // 2
	y_mid_slice = get_mountain_slice(g_y, y_mid)
	g_x = g.mean(axis=0)
	x_mid = g_x.size // 2
	x_mid_slice = get_mountain_slice(g_x, x_mid)
	g[y_mid_slice] = 0
	g[:,x_mid_slice] = 0
	g[:y_mid] = 0 # g has rotational symmetry, so only count it once

	mountain_grid = np.zeros_like(g)

	save_color('test_pre.png', g**3)

	peaks = []
	for i in range(20): #TODO:edit number of tries
		dis_loc = np.unravel_index(g.argmax(), g.shape)
		if is_far_from_edge(g, dis_loc):
			cont_loc = get_interpolated_peak(g, dis_loc)
			peaks.append(cont_loc)
			# print(dis_loc, cont_loc)
		y_peak_slice = get_mountain_slice(g[:,dis_loc[1]], dis_loc[0])
		x_peak_slice = get_mountain_slice(g[dis_loc[0]], dis_loc[1])
		mountain_grid[y_peak_slice, x_peak_slice] = g[y_peak_slice, x_peak_slice]
		g[y_peak_slice, x_peak_slice] = 0

	save_color('test_keep.png', mountain_grid**3)
	save_color('test.png', g**3)

	origin = np.asarray(g.shape) // 2
	peak_offsets = np.stack(peaks) - origin
	peak_y, peak_x = peak_offsets.transpose()
	peak_y = np.abs(peak_y)
	peak_x = np.abs(peak_x)
	# print(peak_y)
	# print(peak_x)

	def get_gcd(xs, init_max_denom=6, init_thresh=1e-1):
		'Returns gcd of inliers and number of inliers.'
		max_denom = init_max_denom
		thresh = init_thresh
		denom_power_factor = 0.9
		thresh_scaling_factor = 0.8

		def is_approx(a, b, thresh=0.1):
			return a + thresh > b and a < b + thresh
		weight = 0
		val = None
		n = 0

		for x in xs:
			if val is None:
				val = x
				weight = 1
				n += 1
			else:
				if x > val:
					f = Fraction(x / val).limit_denominator(int(max_denom))
				else:
					f = 1 / Fraction(val / x).limit_denominator(int(max_denom))
				# print(x, val, float(f), x / val, thresh)
				if is_approx(float(f), x / val, thresh=thresh):
					n += 1
					max_denom **= denom_power_factor
					thresh *= thresh_scaling_factor

					x_weight = f.numerator
					x_val = x / x_weight
					weight *= f.denominator
					val /= f.denominator
					total = val * weight + x_val * x_weight
					weight += x_weight
					val = total / weight
		return val, n

	y, _ = get_gcd(peak_y)
	x, _ = get_gcd(peak_x)

	# stricter threshholds for matching x and y as same size
	v, n = get_gcd([y, x], init_max_denom=1, init_thresh=1e-2)
	if n == 2:
		y /= round(y / v)
		x /= round(x / v)

	return y, x

def make_grid(im, cell_size, origin=(0,0), double_size=False, border='sawtooth', width=1/4):
	'''Make a grid to match against im.'''
	dy, dx = cell_size
	oy, ox = origin
	assert border in ['sawtooth', 'square', 'horizontal', 'vertical']

	bw = make_normalized_mono(im)
	if double_size:
		shape = tuple(2*s-1 for s in bw.shape)
	else:
		shape = bw.shape
	grid = np.zeros(shape, dtype=np.float)

	def make_row(s, dx, ox):
		xs = np.zeros(s, dtype=np.float)
		mid = s // 2
		# range is from -(mid + 0.5) to (mid + 0.5)
		beg = -mid - 0.5
		end = mid + 0.5

		ox %= dx
		if ox > 0:
			ox -= dx

		i = math.floor(beg / dx)
		right = -float('inf')
		while right < end:
			center = i * dx + ox
			left = (i - width) * dx + ox
			right = (i + width) * dx + ox

			for b in range(math.floor(max(beg, left) + 0.5), math.ceil(min(end, right) - 0.5)):
				value = 0
				l = max((b - 0.5, beg, left))
				r = min((b + 0.5, end, center))
				if l < r:
					if border == 'sawtooth':
						lv = (l - left) / (center - left)
						rv = (r - left) / (center - left)
						value += (rv + lv) / 2 * (r - l)
					else:
						value += r - l
				l = max((b - 0.5, beg, center))
				r = min((b + 0.5, end, right))
				if l < r:
					if border == 'sawtooth':
						lv = (right - l) / (right - center)
						rv = (right - r) / (right - center)
						value += (rv + lv) / 2 * (r - l)
					else:
						value += r - l
				xs[mid + b] += value

			i += 1
		return xs

	ys = np.expand_dims(make_row(grid.shape[0], dy, oy), axis=1)
	xs = np.expand_dims(make_row(grid.shape[1], dx, ox), axis=0)
	if border == 'horizontal':
		grid = ys * (1 - xs)
	elif border == 'vertical':
		grid = (1 - ys) * xs
	else:
		grid = 1 - np.maximum(ys, xs)
	return grid

def get_offset(im, cell_size):
	grid = make_grid(im, cell_size, double_size=True)
	save('grid.png', grid)

	bw = make_normalized_mono(im)
	grid = make_normalized_mono(grid)
	cor = scipy.signal.fftconvolve(bw, np.flip(grid), mode='valid')
	cor_mag = np.abs(cor)
	save_color('grid_match.png', cor_mag)

	g = np.array(cor_mag)
	dis_loc = (0, 0)
	while not is_far_from_edge(g, dis_loc):
		g[dis_loc] = 0
		dis_loc = np.unravel_index(g.argmax(), g.shape)
	cont_loc = get_interpolated_peak(g, dis_loc)
	offset = cont_loc - tuple((s-1)/2 for s in cor_mag.shape)
	offset %= cell_size
	return offset

def analyze_grid(im, cell_size, offset):
	dy, dx = cell_size
	im = skimage.img_as_float(im)

	width = 1 / 8
	grid_horiz = make_grid(im, cell_size, offset, border='horizontal', width=width)[..., np.newaxis]
	save('horizontal.png', grid_horiz)
	grid_vert = make_grid(im, cell_size, offset, border='vertical', width=width)[..., np.newaxis]
	save('vertical.png', grid_vert)
	grid_middle = make_grid(im, cell_size, offset, border='square', width=width)[..., np.newaxis]
	save('middle.png', grid_middle)
	tot = np.concatenate((grid_horiz, grid_vert, grid_middle), axis=-1)
	save('tot.png', tot)

	center = tuple(s / 2 for s in im.shape[:2])
	origin = center + offset

	y_sep = np.concatenate((np.arange(origin[0], 0, -dy)[:0:-1], np.arange(origin[0], im.shape[0], dy))).astype(np.int)
	x_sep = np.concatenate((np.arange(origin[1], 0, -dx)[:0:-1], np.arange(origin[1], im.shape[1], dx))).astype(np.int)
	y_cen = np.minimum(y_sep + dy / 2, im.shape[0]-1).astype(np.int)
	x_cen = np.minimum(x_sep + dx / 2, im.shape[1]-1).astype(np.int)
	y_sep = np.insert(y_sep, 0, 0)
	x_sep = np.insert(x_sep, 0, 0)
	y_cen = np.insert(y_cen, 0, 0)
	x_cen = np.insert(x_cen, 0, 0)

	grid_sep = np.array(grid_middle)
	grid_sep[y_sep] = 1
	grid_sep[:,x_sep] = 1
	save('sep.png', grid_sep)
	grid_cen = np.array(grid_middle)
	grid_cen[y_cen] = 1
	grid_cen[:,x_cen] = 1
	save('cen.png', grid_cen)

	def reduceat_sum(grid, ys, xs):
		grid = np.add.reduceat(grid, ys, axis=0)
		grid = np.add.reduceat(grid, xs, axis=1)
		return grid
	def reduceat_mean(im, grid, ys, xs):
		im = im * grid
		im_sum = reduceat_sum(im, ys, xs)
		grid_sum = reduceat_sum(grid, ys, xs)
		ret = np.where(grid_sum == 0, np.zeros_like(im_sum), np.divide(im_sum, grid_sum, where=grid_sum>0))
		return ret
	cells_color = reduceat_mean(im, grid_middle, y_sep, x_sep)
	save('cells_center.png', cells_color, resize=im)
	cells_horiz = reduceat_mean(im, grid_horiz, y_cen, x_sep)
	save('cells_horiz.png', cells_horiz, resize=im)
	cells_vert = reduceat_mean(im, grid_vert, y_sep, x_cen)
	save('cells_vert.png', cells_vert, resize=im)



if __name__ == '__main__':
	import os
	import sys
	root = os.path.dirname(os.path.dirname(__file__))
	# impath = 'wild.png'
	impath = 'wild_big.png'
	# impath = 'smogon.png'
	# impath = 'fill_in_blanks.png'
	# impath = 'smogon_70.png'
	# impath = 'smogon_33.png'
	im = imageio.imread(os.path.join(root, 'img_test', impath))

	cell_size = get_cell_size(im)
	print(cell_size)
	offset = get_offset(im, cell_size)
	print(offset)

	grid = make_grid(im, cell_size, offset, double_size=False)
	grid3 = grid[..., np.newaxis]
	combined = 0.8 * skimage.img_as_float(im)[...,:3] + 0.2 * grid3
	save('combined.png', combined)
	save('c.png', im)

	analyze_grid(im, cell_size, offset)
