from fractions import Fraction
import math
import warnings

import cv2
import imageio
import numpy as np
import scipy.interpolate
import scipy.signal
import skimage


blacker = np.fmin
whiter = np.fmax


def autocorrelate(x):
	'''
	Get autocorrelation of an image.

	Output size is 2s-1 for each dimension of size s.
	'''
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
	'''Get a slice from a peak at index mid to the bottom of both sides.'''
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
	'''Get the continuous location of a peak from a discrete 2D sample.'''
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
	'''Save monocromatic image in color as a colormap.'''
	out = np.asarray(im)
	out = np.interp(out, (out.min(), out.max()), (0, 1))
	with warnings.catch_warnings():
		warnings.simplefilter('ignore') # suppress precision loss warning
		out = skimage.img_as_ubyte(out)
	out = cv2.cvtColor(cv2.applyColorMap(out, cv2.COLORMAP_INFERNO), cv2.COLOR_BGR2RGB)
	imageio.imwrite(fname, out)

def save(fname, im, resize=None):
	'''Save image.'''
	out = np.asarray(im)
	with warnings.catch_warnings():
		warnings.simplefilter('ignore') # suppress precision loss warning
		out = skimage.img_as_ubyte(out)
	if resize is not None:
		out = cv2.resize(out, resize.shape[1::-1], interpolation=cv2.INTER_NEAREST)
	imageio.imwrite(fname, out)

def make_normalized_mono(im):
	'''Convert to single channel (darkest channel) and normalize the image.'''
	bw = skimage.img_as_float(im)
	while bw.ndim > 2:
		bw = blacker.reduce(bw, axis=-1)
	bw -= np.mean(bw)
	return bw

def get_square_size(im):
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

	# save_color('test_pre.png', g**3)

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

	# save_color('test_keep.png', mountain_grid**3)
	# save_color('test.png', g**3)

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

def make_grid(im, square_size, origin=(0,0), double_size=False, waveform='square', area='center', width=1/4):
	'''Make a grid mask.'''
	dy, dx = square_size
	oy, ox = origin
	assert waveform in ['sawtooth', 'square']
	assert area in ['center', 'horizontal', 'vertical', 'QII']

	bw = make_normalized_mono(im)
	if double_size:
		shape = tuple(2*s-1 for s in bw.shape)
	else:
		shape = bw.shape
	grid = np.zeros(shape, dtype=np.float)

	def make_row(s, dx, ox):
		xs = np.zeros(s, dtype=np.float)
		ox = s / 2 + ox

		beg = 0
		end = s
		i = math.floor(-ox / dx)
		right = -float('inf')
		while right < end:
			if area == 'QII':
				left = (i + width) * dx + ox
				center = (i + width) * dx + ox
				right = (i + 1/2) * dx + ox
			else:
				left = (i - width) * dx + ox
				center = i * dx + ox
				right = (i + width) * dx + ox

			for b in range(math.floor(max(beg, left)), math.ceil(min(end, right))):
				value = 0
				if area != 'QII':
					l = max((b, beg, left))
					r = min((b + 1, end, center))
					if l < r:
						if waveform == 'sawtooth':
							lv = (l - left) / (center - left)
							rv = (r - left) / (center - left)
							value += (rv + lv) / 2 * (r - l)
						else:
							value += r - l
				l = max((b, beg, center))
				r = min((b + 1, end, right))
				if l < r:
					if waveform == 'sawtooth':
						lv = (right - l) / (right - center)
						rv = (right - r) / (right - center)
						value += (rv + lv) / 2 * (r - l)
					else:
						value += r - l
				if value != 0:
					xs[b] += value

			i += 1
		return xs

	ys = np.expand_dims(make_row(grid.shape[0], dy, oy), axis=1)
	xs = np.expand_dims(make_row(grid.shape[1], dx, ox), axis=0)
	if area == 'center':
		grid = 1 - np.maximum(ys, xs)
	elif area == 'horizontal':
		grid = ys * (1 - xs)
	elif area == 'vertical':
		grid = (1 - ys) * xs
	elif area == 'QII':
		grid = np.minimum(ys, xs)
	else:
		raise NotImplemented()
	return grid

def get_offset(im, square_size):
	'''Compute the offset in pixels from the center of the image given the square size.'''
	grid = make_grid(im, square_size, waveform='sawtooth', double_size=True)
	# save('grid.png', grid)

	bw = make_normalized_mono(im)
	grid = make_normalized_mono(grid)
	cor = scipy.signal.fftconvolve(bw, np.flip(grid), mode='valid')
	cor_mag = np.abs(cor)
	# save_color('grid_match.png', cor_mag)

	g = np.array(cor_mag)
	dis_loc = (0, 0)
	while not is_far_from_edge(g, dis_loc):
		g[dis_loc] = 0
		dis_loc = np.unravel_index(g.argmax(), g.shape)
	cont_loc = get_interpolated_peak(g, dis_loc)
	offset = cont_loc - tuple((s-1)/2 for s in cor_mag.shape)
	offset %= square_size
	return offset

def cluster_splits(xs, stop_factor=2, stop_base_idx=10, thresh=5e-2):
	'''
	Get split points from clustering.

	Output is in largest gap order.
	'''
	# TODO: cutoff threshold better
	# TODO: split by pixel width
	stop_base_idx = min(stop_base_idx, math.ceil(xs.size / 2))

	xs = np.sort(xs)
	diffs = np.diff(xs) # n - 1 elts
	idxs = np.argpartition(diffs, -stop_base_idx)[-stop_base_idx:]
	idxs = idxs[np.argsort(diffs[idxs])][::-1]
	base = diffs[idxs[-1]]
	splits = []
	for i in idxs:
		if diffs[i] > stop_factor * base and diffs[i] > thresh:
			splits.append(xs[i] + diffs[i] / 2)
	# print('splits:', splits, diffs[idxs])
	return splits


def analyze_grid(im, square_size, offset):
	'''
	Do grid analysis.

	im must be RGB.

	Terminalogy:
	- square: unit of the grid
	- board: squares in the puzzle
	- cell: board square for an entry
	- block: board square without an entry
	'''
	assert im.ndim == 3 and im.shape[-1] == 3
	dy, dx = square_size
	im = skimage.img_as_float(im)
	center = tuple(s / 2 for s in im.shape[:2])
	origin = center + offset

	# create grid masks
	width = 1 / 16
	grid_horiz = make_grid(im, square_size, offset, area='horizontal', width=width)[..., np.newaxis]
	# save('horizontal.png', grid_horiz)
	grid_vert = make_grid(im, square_size, offset, area='vertical', width=width)[..., np.newaxis]
	# save('vertical.png', grid_vert)
	grid_middle = make_grid(im, square_size, offset, area='center', width=width)[..., np.newaxis]
	# save('middle.png', grid_middle)
	grid_qii = make_grid(im, square_size, offset, area='QII', width=width)[..., np.newaxis]
	# save('qii.png', grid_qii)
	tot = np.concatenate((grid_horiz, grid_vert, grid_middle), axis=-1)
	# save('tot.png', tot)

	# compute separators for grid mask regions
	y_sep = np.concatenate((np.arange(origin[0], 0, -dy)[:0:-1], np.arange(origin[0], im.shape[0], dy))).astype(np.int)
	x_sep = np.concatenate((np.arange(origin[1], 0, -dx)[:0:-1], np.arange(origin[1], im.shape[1], dx))).astype(np.int)
	y_cen = np.minimum(y_sep + dy / 2, im.shape[0]-1).astype(np.int)
	x_cen = np.minimum(x_sep + dx / 2, im.shape[1]-1).astype(np.int)
	y_sep = np.insert(y_sep, 0, 0)
	x_sep = np.insert(x_sep, 0, 0)
	y_cen = np.insert(y_cen, 0, 0)
	x_cen = np.insert(x_cen, 0, 0)
	ny = y_sep.size
	nx = x_sep.size
	def get_inserts(xs, s):
		diffs = np.diff(xs, append=s)
		m = np.max(diffs)
		inserts = np.concatenate(tuple(np.full(m - diff, x) for x, diff in zip(xs, diffs)))
		return inserts
	y_insert = get_inserts(y_sep, im.shape[0])
	x_insert = get_inserts(x_sep, im.shape[1])

	if False:
		# visualize grid separators
		grid_sep = np.array(grid_middle)
		grid_sep[y_sep] = 1
		grid_sep[:,x_sep] = 1
		save('sep.png', grid_sep)
		grid_cen = np.array(grid_middle)
		grid_cen[y_cen] = 1
		grid_cen[:,x_cen] = 1
		save('cen.png', grid_cen)

	# operations to get the average/min/max value of masked squares
	def reduceat_op(grid, ys, xs, op=np.add):
		grid = op.reduceat(grid, ys, axis=0)
		grid = op.reduceat(grid, xs, axis=1)
		return grid
	def reduceat_mean(im, grid, ys, xs, background=None, op=np.add, mean=True):
		if background is None:
			background = np.zeros_like(ret)
		if mean:
			im = im * grid
		else:
			im = np.where(grid == 1, im, np.full_like(im, np.nan))
		ret = reduceat_op(im, ys, xs, op=op)
		if mean:
			grid_sum = reduceat_op(grid, ys, xs, op=op)
			ret = np.where(grid_sum == 0, background, np.divide(ret, grid_sum, where=grid_sum>0))
		else:
			ret = np.where(np.isnan(ret), background, ret)
		return ret

	# compute background colors of squares with median
	im_full = im
	im_full = np.insert(im_full, y_insert, np.nan, axis=0)
	im_full = np.insert(im_full, x_insert, np.nan, axis=1)
	# save('im_full.png', im_full)
	im_partitioned = im_full.reshape((ny, im_full.shape[0]//ny, nx, im_full.shape[1]//nx, im_full.shape[2]))
	squares_background = np.nanmedian(im_partitioned, axis=(1, 3))
	save('squares_background.png', squares_background, resize=im_full)

	# compute mean values of masked squares
	# horizontal is bottom, vertical is right
	squares_center = reduceat_mean(im, grid_middle, y_sep, x_sep, background=squares_background)
	# save('squares_center.png', squares_center, resize=im)
	squares_horiz = reduceat_mean(im, grid_horiz, y_cen, x_sep, background=squares_background)
	save('squares_horiz.png', squares_horiz, resize=im)
	squares_vert = reduceat_mean(im, grid_vert, y_sep, x_cen, background=squares_background)
	save('squares_vert.png', squares_vert, resize=im)
	squares_qii = reduceat_mean(im, grid_qii, y_sep, x_sep, background=squares_background, op=blacker, mean=False)
	# save('squares_qii.png', squares_qii, resize=im)

	# compute which square candidates are squares in the actual grid
	squares_topbottom = whiter(np.insert(squares_horiz, 0, squares_background[0], axis=0)[:-1], squares_horiz)
	# save('squares_topbottom.png', squares_topbottom, resize=im)
	squares_leftright = whiter(np.insert(squares_vert, 0, squares_background[:,0], axis=1)[:,:-1], squares_vert)
	# save('squares_leftright.png', squares_leftright, resize=im)
	squares_bordered = whiter(squares_topbottom, squares_leftright)
	# save('squares_bordered.png', squares_bordered, resize=im)

	def separate(mask, values, condition, condition2, default=None, **cluster_split_kwargs):
		'''
		Separate values based on a condition.

		mask: mask for valid squares
		values: values to separate
		condition: direction of separation (blacker or whiter)
		condition2: which split point to use (blacker or whiter)
		default: value to return if no split point (None raises a RuntimeError)
		'''
		values_mono = condition.reduce(values, axis=tuple(range(2, values.ndim)))
		if mask is None:
			valid_mono_list = values_mono.flatten()
		else:
			valid_mono_list = values_mono[mask]
		splits = cluster_splits(valid_mono_list, **cluster_split_kwargs)
		if splits:
			split = condition2.reduce(splits)
			ret = condition(values_mono, split) == values_mono
		else:
			if default is None:
				raise RuntimeError('no splits were found')
			ret = default
		if mask is not None:
			ret = np.logical_and(mask, ret)
		return ret

	# separate outside from board grid
	board = separate(mask=None, values=squares_bordered, condition=blacker, condition2=whiter)
	save('board.png', board, resize=im)
	empty = np.zeros_like(board)

	# separate cells from blocks
	cells = separate(mask=board, values=squares_background, condition=whiter, condition2=blacker, default=board)
	save('cells.png', cells, resize=im)

	# separate numbered cells from empty cells
	numbered_cells = separate(mask=cells, values=squares_qii, condition=blacker, condition2=blacker, default=empty)
	save('numbered_cells.png', numbered_cells, resize=im)

	# separate horizontal borders
	cells_with_below = np.logical_and(cells, np.insert(cells, -1, False, axis=0)[1:])
	cells_border_below = separate(mask=cells_with_below, values=squares_horiz, condition=blacker, condition2=blacker, default=empty, stop_factor=8)
	save('cells_border_below.png', cells_border_below, resize=im)

	# separate vertical borders
	cells_with_right = np.logical_and(cells, np.insert(cells, -1, False, axis=1)[:,1:])
	cells_border_right = separate(mask=cells_with_right, values=squares_vert, condition=blacker, condition2=blacker, default=empty, stop_factor=8)
	save('cells_border_right.png', cells_border_right, resize=im)


if __name__ == '__main__':
	import os
	import sys
	root = os.path.dirname(os.path.dirname(__file__))
	impath = 'wild.png'
	# impath = 'wild_big.png'
	# impath = 'smogon.png'
	# impath = 'fill_in_blanks.png'
	# impath = 'smogon_70.png'
	# impath = 'smogon_33.png'
	im = imageio.imread(os.path.join(root, 'img_test', impath), pilmode='RGB')

	square_size = get_square_size(im)
	print('square size:', square_size)
	offset = get_offset(im, square_size)
	print('offset:', offset)

	grid = make_grid(im, square_size, offset, waveform='sawtooth', double_size=False)
	grid3 = grid[..., np.newaxis]
	combined = 0.8 * skimage.img_as_float(im)[...,:3] + 0.2 * grid3
	save('combined.png', combined)
	save('c.png', im)

	analyze_grid(im, square_size, offset)
