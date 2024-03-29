from fractions import Fraction
import math
import logging
from typing import cast, Any, Dict, List, Optional, Tuple
import warnings

import cv2
import imageio
import numpy as np
import scipy.interpolate
import scipy.ndimage
import scipy.signal
import skimage

from .board import Board

blacker = np.fmin
whiter = np.fmax

RECIPROCAL_BORDER_PROPORTION = 16
SAVE_IMAGES = False
SQUARE_LIMIT = 200


def get_gcd(xs: List[float], init_max_denom: int = 8, init_thresh: float = 1e-1) -> Tuple[float, int]:
	'Returns gcd of inliers and number of inliers.'
	max_denom = init_max_denom
	thresh = init_thresh
	denom_power_factor = 0.9
	thresh_scaling_factor = 0.8

	def is_approx(a: float, b: float, thresh: float = 0.1) -> bool:
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
			if is_approx(float(f), x / val, thresh=thresh):
				n += 1
				max_denom **= denom_power_factor
				thresh *= thresh_scaling_factor

				x_weight = f.numerator
				x_val = x / f.numerator
				weight *= f.denominator
				val /= f.denominator
				total = val * weight + x_val * x_weight
				weight += x_weight
				val = total / weight
	assert val is not None
	return val, n

def autocorrelate(x: np.ndarray) -> np.ndarray:
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

def get_mountain_slice(x: np.ndarray, mid: int) -> slice:
	'''Get a slice from a peak at index mid to the bottom of both sides.'''
	low = mid
	while low-1 > 0 and (x[low-1] < x[low] or x[low-1] == x[mid]):
		low -= 1
	high = mid
	while high+1 < len(x) and (x[high+1] < x[high] or x[high+1] == x[mid]):
		high += 1
	return slice(low, high + 1)

def is_far_from_edge(im: np.ndarray, loc: Tuple[int, ...], thresh: int = 1) -> bool:
	assert im.ndim == len(loc)
	for s, x in zip(im.shape, loc):
		if x - thresh < 0 or x + thresh >= s:
			return False
	return True

def get_interpolated_peak_1d(xs: np.ndarray, dis_loc: int, delta: int = 1, order: int = 2) -> Tuple[float, float]:
	'''Get the continuous location of a peak from a discrete 1D sample and the value.'''
	assert is_far_from_edge(xs, (dis_loc,), delta)
	x_partial = xs[dis_loc-delta:dis_loc+delta+1]
	spline = scipy.interpolate.UnivariateSpline(
		np.arange(dis_loc-delta, dis_loc+delta+1),
		-x_partial, # negative because optimize finds the min
		k=order,
	)
	result = scipy.optimize.minimize(
		lambda x: spline(x).item(),
		dis_loc,
		bounds=((dis_loc-delta, dis_loc+delta),),
	)
	return result.x, -spline(*result.x).item()

def get_interpolated_peak_2d(im: np.ndarray, dis_loc: Tuple[int, int], delta: int = 1, order: int = 2) -> Tuple[np.ndarray, float]:
	'''Get the continuous location of a peak from a discrete 2D sample and the value.'''
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
		lambda x: spline(*x).item(),
		dis_loc,
		bounds=((y-delta, y+delta), (x-delta, x+delta)),
	)
	return result.x, -spline(*result.x).item()

def save_color(fname: str, im: np.ndarray) -> None:
	'''Save monocromatic image in color as a colormap.'''
	if not SAVE_IMAGES:
		return
	out = np.asarray(im)
	out = np.interp(out, (out.min(), out.max()), (0, 1))
	with warnings.catch_warnings():
		warnings.simplefilter('ignore') # suppress precision loss warning
		out = skimage.img_as_ubyte(out)
	out = cv2.cvtColor(cv2.applyColorMap(out, cv2.COLORMAP_INFERNO), cv2.COLOR_BGR2RGB)
	imageio.imwrite(fname, out)

def save(fname: str, im: np.ndarray, resize: Optional[np.ndarray] = None) -> None:
	'''Save image.'''
	if not SAVE_IMAGES:
		return
	out = np.asarray(im)
	if out.min() > -1e-6 and out.max() < 1 + 1e-6:
		out = np.maximum(0, np.minimum(1, out))
	with warnings.catch_warnings():
		warnings.simplefilter('ignore') # suppress precision loss warning
		out = skimage.img_as_ubyte(out)
	if resize is not None:
		out = cv2.resize(out, resize.shape[1::-1], interpolation=cv2.INTER_NEAREST)
	imageio.imwrite(fname, out)

def make_thresh_mono(im: np.ndarray) -> np.ndarray:
	'''Convert to single channel (darkest channel) and threshold the image.'''
	bw = skimage.img_as_float(im)
	while bw.ndim > 2:
		bw = blacker.reduce(bw, axis=-1)
	median = np.median(bw)
	if median < 0.5:
		bw = np.where(bw <= 2 * median, np.zeros_like(bw), np.ones_like(bw))
	else:
		bw = np.where(bw < 1 - 2 * (1 - median), np.zeros_like(bw), np.ones_like(bw))
	return bw

def make_normalized_mono(im: np.ndarray) -> np.ndarray:
	'''Convert to single channel (darkest channel) and normalize the image.'''
	bw = skimage.img_as_float(im)
	while bw.ndim > 2:
		bw = blacker.reduce(bw, axis=-1)
	bw -= np.mean(bw)
	return bw

def make_lrn_mono(im: np.ndarray, sigma: float = 5) -> np.ndarray:
	'''Convert to single channel (darkest channel) and perform local response normalization.'''
	bw = skimage.img_as_float(im)
	while bw.ndim > 2:
		bw = blacker.reduce(bw, axis=-1)
	lr = scipy.ndimage.gaussian_filter(bw, sigma)
	bw -= lr
	return bw

def make_window_mono(im: np.ndarray, w: int = 5) -> np.ndarray:
	'''Convert to single channel (darkest channel) and perform local response normalization.'''
	bw = skimage.img_as_float(im)
	while bw.ndim > 2:
		bw = blacker.reduce(bw, axis=-1)
	lr = scipy.ndimage.maximum_filter(bw, w)
	bw -= lr
	return bw

def get_square_size(im: np.ndarray) -> Tuple[float, float]:
	bw = make_normalized_mono(im)
	ac = autocorrelate(bw)
	ac_mag = np.abs(ac)
	# save_color('ac_mag.png', ac_mag)

	smooth_pixels = 3
	# use g as workspace to find peaks
	g = np.array(ac_mag)
	# remove zero frequencies
	g_y = g.mean(axis=1)
	g_x = g.mean(axis=0)
	for i in range(smooth_pixels):
		g_y = np.convolve(g_y, [1, 2, 1], mode='same')
		g_x = np.convolve(g_x, [1, 2, 1], mode='same')
	y_mid = g_y.size // 2
	y_mid_slice = get_mountain_slice(g_y, y_mid)
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
			cont_loc, _ = get_interpolated_peak_2d(g, dis_loc)
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

	y, _ = get_gcd(peak_y)
	x, _ = get_gcd(peak_x)

	# stricter threshholds for matching x and y as same size
	v, n = get_gcd([y, x], init_max_denom=1, init_thresh=1e-2)
	if n == 2:
		y /= round(y / v)
		x /= round(x / v)

	return y, x

def make_grid(im: np.ndarray, square_size: Tuple[float, float], origin: Tuple[float, float] = (0, 0), double_size: bool = False, waveform: str = 'square', area: str = 'center', width: float = 1/4):
	'''Make a grid mask.'''
	dy, dx = square_size
	oy, ox = origin
	assert waveform in ('sawtooth', 'square')
	assert area in ('center', 'horizontal', 'vertical', 'QII')

	bw = make_normalized_mono(im)
	if double_size:
		shape = tuple(2*s-1 for s in bw.shape)
	else:
		shape = bw.shape
	grid = np.zeros(shape, dtype=float)

	def make_row(s, dx, ox):
		xs = np.zeros(s, dtype=float)
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
		raise NotImplementedError()
	return grid

def get_offset(im: np.ndarray, square_size: Tuple[float, float]) -> Tuple[Tuple[float, float], float]:
	'''Compute the offset in pixels from the center of the image given the square size.'''
	grid = make_grid(im, square_size, waveform='sawtooth', double_size=True)
	# save('grid.png', grid)

	kernel_shape = tuple(max(2, math.ceil(x / 8)) for x in square_size)
	if im.ndim == 3:
		kernel_shape += (1,)
	dilated = scipy.ndimage.maximum_filter(im, kernel_shape)
	mask = skimage.img_as_float(dilated - im)
	# save('mask.png', mask)
	masked_im = (mask.transpose() * skimage.img_as_float(im).transpose()).transpose() + (1 - mask)
	# save('masked_im.png', masked_im)

	bw = make_normalized_mono(masked_im)
	grid = make_normalized_mono(grid)
	cor = scipy.signal.fftconvolve(bw, np.flip(grid), mode='valid')

	g = np.array(cor)
	dis_loc = (0, 0)
	while not is_far_from_edge(g, dis_loc):
		g[dis_loc] = 0
		dis_loc = np.unravel_index(g.argmax(), g.shape)
	cont_loc, peak = get_interpolated_peak_2d(g, dis_loc)
	offset = cont_loc - np.asarray([(s-1)/2 for s in cor.shape])
	offset %= square_size
	return tuple(offset), peak # type: ignore # mypy does not determine size

def cluster_splits(xs: np.ndarray, stop_factor: float = 2, stop_base_idx: int = 10, thresh:float = 5e-2) -> List[float]:
	'''
	Get split points from clustering.

	Output is in largest gap order.
	'''
	# TODO: cutoff threshold better
	stop_base_idx = min(stop_base_idx, math.ceil(xs.size / 2))

	xs = np.sort(xs)
	if xs.size == 0:
		return []
	diffs = np.diff(xs) # n - 1 elts
	if diffs.size == 0:
		return []
	idxs = np.argpartition(diffs, -stop_base_idx)[-stop_base_idx:]
	idxs = idxs[np.argsort(diffs[idxs])][::-1]
	base = diffs[idxs[-1]]
	splits = []
	for i in idxs:
		if diffs[i] > stop_factor * base and diffs[i] > thresh:
			splits.append(xs[i] + diffs[i] / 2)
	# print('splits:', splits, diffs[idxs])
	return splits

def line_detector_seps(im: np.ndarray) -> None:
	bw = make_thresh_mono(im)
	y = 1 - np.average(bw, axis=1)
	x = 1 - np.average(bw, axis=0)

	def get_stride(vs):
		r = np.zeros_like(vs)
		stack = []
		for i, v in enumerate(vs):
			v_used = 0
			while stack and stack[-1][1] <= v:
				pi, pv = stack[-1]
				r[i - pi] += pv - v_used
				v_used = pv
				stack.pop()
			if stack:
				pi, pv = stack[-1]
				r[i - pi] += v - v_used
			stack.append((i, v))
		_slice = get_mountain_slice(r, r.argmax())
		weights = np.zeros_like(r)
		weights[_slice] = r[_slice]
		weights[weights < max(r[_slice.start], r[_slice.stop-1])] = 0
		weights /= weights.sum()
		stride = weights @ np.arange(len(weights))
		return stride

	dy_l = np.pad(np.maximum(0, y[1:] - y[:-1]), (1, 0))
	dy_r = np.pad(np.maximum(0, y[:-1] - y[1:]), (0, 1))
	y_stride = (get_stride(dy_l) + get_stride(dy_r)) / 2
	dx_l = np.pad(np.maximum(0, x[1:] - x[:-1]), (1, 0))
	dx_r = np.pad(np.maximum(0, x[:-1] - x[1:]), (0, 1))
	x_stride = (get_stride(dx_l) + get_stride(dx_r)) / 2

	stride, n = get_gcd([y_stride, x_stride], init_max_denom=1, init_thresh=1e-2)
	if n == 2:
		y_stride /= round(y_stride / stride)
		x_stride /= round(x_stride / stride)

	def get_sep(dv_l, dv_r, v_stride, thresh=1e-1):
		dv_conv = scipy.signal.fftconvolve(dv_r, np.flip(dv_l), mode='full')
		dv_conv[(len(dv_conv)-1)//2+int(v_stride/2)+1:] = 0
		dv_conv[:(len(dv_conv)-1)//2-int(v_stride/2)] = 0
		dv_w = (get_interpolated_peak_1d(dv_conv, dv_conv.argmax())[0][0] - (len(dv_conv)-1)/2) / 2
		if dv_w > 0:
			dv = np.zeros_like(dv_l)
			hw = dv_w / 2
			dv[int(hw):] += (1 - math.modf(hw)[0]) * dv_l[:len(dv)-int(hw)]
			dv[int(hw)+1:] += math.modf(hw)[0] * dv_l[:len(dv)-int(hw)-1]
			dv[:len(dv)-int(hw)] += (1 - math.modf(hw)[0]) * dv_r[int(hw):]
			dv[:len(dv)-int(hw)-1] += math.modf(hw)[0] * dv_r[int(hw)+1:]
		else:
			dv = dv_l + dv_r

		w = 2 + int(dv_w)
		min_stride = max(w, int(v_stride * 2 / 3))
		mav_stride = int(v_stride * 4 / 3) + 1

		n = len(dv)
		cdv = np.cumsum(dv)
		dvcdv = np.cumsum(dv * np.arange(n))
		dp = np.full(n, -np.inf, dtype=float)
		dp[:mav_stride] = 0
		dp_l = np.zeros(n, dtype=int)
		for j in range(n):
			for l in range(min_stride, mav_stride+1):
				i = j - l
				dscore = cdv[j] - (cdv[j-w] if j-w >= 0 else 0)
				score = (dp[i] if i >= 0 else 0) + dscore
				if score > dp[j]:
					dp[j] = score
					dp_l[j] = l
		last = dp.argmax()
		values = []
		while last >= 0:
			p = cdv[last] - (cdv[last-w] if last-w >= 0 else 0)
			s = dvcdv[last] - (dvcdv[last-w] if last-w >= 0 else 0)
			mu = s / p if p else (last - 1) - (w - 1) / 2
			values.append((mu, p))
			l = dp_l[last]
			if l == 0:
				break
			last = last - l
		values = list(reversed(values))
		p_thresh = max((p for mu, p in values[1:-1]), default=0) * thresh
		values = [mu if p > p_thresh else None for mu, p in values]
		v_sep = []
		for mu in values:
			if mu is not None:
				if v_sep:
					skipped = max(1, round((mu - v_sep[-1]) / v_stride))
					delta = (mu - v_sep[-1]) / skipped
					for i in range(skipped):
						v_sep.append(v_sep[-1] + delta)
				else:
					v_sep.append(mu)
		return v_sep

	y_sep = np.asarray(get_sep(dy_l, dy_r, y_stride), dtype=float) + 1/2
	x_sep = np.asarray(get_sep(dx_l, dx_r, x_stride), dtype=float) + 1/2
	return y_sep, x_sep


def rescale(im: np.ndarray, y_sep: List[float], w: int) -> Board:
	'''
	Rescale image according to each pair of y_sep corresponds to w pixels.
	y_sep should be presorted.
	'''
	s = np.pad(np.cumsum(im, axis=0), ((1, 0), *((0, 0),)*(im.ndim-1)))
	out = np.zeros((w*max(0, len(y_sep)-1), *im.shape[1:]), dtype=float)
	y_sep = np.asarray(y_sep)
	ww = (y_sep[1:] - y_sep[:-1]) / w
	a = np.linspace(y_sep[:-1], y_sep[1:], w, endpoint=False, axis=-1)
	b = a + ww[..., None]
	af, ai = np.modf(a)
	ai = ai.astype(int)
	bf, bi = np.modf(b)
	bi = bi.astype(int)
	out = (s[bi] + bf[(..., *(None,)*(im.ndim-1))] * im[bi]) - (s[ai] + af[(..., *(None,)*(im.ndim-1))] * im[ai])
	out /= ww[(..., *(None,)*im.ndim)]
	out = out.reshape((np.prod(out.shape[:2]), *out.shape[2:]))
	return out


def analyze_grid(
		im: np.ndarray,
		square_size: Tuple[float, float],
		offset: Optional[Tuple[float, float]],
		force_number: bool = False,
) -> Board:
	'''
	Do grid analysis. Returns a Board.

	im must be RGB.

	Terminalogy:
	- square: unit of the grid
	- board: squares in the puzzle
	- cell: board square for an entry
	- block: board square without an entry
	'''
	assert im.ndim == 3 and im.shape[-1] == 3
	im = skimage.img_as_float(im)
	dy, dx = square_size
	if offset is None:
		offset = tuple(-s / 2 for s in im.shape[:2])
	center = tuple(s / 2 for s in im.shape[:2])
	origin = np.asarray(center) + np.asarray(offset)
	save('input.png', im)

	# create grid masks
	width = 1 / RECIPROCAL_BORDER_PROPORTION
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
	y_sep = np.concatenate((np.arange(origin[0], 0, -dy)[:0:-1], np.arange(origin[0], im.shape[0], dy))).astype(int)
	x_sep = np.concatenate((np.arange(origin[1], 0, -dx)[:0:-1], np.arange(origin[1], im.shape[1], dx))).astype(int)
	y_cen = np.minimum(y_sep + dy / 2, im.shape[0]-1).astype(int)
	x_cen = np.minimum(x_sep + dx / 2, im.shape[1]-1).astype(int)
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
		grid_sep[y_sep,...,0] = 1
		grid_sep[:,x_sep,...,0] = 1
		save('sep.png', grid_sep)
		grid_cen = np.array(grid_middle)
		grid_cen[y_cen,...,0] = 1
		grid_cen[:,x_cen,...,0] = 1
		save('cen.png', grid_cen)

	# operations to get the average/min/max value of masked squares
	def reduceat_op(grid, splits, op=np.add, op2=None, order=(0,1)):
		if op2 is None:
			op2 = op
		grid = op.reduceat(grid, splits[order[0]], axis=order[0])
		grid = op2.reduceat(grid, splits[order[1]], axis=order[1])
		return grid
	def reduceat_mean(im, grid, ys, xs, background=None, mean=True, **kwargs):
		if background is None:
			background = np.zeros_like(ret)
		if mean:
			im = im * grid
		else:
			im = np.where(grid == 1, im, np.full_like(im, np.nan))
		ret = reduceat_op(im, (ys, xs), **kwargs)
		if mean:
			grid_sum = reduceat_op(grid, (ys, xs), **kwargs)
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
	# TODO: compute by pixel width instead of mean value
	# horizontal is bottom, vertical is right
	squares_center = reduceat_mean(im, grid_middle, y_sep, x_sep, background=squares_background)
	# save('squares_center.png', squares_center, resize=im)
	squares_horiz = reduceat_mean(im, grid_horiz, y_cen, x_sep, background=squares_background, op=np.add, op2=whiter, order=(0,1))
	squares_horiz = np.abs(squares_horiz - squares_background)
	save('squares_horiz.png', squares_horiz, resize=im)
	squares_vert = reduceat_mean(im, grid_vert, y_sep, x_cen, background=squares_background, op=np.add, op2=whiter, order=(1,0))
	squares_vert = np.abs(squares_vert - squares_background)
	save('squares_vert.png', squares_vert, resize=im)
	squares_qii = reduceat_mean(im, grid_qii, y_sep, x_sep, background=squares_background, op=blacker, mean=False)
	save('squares_qii.png', squares_qii, resize=im)

	# compute which square candidates are squares in the actual grid
	squares_topbottom = blacker(np.insert(squares_horiz, 0, 0, axis=0)[:-1], squares_horiz)
	# save('squares_topbottom.png', squares_topbottom, resize=im)
	squares_leftright = blacker(np.insert(squares_vert, 0, 0, axis=1)[:,:-1], squares_vert)
	# save('squares_leftright.png', squares_leftright, resize=im)
	squares_bordered = blacker(squares_topbottom, squares_leftright)
	save('squares_bordered.png', squares_bordered, resize=im)

	def separate(mask: np.ndarray, values: np.ndarray, condition: np.ufunc, condition2: np.ufunc, default: Optional[np.ndarray] = None, aux_output : Optional[Dict[Any, Any]] = None, **cluster_split_kwargs) -> np.ndarray:
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
			elif aux_output is not None:
				aux_output['default'] = True
			ret = default
		if mask is not None:
			ret = np.logical_and(mask, ret)
		return ret

	# separate outside from board grid
	white_background = squares_background == whiter.reduce(squares_background, axis=(0,1))
	while white_background.ndim > 2:
		white_background = white_background.all(axis=-1)
	aux_output = {} # type: Dict[Any, Any]
	board = separate(mask=None, values=squares_bordered, condition=whiter, condition2=blacker, default=white_background, aux_output=aux_output)
	if aux_output.get('default', False):
		logging.warning('Could not get board from borders, using white background instead.')
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
	cells_border_below = separate(mask=cells_with_below, values=squares_horiz, condition=whiter, condition2=whiter, default=empty, stop_factor=8)
	# save('cells_border_below.png', cells_border_below, resize=im)

	# separate vertical borders
	cells_with_right = np.logical_and(cells, np.insert(cells, -1, False, axis=1)[:,1:])
	cells_border_right = separate(mask=cells_with_right, values=squares_vert, condition=whiter, condition2=whiter, default=empty, stop_factor=8)
	# save('cells_border_right.png', cells_border_right, resize=im)

	# trim to board
	board_coords = np.nonzero(board)
	y0 = np.min(board_coords[0])
	x0 = np.min(board_coords[1])
	y1 = np.max(board_coords[0])
	x1 = np.max(board_coords[1])
	cells_background_trimmed = squares_background[y0:y1+1, x0:x1+1]
	cells_trimmed = cells[y0:y1+1, x0:x1+1]
	numbered_cells_trimmed = numbered_cells[y0:y1+1, x0:x1+1]
	cells_border_below_trimmed = cells_border_below[y0:y1+1, x0:x1+1]
	cells_border_right_trimmed = cells_border_right[y0:y1+1, x0:x1+1]

	return Board(cells_trimmed, cells_background_trimmed, numbered_cells_trimmed, cells_border_below_trimmed, cells_border_right_trimmed, force_number=force_number)


def fft_square_size_and_offset(im: np.ndarray) -> Tuple[Tuple[float, float], Tuple[float, float]]:
	square_size = get_square_size(im)
	logging.info('Square size: {}'.format(square_size))
	offset, peak = get_offset(im, square_size)
	square_size_half = cast(Tuple[float, float], tuple(x / 2 for x in square_size))
	offset_half, peak_half = get_offset(im, square_size_half)
	# print(peak, peak_half)
	if peak_half > peak:
		square_size, offset, peak = square_size_half, offset_half, peak_half
		logging.info('Half square size fits better, using square size: {}'.format(square_size))
	logging.info('Square offset: {}'.format(offset))

	grid = make_grid(im, square_size, offset, waveform='sawtooth', double_size=False)
	grid3 = grid[..., np.newaxis]
	combined = 0.8 * skimage.img_as_float(im)[...,:3] + 0.2 * grid3
	save('combined.png', combined)
	save('c.png', im)

	return square_size, offset


def make_board(im: np.ndarray, method: Optional[str] = None, force_number: bool = False) -> Board:
	if im.ndim == 2:
		im = im[..., np.newaxis]
	if im.ndim == 3 and im.shape[-1] == 4:
		# remove alpha channel
		im = im[..., :3]
	if method == None:
		method = 'lines'

	if method == 'fft':
		square_size, offset = fft_square_size_and_offset(im)
	elif method == 'lines':
		w = RECIPROCAL_BORDER_PROPORTION
		im = skimage.img_as_float(im)
		y_sep, x_sep = line_detector_seps(im)
		if len(y_sep) > SQUARE_LIMIT or len(x_sep) > SQUARE_LIMIT:
			raise RuntimeError('line detector found {} vertical lines and {} horizontal lines'.format(len(y_sep), len(x_sep)))
		im_y = rescale(im, y_sep, w)
		im = np.swapaxes(rescale(np.swapaxes(im_y, 0, 1), x_sep, w), 0, 1)
		square_size = (w, w)
		offset = None
	else:
		raise ValueError('method {} not recognized'.format(method))

	board = analyze_grid(im, square_size, offset, force_number=force_number)

	logging.info('board shape: {}x{}'.format(*board.shape))

	return board


if __name__ == '__main__':
	import os
	import sys
	from clipboard_qt import set_clipboard

	root = os.path.dirname(os.path.dirname(__file__))
	impath = 'wild.png'
	# impath = 'wild_big.png'
	# impath = 'smogon.png'
	# impath = 'fill_in_blanks.png'
	# impath = 'smogon_70.png'
	# impath = 'smogon_33.png'
	# impath = 'mashup.png'
	# impath = 'caged.png'
	im = imageio.imread(os.path.join(root, 'test_cases', impath))

	board = make_board(im)
	output = board.format()
	set_clipboard(html=output)

	board.update_cells()
	board.update_entries()
