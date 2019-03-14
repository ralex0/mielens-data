"""
For reference, the xy projections are at:
    below = 69 / 276 =  0.25 of the way from the bottom
    focus = 148 / 276 = 0.536 of the way from the bottom
    above = 207 / 276 = 0.75 of the way from the bottom

"""
import os
import sys
import json
from collections import OrderedDict

import numpy as np
from scipy.interpolate import interp1d
import matplotlib as mpl
import matplotlib.pyplot as plt

import holopy
from holopy.scattering import calc_holo, Sphere
from holopy.scattering.theory import MieLens, Mie

import monkeyrc
from fit_data import load_few_PS_data_Jan10

HERE = os.path.dirname(__file__)


class BigFigure(object):
    _imshow_kwargs = {'vmin': 0, 'vmax': 1.8, 'cmap': 'gray',
                      'interpolation': 'nearest'}

    def __init__(self, data, mieonly, mielens, figsize=[6, 3]):
        for ar in [data, mieonly, mielens]:
            if np.ndim(ar) != 3:
                raise ValueError("data, mieonly, mielens must be 3D arrays!")
        for ar in [mieonly, mielens]:
            if ar.shape != data.shape:
                raise ValueError("data, mieonly, mielens must be same shape!")
        self.data = data
        self.mieonly = mieonly
        self.mielens = mielens
        self.figsize = figsize

        self._get_slices()
        self._setup_figure()
        self.draw_figure()

    def _setup_figure(self):
        self.figure = plt.figure(figsize=self.figsize)
        bboxes = [(i / 3, 0, 1 / 3, 1) for i in range(3)]
        left_axes, center_axes, right_axes = [
            self.create_xy_xz_axes(bbox) for bbox in bboxes]
        axes_dict = {
            'left_lower_xy': left_axes[0],
            'left_center_xy': left_axes[1],
            'left_upper_xy': left_axes[2],
            'left_xz': left_axes[3],
            'center_lower_xy': center_axes[0],
            'center_center_xy': center_axes[1],
            'center_upper_xy': center_axes[2],
            'center_xz': center_axes[3],
            'right_lower_xy': right_axes[0],
            'right_center_xy': right_axes[1],
            'right_upper_xy': right_axes[2],
            'right_xz': right_axes[3],
            }

        self.axes_dict = axes_dict

    def create_xy_xz_axes(self, outer_bbox, xpad=0.01, ypad=0.01):
        if len(outer_bbox) != 4:
            msg = '{} must be (left, bottom, width, height)'.format(outer_bbox)
            raise ValueError(msg)
        left, bottom, width, height = outer_bbox
        halfwidth = width / 2
        thirdheight = height / 3

        axes_width = halfwidth - 2 * xpad
        left_axes_bottoms = [bottom + ypad + i * thirdheight for i in range(3)]
        left_axes_height = thirdheight - 2 * ypad
        left_axes_positions = [
            (left + xpad, b, axes_width, left_axes_height)
            for b in left_axes_bottoms]

        right_axes_position = (
            left + halfwidth + xpad,
            bottom + ypad,
            axes_width,
            height - 2 * ypad)

        all_positions = left_axes_positions + [right_axes_position]
        axes = [self.figure.add_axes(position) for position in all_positions]
        for ax in axes:
            self.clean_ax(ax)
        return axes

    def clean_ax(self, ax):
        ax.set_xticks([])
        ax.set_yticks([])

    def draw_figure(self):
        key_prefixes = ['left', 'center', 'right']
        arrays = [self.data, self.mieonly, self.mielens]
        for prefix, array in zip(key_prefixes, arrays):
            slices = [
                self.xy_slice_below, self.xy_slice_focus, self.xy_slice_above]
            suffixes = ['_lower_xy', '_center_xy', '_upper_xy']
            # 1. xy cross-section:
            for aslice, suffix in zip(slices, suffixes):
                im = array[aslice].squeeze()
                key = prefix + suffix
                self.axes_dict[key].imshow(im, **self._imshow_kwargs)
            # 2. xz cross-section:
            xzkey = prefix + '_xz'
            xzim = array[self.xz_slice].squeeze()[::-1, :]
            self.axes_dict[xzkey].imshow(xzim, **self._imshow_kwargs)

    def _get_slices(self):
        self.xy_slice_below = (slice(69, 70), slice(None), slice(None))
        self.xy_slice_focus = (slice(148, 149), slice(None), slice(None))
        self.xy_slice_above = (slice(207, 208), slice(None), slice(None))

        y_middle = self.data.shape[1] // 2
        self.xz_slice = (
            slice(None), slice(None), slice(y_middle, y_middle + 1))


def get_data():
    return load_few_PS_data_Jan10()[0]


class ImageCalculator(object):
    @classmethod
    def create_model_images(cls, all_data, all_params, theory_type='mielens'):
        models = []
        for data, params in zip(all_data, all_params):
            model = cls.calc_model(data, params, theory_type=theory_type)
            models.append(model.values.squeeze())
        return np.array(models)

    @classmethod
    def _make_scatterer(cls, params):
        x, y, z = [params[k] for k in 'xyz']
        center = (x, y, z)
        index = params['n']
        radius = params['r']
        scatterer = Sphere(n=index, r=radius, center=center)
        return scatterer

    @classmethod
    def calc_model(cls, data, params, theory_type='mielens'):
        scatterer = cls._make_scatterer(params)
        if theory_type == 'mielens':
            theory = MieLens(lens_angle=params['lens_angle'])
            scaling = 1.0
        else:
            theory = Mie()
            scaling = params['alpha']
        model = calc_holo(data, scatterer, theory=theory, scaling=scaling)
        return model


class ImageResampler(object):
    def __init__(self, sampled_zs_um, any_hologram):
        # sampled_zs_um = np.array([f['z'] for f in self.fit_parameters])
        self.sampled_zs_um = sampled_zs_um

        xy_px_size = np.diff(any_hologram.x.values).mean()
        self.xy_px_size = xy_px_size

        self._sampled_zs_px = sampled_zs_um / xy_px_size
        self._resampled_zs_px = np.arange(
            self._sampled_zs_px.min(), self._sampled_zs_px.max(), 1.0)

    def resample(self, array, kind='nearest'):
        interpolator = interp1d(self._sampled_zs_px, array, axis=0, kind=kind)
        return interpolator(self._resampled_zs_px)


def transform_mielens_fit_to_mieonly(params):
    d = {k: v for k, v in params.items()}
    d.pop('lens_angle')
    d.update({'alpha': 1.0})
    return d


def get_mielens_fits():
    fits = json.load(open(os.path.join(HERE, 'finalized-fits.json')),
                     object_pairs_hook=OrderedDict)
    fits_list = [v for v in fits.values()]
    indices = np.array([f['n'] for f in fits_list])
    radii = np.array([f['r'] for f in fits_list])
    angles = np.array([f['lens_angle'] for f in fits_list])
    best_radius = radii.mean()  # 0.4869
    best_index = indices.mean()  # 1.566
    best_angle = angles.mean()  # 0.613
    # Then we re-sample the zs to be evenly spaced, to avoid artifacts
    # near the focus:
    firstz = fits_list[0]['z']
    lastz = fits_list[-1]['z']
    shifts = np.linspace(0, 1, len(fits_list))
    for shift, fit in zip(shifts, fits_list):
        fit['z'] = (1 - shift) * firstz + shift * lastz
        # Then let's also update index, angle to reasonable values:
        fit['n'] = best_index
        fit['r'] = best_radius
        fit['lens_angle'] = best_angle
    return fits_list


def get_resampled_data_and_models():
    mielens_fit_parameters = get_mielens_fits()
    mieonly_fit_parameters = [
        transform_mielens_fit_to_mieonly(d) for d in mielens_fit_parameters]

    rawdata = get_data()  # 4.2 s
    rawdata_ar = np.array([d.values.squeeze() for d in rawdata])
    rawmielens = ImageCalculator.create_model_images(
        rawdata, mielens_fit_parameters, theory_type='mielens')  # 6.01 s
    rawmieonly = ImageCalculator.create_model_images(
        rawdata, mieonly_fit_parameters, theory_type='mieonly')  # 7.03 s

    sampled_zs_um = np.array([f['z'] for f in mielens_fit_parameters])
    resampler = ImageResampler(sampled_zs_um, rawdata[0])
    data_resampled = resampler.resample(rawdata_ar)
    mielens_resampled = resampler.resample(rawmielens)
    mieonly_resampled = resampler.resample(rawmieonly)
    return data_resampled, mielens_resampled, mieonly_resampled


if __name__ == '__main__':
    data, mielens, mieonly = get_resampled_data_and_models()
    bf = BigFigure(
        data[:, 8:-8, 8:-8], mieonly[:, 8:-8, 8:-8], mielens[:, 8:-8, 8:-8])
    plt.show()
    bf.figure.savefig('./data-mieonly-mielens-comparison.svg')

