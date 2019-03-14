"""
    General comments about this:
    Using the best-fits obviously gives a nice result, but the best fits
    are tricky to interpolate since the z fluctuates.
    But using a linear z interpolation for the model doesn't give a
    good result, since the particle radius, index, etc vary.

    another option is to fix (n, r, lens_angle), and do a continuously
    varying z, to show a similar plot. I think that might be the
    clearest.

    What I've settled on here is using the mean radius, index, lens angle
    from all the fits. This gives a big bright spot in the center.

    I don't see a big difference in the way it looks from that and
    using an attempt at fitting all the frames at once. Since fitting
    all the frames gives slightly weirder values (to avoid the bright
    spot in the middle...), I am just using mean from the fits.
"""
import os
import json
from collections import OrderedDict

import numpy as np
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt

import mielensfit as mlf
from fit_data import load_few_PS_data_Jan10, compare_imgs
import monkeyrc as mrc


HERE = os.path.dirname(__file__)


class XZFigure(object):
    figsize = [4, 4]
    cmap = 'GraySaturated'

    def __init__(self, fit_parameters, holos=None):
        self.fit_parameters = fit_parameters
        if holos is None:
            self._holos, _ = load_few_PS_data_Jan10()
        else:
            self._holos = holos
        self._raw_data = np.array([h.values.squeeze() for h in self._holos])
        self._raw_model = self.create_model_images()

        self._xy_px_size = np.diff(self._holos[0].x.values).mean()

        self._setup_z_values()
        self.resampled_data = self.resample(self._raw_data, kind='nearest')
        self.resampled_model = self.resample(self._raw_model, kind='linear')

    def create_model_images(self):
        models = []
        for holo, params in zip(self._holos, self.fit_parameters):
            fitter = mlf.Fitter(holo, params)
            models.append(fitter.evaluate_model(params).values.squeeze())
        return np.array(models)

    def grab_xz_slice(self, array):
        return array[:, 50]

    def make_plot(self):
        data = self.grab_xz_slice(self.resampled_data)
        model = self.grab_xz_slice(self.resampled_model)
        # vmax = max(data.max(), model.max())
        # -- the model is much brighter than the data at the focus.
        # Some of this is that the data is saturated near the focus
        vmax = data.max()
        vmin = 0
        fig = plt.figure(figsize=self.figsize)

        ax1 = fig.add_subplot(1, 2, 1)
        ax2 = fig.add_subplot(1, 2, 2)
        for ax, im, title in zip([ax1, ax2], [data, model], ['Data', 'Model']):
            ax.imshow(im, interpolation='nearest', cmap=self.cmap, vmin=vmin,
                      vmax=vmax)
            ax.set_xticks([])
            ax.set_yticks([])
            # ax.set_title(title)
        return fig, [ax1, ax2]

    def _setup_z_values(self):
        sampled_zs_um = np.array([f['z'] for f in self.fit_parameters])
        self._sampled_zs_px = sampled_zs_um / self._xy_px_size
        self._resampled_zs_px = np.arange(
            self._sampled_zs_px.min(), self._sampled_zs_px.max(), 1.0)

    def resample(self, array, kind='nearest'):
        interpolator = interp1d(self._sampled_zs_px, array, axis=0, kind=kind)
        return interpolator(self._resampled_zs_px)


def find_best_global_index_radius_lensangle():
    fits = json.load(open(os.path.join(HERE, 'finalized-fits.json')),
                     object_pairs_hook=OrderedDict)
    fits_list = [v for v in fits.values()]
    # Then we re-sample the zs to be evenly spaced, to avoid artifacts
    # near the focus:
    firstz = fits_list[0]['z']
    lastz = fits_list[-1]['z']
    shifts = np.linspace(0, 1, len(fits_list))
    for shift, fit in zip(shifts, fits_list):
        fit['z'] = (1 - shift) * firstz + shift * lastz

    fig = XZFigure(fits_list)
    data = fig._raw_data

    def get_residuals(index_radius_angle):
        index, radius, angle = index_radius_angle
        for f in fits_list:
            f['n'] = index
            f['r'] = radius
            f['lens_angle'] = angle
        fig.fit_parameters = fits_list
        model = fig.create_model_images()
        return np.ravel(model - data)

    indices = np.array([f['n'] for f in fits_list])
    radii = np.array([f['r'] for f in fits_list])
    angles = np.array([f['lens_angle'] for f in fits_list])
    p0 = np.array([indices.mean(), radii.mean(), angles.mean()])
    res = leastsq(get_residuals, p0, maxfev=75)
    return res  # 1.52, 0.504, 0.496


def get_fits():
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


if __name__ == '__main__':
    fits = get_fits()
    holos = load_few_PS_data_Jan10()[0]
    fig = XZFigure(fits, holos=holos)
    fig.make_plot()
    plt.show()
