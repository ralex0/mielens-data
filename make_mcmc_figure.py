import os

import numpy as np
from scipy.stats import gaussian_kde

import matplotlib as mpl; mpl.rcdefaults()
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt; plt.close('all')

import seaborn as sns

from pandas import DataFrame

import inout



class KDEEstimator(object):
    def __init__(self, samples, names):
        self.samples = samples
        self.names = list(names)

    def estimate_univariate_density(self, name):
        index = self.names.index(name)
        samples = self.samples[:, index]
        kernel_callable = gaussian_kde(samples)
        return kernel_callable

    def estimate_bivariate_density(self, name1, name2):
        index1 = self.names.index(name1)
        index2 = self.names.index(name2)
        samples = self.samples[:, [index1, index2]]
        kernel_callable = gaussian_kde(samples)
        return kernel_callable



class MCMCFigure(object):
    figsize = (5.25, 1)

    def __init__(self, samples, mode='mielens'):
        self.samples = samples
        self.mode = mode
        if mode == 'mielens':
            names = {'n', 'r', 'lens_angle'}
        elif mode == 'mieonly':
            names = {'n', 'r', 'alpha'}
        else:
            raise ValueError("mode must be one of mielens, mieonly")
        self.names = names

        self.figure = plt.figure(figsize=self.figsize)
        self.axes_dict = self._setup_axes()


class AxesSetup(object):
    def __init__(self, figsize=(8, 6), xpad=0.05, ypad=0.20):
        self.figsize = figsize
        self.figure = plt.figure(figsize=self.figsize)
        self.xpad = xpad
        self.ypad = ypad
        self.axes = self._setup_axes()

    def _setup_axes(self):
        # Should make 2 sets of 6 axes, one for each pair each joint.
        # Cohesion suggests this should be a separate object
        #
        axes_config = self._get_axes_config()
        axes_left = self._transform_axes_to_box(
            axes_config,
            (0.5 * self.xpad, 0.5 * self.ypad, 0.5 - self.xpad, 1 - self.ypad))
        axes_right = self._transform_axes_to_box(
            axes_config,
            (.5 + .5 * self.xpad, .5 * self.ypad, .5 - self.xpad,
             1 - self.ypad))
        axes_dict = {}
        axes_names = ['left', 'right']
        for name, d in zip(axes_names, [axes_left, axes_right]):
            for key, box in d.items():
                ax = self.figure.add_axes(box)
                axes_dict.update({'{}_{}'.format(name, key): ax})
        return axes_dict

    def _get_axes_config(self):
        # axes boxes are left, bottom, width, height
        box_height = 1.0 / 2.3
        box_width = 0.5
        half_height = box_height / 2.0
        half_width = 2 * half_height * self.figsize[1] / self.figsize[0]
        axes_positions = {}

        axes_positions.update(
            {'lower_left_box': (0, 0, box_width, box_height),
             'upper_left_box': (0, box_height, box_width, box_height),
             'upper_right_box': (box_width, box_height, box_width, box_height)})

        axes_positions.update(
            {'lower_left_half': (box_width, 0, half_width, box_height),
             'upper_left_half': (0, 2 * box_height, box_width, half_height),
             'upper_right_half':
                (box_width, 2 * box_height, box_width, half_height)})

        return axes_positions

    def _transform_axes_to_box(self, axes, box):
        box_left, box_bottom, box_width, box_height = box
        new_axes = {}
        for key, value in axes.items():
            old_left, old_bottom, old_width, old_height = value

            new_left = box_left + old_left * box_width
            new_bottom = box_bottom + old_bottom * box_height
            new_width = old_width * box_width
            new_height = old_height * box_height

            new_axes.update(
                {key: (new_left, new_bottom, new_width, new_height)})
        return new_axes


def fuckme():
    # data = inout.load_mcmc_result_PS_mielens()
    fitfolder = './fits/sedimentation/mcmc/'
    figures = []
    modes = {
        'mielens': {
            'variables': [
                'n', 'r', 'lens_angle'],
            'labels': [
                "refractive index", "radius (μm)", "angular aperture (rad)"],
            },
        'mieonly': {
            'variables': [
                'n', 'r', 'alpha'],
            'labels': [
                "refractive index", "radius (μm)", "field rescale"],
            },
        }

    for mode in modes.keys():
        filename = os.path.join(
            fitfolder,
            'polystyrene-{}-frame=2-size=250-npx=10000-mcmc.pkl'.format(mode))
        data = inout.load_pickle(filename)
        plotter = MCMCJointPlotFigure(data, burnin=-500)
        fig = plotter.plot(
            *modes[mode]['variables'], labels=modes[mode]['labels'])
        figures.append(fig)
    plt.show()


if __name__ == '__main__':
    axes_setup = AxesSetup(figsize=(5.25, 3))
    plt.show()

