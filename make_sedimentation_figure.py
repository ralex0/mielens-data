import os
import time
import warnings
import json
from collections import OrderedDict

import numpy as np

import matplotlib.pyplot as plt; plt.close('all')
from matplotlib import rc

from mpl_toolkits import mplot3d

import holopy as hp
from holopy.scattering import calc_holo
from holopy.scattering.theory import MieLens

import mielensfit as mlf
import figures
import monkeyrc


class TrackingSedimentationFigure(object):
    _figsize = (5.25, 4.0) #  -- true figsize needs to be 5.25, x

    def __init__(self, data, mielens_fits, mieonly_fits, frame_times=None):
        self.data = data
        self.mielens_fits = mielens_fits
        self.mieonly_fits = mieonly_fits
        self.frame_times = self._setup_frame_times(frame_times)

    def _setup_frame_times(self, frame_times):
        if frame_times is None:
            frame_times = range(len(self.mielens_fits))
        return frame_times

    def make_figure(self, holonums):
        assert len(holonums) == 3
        self.fig = plt.figure(figsize=self._figsize)
        self._make_axes(self.fig)
        self._plot_holograms(holonums)
        self._plot_sedimentation(accent_these=holonums)
        self._plot_parameters()
        return self.fig

    def _make_axes(self, fig):
        # 1. Define the positions for all the axes:
        xpad = 0.01
        # make ypad the same as xpad in real units:
        ypad = xpad * self._figsize[0] / self._figsize[1]

        width_holo = 0.22
        height_holo = (1 - 4 * ypad) / 3.
        width_plot = 0.2
        height_plot = height_holo - 5 * ypad  # extra space for labels
        width_sedplt = (1 - 4 * xpad - width_plot - width_holo)

        bottom_holo_top = 1 - (ypad + height_holo)
        bottom_holo_mid = 1 - 2 * (ypad + height_holo)
        bottom_holo_bot = 1 - 3 * (ypad + height_holo)
        # We set the _top_ of the plot axes to be equal to the hologram top:
        bottom_plot_top = bottom_holo_top + (height_holo - height_plot)
        bottom_plot_mid = bottom_holo_mid + (height_holo - height_plot)
        bottom_plot_bot = bottom_holo_bot + (height_holo - height_plot)

        left_sedplt = width_holo + xpad
        left_plot = 1 - xpad - width_plot

        # 2. Make the axes.
        # We make the 3D plot first so it is on the bottom; otherwise it
        # overlaps the other axes.
        # ax_sedplt = fig.add_axes(
        #     [left_sedplt, 0.025, width_sedplt, 1.0], projection='3d',
        #     label="sedplot")
        self.ax_sed = fig.add_axes(
            [left_sedplt, 0.025, width_sedplt, 1.0], label="sedplot")

        self.ax_topholo = fig.add_axes(
            [xpad, bottom_holo_top, width_holo, height_holo],
            label="topholo")
        self.ax_midholo = fig.add_axes(
            [xpad, bottom_holo_mid, width_holo, height_holo],
            label="midholo")
        self.ax_btmholo = fig.add_axes(
            [xpad, bottom_holo_bot, width_holo, height_holo],
            label="bottomholo")

        self.ax_n = fig.add_axes(
            [left_plot, bottom_plot_top, width_plot, height_plot],
            label="nplot")
        self.ax_r = fig.add_axes(
            [left_plot, bottom_plot_mid, width_plot, height_plot],
            label="rplot")
        self.ax_z = fig.add_axes(
            [left_plot, bottom_plot_bot, width_plot, height_plot],
            label="zplot")

    def _plot_holograms(self, indices):
        axes = [self.ax_topholo, self.ax_midholo, self.ax_btmholo]
        holos = [self.data[num].values.squeeze() for num in indices]
        for num, ax in enumerate(axes):
            ax.imshow(holos[num], vmin=np.min(holos), vmax=np.max(holos),
                        interpolation='nearest', cmap='gray')
            ax.axis('off')

    def _plot_sedimentation(self, accent_these=None):
        positions = {
            k1: np.array(
                [fit['center.{}'.format(k2)]
                 for fit in self.mielens_fits.values()])
            for k1, k2 in zip(['x', 'y', 'z'], [0, 1, 2])}

        plotter = figures.ThreeDPlot(
            self.ax_sed, azimuth_elevation=(0.75*np.pi, 0.1*np.pi))
        plotter.plot(
            positions['x'], positions['y'], positions['z'], color='#8080F0',
            lw=2)
        if accent_these is not None:
            accent_x = positions['x'][accent_these]
            accent_y = positions['y'][accent_these]
            accent_z = positions['z'][accent_these]
            plotter.plot(
                accent_x, accent_y, accent_z, color='#6060A0', marker='o',
                linestyle='', rescale=False)
        # axes.set_xlabel('x', {'size': 8})
        self.ax_sed.set_xticklabels([])
        # axes.set_ylabel('y', {'size': 8})
        self.ax_sed.set_yticklabels([])
        # axes.set_zlabel('z', {'size': 8})
        # axes.set_title("Best Fit Position", {'size': 8})
        self.ax_sed.set_aspect('equal')  # , 'box')

    def _plot_parameters(self):
        mielens_index = [fit['n'] for fit in self.mielens_fits.values()]
        mieonly_index = [fit['n'] for fit in self.mieonly_fits.values()]
        mielens_rad = [fit['r'] for fit in self.mielens_fits.values()]
        mieonly_rad = [fit['r'] for fit in self.mieonly_fits.values()]
        mielens_z = [fit['center.2'] for fit in self.mielens_fits.values()]
        mieonly_z = [fit['center.2'] for fit in self.mieonly_fits.values()]
        mielens_times = [t for t, n in zip(self.frame_times, mielens_index)]
        mieonly_times = [t for t, n in zip(self.frame_times, mieonly_index)]

        self.ax_n.set_ylabel('Refractive Index', {'size': 8}, labelpad=0)
        self.ax_n.scatter(
            mielens_times, mielens_index, color='green', s=4, marker='o',
            label="with lens")
        self.ax_n.scatter(
            mieonly_times, mieonly_index, color='red', s=4, marker='^',
            label="without lens")
        self.ax_n.tick_params(labelsize=7)

        self.ax_r.set_ylabel('Radius', {'size': 8}, labelpad=1)
        self.ax_r.scatter(
            mielens_times, mielens_rad, color='green', s=4, marker='o',
            label="with lens")
        self.ax_r.scatter(
            mieonly_times, mieonly_rad, color='red', s=4, marker='^',
            label="without lens")
        self.ax_r.tick_params(labelsize=7)

        self.ax_z.set_xlabel('Elapsed time (s)', {'size': 8}, labelpad=2)
        self.ax_z.set_ylabel('z-position  ($\mu m$)', {'size': 8}, labelpad=-4)
        self.ax_z.scatter(
            mielens_times, mielens_z, color='green', s=4, marker='o',
            label="with lens")
        self.ax_z.scatter(
            mieonly_times, mieonly_z, color='red', s=4, marker='^',
            label="without lens")
        self.ax_z.tick_params(labelsize=7)


def load_Si_sedemintation_data_Feb15():
    camera_resolution = 5.6983 # px / um
    metadata = {'spacing' : 1 / camera_resolution,
                    'medium_index' : 1.33,
                    'illum_wavelen' : .660,
                    'illum_polarization' : (1, 0)}
    position = [553, 725]

    holonums = range(100)
    zpos = np.linspace(38, -12, 100)
    paths = ["data/Silica1um-60xWater-021519/raw/image"
             +  zfill(num, 4) + ".tif" for num in holonums]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        refimg = hp.load_image(paths[0], **metadata)
        bkg = mlf.load_bkg(
            "data/Silica1um-60xWater-021519/raw/bg/",
            bg_prefix='bg', refimg=refimg)  # 10 s! all holopy
        dark = mlf.load_dark(
            "data/Silica1um-60xWater-021519/raw/dark/",
            df_prefix='dark', refimg=refimg)  # 8.7 s! all holopy
        holos = [mlf.load_bgdivide_crop_v2(path=path, metadata=metadata,
                                        particle_position=position,
                                        bkg=bkg, dark=dark, size=140)
                 for path in paths]
    return holos, zpos


def load_few_PS_sedemintation_data_Jan24():
    camera_resolution = 5.6983 # px / um
    metadata = {'spacing' : 1 / camera_resolution,
                    'medium_index' : 1.33,
                    'illum_wavelen' : .660,
                    'illum_polarization' : (1, 0)}
    position = [242, 266]

    holonums = range(50)
    zpos = np.linspace(20, -12, 50)
    paths = ["data/Polystyrene2-4um-60xWater-012419/raw/image"
             +  zfill(num, 4) + ".tif" for num in holonums]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        holos = [mlf.load_bgdivide_crop(path=path, metadata=metadata,
                                        particle_position=position,
                                        bg_prefix="bg", df_prefix="dark")
             for path in paths]
    return holos, zpos


def zfill(n, nzeros=4):
    return str(n).rjust(nzeros, '0')


def clip_data_to(fits, max_frame):
    clipped_data = OrderedDict()
    for k, v in fits.items():
        if int(k) <= max_frame:
            clipped_data.update({k: v})
    return clipped_data


if __name__ == '__main__':
    Si_data = load_Si_sedemintation_data_Feb15()[0]
    Si_times = np.load("./fits/sedimentation/Si_frame_times.npy")
    mofit_Si_full = json.load(
        open("fits/sedimentation/mieonly_sedimentation_fits_Si.json", 'r'),
        object_pairs_hook=OrderedDict)
    mofit_Si_clipped = clip_data_to(mofit_Si_full, 45)
    mlfit_Si = json.load(
        open("fits/sedimentation/mielens_sedimentation_fits_Si.json", 'r'),
        object_pairs_hook=OrderedDict)
    figure_Si = TrackingSedimentationFigure(
        Si_data, mlfit_Si, mofit_Si_clipped, Si_times)
    fig_si = figure_Si.make_figure(holonums=[0, 45, 99])
    # Then we have to rescale the 3d plot b/c fuck matplotlib:
    figure_Si.ax_sed.set_ylim(-70, -4)

    figure_Si.ax_n.legend(fontsize=6, loc='upper right')
    figure_Si.ax_n.set_yticks([1.3, 1.4, 1.5, 1.6])
    figure_Si.ax_n.set_ylim([1.3, 1.6])
    figure_Si.ax_r.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0, 1.2])
    figure_Si.ax_r.set_ylim([0.2, 1.2])
    figure_Si.ax_z.set_yticks([-20, 0, 20, 40])
    figure_Si.ax_z.set_ylim(-20, 40)
    for ax in [figure_Si.ax_r, figure_Si.ax_n, figure_Si.ax_z]:
        ax.set_xlim(0, 60)
        ax.set_xticks([0, 30, 60])


    PS_data = load_few_PS_sedemintation_data_Jan24()[0]
    PS_times = np.load("./fits/sedimentation/PS_frame_times.npy")
    mofit_PS_full = json.load(
        open("fits/sedimentation/mieonly_sedimentation_fits_PS.json", 'r'),
        object_pairs_hook=OrderedDict)
    mofit_PS_clipped = clip_data_to(mofit_PS_full, 19)
    mlfit_PS = json.load(
        open("fits/sedimentation/mielens_sedimentation_fits_PS.json", 'r'),
        object_pairs_hook=OrderedDict)
    figure_PS = TrackingSedimentationFigure(
        PS_data, mlfit_PS, mofit_PS_clipped, PS_times)
    fig_ps = figure_PS.make_figure(holonums=[0, 20, 49])
    figure_PS.ax_sed.set_ylim(-36, 6)

    figure_PS.ax_z.legend(fontsize=6, loc='upper right')
    figure_PS.ax_n.set_yticks([1.5, 1.6, 1.7])
    figure_PS.ax_n.set_ylim([1.5, 1.7])
    figure_PS.ax_r.set_yticks([0.8, 1.0, 1.2])
    figure_PS.ax_r.set_ylim([0.8, 1.2])
    figure_PS.ax_z.set_yticks([-15, 0, 15, 30])
    figure_PS.ax_z.set_ylim(-15, 30)
    for ax in [figure_PS.ax_r, figure_PS.ax_n, figure_PS.ax_z]:
        ax.set_xlim(0, 300)
        ax.set_xticks([0, 150, 300])

    plt.show()
    fig_si.savefig('./silica-sedimentation.svg')
    fig_ps.savefig('./polystyrene-sedimentation.svg')
