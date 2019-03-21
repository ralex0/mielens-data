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

        left_holo = xpad + 0.015
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

        left_sedplt = width_holo - left_holo + xpad
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
            [left_holo, bottom_holo_top, width_holo, height_holo],
            label="topholo")
        self.ax_midholo = fig.add_axes(
            [left_holo, bottom_holo_mid, width_holo, height_holo],
            label="midholo")
        self.ax_btmholo = fig.add_axes(
            [left_holo, bottom_holo_bot, width_holo, height_holo],
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
        excursion = max([1 - np.min(holos), np.max(holos) - 1])
        vmin = 1 - excursion
        vmax = 1 + excursion
        for num, ax in enumerate(axes):
            ax.imshow(holos[num], vmin=vmin, vmax=vmax,
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
            positions['x'], positions['y'], positions['z'],
            color=monkeyrc.COLORS['blue'], lw=2)
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
        self.plotter_sed = plotter

    def _plot_parameters(self):
        mielens_index = [fit['n'] for fit in self.mielens_fits.values()]
        mieonly_index = [fit['n'] for fit in self.mieonly_fits.values()]
        mielens_rad = [fit['r'] for fit in self.mielens_fits.values()]
        mieonly_rad = [fit['r'] for fit in self.mieonly_fits.values()]
        mielens_z = [fit['center.2'] for fit in self.mielens_fits.values()]
        mieonly_z = [fit['center.2'] for fit in self.mieonly_fits.values()]
        mielens_times = [t for t, n in zip(self.frame_times, mielens_index)]
        mieonly_times = [t for t, n in zip(self.frame_times, mieonly_index)]

        self.ax_n.set_ylabel('Refractive Index', labelpad=0)
        self.ax_n.scatter(
            mielens_times, mielens_index, color=monkeyrc.COLORS['blue'], s=4,
            marker='o', label="With Lens")
        self.ax_n.scatter(
            mieonly_times, mieonly_index, color=monkeyrc.COLORS['red'], s=4,
            marker='^', label="Without Lens")
        self.ax_n.tick_params(labelsize=7)

        self.ax_r.set_ylabel('Radius', labelpad=1)
        self.ax_r.scatter(
            mielens_times, mielens_rad, color=monkeyrc.COLORS['blue'], s=4,
            marker='o', label="With Lens")
        self.ax_r.scatter(
            mieonly_times, mieonly_rad, color=monkeyrc.COLORS['red'], s=4,
            marker='^', label="Without Lens")
        self.ax_r.tick_params(labelsize=7)

        self.ax_z.set_xlabel('Elapsed time (s)', labelpad=2)
        self.ax_z.set_ylabel('z-position  ($\mu m$)', labelpad=-4)
        self.ax_z.scatter(
            mielens_times, mielens_z, color=monkeyrc.COLORS['blue'], s=4,
            marker='o', label="With Lens")
        self.ax_z.scatter(
            mieonly_times, mieonly_z, color=monkeyrc.COLORS['red'], s=4,
            marker='^', label="Without Lens")
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


def make_si_figure(si_data=None):
    if si_data is None:
        si_data = load_Si_sedemintation_data_Feb15()[0]
    si_times = np.load("./fits/sedimentation/Si_frame_times.npy")
    mofit_si_full = json.load(
        open("fits/sedimentation/mieonly_sedimentation_fits_Si.json", 'r'),
        object_pairs_hook=OrderedDict)
    mofit_si_clipped = clip_data_to(mofit_si_full, 45)
    mlfit_si = json.load(
        open("fits/sedimentation/mielens_sedimentation_fits_Si.json", 'r'),
        object_pairs_hook=OrderedDict)
    figure_si = TrackingSedimentationFigure(
        si_data, mlfit_si, mofit_si_clipped, si_times)
    fig_si = figure_si.make_figure(holonums=[0, 45, 99])
    # Then we have to rescale the 3d plot b/c fuck matplotlib:
    figure_si.ax_sed.set_ylim(-69.8, -4.2)

    figure_si.ax_r.legend(fontsize=6, loc='upper left')
    figure_si.ax_n.set_yticks([1.3, 1.4, 1.5])
    figure_si.ax_n.set_ylim([1.3, 1.5])

    figure_si.ax_r.set_yticks([0.6, 0.8, 1.0])
    figure_si.ax_r.set_ylim([0.6, 1.0])

    figure_si.ax_z.set_yticks([-20, 0, 20, 40])
    figure_si.ax_z.set_ylim(-20, 40)

    for ax in [figure_si.ax_r, figure_si.ax_n, figure_si.ax_z]:
        ax.set_xlim(0, 60)
        ax.set_xticks([0, 30, 60])
    return figure_si, fig_si


def make_ps_figure(ps_data=None):
    if ps_data is None:
        ps_data = load_few_PS_sedemintation_data_Jan24()[0]
    ps_times = np.load("./fits/sedimentation/PS_frame_times.npy")
    mofit_ps_full = json.load(
        open("fits/sedimentation/mieonly_sedimentation_fits_PS.json", 'r'),
        object_pairs_hook=OrderedDict)
    mofit_ps_clipped = clip_data_to(mofit_ps_full, 19)
    mlfit_ps = json.load(
        open("fits/sedimentation/mielens_sedimentation_fits_PS.json", 'r'),
        object_pairs_hook=OrderedDict)
    figure_ps = TrackingSedimentationFigure(
        ps_data, mlfit_ps, mofit_ps_clipped, ps_times)
    fig_ps = figure_ps.make_figure(holonums=[0, 20, 49])
    figure_ps.plotter_sed.set_xlim(36, 53)
    figure_ps.plotter_sed.set_ylim(34, 51)
    figure_ps.ax_sed.set_ylim(-37.5, 7.5)

    figure_ps.ax_z.legend(fontsize=6, loc='upper right')
    figure_ps.ax_n.set_yticks([1.5, 1.6, 1.7])
    figure_ps.ax_n.set_ylim([1.5, 1.7])
    figure_ps.ax_r.set_yticks([0.8, 1.0, 1.2])
    figure_ps.ax_r.set_ylim([0.8, 1.2])
    figure_ps.ax_z.set_yticks([-15, 0, 15, 30])
    figure_ps.ax_z.set_ylim(-15, 30)
    for ax in [figure_ps.ax_r, figure_ps.ax_n, figure_ps.ax_z]:
        ax.set_xlim(0, 300)
        ax.set_xticks([0, 150, 300])
    return figure_ps, fig_ps


if __name__ == '__main__':
    si_data = load_Si_sedemintation_data_Feb15()[0]
    ps_data = load_few_PS_sedemintation_data_Jan24()[0]

    figure_si, fig_si = make_si_figure(si_data)
    figure_ps, fig_ps = make_ps_figure(ps_data)

    plt.show()
    fig_si.savefig('./silica-sedimentation.svg')
    fig_ps.savefig('./polystyrene-sedimentation.svg')

