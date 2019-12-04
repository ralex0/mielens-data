import os
import time
import warnings
import json
from collections import OrderedDict, namedtuple

import numpy as np

import matplotlib as mpl
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt; plt.close('all')
from matplotlib import rc
mpl.rcParams['figure.dpi'] = 600

from mpl_toolkits import mplot3d

import holopy as hp
from holopy.scattering import calc_holo, Sphere
from holopy.scattering.theory import MieLens
from holopy.core.process import normalize

import mielensfit as mlf
import figures
import monkeyrc
import inout

PS_FIT_DIR = 'fits/Polystyrene2-4um-60xWater-042919/'
PS_DATA_DIR = 'data/Polystyrene2-4um-60xWater-042919/'

SI_FIT_DIR = 'fits/Silica1um-60xWater-080619/'
SI_DATA_DIR = 'data/Silica1um-60xWater-080619/'

Particle = namedtuple("Particle", ["radius", "density"])
# We take the radii as the median radii from mielens. We use the
# median and not the mean to avoid the bad-fit outliers.
SILICA_PARTICLE = Particle(radius=0.5015611007969236, density=2.0)
# ..except the radii aren't correct for ml PS, so we use median(mo)
# for the PS
POLYSTYRENE_PARTICLE = Particle(radius=1.1624033452698568, density=1.05)
VISCOSITY_WATER = 8.9e-4  # in mks units = Pa*s

LABEL_FONT = {'size': 6, 'family': 'Times New Roman'}
TICK_FONT = {'family': 'Times New Roman', 'size': 6}
FIGLABEL_FONT = {'family': 'Times New Roman', 'size': 9}

class TrackingSedimentationFigure(object):
    _figsize = (5.25, 4.0) #  -- true figsize needs to be 5.25, x

    def __init__(self, data, mielens_fits, mieonly_fits, frame_times=None, xy_pos=None):
        self.data = data
        self.mielens_fits = mielens_fits
        self.mieonly_fits = mieonly_fits
        self.frame_times = self._setup_frame_times(frame_times)
        self.xy_pos = xy_pos

    def _setup_frame_times(self, frame_times):
        if frame_times is None:
            frame_times = range(len(self.mielens_fits))
        return frame_times

    def make_figure(self, holonums):
        assert len(holonums) == 3
        self.fig = plt.figure(figsize=self._figsize)
        self._make_axes()
        self._plot_holograms(holonums)
        self._plot_sedimentation(accent_these=holonums)
        self._plot_z()
        return self.fig

    def _make_axes(self):
        fig = self.fig
        # 1. Define the positions for all the axes:
        xpad = 0.01
        # make ypad the same as xpad in real units:
        ypad = xpad * self._figsize[0] / self._figsize[1]

        width_holo = 0.23
        width_plot = 0.22
        width_sedplot = 1 - (width_holo + width_plot + 4 * xpad  + 0.07)

        left_holo = xpad + 0.015
        left_sedplot = 2 * xpad + width_holo
        left_plot = 3 * xpad + width_holo + width_sedplot + 0.05

        height_holo = (1 - 4 * ypad) / 3.
        height_plot = 1.2 * height_holo

        bottom_holo_top = 1 - (ypad + height_holo)
        bottom_holo_mid = 1 - 2 * (ypad + height_holo)
        bottom_holo_bot = 1 - 3 * (ypad + height_holo)
        # We set the _top_ of the plot axes to be equal to the hologram top:
        bottom_plot_mid = bottom_holo_mid + 0.5 * (height_holo - height_plot)

        # 2. Make the axes.
        # We make the 3D plot first so it is on the bottom; otherwise it
        # overlaps the other axes.
        self.ax_sed = fig.add_axes(
            [left_sedplot, 0.025, width_sedplot, 1.0], label="sedplot")

        self.ax_topholo = fig.add_axes(
            [left_holo, bottom_holo_top, width_holo, height_holo],
            label="topholo")
        self.ax_midholo = fig.add_axes(
            [left_holo, bottom_holo_mid, width_holo, height_holo],
            label="midholo")
        self.ax_btmholo = fig.add_axes(
            [left_holo, bottom_holo_bot, width_holo, height_holo],
            label="bottomholo")

        self.ax_z = fig.add_axes(
            [left_plot, bottom_plot_mid, width_plot, height_plot],
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
        p_x = self.xy_pos[:,0]
        p_y = self.xy_pos[:,1]
        p_z = np.array([fit['z'] for fit in self.mielens_fits.values()])
        positions = {'x': p_x, 'y': p_y, 'z': p_z}
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
        self.ax_sed.set_xticklabels([])
        self.ax_sed.set_yticklabels([])
        self.ax_sed.set_aspect('equal')
        self.plotter_sed = plotter

    def _plot_z(self):
        mielens_z = [fit['z'] for fit in self.mielens_fits.values()]
        mieonly_z = [fit['z'] for fit in self.mieonly_fits.values()]

        mielens_times = self.frame_times
        mieonly_times = self.frame_times[:len(mieonly_z)]

        self.ax_z.set_ylabel('z position (μm)', **LABEL_FONT, labelpad=-1)
        self.ax_z.scatter(
            mielens_times, mielens_z, color=monkeyrc.COLORS['blue'], s=4,
            marker='o', label="With Lens", zorder=3)
        self.ax_z.scatter(
            mieonly_times, mieonly_z, color=monkeyrc.COLORS['red'], s=4,
            marker='^', label="Without Lens", zorder=3)
        self.ax_z.set_xlabel('Elapsed time (s)', **LABEL_FONT, labelpad=2)


class CharacterizationFigure(object):
    _figsize = (5.25, 1.5) #  -- true figsize needs to be 5.25, x

    def __init__(self, mielens_fits, mieonly_fits):
        self.mielens_fits = mielens_fits
        self.mieonly_fits = mieonly_fits
        self.fig = plt.figure(figsize=self._figsize)

    def make_figure(self):
        self._make_axes()
        self._plot_var(self.ax_n, 'n', 'Refractive Index')
        self._plot_var(self.ax_r, 'r', 'Radius (μm)')
        self._plot_var(self.ax_alpha, 'alpha', 'Alpha')
        self._plot_var(self.ax_lens, 'lens_angle', 'Lens Angle (radians)')
        self.fig.tight_layout()
        return self.fig

    def _make_axes(self):
        gs = gridspec.GridSpec(1, 4)

        self.ax_n = plt.Subplot(self.fig, gs[0, 0])
        self.ax_r = plt.Subplot(self.fig, gs[0, 1])
        self.ax_alpha = plt.Subplot(self.fig, gs[0, 2])
        self.ax_lens = plt.Subplot(self.fig, gs[0, 3])

        self.fig.add_subplot(self.ax_n)
        self.fig.add_subplot(self.ax_r)
        self.fig.add_subplot(self.ax_alpha)
        self.fig.add_subplot(self.ax_lens)
        self._axes = [self.ax_n, self.ax_r, self.ax_alpha, self.ax_lens]

    def _plot_var(self, axes, key, label):
        mielens_x = np.array([fit['z'][0] for fit in self.mielens_fits.values()])
        sort = np.argsort(mielens_x)
        mielens_x.sort()
        mielens_y = np.array([fit[key][0] for fit in self.mielens_fits.values()])[sort]
        mielens_yerr = np.array([fit[key][1] for fit in self.mielens_fits.values()])[sort]

        # axes.errorbar(
        #     mielens_x, mielens_y, mielens_yerr, color=monkeyrc.COLORS['blue'],
        #     marker='o', label="With Lens", ls='None', ms=2, elinewidth=0.5, fillstyle='none', markeredgewidth=0.5)
        axes.scatter(mielens_x, mielens_y, color=monkeyrc.COLORS['blue'],
                     marker='o', label="With Lens", s=2, ls='None')
        axes.fill_between(mielens_x,
                          mielens_y - mielens_yerr,
                          mielens_y + mielens_yerr,
                          interpolate=True,
                          color=monkeyrc.COLORS['blue'], alpha=0.5,
                          lw=1)
        if key != 'lens_angle':
            mieonly_x = np.array([fit['z'][0] for fit in self.mieonly_fits.values()])
            sort = np.argsort(mieonly_x)
            mieonly_x.sort()
            mieonly_y = np.array([fit[key][0] for fit in self.mieonly_fits.values()])[sort]
            mieonly_yerr = np.array([fit[key][1] for fit in self.mieonly_fits.values()])[sort]

            # axes.errorbar(
            #     mieonly_x, mieonly_y, mieonly_yerr, color=monkeyrc.COLORS['red'],
            #     marker='^', label="Without Lens", ls='None', ms=2, elinewidth=0.5, fillstyle='none', markeredgewidth=0.5)
            axes.scatter(mieonly_x, mieonly_y, color=monkeyrc.COLORS['red'],
                         marker='^', label="Without Lens", ls='None', s=2)
            axes.fill_between(mieonly_x,
                              mieonly_y - mieonly_yerr,
                              mieonly_y + mieonly_yerr,
                              interpolate=True,
                              color=monkeyrc.COLORS['red'], alpha=0.5,
                              lw=1)

        axes.set_ylabel(label, **LABEL_FONT)
        axes.set_xlabel('z position (μm)', **LABEL_FONT)


def zfill(n, nzeros=4):
    return str(n).rjust(nzeros, '0')


def clip_data_to(fits, max_frame):
    clipped_data = OrderedDict()
    for k, v in fits.items():
        if int(k) <= max_frame:
            clipped_data.update({k: v})
    return clipped_data


def update_z_vs_t_plot_with_expected_sedimentation(
        axes, times, particle, initial_z_position):
    # 1. Calculate the velocity, using meter-kilogram-second units:
    radius = particle.radius * 1e-6
    density = (particle.density - 1) * 1e3  # 1 g / cc = 1e3 kg / m^3
    volume = 4 * np.pi / 3. * radius**3
    mass = density * volume
    gravity = 9.8  # mks
    force = mass * gravity
    drag = 6 * np.pi * VISCOSITY_WATER * radius
    velocity_meters_per_second = force / drag
    velocity_microns_per_second = 1e6 * velocity_meters_per_second

    # 2. Calculate the trajectory:
    trajectory = initial_z_position - times * velocity_microns_per_second
    line = axes.plot(times, trajectory, '--', color='#404040', zorder=1)
    return line


def make_ps_figure(ps_data=None, mofit_ps=None, mlfit_ps=None):
    ps_data = _thin_ps(ps_data)
    ps_times = np.array(_thin_ps(np.load(PS_DATA_DIR + 'PS_frame_times.npy')))
    xy_pos = np.array(_thin_ps(np.load(PS_DATA_DIR + 'processed-256-uncentered/xy-positions.npy')))
    figure_ps = TrackingSedimentationFigure(
        ps_data, mlfit_ps, mofit_ps, ps_times, xy_pos)
    fig_ps = figure_ps.make_figure(holonums=[0, 37, 99])
    figure_ps.plotter_sed.set_xlim(-6.25, 9.25)
    figure_ps.plotter_sed.set_ylim(-1.25, 14.25)
    figure_ps.plotter_sed.set_zlim(-13, 20)
    figure_ps.ax_sed.set_ylim(-18., 22)

    figure_ps.ax_z.legend(fontsize=6, loc='upper right')
    yticks = [-14, 0, 14]
    ylabels = [str(i) for i in yticks]
    figure_ps.ax_z.set_yticks(yticks)
    figure_ps.ax_z.set_yticklabels(ylabels, **TICK_FONT)
    figure_ps.ax_z.set_ylim(-14, 17)

    xticks = [0, 80, 160]
    xlabels = [str(i) for i in xticks]
    figure_ps.ax_z.set_xlim(0, 160)
    figure_ps.ax_z.set_xticks(xticks)
    figure_ps.ax_z.set_xticklabels(xlabels, **TICK_FONT)

    initial_z = mlfit_ps['0']['z']
    _ = update_z_vs_t_plot_with_expected_sedimentation(
        figure_ps.ax_z, ps_times, POLYSTYRENE_PARTICLE, initial_z)

    return figure_ps, fig_ps

def make_chr_figure(mofit_ps=None, mlfit_ps=None):
    figure_chr = CharacterizationFigure(mlfit_ps, mofit_ps)
    fig_chr = figure_chr.make_figure()

    xlim = [18, -15]
    xticks  = [-10, 0, 10]
    xlabels = [str(i) for i in xticks]

    for ax in figure_chr._axes:
        ax.set_xlim(xlim)
        ax.set_xticks(xticks)
        ax.set_xticklabels(xlabels, **TICK_FONT)

    figure_chr.ax_n.set_ylim([1.4, 1.8])
    yticks_n = [1.5, 1.6, 1.7]
    figure_chr.ax_n.set_yticks(yticks_n)
    figure_chr.ax_n.set_yticklabels([str(i) for i in yticks_n], **TICK_FONT)
    figure_chr.ax_n.legend(fontsize=4, loc=(.45, .8))

    figure_chr.ax_r.set_ylim([0.8, 1.6])
    yticks_r = [1.0, 1.2, 1.4]
    figure_chr.ax_r.set_yticks(yticks_r)
    figure_chr.ax_r.set_yticklabels([str(i) for i in yticks_r], **TICK_FONT)

    figure_chr.ax_alpha.set_ylim([0.2, 1.2])
    yticks_alpha = [0.25, 0.5, 0.75, 1.0]
    figure_chr.ax_alpha.set_yticks(yticks_alpha)
    figure_chr.ax_alpha.set_yticklabels([str(i) for i in yticks_alpha], **TICK_FONT)

    figure_chr.ax_lens.set_ylim([0.2, 1.2])
    yticks_lens = [0.25, 0.5, 0.75, 1.0]
    figure_chr.ax_lens.set_yticks(yticks_lens)
    figure_chr.ax_lens.set_yticklabels([str(i) for i in yticks_lens], **TICK_FONT)

    return figure_chr, fig_chr

def _thin_ps(data):
    nums = np.arange(0, 1000, 10)
    return [data[num] for num in nums]

if __name__ == '__main__':
    # ps_data = inout.fastload_polystyrene_sedimentation_data(size=256, recenter=False)

    # ps_fits_mo = inout.load_json('PTmcmc_results_PS_mieonly_last100.json')
    # ps_fits_ml = inout.load_json('PTmcmc_results_PS_mielensalpha_last100.json')

    ps_fits_witherr_mo = inout.load_json('PTmcmc_results_PS_mieonly_last100_werror.json')
    ps_fits_witherr_ml = inout.load_json('PTmcmc_results_PS_mielensalpha_last100_werror.json')

    # sed_figure_ps, sed_fig_ps = make_ps_figure(ps_data, ps_fits_mo, ps_fits_ml)
    chr_figure_ps, chr_fig_ps = make_chr_figure(ps_fits_witherr_mo, ps_fits_witherr_ml)

    # sed_fig_ps.savefig('./polystyrene-sedimentation.png', dpi=600)
    # chr_fig_ps.savefig('./polystyrene-characterization.png', dpi=600)

    plt.show()
