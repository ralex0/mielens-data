import os
import time
import warnings
import json
from collections import OrderedDict, namedtuple

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

        left_holo = xpad
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


def make_si_figure(si_data=None, mofit_si=None, mlfit_si=None):
    si_times = np.load(SI_DATA_DIR + 'Si_frame_times.npy')
    xy_pos = np.load(SI_DATA_DIR + 'processed0-256-uncentered/xy-positions.npy')
    figure_si = TrackingSedimentationFigure(
        si_data, mlfit_si, mofit_si, si_times, xy_pos)
    fig_si = figure_si.make_figure(holonums=[0, 500, 999])
    # Then we have to rescale the 3d plot b/c fuck matplotlib:
    figure_si.ax_sed.set_ylim(-30., 9.)

    figure_si.ax_z.legend(fontsize=6, loc='upper right')
    yticks = [-20, -10, 0, 10, 20]
    ylabels = [str(i) for i in yticks]
    figure_si.ax_z.set_yticks(yticks)
    figure_si.ax_z.set_yticklabels(ylabels, **TICK_FONT)
    figure_si.ax_z.set_ylim(-30., 9.)

    max_time = np.ceil(np.max(si_times))
    figure_si.ax_z.set_xlim(0, max_time)
    xticks = max_time * np.array([0, .50, 1.0])
    xlabels = [str(i) for i in xticks]
    figure_si.ax_z.set_xticks(xticks)
    figure_si.ax_z.set_xticklabels(xlabels, **TICK_FONT)

    initial_z = mlfit_si['0']['z']
    _ = update_z_vs_t_plot_with_expected_sedimentation(
        figure_si.ax_z, si_times, SILICA_PARTICLE, initial_z)

    return figure_si, fig_si


def make_ps_figure(ps_data=None, mofit_ps=None, mlfit_ps=None):
    ps_times = np.load(PS_DATA_DIR + 'PS_frame_times.npy')
    xy_pos = np.load(PS_DATA_DIR + 'processed-256-uncentered/xy-positions.npy')
    figure_ps = TrackingSedimentationFigure(
        ps_data, mlfit_ps, mofit_ps, ps_times, xy_pos)
    fig_ps = figure_ps.make_figure(holonums=[0, 500, 999])
    figure_ps.plotter_sed.set_xlim(-20., 8.)
    figure_ps.plotter_sed.set_ylim(-6., 9.)
    figure_ps.ax_sed.set_ylim(-14., 17.)

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


if __name__ == '__main__':
    si_data = inout.fastload_silica_sedimentation_data(size=256, recenter=False)
    ps_data = inout.fastload_polystyrene_sedimentation_data(size=256, recenter=False)

    si_fits_mo, si_fits_ml = inout.load_silica_sedimentation_params()
    ps_fits_mo, ps_fits_ml = inout.load_polystyrene_sedimentation_params()

    figure_si, fig_si = make_si_figure(si_data, si_fits_mo, si_fits_ml)
    figure_ps, fig_ps = make_ps_figure(ps_data, ps_fits_mo, ps_fits_ml)

    plt.show()
    #fig_si.savefig('./silica-sedimentation.svg')
    #fig_ps.savefig('./polystyrene-sedimentation.svg')
