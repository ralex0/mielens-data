import os

import numpy as np
from scipy.stats import gaussian_kde

import matplotlib as mpl; mpl.rcdefaults()
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt; plt.close('all')

import seaborn as sns
from seaborn import cubehelix_palette

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


class MCMCJointPlotFigure_v2(object):
    def __init__(self, result_mo, result_ml, burnin=0):
        self._chain_mo = self._burn_in(result_mo.chain, burnin)
        self._chain_ml = self._burn_in(result_ml.chain, burnin)
        self.data_mo = DataFrame(self._chain_mo.reshape(-1, result_mo.nvarys), 
                                 columns=result_mo.var_names)
        self.data_ml = DataFrame(self._chain_ml.reshape(-1, result_ml.nvarys), 
                                 columns=result_ml.var_names)
        self.fig = plt.figure(figsize=(5.25, 2.5))
        self._setup_axes()

    def _burn_in(self, chain, n):
        return chain[:, n:, :]

    def _setup_axes(self):
        gs = gridspec.GridSpec(1, 2)
        self._setup_axes_mieonly(gs[0])
        self._setup_axes_mielens(gs[1])
        all_axes = [self._axMO_nr, self._axMO_na, self._axMO_ar, 
                    self._axMO_n, self._axMO_r, self._axMO_a,
                    self._axML_nr, self._axML_nl, self._axML_lr, 
                    self._axML_n, self._axML_r, self._axML_l]
        for ax in all_axes:
            self.fig.add_subplot(ax)

    def _setup_axes_mieonly(self, subplotspec):
        (self._axMO_nr, self._axMO_na, 
         self._axMO_ar, self._axMO_n, 
         self._axMO_r, self._axMO_a) = self._setup_jointplot_axes(subplotspec)

    def _setup_axes_mielens(self, subplotspec):
        (self._axML_nr, self._axML_nl,
         self._axML_lr, self._axML_n, 
         self._axML_r, self._axML_l) = self._setup_jointplot_axes(subplotspec)

    def _setup_jointplot_axes(self, subplotspec):
        gs = gridspec.GridSpecFromSubplotSpec(11, 11, subplot_spec=subplotspec,
                                              hspace=0, wspace=0)
        # Joint marginal axes
        ax01 = plt.Subplot(self.fig, gs[1:6, :5])
        ax02 = plt.Subplot(self.fig, gs[6:, :5], sharex=ax01)
        ax21 = plt.Subplot(self.fig, gs[1:6, 5:10])
        # Fully marginalized axes
        ax0 = plt.Subplot(self.fig, gs[0, :5], sharex=ax01)
        ax1 = plt.Subplot(self.fig, gs[1:6, -1], sharey=ax01)
        ax2 = plt.Subplot(self.fig, gs[6:, 5], sharey=ax02)
        return ax01, ax02, ax21, ax0, ax1, ax2

    def plot(self):
        self._plot_joints()
        self._plot_marginals()
        self._set_axes_style()
        self._add_figlabels()
        return self.fig

    def _plot_joints(self):
        self._plot_joint(self._axMO_nr, self.data_mo['n'], self.data_mo['r'])
        self._plot_joint(self._axMO_na, self.data_mo['n'], self.data_mo['alpha'])
        self._plot_joint(self._axMO_ar, self.data_mo['alpha'], self.data_mo['r'])
        self._plot_joint(self._axML_nr, self.data_ml['n'], self.data_ml['r'])
        self._plot_joint(self._axML_nl, self.data_ml['n'], self.data_ml['lens_angle'])
        self._plot_joint(self._axML_lr, self.data_ml['lens_angle'], self.data_ml['r'])

    def _plot_joint(self, ax, datax, datay):
        plt.sca(ax)
        
        cmap = cubehelix_palette(n_colors=12, start=2.95, rot=-0.10, light=1.00, 
                                 dark=0.20, as_cmap=True)
        #sns.kdeplot(datax, datay, shade=True, cmap=cmap, n_levels=12, shade_lowest=False)
        plt.hexbin(datax, datay, cmap=cmap)

    def _plot_marginals(self):
        self._plot_marginal(self._axMO_n, self.data_mo['n'])
        self._plot_marginal(self._axMO_r, self.data_mo['r'], vertical=True)
        self._plot_marginal(self._axMO_a, self.data_mo['alpha'], vertical=True)
        self._plot_marginal(self._axML_n, self.data_ml['n'])
        self._plot_marginal(self._axML_r, self.data_ml['r'], vertical=True)
        self._plot_marginal(self._axML_l, self.data_ml['lens_angle'], vertical=True)

    def _plot_marginal(self, ax, data, vertical=False):
        plt.sca(ax)
        sns.kdeplot(np.array(data), shade=True, vertical=vertical, legend=False)

    def _set_axes_style(self):
        self._set_joint_axes_style()
        self._set_marginal_axes_style()
        self.fig.tight_layout()

    def _set_joint_axes_style(self):
        label_font = {'size': 6, 'family': 'Times New Roman'}
        tick_font = {'family': 'Times New Roman', 'size': 6}

        n_lim_mo = self._get_lims(self.data_mo['n'])
        r_lim_mo = self._get_lims(self.data_mo['r'])
        a_lim_mo = self._get_lims(self.data_mo['alpha'])

        n_lim_ml = self._get_lims(self.data_ml['n'])
        r_lim_ml = self._get_lims(self.data_ml['r'])
        l_lim_ml = self._get_lims(self.data_ml['lens_angle'])

        # Mieonly n/r joint
        plt.sca(self._axMO_nr)
        plt.xlim(n_lim_mo)
        plt.ylim(r_lim_mo)
        plt.yticks(*self._get_ticks_labels(r_lim_mo), **tick_font)
        plt.ylabel("Radius (μm)", **label_font)

        # Mieonly n/alpha joint
        self._axMO_na.set_ylim(a_lim_mo)
        ticks, labels = self._get_ticks_labels(n_lim_mo)
        self._axMO_na.set_xticks(ticks)
        self._axMO_na.set_xticklabels(labels, **tick_font, rotation='vertical')
        self._axMO_na.set_xlabel("Refractive index", **label_font)
        ticks, labels = self._get_ticks_labels(a_lim_mo)
        self._axMO_na.set_yticks(ticks)
        self._axMO_na.set_yticklabels(labels, **tick_font)
        self._axMO_na.set_ylabel("Alpha", **label_font)

        # Mieonly alpha/r joint
        self._axMO_ar.set_xlim(a_lim_mo)
        self._axMO_ar.set_ylim(r_lim_mo)
        ticks, labels = self._get_ticks_labels(a_lim_mo)
        self._axMO_ar.set_xticks(ticks)
        self._axMO_ar.set_xticklabels(labels, **tick_font, rotation='vertical')
        self._axMO_ar.set_xlabel("Alpha", **label_font)
        self._axMO_ar.set_yticks([])
        self._axMO_ar.set_yticklabels([])
        self._axMO_ar.set_ylabel('')

        # Mielens n/r joint
        self._axML_nr.set_xlim(n_lim_ml)
        self._axML_nr.set_ylim(r_lim_ml)
        ticks, labels = self._get_ticks_labels(r_lim_ml)
        self._axML_nr.set_yticks(ticks)
        self._axML_nr.set_yticklabels(labels, **tick_font)
        self._axML_nr.set_ylabel("Radius (μm)", **label_font)

        # Mielens n/lens_angle joint
        self._axML_nl.set_ylim(l_lim_ml)
        ticks, labels = self._get_ticks_labels(n_lim_ml)
        self._axML_nl.set_xticks(ticks)
        self._axML_nl.set_xticklabels(labels, **tick_font, rotation='vertical')
        self._axML_nl.set_xlabel("Refractive index", **label_font)
        ticks, labels = self._get_ticks_labels(l_lim_ml)
        self._axML_nl.set_yticks(ticks)
        self._axML_nl.set_yticklabels(labels, **tick_font)
        self._axML_nl.set_ylabel("Acceptance\nangle (rad)", **label_font)

        # Mielens lens_angle/r joint
        self._axML_lr.set_xlim(l_lim_ml)
        self._axML_lr.set_ylim(r_lim_ml)
        ticks, labels = self._get_ticks_labels(l_lim_ml)
        self._axML_lr.set_xticks(ticks)
        self._axML_lr.set_xticklabels(labels, **tick_font, rotation='vertical')
        self._axML_lr.set_xlabel("Acceptance\nangle (rad)", **label_font)
        self._axML_lr.set_yticks([])
        self._axML_lr.set_yticklabels([])
        self._axML_lr.set_ylabel('')


    def _get_lims(self, data):
        median = np.median(data)
        std = np.std(data)
        high = min(median + std * 2.5, max(data))
        low = max(median - std * 2.5, min(data))
        return low, high

    def _get_ticks_labels(self, lims, numticks=3):
        ticks = np.linspace(*lims, numticks+2)[1:-1]
        labels = ["{:.2f}".format(tick) for tick in ticks]
        return ticks, labels

    def _set_marginal_axes_style(self):
        margx_axes = [self._axMO_n, self._axML_n]
        margy_axes0 = [self._axMO_r, self._axML_r]
        margy_axes1 = [self._axMO_a, self._axML_l]
        for ax in margx_axes + margy_axes0 + margy_axes1:
            plt.setp(ax.get_yticklabels(), visible=False)
            plt.setp(ax.get_xticklabels(), visible=False)
            plt.setp(ax.xaxis.get_majorticklines(), visible=False)
            plt.setp(ax.xaxis.get_minorticklines(), visible=False)
            plt.setp(ax.yaxis.get_majorticklines(), visible=False)
            plt.setp(ax.yaxis.get_minorticklines(), visible=False)
            #ax.xaxis.grid(False)
        for ax in margx_axes:
            sns.utils.despine(ax=ax, left=True)
        for ax in margy_axes0:
            sns.utils.despine(ax=ax, bottom=True)
        for ax in margy_axes1:
            sns.utils.despine(ax=ax, top=False, bottom=True)
            ax.set_xticks([])
            ax.set_xticklabels([])

    def _add_figlabels(self):
        figlabel_font = {'family': 'Times New Roman', 'size': 9}

        self._ax_labela = self.fig.add_axes([.05, .95, .05, .05])
        plt.sca(self._ax_labela)
        plt.text(0, 0, "a)", **figlabel_font)
        plt.xticks([])
        plt.yticks([])

        self._ax_labelb = self.fig.add_axes([.55, .95, .05, .05])
        plt.sca(self._ax_labelb)
        plt.text(0, 0, "b)", **figlabel_font)
        plt.xticks([])
        plt.yticks([])

        sns.utils.despine(ax=self._ax_labela, bottom=True, left=True)
        sns.utils.despine(ax=self._ax_labelb, bottom=True, left=True)


def plot_mcmc_samples(mcmc_result):
    fig = plt.figure()
    for i, var in enumerate(mcmc_result.var_names):
        plt.subplot(len(mcmc_result.var_names), 1, i+1)
        for chain in mcmc_result.chain.T[i].T:
            plt.plot(chain)
        plt.ylabel(var)
    return fig


if __name__ == '__main__':
    result_ml = inout.load_mcmc_result_Si_mielens()
    result_mo = inout.load_mcmc_result_Si_mieonly()
    plotter = MCMCJointPlotFigure_v2(result_mo, result_ml, burnin=200)
    #labels = ["Refractive index", "Radius (μm)", "Acceptance angle (rad)"]
    fig = plotter.plot()
    plt.show()

