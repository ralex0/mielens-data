import os

import numpy as np

import matplotlib as mpl; mpl.rcdefaults()
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt; plt.close('all')

import seaborn as sns

from pandas import DataFrame

import inout


class MCMCJointPlotFigure(object):
    def __init__(self, result, burnin=0, joint_kdes=None):
        """
        Parameters
        ----------
        result : lmfit.minimizer.MinimizerResult
        burning : int
            Can also be negative. e.g. burnin=100 throws out the first 100
            steps, burnin=-100 takes only the last 100 steps.
        """
        self.result = result
        self.burnin = burnin

        self.flatchain = self._create_flat_chain()

    def _create_flat_chain(self):
        flattened = self._burn_in()
        return DataFrame(flattened, columns=self.result.var_names)

    def _burn_in(self):
        if self.result.chain.ndim == 4:  # parallel tempering
            samples = self.result.chain[0]
        elif self.result.chain.ndim == 3:  # normal mcmc
            samples = self.result.chain
        else:
            raise ValueError("chain dimensions not understood")
        flattened = samples[:, self.burnin:, :].transpose((1, 0, 2)).reshape(
            (-1, self.result.nvarys))
        return flattened

    def plot(self, var0, var1, var2, labels=None):
        self.fig = plt.figure(figsize=(5.25, 1.75))
        if labels is None:
            labels = [var0, var1, var2]
        self._setup_axes()
        self._set_axes_style()
        self._plot_joints(var0, var1, var2)  # this is the slow part
        self._plot_marginals(var0, var1, var2)
        self._setlabels(labels)
        return self.fig

    def _setup_axes(self):
        gs = gridspec.GridSpec(1, 3)
        gs01 = gridspec.GridSpecFromSubplotSpec(
            6, 6, subplot_spec=gs[0], hspace=0, wspace=0)
        gs02 = gridspec.GridSpecFromSubplotSpec(
            6, 6, subplot_spec=gs[1], hspace=0, wspace=0)
        gs12 = gridspec.GridSpecFromSubplotSpec(
            6, 6, subplot_spec=gs[2], hspace=0, wspace=0)

        self._ax01_joint = plt.Subplot(
            self.fig, gs01[1:, :-1])
        self._ax01_margx = plt.Subplot(
            self.fig, gs01[0, :-1], sharex=self._ax01_joint)
        self._ax01_margy = plt.Subplot(
            self.fig, gs01[1:, -1], sharey=self._ax01_joint)

        self._ax02_joint = plt.Subplot(self.fig, gs02[1:, :-1])
        self._ax02_margx = plt.Subplot(self.fig, gs02[0, :-1],
                                       sharex=self._ax02_joint)
        self._ax02_margy = plt.Subplot(self.fig, gs02[1:, -1],
                                       sharey=self._ax02_joint)

        self._ax12_joint = plt.Subplot(self.fig, gs12[1:, :-1])
        self._ax12_margx = plt.Subplot(self.fig, gs12[0, :-1],
                                       sharex=self._ax12_joint)
        self._ax12_margy = plt.Subplot(self.fig, gs12[1:, -1],
                                       sharey=self._ax12_joint)

        all_axes = [self._ax01_joint, self._ax01_margx, self._ax01_margy,
                    self._ax02_joint, self._ax02_margx, self._ax02_margy,
                    self._ax12_joint, self._ax12_margx, self._ax12_margy]

        for ax in all_axes:
            self.fig.add_subplot(ax)

    def _plot_joints(self, var0, var1, var2):
        data = [np.array(self.flatchain[v].values) for v in [var0, var1, var2]]
        data_paired = [
            (data[0], data[1]),
            (data[0], data[2]),
            (data[1], data[2]),
            ]

        joint_axes = [self._ax01_joint, self._ax02_joint, self._ax12_joint]
        for ax, data_pair in zip(joint_axes, data_paired):
            plt.sca(ax)
            sns.kdeplot(
                data_pair[0], data_pair[1], shade=True, shade_lowest=False,
                gridsize=50)
            self._set_joint_axes_style(ax, *data_pair)

    def _set_joint_axes_style(self, ax, x, y):
        x_bw, y_bw = np.array((x.std(), y.std())) * 2.5
        xmin = max(np.median(x) - x_bw, x.min())
        xmax = min(np.median(x) + x_bw, x.max())
        ymin = max(np.median(y) - y_bw, y.min())
        ymax = min(np.median(y) + y_bw, y.max())

        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)

        xticks = np.linspace(xmin, xmax, 5)
        yticks = np.linspace(ymin, ymax, 5)

        xticklabels = ["{:.2f}".format(tick) for tick in xticks]
        yticklabels = ["{:.2f}".format(tick) for tick in yticks]

        ax.set_xticks(xticks)
        ax.set_yticks(yticks)
        ax.set_xticklabels(xticklabels, {'family': 'serif', 'size': 6})
        ax.set_yticklabels(yticklabels, {'family': 'serif', 'size': 6})

        ax.margins(x=0, y=0)

    def _plot_marginals(self, var0, var1, var2):
        margx_axes = [self._ax01_margx, self._ax02_margx, self._ax12_margx]
        margy_axes = [self._ax01_margy, self._ax02_margy, self._ax12_margy]

        datax = [np.array(self.flatchain[v]) for v in [var0, var0, var1]]
        datay = [np.array(self.flatchain[v]) for v in [var1, var2, var2]]

        for ax, data in zip(margx_axes, datax):
            plt.sca(ax)
            sns.kdeplot(data, shade=True,)

        for ax, data in zip(margy_axes, datay):
            plt.sca(ax)
            sns.kdeplot(data, shade=True, vertical=True)

    def _setlabels(self, labels):
        font = {'size': 8,
                'family': 'serif'}
        self._ax01_joint.set_xlabel(labels[0], font)
        self._ax01_joint.set_ylabel(labels[1], font)
        self._ax02_joint.set_xlabel(labels[0], font)
        self._ax02_joint.set_ylabel(labels[2], font)
        self._ax12_joint.set_xlabel(labels[1], font)
        self._ax12_joint.set_ylabel(labels[2], font)

    def _set_axes_style(self):
        self._set_marginal_axes_style()
        self.fig.tight_layout()

    def _set_marginal_axes_style(self):
        margx_axes = [self._ax01_margx, self._ax02_margx, self._ax12_margx]
        margy_axes = [self._ax01_margy, self._ax02_margy, self._ax12_margy]

        for ax in margx_axes:
            plt.setp(ax.get_xticklabels(), visible=False)
            plt.setp(ax.yaxis.get_majorticklines(), visible=False)
            plt.setp(ax.yaxis.get_minorticklines(), visible=False)
            plt.setp(ax.get_yticklabels(), visible=False)
            ax.yaxis.grid(False)

        for ax in margy_axes:
            plt.setp(ax.get_yticklabels(), visible=False)
            plt.setp(ax.xaxis.get_majorticklines(), visible=False)
            plt.setp(ax.xaxis.get_minorticklines(), visible=False)
            plt.setp(ax.get_xticklabels(), visible=False)
            ax.xaxis.grid(False)

        sns.utils.despine(self.fig)

        for ax in margx_axes:
            sns.utils.despine(ax=ax, left=True)
        for ax in margy_axes:
            sns.utils.despine(ax=ax, bottom=True)


def plot_mcmc_samples(mcmc_result):
    fig = plt.figure()
    for i, var in enumerate(mcmc_result.var_names):
        plt.subplot(len(mcmc_result.var_names), 1, i+1)
        for chain in mcmc_result.chain.T[i].T:
            plt.plot(chain)
        plt.ylabel(var)
    return fig



if __name__ == '__main__':
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


