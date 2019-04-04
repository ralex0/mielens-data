import numpy as np

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt

import seaborn as sns

from pandas import DataFrame

import inout

class MCMCJointPlotFigure(object):
    def __init__(self, result, burnin=0, jointfunc=sns.kdeplot, margfunc=sns.kdeplot):
        self.chain = self._burn_in(result.chain, burnin)
        self.flatchain = DataFrame(self.chain.reshape(-1, result.nvarys), 
                                   columns=result.var_names)
        self.jointfunc = jointfunc
        self.margfunc = margfunc

    def _burn_in(self, chain, n):
        return chain[:, n:, :]

    def plot(self, var0, var1, var2, labels=None):
        self.fig = plt.figure(figsize=(5.25, 1.75))
        if labels is None:
            labels = [var0, var1, var2]
        self._setup_axes()
        self._plot_joints(var0, var1, var2)
        self._plot_marginals(var0, var1, var2)
        self._setlabels(labels)
        self._set_axes_style()
        return self.fig

    def _setup_axes(self):
        gs = gridspec.GridSpec(1, 3)

        gs01 = gridspec.GridSpecFromSubplotSpec(6, 6, subplot_spec=gs[0], hspace=0, wspace=0)
        gs02 = gridspec.GridSpecFromSubplotSpec(6, 6, subplot_spec=gs[1], hspace=0, wspace=0)
        gs12 = gridspec.GridSpecFromSubplotSpec(6, 6, subplot_spec=gs[2], hspace=0, wspace=0)

        self._ax01_joint = plt.Subplot(self.fig, gs01[1:, :-1])
        self._ax01_margx = plt.Subplot(self.fig, gs01[0, :-1], sharex=self._ax01_joint)
        self._ax01_margy = plt.Subplot(self.fig, gs01[1:, -1], sharey=self._ax01_joint)

        self._ax02_joint = plt.Subplot(self.fig, gs02[1:, :-1])
        self._ax02_margx = plt.Subplot(self.fig, gs02[0, :-1], sharex=self._ax02_joint)
        self._ax02_margy = plt.Subplot(self.fig, gs02[1:, -1], sharey=self._ax02_joint)

        self._ax12_joint = plt.Subplot(self.fig, gs12[1:, :-1])
        self._ax12_margx = plt.Subplot(self.fig, gs12[0, :-1], sharex=self._ax12_joint)
        self._ax12_margy = plt.Subplot(self.fig, gs12[1:, -1], sharey=self._ax12_joint)

        all_axes = [self._ax01_joint, self._ax01_margx, self._ax01_margy, 
                    self._ax02_joint, self._ax02_margx, self._ax02_margy, 
                    self._ax12_joint, self._ax12_margx, self._ax12_margy]

        for ax in all_axes:
            self.fig.add_subplot(ax)


    def _plot_joints(self, var0, var1, var2):
        data = [np.array(self.flatchain[v]) for v in [var0, var1, var2]]
        data_paired = [(data[0], data[1]), (data[0], data[2]), (data[1], data[2])]

        joint_axes = [self._ax01_joint, self._ax02_joint, self._ax12_joint]
        for ax, data_pair in zip(joint_axes, data_paired):
            self.jointfunc(data_pair[0], data_pair[1], ax=ax)

    def _plot_marginals(self, var0, var1, var2):
        margx_axes = [self._ax01_margx, self._ax02_margx, self._ax12_margx]
        margy_axes = [self._ax01_margy, self._ax02_margy, self._ax12_margy]

        datax = [np.array(self.flatchain[v]) for v in [var0, var0, var1]]
        datay = [np.array(self.flatchain[v]) for v in [var1, var2, var2]]

        for ax, data in zip(margx_axes, datax):
            self.margfunc(data, ax=ax)

        for ax, data in zip(margy_axes, datay):
            self.margfunc(data, ax=ax, vertical=True)

    def _setlabels(self, labels):
        self._ax01_joint.set_xlabel(labels[0])
        self._ax01_joint.set_ylabel(labels[1])
        self._ax02_joint.set_xlabel(labels[0])
        self._ax02_joint.set_ylabel(labels[2])
        self._ax12_joint.set_xlabel(labels[1])
        self._ax12_joint.set_ylabel(labels[2])
        
    def _set_axes_style(self):
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

        self.fig.tight_layout()


def plot_mcmc_samples(mcmc_result):
    fig = plt.figure()
    for i, var in enumerate(mcmc_result.var_names):
        plt.subplot(len(mcmc_result.var_names), 1, i+1)
        for chain in mcmc_result.chain.T[i].T:
            plt.plot(chain)
        plt.ylabel(var)
    return fig


if __name__ == '__main__':
    data = inout.load_mcmc_result_PS_mieonly()
    plotter = MCMCJointPlotFigure(data, burnin=200, jointfunc=sns.scatterplot)
    figure = plotter.plot('n', 'r', 'alpha')
    plt.show()