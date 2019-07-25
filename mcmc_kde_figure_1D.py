import numpy as np

from scipy.stats import gaussian_kde

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from pandas import DataFrame

import inout
import monkeyrc
from monkeyrc import COLORS

LABEL_FONT = {'size': 6, 'family': 'Times New Roman'}
TICK_FONT = {'family': 'Times New Roman', 'size': 6}

class MCMCKDEFigure:
    def __init__(self, result_mo, result_ml, burnin=0):
        self._chain_mo = self._burn_in(result_mo.chain, burnin)
        self._chain_ml = self._burn_in(result_ml.chain, burnin)
        self.data_mo = DataFrame(self._chain_mo.reshape(-1, result_mo.nvarys),
                                 columns=result_mo.var_names)
        self.data_ml = DataFrame(self._chain_ml.reshape(-1, result_ml.nvarys),
                                 columns=result_ml.var_names)
        self.fig = plt.figure(figsize=(5.25, 1.25))
        self._setup_axes()

    def _burn_in(self, chain, n):
        if len(chain.shape) == 3:
            return chain[:, n:, :]
        elif len(chain.shape) == 4:
            return chain[0, :, n:, :]

    def _setup_axes(self):
        gs = gridspec.GridSpec(1, 4)

        self._ax_radius = plt.Subplot(self.fig, gs[:, 0])
        self._ax_index = plt.Subplot(self.fig, gs[:, 1])
        self._ax_alpha = plt.Subplot(self.fig, gs[:, 2])
        self._ax_lens_angle = plt.Subplot(self.fig, gs[:, 3])

        self._all_axes = [self._ax_radius, self._ax_index,
                          self._ax_alpha, self._ax_lens_angle]

        for ax in self._all_axes:
            self.fig.add_subplot(ax)

    def plot(self):
        self._plot_kdes()
        self._set_axes_style()
        return self.fig

    def _plot_kdes(self):
        self._plot_kde(self._ax_radius, 'r', legend=True)
        self._plot_kde(self._ax_index, 'n')
        self._plot_kde(self._ax_alpha, 'alpha')
        self._plot_kde(self._ax_lens_angle, 'lens_angle')

    def _plot_kde(self, ax, param, legend=False):
        labels = {'r': 'radius (μm)', 'n': 'refractive index',
                  'alpha': 'alpha', 'lens_angle': 'angular aperture (rad)'}
        xlim = self._get_xlim(param)
        ticks, tick_labels = self._get_xticks(param)

        bw = self.data_ml[param].size**(-1./(1+4)) # Scott's bw
        x = np.linspace(*xlim, 200) # 200 is magic

        kde_mo = gaussian_kde(self.data_mo[param], bw_method=bw*2) if param != 'lens_angle' else None
        kde_ml = gaussian_kde(self.data_ml[param], bw_method=bw*2)

        label_mo = ' without lens' if legend else None
        label_ml = ' with lens' if legend else None

        if kde_mo: ax.plot(x, kde_mo(x), label=label_mo, c=COLORS['red'])
        ax.plot(x, kde_ml(x), label=label_ml, c=COLORS['blue'])
        ax.spines['left'].set_visible(False)
        if param != 'lens_angle':
            ax.axvline(self.data_mo[param].mean(), ls=':', c=COLORS['red'])
        ax.axvline(self.data_ml[param].mean(), ls=':', c=COLORS['blue'])

        ax.set_xlim(xlim)
        ax.set_xticks(ticks)
        ax.set_xticklabels(tick_labels, **TICK_FONT)
        ax.set_xlabel(labels[param], **LABEL_FONT)
        ax.set_yticks([])
        ax.set_yticklabels([])
        ax.set_ylabel('')

    def _get_xlim(self, param):
        if param == 'lens_angle':
            low, high = self._get_lims(self.data_ml[param])
        else:
            low = np.min([self._get_lims(self.data_mo[param]),
                          self._get_lims(self.data_ml[param])])
            high = np.max([self._get_lims(self.data_mo[param]),
                           self._get_lims(self.data_ml[param])])
        return low, high

    def _set_axes_style(self):
        plt.sca(self._ax_radius)
        plt.legend(loc=(.05, .95), fontsize='x-small', numpoints=7)
        self.fig.tight_layout()

    def _get_lims(self, data):
        low, high = np.quantile(data, [.001, .999])
        return np.array([low, high])

    def _get_xticks(self, param):
        if param == 'r':
            ticks = [1.16, 1.165, 1.17]
        elif param == 'n':
            ticks = [1.598, 1.6, 1.602]
        elif param == 'alpha':
            ticks = [.665, .675, .685]
        elif param == 'lens_angle':
            ticks = [.83, .832, .834]
        labels = ["{:.3f}".format(tick) for tick in ticks]
        return ticks, labels

if __name__ == '__main__':
    result_ml = inout.load_mcmc_result_PS_mielens()
    result_mo = inout.load_mcmc_result_PS_mieonly()

    plotter = MCMCKDEFigure(result_mo, result_ml, burnin=512)
    fig = plotter.plot()
    plt.show()
