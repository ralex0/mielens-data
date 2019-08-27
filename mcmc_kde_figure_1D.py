import numpy as np

from scipy.stats import gaussian_kde

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from pandas import DataFrame

import holopy as hp

import inout
import monkeyrc
from monkeyrc import COLORS

LABEL_FONT = {'size': 6, 'family': 'Times New Roman'}
TICK_FONT = {'family': 'Times New Roman', 'size': 6}
FIGLABEL_FONT = {'family': 'Times New Roman', 'size': 9}

class MCMCKDEFigure:
    def __init__(self, result_mo, result_ml, holos, burnin=0):
        self._chain_mo = [self._burn_in(result.chain, burnin) for result in result_mo]
        self._chain_ml = [self._burn_in(result.chain, burnin) for result in result_ml]
        self.data_mo = [DataFrame(chain.reshape(-1, result.nvarys),
                                  columns=result.var_names)
                        for chain, result in zip(self._chain_mo, result_mo)]
        self.data_ml = [DataFrame(chain.reshape(-1, result.nvarys),
                                  columns=result.var_names)
                        for chain, result in zip(self._chain_ml, result_ml)]
        self.holos = holos
        self.fig = plt.figure(figsize=(5.25, 2.5))
        self._setup_axes()

    def _burn_in(self, chain, n):
        if len(chain.shape) == 3:
            return chain[:, n:, :]
        elif len(chain.shape) == 4:
            return chain[0, :, n:, :]

    def _setup_axes(self):
        gs = gridspec.GridSpec(2, 5)
        self._ax_holo = [plt.Subplot(self.fig, gs[i, 0]) for i in range(2)]
        self._ax_z = [plt.Subplot(self.fig, gs[i, 1]) for i in range(2)]
        self._ax_radius = [plt.Subplot(self.fig, gs[i, 2]) for i in range(2)]
        self._ax_index = [plt.Subplot(self.fig, gs[i, 3]) for i in range(2)]
        self._ax_alpha = [plt.Subplot(self.fig, gs[i, 4]) for i in range(2)]

        self._all_axes = [*self._ax_holo, *self._ax_radius, *self._ax_index,
                          *self._ax_alpha, *self._ax_z]

        for ax in self._all_axes:
            self.fig.add_subplot(ax)

    def plot(self):
        self._plot_holos()
        self._plot_kdes()
        self._set_axes_style()
        self._add_figlabels()
        return self.fig

    def _plot_holos(self):
        vmin = np.min(holos)
        vmax = np.max(holos)
        for i in range(2):
            ax = self._ax_holo[i]
            ax.imshow(self.holos[i], interpolation='nearest',
                      vmin=vmin, vmax=vmax, cmap='gray')
            ax.set_xticks([])
            ax.set_yticks([])
            ax.spines['left'].set_visible(False)
            ax.spines['bottom'].set_visible(False)


    def _plot_kdes(self):
        for i in range(2):
            self._plot_kde(self._ax_radius[i], 'r', i)
            self._plot_kde(self._ax_index[i], 'n', i, legend=True)
            self._plot_kde(self._ax_alpha[i], 'alpha', i)
            self._plot_kde(self._ax_z[i], 'z', i)

    def _plot_kde(self, ax, param, data_index, legend=False):
        labels = {'r': 'radius (μm)', 'n': 'refractive index',
                  'alpha': 'alpha', 'z': 'z position (μm)'}
        xlim = self._get_xlim(param, data_index)
        ticks, tick_labels = self._get_xticks(param, data_index)

        bw = self.data_ml[data_index][param].size**(-1./(1+4)) # Scott's bw
        x = np.linspace(*xlim, 200) # 200 is magic

        kde_mo = gaussian_kde(self.data_mo[data_index][param], bw_method=bw*2)
        kde_ml = gaussian_kde(self.data_ml[data_index][param], bw_method=bw*2)

        label_mo = ' without lens' if legend else None
        label_ml = ' with lens' if legend else None

        ax.plot(x, kde_mo(x), label=label_mo, c=COLORS['red'], lw=1)
        ax.plot(x, kde_ml(x), label=label_ml, c=COLORS['blue'], lw=1)
        ax.spines['left'].set_visible(False)
        ax.axvline(self.data_mo[data_index][param].mean(), lw=.5, c=COLORS['red'])
        ax.axvline(self.data_ml[data_index][param].mean(), lw=.5, c=COLORS['blue'])

        ax.set_xlim(xlim)
        ax.set_xticks(ticks)
        ax.set_xticklabels(tick_labels, **TICK_FONT)
        if data_index == 1: ax.set_xlabel(labels[param], **LABEL_FONT)
        ax.set_yticks([])
        ax.set_yticklabels([])
        ax.set_ylabel('')

    def _get_xlim(self, param, data_index):
        lims = {'r': [[1.12, 1.2], [1.12, 1.2]],
                 'n': [[1.45, 1.61], [1.45, 1.61]],
                 'z': [[15.99, 16.21], [3.54, 3.76]],
                 'alpha': [[.64, .86], [.64, .86]]}
        return lims[param][data_index]

    def _set_axes_style(self):
        plt.sca(self._ax_index[0])
        plt.legend(loc=(.05, .65), fontsize='x-small', numpoints=7)
        self.fig.tight_layout(w_pad=0.001, h_pad=0.05)

    def _get_lims(self, data):
        low, high = np.quantile(data, [.001, .999])
        return np.array([low, high])

    def _get_xticks(self, param, data_index):
        ticks = {'r': [[1.13, 1.16, 1.19], [1.13, 1.16, 1.19]],
                 'n': [[1.46, 1.53, 1.60], [1.46, 1.53, 1.60]],
                 'z': [[16.0, 16.1, 16.2], [3.55, 3.65, 3.75]],
                 'alpha': [[.65, .75, .85], [.65, .75, .85]]}
        labels = ["{:.2f}".format(tick) for tick in ticks[param][data_index]]
        return ticks[param][data_index], labels

    def _add_figlabels(self):
        self._ax_labela = self.fig.add_axes([0.01, .9, .01, .01])
        plt.sca(self._ax_labela)
        plt.text(0, 0, "a)", **FIGLABEL_FONT)
        plt.xticks([])
        plt.yticks([])

        self._ax_labelb = self.fig.add_axes([0.01, .47, .01, .01])
        plt.sca(self._ax_labelb)
        plt.text(0, 0, "b)", **FIGLABEL_FONT)
        plt.xticks([])
        plt.yticks([])

        for dir in ['left', 'right', 'top', 'bottom']:
            self._ax_labela.spines[dir].set_visible(False)
            self._ax_labelb.spines[dir].set_visible(False)

def _load_holos():
    folder = 'data/Polystyrene2-4um-60xWater-042919/processed-128-thin/'
    file0 = 'im0001.tif'
    file1 = 'im0036.tif'
    holo0 = hp.load(folder + file0)
    holo1 = hp.load(folder + file1)
    return holo0.values.squeeze(), holo1.values.squeeze()

if __name__ == '__main__':
    holonums = [1, 36]
    holos = _load_holos()
    result_ml = [inout.load_mcmc_result_PS_mielens(frame) for frame in holonums]
    result_mo = [inout.load_mcmc_result_PS_mieonly(frame) for frame in holonums]

    plotter = MCMCKDEFigure(result_mo, result_ml, holos, burnin=512)
    fig = plotter.plot()
    plt.show()
