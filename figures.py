import json
from collections import OrderedDict

import numpy as np
import matplotlib.pyplot as plt

import mielensfit as mlf
from fit_data import load_few_PS_data_Jan10, compare_imgs



# FIXME this does not do interpolation yet!
# Ron did it by doing
# z_positions=np.linspace(26.5, -23.5, 51))
# I think it needs to be interpolated though b/c the z positions need to
# be ~10x finer than this.
class XZFigure(object):
    def __init__(self, fit_parameters):
        self.fit_parameters = fit_parameters
        self._load_data()
        self.models = self.create_model_images()

    def _load_data(self):
        holos, zpos = load_few_PS_data_Jan10()
        data = [h.values.squeeze() for h in holos]
        self.holos = holos
        self.data = np.array(data)

    def create_model_images(self):
        models = []
        for holo, params in zip(self.holos, self.fit_parameters):
            fitter = mlf.Fitter(holo, params)
            models.append(fitter.evaluate_model(params).values.squeeze())
        return np.array(models)

    def grab_xz_slice(self, array):
        return array[:, 50]

    def make_plot(self):
        data = self.grab_xz_slice(self.data)
        model = self.grab_xz_slice(self.models)
        vmax = max(data.max(), model.max())
        vmin = 0
        fig = plt.figure(figsize=[6, 4])

        ax1 = fig.add_subplot(1, 2, 1)
        ax2 = fig.add_subplot(1, 2, 2)
        for ax, im, title in zip([ax1, ax2], [data, model], ['Data', 'Model']):
            ax.imshow(im, interpolation='nearest', cmap='gray', vmin=vmin,
                      vmax=vmax)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_title(title)
        return fig, [ax1, ax2]

    def get_z_spacing(self):
        zs = [f['z'] for f in self.fit_parameters]
        return np.abs(np.diff(zs).mean())


if __name__ == '__main__':
    fits = json.load(open('./finalized-fits.json'),
                     object_pairs_hook=OrderedDict)
    fits_list = [v for v in fits.values()]
    fig = XZFigure(fits_list)
    fig.make_plot()
    plt.show()
