import json
from collections import OrderedDict

import numpy as np
from holopy.scattering import calc_holo
from holopy.scattering.theory import MieLens
import holopy.scattering.errors

import mielensfit as mlf
from fit_data import load_few_PS_data_Jan10

GOOD_FITS = json.load(
    open('./good-fit-parameters.json'), object_pairs_hook=OrderedDict)

# Bonus: Since it's saved and loaded with an ordered dict, they are in order:
_all_radii = np.array([v['r'] for v in GOOD_FITS.values()])
_all_indices = np.array([v['n'] for v in GOOD_FITS.values()])
_all_lens_angles = np.array([v['lens_angle'] for v in GOOD_FITS.values()])

_all_zs = np.array([v['center.2'] for v in GOOD_FITS.values()])
t = np.arange(_all_zs.size)
perfect_spacing_fit = np.polyfit(t, _all_zs, 1)
perfect_spacing_zs = np.polyval(perfect_spacing_fit, t)
_z_deviates = _all_zs - perfect_spacing_zs


def randomly_refit(data, current_best_params, n_attempts=12):
    refitter = RandomRefitter(data, current_best_params, n_attempts=n_attempts)
    return refitter.refit()


def parse_result(parameters):
    holopy_keys = [
        'n', 'r', 'lens_angle', 'center.0', 'center.1', 'center.2']
    my_keys = ['n', 'r', 'lens_angle', 'x', 'y', 'z']
    parsed = {mk: parameters[hk] for hk, mk in zip(holopy_keys, my_keys)}
    # Then we need to clip values, since the fucking fitter doesn't
    parsed['n'] = np.clip(
        parsed['n'], mlf.Fitter._min_index, mlf.Fitter._max_index)
    parsed['r'] = np.clip(
        parsed['r'], mlf.Fitter._min_radius, mlf.Fitter._max_radius)
    return parsed


class RandomRefitter(object):
    _radius_std = 2 * _all_radii.std()
    _index_std = 2 * _all_indices.std()
    _z_std = 2 * _z_deviates.std()
    _lens_angle_std = 2 * _all_lens_angles.std()
    _xy_std = 0.5  # shouldn't really matter

    _log = []

    def __init__(self, data, current_best_params, n_attempts=12):
        self.data = data
        self.current_best_params = current_best_params
        self.n_attempts = n_attempts

    def refit(self):
        new_params = [self._refit_once() for _ in range(self.n_attempts)]
        # We remove the None's:
        new_params = [p for p in new_params if p is not None]
        all_params = [self.current_best_params] + new_params
        all_chisqs = [self.evaluate_chisq(p) for p in all_params]
        self._log.append(all_chisqs)
        best_index = np.argmin(all_chisqs)
        return all_params[best_index]

    def evaluate_residuals(self, params):
        """params is a dict-like"""
        fitter = mlf.Fitter(self.data, params)
        scatterer = fitter.make_guessed_scatterer()
        theory = MieLens(lens_angle=params['lens_angle'])
        model = calc_holo(self.data, scatterer, theory=theory)
        return self.data.values.squeeze() - model.values.squeeze()

    def evaluate_chisq(self, params):
        residuals = self.evaluate_residuals(params)
        return np.sum(residuals**2)

    def _refit_once(self):
        random_initial_guess = self._randomize_parameters()
        fitter = mlf.Fitter(self.data, random_initial_guess)
        try:
            result = fitter.fit()
        except:
            result = None
        parameters = (parse_result(result.parameters)
                      if result is not None else None)
        return parameters

    def _randomize_parameters(self):
        old = self.current_best_params
        # We clip some of the parameters so things don't crash
        # -- the clipvalues depend on stuff in mielensfit.
        new_params = {
            'r': np.clip(
                old['r'] + np.random.randn() * self._radius_std, 0.2, 4.0),
            'n': np.clip(
                old['n'] + np.random.randn() * self._index_std, 1.331, 2.31),
            'lens_angle': np.clip(
                old['lens_angle'] + np.random.randn() * self._lens_angle_std,
                0.15, 1.0),
            'z': old['z'] + np.random.randn() * self._z_std,
            'x': old['x'] + np.random.randn() * self._xy_std,
            'y': old['y'] + np.random.randn() * self._xy_std,
            }
        return new_params


if __name__ == '__main__':
    all_data, _ = load_few_PS_data_Jan10()
    np.random.seed(213)

    best_fits = []
    for i, data in enumerate(all_data):
        this_best = randomly_refit(data, parse_result(GOOD_FITS[str(i)]))
        best_fits.append(this_best)
    fits_dict = OrderedDict()
    for key, value in enumerate(best_fits):
        fits_dict.update({str(key): value})
    json.dump(fits_dict, open("./polished-fits.json", "w"), indent=4)

