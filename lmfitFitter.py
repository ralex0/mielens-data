import warnings

import numpy as np

from scipy.ndimage.filters import gaussian_filter

from holopy.core.metadata import get_extents, get_spacing, make_subset_data
from holopy.core.metadata import update_metadata
from holopy.core.process import center_find
from holopy.scattering import Sphere, calc_holo
from holopy.scattering.theory import MieLens, Mie

from lmfit import Minimizer, Parameter, Parameters, report_fit


class ResidualsCalculator(object):
    """Stores the best-fit as self.best_fit_params, best_fit_chisq
    """

    def __init__(self, data, theory='mielens', noise=1.0):
        self.data = data
        self.theory = theory
        self.best_params = []
        self.best_chisq = np.inf

    def calc_model(self, params, metadata):
        sphere = self.create_sphere_from(params)  # 0.036 ms
        theory, scaling = self._get_theory_and_scaling(params)
        if 'illum_wavelen' in params:
            wavelength = float(params['illum_wavelen'])
            metadata = update_metadata(metadata, illum_wavelen=wavelength)
        return calc_holo(metadata, sphere, theory=theory, scaling=scaling)

    def create_sphere_from(self, params):
        params = {name: param for name, param in params.items()}
        sphere = Sphere(n=params['n'], r=params['r'],
                        center=(params['x'], params['y'], params['z']))
        return sphere

    def calc_residuals(self, params, *, data=None, noise=1.0):  # 170 ms
        model = self.calc_model(params, data)  # 134 ms
        residuals = (model - data).values  # 2.8 ms
        chisq = np.linalg.norm(residuals)**2  # 0.5 ms
        if chisq < self.best_chisq:
            self.best_chisq = chisq
            self.best_params = params
        return residuals / noise

    def _get_theory_and_scaling(self, params):
        if self.theory == 'mielens':
            theory = MieLens(lens_angle=params['lens_angle'])  # 0.0015 ms
            scaling = 1.0
        elif self.theory == 'mieonly':
            theory = Mie()
            scaling = params['alpha']
        elif self.theory == 'mielensalpha':
            theory = MieLens(lens_angle=params['lens_angle'])  # 0.0015 ms
            scaling = params['alpha']
        return theory, scaling


class Fitter(object):
    DEFAULT_MCMC_PARAMS = {}

    def __init__(self, theory="mielens", method="leastsq", quiet=False):
        self.theory = theory
        self.method = method
        self.quiet = quiet

    def fit(self, data, initial_guess):
        params = self._setup_params_from(initial_guess, data)
        cost_kwargs = {'data': data, 'noise': self._estimate_noise_from(data)}

        residuals_calculator = ResidualsCalculator(data, theory=self.theory)
        minimizer = Minimizer(
            residuals_calculator.calc_residuals,
            params,
            nan_policy='omit',
            fcn_kws=cost_kwargs)
        fit_result = minimizer.minimize(params=params, method=self.method)
        return fit_result

    # FIXME why is the param order backwards from fit?
    def mcmc(self, initial_guesses, data, mcmc_kws=None, npixels=100):
        if not self.quiet:
            print("Getting best fit with {}".format(self.method))
        best_fit = self.fit(data, initial_guesses)
        if not self.quiet:
            print(report_fit(best_fit))
        result = self._mcmc(
            best_fit, data, mcmc_kws=mcmc_kws, npixels=npixels)
        return result

    def _mcmc(self, best_fit, data, mcmc_kws=None, npixels=100):
        if mcmc_kws is None:
            mcmc_kws = self.DEFAULT_MCMC_PARAMS.copy()
        subset_data = make_subset_data(data, pixels=npixels)
        noise = self._estimate_noise_from(data)
        params = best_fit.params
        params.add(
            '__lnsigma', value=np.log(noise), min=np.log(noise / 10),
            max=np.log(noise * 10))

        residuals_calculator = ResidualsCalculator(
            subset_data, theory=self.theory)
        minimizer = Minimizer(
            residuals_calculator.calc_residuals,
            params,
            nan_policy='omit',
            fcn_kws={'data': subset_data})

        if not self.quiet:
            print("Sampling with emcee ({}, npixels: {})".format(
                mcmc_kws, npixels))
        self._update_mcmc_kwargs_with_pos(mcmc_kws, params)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            mcmc_result = minimizer.minimize(
                params=params, method='emcee', float_behavior='chi2',
                is_weighted=False, **mcmc_kws)
        if not self.quiet:
            print(report_fit(mcmc_result.params))
        # Then the whole point of the ResidualsCalculator is storing the
        # best-fit value:
        best_params = {'parameters': residuals_calculator.best_params,
                       'chisq': residuals_calculator.best_chisq}
        result = {
            'mcmc_result': mcmc_result,
            'lmfit_result': best_fit,
            'best_result': best_params}
        return result

    def _setup_minimizer(self, params, cost_kwargs=None):
        cost_function = self._setup_cost_function()
        return Minimizer(cost_function, params, nan_policy='omit',
                         fcn_kws=cost_kwargs)

    def _setup_params_from(self, initial_guess, data):  # 5 ms
        params = Parameters()
        x, y, z = self._make_center_priors(data, initial_guess)  # 4.47 ms
        min_index = data.medium_index * 1.001
        n = Parameter(
            name='n', value=initial_guess['n'], min=min_index, max=2.33)
        r = Parameter(
            name='r', value=initial_guess['r'], min=0.05, max=5)
        params.add_many(x, y, z, n, r)
        if self.theory == 'mieonly':
            alpha_val = self._alpha_guess(initial_guess)
            params.add(name='alpha', value=alpha_val, min=0.05, max=1.0)
        elif self.theory == 'mielens':
            angle_val = self._lens_guess(initial_guess)
            params.add(name='lens_angle', value=angle_val, min=0.05, max=1.1)
        if self.theory == 'mielensalpha':
            alpha_val = self._alpha_guess(initial_guess)
            angle_val = self._lens_guess(initial_guess)
            params.add(name='alpha', value=alpha_val, min=0.05, max=1.0)
            params.add(name='lens_angle', value=angle_val, min=0.05, max=1.1)

        if 'illum_wavelen' in initial_guess:
            wavelength = initial_guess['illum_wavelen']
            params.add(name='illum_wavelen', value=wavelength,
                       min=.1, max=2.000)
        return params

    def _make_center_priors(self, data, guess):
        image_x_values = data.x.values
        image_min_x = image_x_values.min()
        image_max_x = image_x_values.max()

        image_y_values = data.y.values
        image_min_y = image_y_values.min()
        image_max_y = image_y_values.max()

        if ('x' in guess) and ('y' in guess):
            x_guess = guess['x']
            y_guess = guess['y']
        elif ('center.0' in guess) and ('center.1' in guess):
            x_guess = guess['center.0']
            y_guess = guess['center.1']
        else:
            pixel_spacing = get_spacing(data)
            image_lower_left = np.array([image_min_x, image_min_y])
            x_guess, y_guess = (center_find(data) * pixel_spacing +
                                image_lower_left)

        extents = get_extents(data)
        # FIXME: 5 is a magic number.
        zextent = 5 * max(extents['x'], extents['y'])
        z_guess = guess['z'] if 'z' in guess else guess['center.2']

        x = Parameter(name='x', value=x_guess,
                      min=image_min_x, max=image_max_x)
        y = Parameter(name='y', value=y_guess,
                      min=image_min_y, max=image_max_y)
        z = Parameter(name='z', value=z_guess, min=-zextent, max=zextent)
        return x, y, z

    def _lens_guess(self, guess):
        lens_angle = (
            guess['lens_angle'] if 'lens_angle' in guess
            else np.arcsin(1.2 / 2))
        return lens_angle

    def _alpha_guess(self, guess):
        return guess['alpha'] if 'alpha' in guess else 0.8

    def _setup_cost_function(self):
        return self._calc_residuals

    def _calc_square_residuals_mieonly(self, params, *, data=None, noise=1.0):
        return self._calc_residuals(params, data=data, noise=noise) ** 2

    def _estimate_noise_from(self, data):
        if data.noise_sd is None:
            return estimate_noise_from(data)
        return float(data.noise_sd)

    def _update_mcmc_kwargs_with_pos(self, mcmc_kws, params):
        # The emcee code looks like it takes a starting guess (the pos kwarg)
        # as params * (1 + 1e-4 * randn(*params.shape)), which might be a
        # little small. We probably want something like delta_xyz ~ 0.03,
        # delta_n ~ 0.03, delta_r ~ 0.03, delta_alpha / lensangle ~ 0.1,
        # since those seem to be the scale of local minima...
        # from the nature of the affine invariant stuff, IMO it will be
        # easier to start broad and go down.
        nwalkers = mcmc_kws['nwalkers'] if 'nwalkers' in mcmc_kws else 100
        nwalkers = mcmc_kws['nwalkers'] if 'nwalkers' in mcmc_kws else 1
        nparams = len(params)
        if 'ntemps' in mcmc_kws:
            ntemps = mcmc_kws['ntemps']
            noise_shape = (ntemps, nwalkers, nparams)
            sigma_shape = (1, 1, nparams)
        else:
            noise_shape = (nwalkers, nparams)
            sigma_shape = (1, nparams)
        noise_sigma = np.full(sigma_shape, 0.05)
        noise = np.random.randn(*noise_shape) * noise_sigma

        pos_center = np.reshape([params[k].value for k in params], sigma_shape)
        pos_randomized = pos_center + noise
        mcmc_kws.update({'pos': pos_randomized})


def estimate_noise_from(data):
    data = data.values.squeeze()
    smoothed_data = gaussian_filter(data, sigma=1)
    noise = np.std(data - smoothed_data)
    return noise
