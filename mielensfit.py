import os

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import imread

import holopy as hp
from holopy.scattering import calc_holo, Sphere
from holopy.scattering.theory import MieLens
from holopy.core.io import load_average
from holopy.core.metadata import get_extents, get_spacing
from holopy.core.process import subimage, normalize, center_find  # , bg_correct
from holopy.inference import prior, AlphaModel, NmpfitStrategy, TemperedStrategy
from holopy.inference.scipyfit import LeastSquaresScipyStrategy
from holopy.inference.model import PerfectLensModel


RGB_CHANNEL = 1
HOLOGRAM_SIZE = 100


def bg_correct(raw, bg, df=None):
    if df is None:
        df = raw.copy()
        df[:] = 0
    denominator = bg - df
    denominator.values = np.clip(denominator.values, 1e-7, np.inf)
    holo = (raw - df) / (bg - df)
    holo = hp.core.copy_metadata(raw, holo)
    return holo


def _get_bounds(hologram, guess_parameters):
    sphere_priors, lens_prior = _make_priors(hologram, guess_parameters)
    xbounds = (sphere_priors.x.lower_bound, sphere_priors.x.upper_bound)
    ybounds = (sphere_priors.y.lower_bound, sphere_priors.y.upper_bound)
    zbounds = (sphere_priors.z.lower_bound, sphere_priors.z.upper_bound)
    rbounds = (0.05, 10)
    nbounds = (1.35, 2.3)
    lens_bounds = (lens_prior.lower_bound, lens_prior.upper_bound)
    return [xbounds, ybounds, zbounds, rbounds, nbounds, lens_bounds]


# ~~~ fitting

class Fitter(object):
    _default_lens_angle = 0.8

    def __init__(self, data, guess):
        """for now, this is for the lens model only"""
        self.data = data
        self.guess = guess

    def fit(self):
        sphere_priors, lens_priors = self._make_priors()
        # for prior_function in [_make_priors, _make_position_only_priors]:
        sphere_priors, lens_prior = self._make_priors()
        model = PerfectLensModel(
            sphere_priors, noise_sd=self.data.noise_sd, lens_angle=lens_prior)
        optimizer = NmpfitStrategy()
        result = optimizer.fit(model, self.data)
        # result = hp.fitting.fit(model, self.data, minimizer=optimizer)
        return result

    def _make_priors(self):
        center = self._make_center_priors()
        scatterer = Sphere(n=prior.Uniform(1.33, 2.3, guess=self.guess['n']),
                           r=prior.Uniform(0.05, 5, guess=self.guess['r']),
                           center=center)
        lens_guess = self._guess_lens_angle()
        lens_prior = prior.Uniform(0, 1.2, guess=lens_guess)
        return scatterer, lens_prior

    def _make_center_priors(self):
        image_x_values = self.data.x.values
        image_min_x = image_x_values.min()
        image_max_x = image_x_values.max()

        image_y_values = self.data.y.values
        image_min_y = image_y_values.min()
        image_max_y = image_y_values.max()

        pixel_spacing = get_spacing(self.data)
        image_lower_left = np.array([image_min_x, image_min_y])
        center = center_find(self.data) * pixel_spacing + image_lower_left

        xpar = prior.Uniform(image_min_x, image_max_x, guess=center[0])
        ypar = prior.Uniform(image_min_y, image_max_y, guess=center[1])

        extents = get_extents(self.data)
        extent = max(extents['x'], extents['y'])
        zextent = 5
        zpar = prior.Uniform(
            -extent * zextent, extent * zextent, guess=self.guess['z'])
        return xpar, ypar, zpar

    def _guess_lens_angle(self):
        lens_angle = (self.guess['lens_angle'] if 'lens_angle' in self.guess
                      else self._default_lens_angle)
        return lens_angle


def fit_mielens(hologram, guess_parameters):
    fitter = Fitter(hologram, guess_parameters)
    return fitter.fit()


def fit_mieonly(hologram, guess_parameters):
    priors, _ = _make_priors(hologram, guess_parameters)
    #priors.center[2].lower_bound = 0
    model = AlphaModel(priors, noise_sd=hologram.noise_sd, alpha=prior.Uniform(0, 1.0, guess=0.6))
    result = NmpfitStrategy().optimize(model, hologram)
    return result

# ~~~ priors



def _make_position_only_priors(hologram, guess_parameters):
    center = _make_center_priors(hologram, guess_parameters)
    s = Sphere(n=guess_parameters['n'],
               r=guess_parameters['r'],
               center=center)
    lens_guess = get_guess_angle(guess_parameters)
    return s, lens_guess


def get_guess_scatterer(data, guesses):
    return _make_priors(data, guesses)[0].guess

def get_guess_angle(guesses):
    return guesses['lens_angle'] if 'lens_angle' in guesses else 0.8

def calc_residual(data, scatterer, theory='mielens', **kwargs):
    dt = data.values.squeeze()
    if theory == 'mielens':
        lens_angle = kwargs['lens_angle']
        fit = calc_holo(data, scatterer, theory=MieLens(lens_angle=lens_angle)).values.squeeze()
    elif theory == 'mieonly':
        scaling = kwargs['alpha']
        fit = calc_holo(data, scatterer, scaling=scaling).values.squeeze()
    return fit - dt


def calc_err_sq(data, scatterer, theory='mielens', **kwargs):
    residual = calc_residual(data, scatterer, theory=theory, **kwargs)
    return np.sum(residual ** 2)


# ~~~ loading data

def load_bgdivide_crop(
        path, metadata, particle_position, bg_prefix="bg", df_prefix=None,
        channel=RGB_CHANNEL, size=HOLOGRAM_SIZE):
    data = hp.load_image(path, channel=channel, **metadata)
    bkg = load_bkg(path, bg_prefix, refimg=data)
    dark = None  # load_dark(path, df_prefix, refimg=data)
    data = bg_correct(data, bkg, dark)
    data = subimage(data, particle_position[::-1], size)
    data = normalize(data)
    return data

def load_bkg(path, bg_prefix, refimg):
    bkg_paths = get_bkg_paths(path, bg_prefix)
    bkg = load_average(bkg_paths, refimg=refimg, channel=RGB_CHANNEL)
    return bkg

def load_dark(path, df_prefix, refimg):
    return load_bkg(path, df_prefix, refimg) if df_prefix is not None else None

def get_bkg_paths(path, bg_prefix):
    subdir = os.path.dirname(path)
    bkg_paths = [subdir + '/' + pth for pth in os.listdir(subdir) if bg_prefix in pth]
    return bkg_paths

def load_bgdivide_crop_v2(
        path, metadata, particle_position, bkg, dark, channel=RGB_CHANNEL,
        size=HOLOGRAM_SIZE):
    data = hp.load_image(path, channel=channel, **metadata)
    data = bg_correct(data, bkg, dark)
    data = subimage(data, particle_position[::-1], size)
    data = normalize(data)
    return data

def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.299, 0.587, 0.114])

