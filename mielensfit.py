import os

import numpy as np

from scipy.optimize import minimize, differential_evolution

import matplotlib.pyplot as plt
from matplotlib.pyplot import imread

import holopy as hp
from holopy.scattering import calc_holo, Sphere
from holopy.scattering.theory import MieLens
from holopy.core.io import load_average
from holopy.core.metadata import get_extents, get_spacing
from holopy.core.process import bg_correct, subimage, normalize, center_find
from holopy.inference import prior, AlphaModel, NmpfitStrategy, TemperedStrategy, GradientFreeStrategy
from holopy.inference.scipyfit import LeastSquaresScipyStrategy
from holopy.inference.model import PerfectLensModel

RGB_CHANNEL = 1
HOLOGRAM_SIZE = 100

def minimize_mielens_de(hologram, guess_parameters, **kwargs):
    def cost(model_parameters):
        x, y, z, radius, index, lens_angle = model_parameters
        scatterer = Sphere(center=(x, y, z), r=radius, n=index)
        return calc_err_sq(hologram, scatterer, lens_angle=lens_angle)

    bounds = _get_bounds(hologram, guess_parameters)
    result = differential_evolution(cost, bounds, **kwargs)
    return result

def _get_bounds(hologram, guess_parameters):
    sphere_priors, lens_prior = _make_priors(hologram, guess_parameters)
    xbounds = (sphere_priors.x.lower_bound, sphere_priors.x.upper_bound)
    ybounds = (sphere_priors.y.lower_bound, sphere_priors.y.upper_bound)
    zbounds = (sphere_priors.z.lower_bound, sphere_priors.z.upper_bound)
    rbounds = (0.05, 10)
    nbounds = (1.35, 2.3)
    lens_bounds = (lens_prior.lower_bound, lens_prior.upper_bound)
    return [xbounds, ybounds, zbounds, rbounds, nbounds, lens_bounds]

def minimize_mielens(hologram, guess_parameters, **kwargs):
    def cost(model_parameters):
        x, y, z, radius, index, lens_angle = model_parameters
        scatterer = Sphere(center=(x, y, z), r=radius, n=index)
        return calc_err_sq(hologram, scatterer, lens_angle=lens_angle)

    intial_guess = _make_minimize_initial_guess(hologram, guess_parameters)
    kwargs['bounds'] = _get_bounds(hologram, guess_parameters)
    result = minimize(cost, intial_guess, **kwargs)
    return result

def _make_minimize_initial_guess(data, guess_parameters):
    guess_sphere = get_guess_scatterer(data, guess_parameters)
    lens_guess = get_guess_angle(guess_parameters)
    guess = np.hstack((guess_sphere.center, guess_sphere.r, guess_sphere.n, lens_guess))
    return guess

def fit_mielens(hologram, guess_parameters, strategy='nmp'):
    sphere_priors, lens_prior = _make_priors(hologram, guess_parameters)
    model = PerfectLensModel(sphere_priors, noise_sd=hologram.noise_sd, lens_angle=lens_prior)
    if strategy == 'nmp':
        result = NmpfitStrategy().optimize(model, hologram)
    elif strategy == 'leastsquares':
        result = LeastSquaresScipyStrategy().optimize(model, hologram)
    return result

def fit_mieonly(hologram, guess_parameters):
    priors, _ = _make_priors(hologram, guess_parameters)
    priors.center[2].lower_bound = 0
    model = AlphaModel(priors, noise_sd=hologram.noise_sd, alpha=prior.Uniform(0, 1.0, guess=0.6))
    result = NmpfitStrategy().optimize(model, hologram)
    return result

def mcmc_inference_mielens(hologram, guess_parameters):
    sphere_priors, lens_prior = _make_priors(hologram, guess_parameters)
    model = PerfectLensModel(sphere_priors, noise_sd=hologram.noise_sd, lens_angle=lens_prior)
    result = TemperedStrategy().optimize(model, hologram)
    return result

def globalop_mieonly(hologram, guess_parameters):
    priors, _ = _make_priors(hologram, guess_parameters)
    priors.center[2].lower_bound = 0
    model = AlphaModel(priors, noise_sd=hologram.noise_sd, alpha=prior.Uniform(0, 1.0, guess=0.6))
    result = GradientFreeStrategy().optimize(model, hologram)
    return result

def globalop_mielens(hologram, guess_parameters):
    sphere_priors, lens_prior = _make_priors(hologram, guess_parameters)
    model = PerfectLensModel(sphere_priors, noise_sd=hologram.noise_sd, lens_angle=lens_prior)
    result = GradientFreeStrategy().optimize(model, hologram)
    return result

def _make_priors(hologram, guess_parameters):
    center = _make_center_priors(hologram, guess_parameters)
    s = Sphere(n=prior.Uniform(1.33, 2.3, guess=guess_parameters['n']), 
               r=prior.Uniform(0.05, 5, guess=guess_parameters['r']),
               center=center)
    lens_guess = get_guess_angle(guess_parameters)
    lens_prior = prior.Uniform(0, 1.2, guess=lens_guess)
    return s, lens_prior

def _make_center_priors(im, guess_parameters, zextent=5):
    extents = get_extents(im)
    extent = max(extents['x'], extents['y'])
    
    spacing = get_spacing(im)
    center = center_find(im) * spacing + [im.x[0], im.y[0]]

    xpar = prior.Uniform(im.x.values.min(), im.x.values.max(), guess=center[0])
    ypar = prior.Uniform(im.y.values.min(), im.y.values.max(), guess=center[1])
    zpar = prior.Uniform(-extent * zextent, extent * zextent, guess=guess_parameters['z'])
    return xpar, ypar, zpar

def get_guess_scatterer(data, guesses):
    return _make_priors(data, guesses)[0].guess

def get_guess_angle(guesses):
    return guesses['lens_angle'] if 'lens_angle' in guesses else 0.8

def calc_chisq_B(data, fitresult, theory='mielens', **kwargs):
    dt = data.values.squeeze()
    residual = calc_residual(data, fitresult.scatterer, theory, **fitresult.parameters)
    return np.std(residual) / np.std(dt)

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

def load_bgdivide_crop(path, metadata, particle_position, bg_prefix="bg", df_prefix=None, channel=RGB_CHANNEL, size=HOLOGRAM_SIZE):
    data = hp.load_image(path, channel=channel, **metadata)
    bkg = load_bkg(path, bg_prefix, refimg=data)
    dark = load_dark(path, df_prefix, refimg=data)
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

def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.299, 0.587, 0.114])

def decode_OptimizeResult(result):
    x, y, z, radius, index, lens_angle = result.x
    scatterer = Sphere(center=(x, y, z), r=radius, n=index)
    return scatterer, lens_angle

