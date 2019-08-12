import os

import numpy as np

import holopy as hp
from holopy.scattering import calc_holo, Sphere
from holopy.scattering.theory import MieLens, Mie
from holopy.core.io import load_average
from holopy.core.metadata import get_extents, get_spacing
from holopy.core.process import subimage, normalize, center_find  # , bg_correct
from holopy.inference import prior, AlphaModel, NmpfitStrategy
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
    _min_lens_angle = 0.0
    _max_lens_angle = 1.2
    _default_alpha = 1.0
    _min_alpha = 0.0
    _max_alpha = 2.0
    _min_index = 1.33
    _max_index = 2.3
    _min_radius = 0.05
    _max_radius = 5.0

    def __init__(self, data, guess, theory='mielens'):
        """for now, this is for the lens model only

        Parameters
        ----------
        data : hologram
        guess : dict
            Must have keys `'n'`, `'r'`, `'z'`. An additional key of
            `'lens_angle'` is optional.
        """
        self.data = data
        self.guess = guess
        self.theory = theory

    def fit(self):
        sphere_priors = self.make_guessed_scatterer()
        if self.theory == 'mielens':
            lens_prior = self.guess_lens_angle()
            model = PerfectLensModel(
                sphere_priors, noise_sd=self.data.noise_sd,
                lens_angle=lens_prior)
        elif self.theory == 'mieonly':
            alpha_prior = self.guess_alpha()
            model = AlphaModel(
                sphere_priors, noise_sd=self.data.noise_sd, alpha=alpha_prior)
        optimizer = NmpfitStrategy()
        result = optimizer.optimize(model, self.data)
        # FIXME this result sometimes leaves the allowed ranges. To get
        # result = hp.fitting.fit(model, self.data, minimizer=optimizer)
        return result

    def evaluate_model(self, params):
        scatterer = self.make_scatterer(params)
        if self.theory == 'mieonly':
            theory = Mie()
        else:
            theory = MieLens(lens_angle=params['lens_angle'])
        if self.theory == 'mielens':
            scaling = 1.0
        else:
            scaling = params['alpha']

        model = calc_holo(self.data, scatterer, theory=theory, scaling=scaling)
        return model

    def evaluate_residuals(self, params):
        """params is a dict-like"""
        model = self.evaluate_model(params)
        return self.data.values.squeeze() - model.values.squeeze()

    def evaluate_chisq(self, params):
        residuals = self.evaluate_residuals(params)
        return np.sum(residuals**2)

    def make_guessed_scatterer(self):
        return self.make_scatterer(self.guess)

    def make_scatterer(self, params):
        center = self._make_center_priors(params)
        index = prior.Uniform(
            self._min_index, self._max_index, guess=params['n'])
        radius = prior.Uniform(
            self._min_radius, self._max_radius, guess=params['r'])
        scatterer = Sphere(n=index, r=radius, center=center)
        return scatterer

    def guess_lens_angle(self):
        lens_angle = (self.guess['lens_angle'] if 'lens_angle' in self.guess
                      else self._default_lens_angle)
        lens_prior = prior.Uniform(
            self._min_lens_angle, self._max_lens_angle, guess=lens_angle)
        return lens_prior

    def guess_alpha(self):
        alpha = (self.guess['alpha'] if 'alpha' in self.guess
                      else self._default_alpha)
        alpha_prior = prior.Uniform(
            self._min_alpha, self._max_alpha, guess=alpha)
        return alpha_prior

    def _make_center_priors(self, params):
        image_x_values = self.data.x.values
        image_min_x = image_x_values.min()
        image_max_x = image_x_values.max()

        image_y_values = self.data.y.values
        image_min_y = image_y_values.min()
        image_max_y = image_y_values.max()

        if ('x' not in params) or ('y' not in params):
            pixel_spacing = get_spacing(self.data)
            image_lower_left = np.array([image_min_x, image_min_y])
            center = center_find(self.data) * pixel_spacing + image_lower_left
        else:
            center = [params['x'], params['y']]

        xpar = prior.Uniform(image_min_x, image_max_x, guess=center[0])
        ypar = prior.Uniform(image_min_y, image_max_y, guess=center[1])

        extents = get_extents(self.data)
        extent = max(extents['x'], extents['y'])
        zextent = 5
        zpar = prior.Uniform(
            -extent * zextent, extent * zextent, guess=params['z'])
        return xpar, ypar, zpar


def fit_mielens(hologram, guess_parameters):
    fitter = Fitter(hologram, guess_parameters, theory='mielens')
    return fitter.fit()


def fit_mieonly(hologram, guess_parameters):
    fitter = Fitter(hologram, guess_parameters, theory='mieonly')
    return fitter.fit()


# ~~~ loading data

class NormalizedDataLoader(object):
    def __init__(self, data_filenames, metadata, particle_position,
                 background_prefix="bg", darkfield_prefix=None):
        self.data_filenames = list(data_filenames)
        self.metadata = metadata
        self.particle_position = particle_position
        self.background_prefix = background_prefix
        self.darkfield_prefix = darkfield_prefix

        self.root_folder = os.path.dirname(self.data_filenames[0])
        self._reference_image = hp.load_image(
            self.data_filenames[0], channel=RGB_CHANNEL, **self.metadata)
        self._background = self._load_background()
        self._darkfield = self._load_darkfield()

    def load_all_data(self):
        return [self._load_data(nm) for nm in self.data_filenames]

    def _load_data(self, name):  # need metadata, particle_position!
        data = hp.load_image(name, channel=RGB_CHANNEL, **self.metadata)
        data = bg_correct(data, self._background, self._darkfield)
        data = subimage(data, self.particle_position[::-1], HOLOGRAM_SIZE)
        data = normalize(data)
        return data

    def _load_background(self):
        names = self._get_filenames_which_contain(self.background_prefix)
        background = load_average(
            names, refimg=self._reference_image, channel=RGB_CHANNEL)
        return background

    def _load_darkfield(self):
        if self.darkfield_prefix is not None:
            names = self._get_filenames_which_contain(self.darkfield_prefix)
            darkfield = load_average(
                names, refimg=self._reference_image, channel=RGB_CHANNEL)
        else:
            darkfield = None
        return darkfield

    def _get_filenames_which_contain(self, prefix):
        paths = [os.path.join(self.root_folder, name)
                 for name in os.listdir(self.root_folder)
                 if prefix in name]
        return paths


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


def load_bgdivide_crop_all_images(paths, metadata, particle_position, **kwargs):
    loader = NormalizedDataLoader(paths, metadata, particle_position, **kwargs)
    return loader.load_all_data()


def load_bkg(path, bg_prefix, refimg):
    subdir = os.path.dirname(path)
    bkg_paths = [
        subdir + '/' + pth for pth in os.listdir(subdir) if bg_prefix in pth]

    bkg = load_average(bkg_paths, refimg=refimg, channel=RGB_CHANNEL)
    return bkg


def load_dark(path, df_prefix, refimg):
    return load_bkg(path, df_prefix, refimg) if df_prefix is not None else None


def load_bgdivide_crop_v2(
        path, metadata, particle_position, bkg, dark, channel=RGB_CHANNEL,
        size=HOLOGRAM_SIZE):
    data = hp.load_image(path, channel=channel, **metadata)
    data = bg_correct(data, bkg, dark)
    data = subimage(data, particle_position[::-1], size)
    data = normalize(data)
    return data

