import numpy as np
import matplotlib.pyplot as plt  # imread
import scipy.ndimage as nd

import holopy as hp
from holopy.scattering import calc_holo, Sphere
from holopy.scattering.theory import mielensfunctions

mielens_fit = {
    'n': 1.3730489224781681,
    'r': 0.7375785547230772,
    'center.0': 121.19973880025405,
    'center.1': 100.7848729508633,
    'center.2': 22.833824553074496,
    'lens_angle': 0.5050657126584215,
    }

mieonly_fit = {
    'n': 1.4316413654808557,
    'r': 0.7729874802343999,
    'center.0': 121.19251826838352,
    'center.1': 100.78707490116628,
    'center.2': 23.852248295641104,
    'alpha': 0.4008335505754322,
    }

x_range = [114.946563, 139.339803]
y_range = [84.762122, 109.155362]
shape = (1, 140, 140)
px_in_um = (x_range[1] - x_range[0]) / (shape[1] - 1)

t = np.arange(140) * px_in_um
x = t.reshape(-1, 1) + x_range[0]
y = t.reshape(1, -1) + y_range[0]

index_water = 1.33
k = 2 * np.pi * index_water / 0.66


def calc_holo_mielens(center0, center1, center2, n, r, lens_angle):
    params = {
        'particle_kz': k * center2,
        'index_ratio': n / index_water,
        'size_parameter': k * r,
        'lens_angle': lens_angle,
        }
    calc = mielensfunctions.MieLensCalculator(**params)
    dx = x - center0
    dy = y - center1
    rho = np.sqrt(dx**2 + dy**2)
    phi =np.arctan2(dy, dx)
    return calc.calculate_total_intensity(k*rho, phi)


def calc_holo_mieonly(center0, center1, center2, n, r, alpha):
    sph = Sphere(center = (center0, center1, center2), n=n, r=r)
    camera_resolution = 5.6983 # px / um
    metadata = {'spacing' : 1 / camera_resolution,
                    'medium_index' : 1.33,
                    'illum_wavelen' : .660,
                    'illum_polarization' : (1, 0)}
    return calc_holo(hp.load_image('image0029.tif', **metadata), sph, scaling=alpha).values.squeeze()


def calculate_gradmodel(params, delta=1e-5):
    params = np.array(params)
    m0 = calc_holo_mielens(*params)
    gradmodel = np.zeros([params.size, m0.size], dtype='float')
    dp = np.zeros_like(params)
    for i, _ in enumerate(params):
        dp *= 0
        dp[i] = delta
        model_plus = calc_holo_mielens(*(params + dp)).ravel()
        model_minus = calc_holo_mielens(*(params - dp)).ravel()
        gradmodel[i] = (model_plus - model_minus) / (2 * delta)
    return gradmodel


def calculate_gradmodel_mieonly(params, delta=1e-5):
    params = np.array(params)
    m0 = calc_holo_mieonly(*params)
    gradmodel = np.zeros([params.size, m0.size], dtype='float')
    dp = np.zeros_like(params)
    for i, _ in enumerate(params):
        dp *= 0
        dp[i] = delta
        model_plus = calc_holo_mieonly(*(params + dp)).ravel()
        model_minus = calc_holo_mieonly(*(params - dp)).ravel()
        gradmodel[i] = (model_plus - model_minus) / (2 * delta)
    return gradmodel


def get_sigma():
    bkg = plt.imread('bkg0001.tif')
    img = plt.imread('image0029.tif')
    holo = img / bkg
    subholo = holo[655:795, 483:623]
    kernel = np.ones([3, 3]); kernel /= kernel.sum()
    sub_smoothed = nd.convolve(subholo, kernel)
    difference = subholo - sub_smoothed
    # Now the difference is x - xbar, where xbar is the sample mean
    # over a 9x9 window. So if I take difference.std(), this is the
    # same as the sample variance, with N=9. So the N-1 correction
    # is important, so we rescale by sqrt(N/N-1) = sqrt(9/8)
    return difference.std() * np.sqrt(9 / 8.)


if __name__ == "__main__":
    keys = ['center.0', 'center.1', 'center.2', 'n', 'r', 'lens_angle']
    params = np.array([mielens_fit[k] for k in keys])
    gradmodel = calculate_gradmodel(params)
    # gradmodel_v2 = calculate_gradmodel(params, delta=1e-4)
    jtj = gradmodel.dot(gradmodel.T)
    jtj_inv = np.linalg.inv(jtj)
    image_sigma = get_sigma()
    covariance = jtj_inv * image_sigma**2
    uncertainties = np.sqrt(np.diag(covariance))
    print("Fit Uncertainties:")
    for (k, p, e) in zip(keys, params, uncertainties):
        ps = '{:0.5f}'.format(p).rjust(10)
        print("{}:\t{}\t+-\t{:0.5f}".format(k.ljust(10), ps, e))

    keys = ['center.0', 'center.1', 'center.2', 'n', 'r', 'alpha']
    params = np.array([mieonly_fit[k] for k in keys])
    gradmodel = calculate_gradmodel_mieonly(params)
    # gradmodel_v2 = calculate_gradmodel(params, delta=1e-4)
    jtj = gradmodel.dot(gradmodel.T)
    jtj_inv = np.linalg.inv(jtj)
    image_sigma = get_sigma()
    covariance_mieonly = jtj_inv * image_sigma**2
    uncertainties_mieonly = np.sqrt(np.diag(covariance_mieonly))
    print("Fit Uncertainties:")
    for (k, p, e) in zip(keys, params, uncertainties_mieonly):
        ps = '{:0.5f}'.format(p).rjust(10)
        print("{}:\t{}\t+-\t{:0.5f}".format(k.ljust(10), ps, e))


"""prints:
Fit Uncertainties:
center.0  :  121.19974  +-  0.00249
center.1  :  100.78487  +-  0.00237
center.2  :   22.83382  +-  0.01364
n         :    1.37305  +-  0.00039
r         :    0.73758  +-  0.00285
lens_angle:    0.50507  +-  0.00271
"""