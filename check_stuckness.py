from polish_fit import *

good_fits = GOOD_FITS
polished_fits = json.load(
    open('./polished-fits.json'),
    object_pairs_hook=OrderedDict)


def get_chisq(data, params):
    rf = RandomRefitter(data, params)
    return rf.evaluate_chisq(params)


IND = 33
# #33 is a good hologram (well below the focus) with a large change
# in error
all_data, _ = load_few_PS_data_Jan10()
data = all_data[IND]

good_params = parse_result(good_fits[str(IND)])
polished_params = polished_fits[str(IND)]


def interpolate_chisq(t):
    """
    t: float, on (0, 1) to interpolate
    t=0 means good_params, t=1 means polished params
    """
    new_params = {
        k: (1 - t) * good_params[k] + t * polished_params[k]
        for k in good_params.keys()}
    return get_chisq(data, new_params)


ts = np.linspace(-0.1, 1.1, 301)
chisqs = np.array([interpolate_chisq(t) for t in ts])

import matplotlib.pyplot as plt
plt.figure()
plt.plot(ts, chisqs)
plt.plot(ts, chisqs, 'k.')
plt.ylabel("Chi^2 Error")
plt.xlabel("Distance between ``good`` fit and polished fit")
plt.title("Data # 33")
plt.savefig('./fit-landscape-cut.png')
