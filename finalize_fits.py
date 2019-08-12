import json
from collections import OrderedDict

import numpy as np

import mielensfit as mlf
from fit_data import load_few_PS_data_Jan10
from polish_fit import parse_result

POLISHED_FITS = json.load(
    open('./polished-fits.json'), object_pairs_hook=OrderedDict)


def refit(data, initial_guess):
    fitter = mlf.Fitter(data, initial_guess)
    result = fitter.fit()
    return parse_result(result.parameters)


if __name__ == '__main__':
    all_data, _ = load_few_PS_data_Jan10()

    final_fits = []
    for i, data in enumerate(all_data):
        this_fit = refit(data, POLISHED_FITS[str(i)])
        final_fits.append(this_fit)
    fits_dict = OrderedDict()
    for key, value in enumerate(final_fits):
        fits_dict.update({str(key): value})
    json.dump(fits_dict, open("./finalized-fits.json", "w"), indent=4)

