import EstimationMethods
import numpy as np

rightwindow = False


def moving_window(timeseries, method, windowsize, leap=1, **args):
    if method == "var":
        method = EstimationMethods.calculate_var
    elif method == "ac1":
        method = EstimationMethods.calculate_acor1
    elif method == "acs":
        method = EstimationMethods.lambda_acs
    elif method == "psd":
        method = EstimationMethods.lambda_psd
    elif method == "gls":
        method = EstimationMethods.phi_gls
    elif method == "lse":
        method = EstimationMethods.phi_lse
    result = np.array([method(timeseries[i - round(windowsize / 2): i + round(windowsize / 2)], **args)
                       for i in range(round(windowsize / 2), len(timeseries) - round(windowsize / 2)+1, leap)])
    return result


def moving_window_filled(timeseries, method, windowsize, leap=1, **args):
    if method == "var":
        method = EstimationMethods.calculate_var
    elif method == "ac1":
        method = EstimationMethods.calculate_acor1
    elif method == "acs":
        method = EstimationMethods.lambda_acs
    elif method == "psd":
        method = EstimationMethods.lambda_psd
    elif method == "gls":
        method = EstimationMethods.phi_gls
    elif method == "lse":
        method = EstimationMethods.phi_lse
    result = [1e20 for i in range(round(windowsize / 2))]
    result += [x for sublist in [[method(timeseries[i - round(windowsize / 2): i + round(windowsize / 2)], **args)]
                                 + [1e20 for j in range(leap - 1)] for i in range(round(windowsize / 2), len(timeseries)
                                                                                  - round(windowsize / 2)+1, leap)]
               for x in sublist]
    if leap > 1:
        result = result[:-leap]
    result += [1e20 for i in range(round(windowsize / 2))]
    result = np.ma.masked_values(result, 1e20)
    if rightwindow:
        result = result[:-round(windowsize / 2)]
        result = np.ma.concatenate((np.ma.masked_values([1e20 for i in range(round(windowsize / 2))], 1e20), result))
    return result