import numpy as np


def generate_path(n, lambda_, theta_, kappa_, oversampling=1, x0=None):
    if (type(lambda_) == float or type(lambda_) == int) and (type(theta_) == float or type(theta_) == int) and \
            (type(kappa_) == float or type(kappa_) == int):
        return generate_path_fixed_parameters(n=n, lambda_=lambda_, theta_=theta_, kappa_=kappa_,
                                              oversampling=oversampling, x0=x0)
    n = n * oversampling
    delta = 1/oversampling
    lambda_ = np.repeat(lambda_, oversampling)
    theta_ = np.repeat(theta_, oversampling)
    kappa_ = np.repeat(kappa_, oversampling)
    if x0 is None:
        x = [np.sqrt(kappa_[0]**2 / (2*lambda_[0])) * np.random.normal(0, 1)]
    else:
        x = [x0]
    ar1_coefficient = np.exp(-theta_ * delta)
    white_noise_array = np.sqrt(1 / (2 * theta_) * (1 - np.exp(-2 * theta_ * delta))) * np.random.normal(0, 1, size=n)
    red_noise_array = [np.sqrt(1 / (2 * theta_[0])) * np.random.normal(0, 1)]
    for i in range(n - 1):
        red_noise_array.append(ar1_coefficient[i] * red_noise_array[-1] + white_noise_array[i])
    for i in range(n - 1):
        x.append(x[-1] - lambda_[i] * x[-1] * delta + kappa_[i] * red_noise_array[i] * delta)
    x = x[::oversampling]
    return np.array(x)


def generate_path_fixed_parameters(n, lambda_, theta_, kappa_, oversampling=1, x0=None):
    n = n * oversampling
    delta = 1/oversampling
    if x0 is None:
        x = [np.sqrt(kappa_**2 / (2*lambda_)) * np.random.normal(0, 1)]
    else:
        x = [x0]
    ar1_coefficient = np.exp(-theta_ * delta)
    white_noise_array = np.sqrt(1 / (2 * theta_) * (1 - np.exp(-2 * theta_ * delta))) * np.random.normal(0, 1, size=n)
    red_noise_array = [np.sqrt(1 / (2 * theta_)) * np.random.normal(0, 1)]
    for i in range(n - 1):
        red_noise_array.append(ar1_coefficient * red_noise_array[-1] + white_noise_array[i])
    for i in range(n - 1):
        x.append(x[-1] - lambda_ * x[-1] * delta + kappa_ * red_noise_array[i] * delta)
    x = x[::oversampling]
    return np.array(x)
