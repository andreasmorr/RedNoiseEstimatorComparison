import matplotlib.pyplot as plt
from scipy import optimize
import numpy as np
import statsmodels.api as sm

lse_negative_count = 0


def detrend_quadratic(timeseries):
    fit = np.polyfit(range(len(timeseries)), timeseries, 2)
    timeseries = timeseries - [x**2 * fit[0] + x * fit[1] + fit[2] for x in range(len(timeseries))]
    return timeseries


def power_spectrum(timeseries):
    fouriertransform = np.fft.rfft(timeseries)[1:round(len(timeseries) / 2)]
    return abs(fouriertransform) ** 2 / len(timeseries)


def calculate_var(timeseries, **kwargs):
    return sum([x**2 for x in timeseries]) / len(timeseries)


def calculate_acov(timeseries, lag=1):
    return sum([timeseries[i] * timeseries[i + lag] for i in range(len(timeseries) - lag)]) / (len(timeseries) - lag)


def calculate_acor1(timeseries, **kwargs):
    return calculate_acov(timeseries, lag=1)/calculate_var(timeseries)


def frequency_mean(timeseries, band_size=10):
    if band_size > 1:
        return np.array([sum(timeseries[i * band_size:(i + 1) * band_size]) / band_size for i in
                         range(round(len(timeseries) / band_size - 1))])
    else:
        return timeseries


def error_psd(params, n, observed_psd):
    lambda_ = abs(params[0])
    theta_ = abs(params[1])
    kappa_ = abs(params[2])
    lambda_ = min(lambda_,5)
    theta_ = min(theta_,5)
    elambda_ = np.exp(lambda_)
    etheta_ = np.exp(theta_)
    cos_array = [np.cos(2*np.pi*f) for f in np.fft.fftfreq(n)[1:round(n/2)]]
    if abs(lambda_ - theta_) < 0.01:
        theoretical_psd = np.array([kappa_ ** 2 / (4 * lambda_**3) * (-lambda_-cos_array[i]*np.sinh(lambda_) +
                                                                np.cosh(lambda_) * (lambda_*cos_array[i] +
                                                                                      np.sinh(lambda_)))
                                    / (cos_array[i]-np.cosh(lambda_))**2 for i in range(len(cos_array))])
    else:
        theoretical_psd = np.array([kappa_**2/(2*lambda_*theta_*(lambda_**2-theta_**2))*(lambda_*(1 - etheta_ ** -2)
                                                                                         / (1 + etheta_ ** -2 - 2
                                                                                            * etheta_ ** -1
                                                                                            * cos_array[i]) - theta_
                                                                                         * (1 - elambda_ ** -2)
                                                                                         / (1 + elambda_ ** -2 - 2 *
                                                                                            elambda_ ** -1
                                                                                            * cos_array[i]))
                                    for i in range(len(cos_array))])
    theoretical_psd = np.log(frequency_mean(theoretical_psd))
    if 0:
        plt.loglog(np.exp(theoretical_psd))
        plt.loglog(np.exp(observed_psd))
        plt.show()
    return sum(abs(observed_psd - theoretical_psd) ** 2)


def lambda_psd(timeseries, **kwargs):
    observed_psd = np.log(frequency_mean(power_spectrum(timeseries)))
    params = optimize.fmin(error_psd, kwargs["initial"], args=(len(timeseries), observed_psd), maxiter=100000, disp=False)
    if "return_all_params" in kwargs and kwargs.get("return_all_params"):
        return [min([abs(params[0]), abs(params[1])]), max([abs(params[0]), abs(params[1])]),
                abs(params[2])]
    else:
        return min([abs(params[0]), abs(params[1])])


def acor_struc(timeseries, relevant_lags=1):
    struc = [calculate_acov(timeseries, lag) / calculate_var(timeseries) for lag in range(relevant_lags)]
    return np.array(struc)


def error_acs(params, observed_ac):
    lambda_ = abs(params[0])
    theta_ = abs(params[1])
    lambda_ = min(lambda_,5)
    theta_ = min(theta_,5)
    if abs(lambda_ - theta_) < 0.01:
        theoretical_ac = np.array([np.exp(-lambda_ * x) * (1 + lambda_ * x) for x in range(len(observed_ac))])
    else:
        theoretical_ac = np.array([(theta_ * np.exp(-lambda_ * x) - lambda_ * np.exp(-theta_ * x)) / (theta_ - lambda_)
                                   for x in range(len(observed_ac))])
    if 0:
        plt.plot(observed_ac)
        plt.plot(theoretical_ac)
        plt.show()
        print(params)
    return sum(abs(observed_ac - theoretical_ac) ** 2)


def lambda_acs(timeseries, **kwargs):
    observed_ac = acor_struc(timeseries, kwargs["relevant_lags"])
    params = optimize.fmin(error_acs, kwargs["initial"], args=(observed_ac,), maxiter=100000, disp=False)
    if "return_all_params" in kwargs and kwargs.get("return_all_params"):
        return [min([abs(params[0]), abs(params[1])]), max([abs(params[0]), abs(params[1])])]
    else:
        return min([np.abs(params[0]), abs(params[1])])
    

def phi_gls(timeseries, **kwargs):
    diff = timeseries[1:] - timeseries[:-1]
    #timeseries = sm.add_constant(timeseries)
    model = sm.GLSAR(diff, timeseries[:-1], rho=1)
    result = model.iterative_fit(maxiter=100)
    #print(result.summary())
    if "return_all_params" in kwargs and kwargs.get("return_all_params"):
        return [result.params[0]+1, model.rho[0]]
    else:
        return result.params[0]+1
    

def phi_lse(timeseries, **kwargs):
    global lse_negative_count
    phi_b = calculate_acor1(timeseries)
    V = timeseries[:-1] - phi_b*timeseries[1:]
    rho_b = calculate_acor1(V)
    a = phi_b + rho_b
    b = rho_b/phi_b
    if a**2-4*b>0:
        phi_a = (a+np.sqrt(a**2-4*b))/2
    else:
        lse_negative_count+=1
        #print(lse_negative_count)
        return 0
    rho_a = rho_b/(phi_a*phi_b)
    if "return_all_params" in kwargs and kwargs.get("return_all_params"):
        return [phi_a, rho_a]
    else:
        return phi_a