import numpy as np
import pandas as pd
from scipy import stats
from datetime import datetime
import EstimationMethods
import SampleGeneration
import WindowEstimation
import MethodComparisons
import sys
import os


number_of_windows = 20
windowsizes = [100,200,350,500,700,900,1100,1300,1500]
observation_lengths = [0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1]
oversampling = 10
scenario_size = 1000




def comparison_taus(n, windowsize, leap, oversampling, scenario_size, observation_length):
    lambda_pos = np.array([np.sqrt(1-i/n) for i in range(n)])
    lambda_neg = np.array([1 for i in range(n)])
    lambda_scale_min = 0.3
    lambda_scale_max = 0.5
    theta_min = 0.5
    theta_max = 4
    kappa_min = 0.5
    kappa_max = 4
    taus_pos = [[], [], [], []]
    taus_neg = [[], [], [], []]
    start = datetime.now()
    for j in range(scenario_size):
        if j == 0:
            print("Start time: " + str(pd.to_datetime(start).round("1s")) + "; Progress: " + str(round(100*j/scenario_size)) + "%")
        elif j == 10:
            now = datetime.now()
            end = start + (now-start)*scenario_size/j
            print("End time: " + str(pd.to_datetime(end).round("1s")) + "; Progress: " + str(round(100*j/scenario_size)) + "%")
        lambda_scale = np.random.uniform(lambda_scale_min, lambda_scale_max)
        [theta_start, theta_end] = np.random.uniform(theta_min, theta_max, 2)
        [kappa_start, kappa_end] = np.random.uniform(kappa_min, kappa_max, 2)
        short_n = round(n * observation_length)
        short_lambda_pos = lambda_pos[:short_n]
        short_lambda_neg = lambda_neg[:short_n]
        theta_ = np.array([theta_start * (1 - i / short_n) + theta_end * i / short_n for i in range(short_n)])
        kappa_ = np.array([kappa_start * (1 - i / short_n) + kappa_end * i / short_n for i in range(short_n)])
        sample_pos = SampleGeneration.generate_path(short_n, lambda_scale * short_lambda_pos, theta_, kappa_,
                                                    oversampling=oversampling)
        sample_neg = SampleGeneration.generate_path(short_n, lambda_scale * short_lambda_neg, theta_, kappa_,
                                                    oversampling=oversampling)
        for method_number in range(4):
            methods = [EstimationMethods.calculate_var, EstimationMethods.calculate_acor1, EstimationMethods.lambda_acs,
                       EstimationMethods.lambda_psd]
            method = methods[method_number]
            results_pos = WindowEstimation.moving_window(timeseries=sample_pos, method=method, windowsize=windowsize,
                                                         leap=leap, initial=[1,1,1], relevant_lags=3)
            results_neg = WindowEstimation.moving_window(timeseries=sample_neg, method=method, windowsize=windowsize,
                                                         leap=leap, initial=[1, 1, 1], relevant_lags=3)
            if method_number == 2 or method_number == 3:
                results_pos = -1*results_pos
                results_neg = -1 * results_neg
            taus_pos[method_number].append(stats.kendalltau(range(len(results_pos)),
                                                                  results_pos)[0])
            taus_neg[method_number].append(stats.kendalltau(range(len(results_neg)),
                                                                  results_neg)[0])
    return [taus_pos,taus_neg]


def get_tpr_fpr_auc(i_,j_):
    if not os.path.exists("tpr_fpr_auc"):
        os.makedirs("tpr_fpr_auc")
    windowsize=windowsizes[i_]
    observation_length=observation_lengths[j_]
    n = number_of_windows*windowsize
    leap = windowsize
    taus = MethodComparisons.comparison_taus(n, windowsize, leap, oversampling, scenario_size, observation_length)
    for method in range(5):
        roc_data = MethodComparisons.roc_curve(np.array(taus[0][method]), np.array(taus[1][method]),probe_count=200)
        roc_data = [roc_data[0], roc_data[1], np.array([roc_data[2]])]
        pd.DataFrame(roc_data,index=["tpr", "fpr", "auc"]).to_csv("tpr_fpr_auc/" + str(method) + "_" + str(i_) + "_" + str(j_) + ".csv")
get_tpr_fpr_auc(0,0)