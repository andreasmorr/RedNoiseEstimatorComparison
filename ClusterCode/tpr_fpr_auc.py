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
scenario_size = 5000



def get_tpr_fpr_auc(i_,j_):
    windowsize=windowsizes[i_]
    observation_length=observation_lengths[j_]
    n = number_of_windows*windowsize
    leap = windowsize
    taus = MethodComparisons.comparison_taus(n, windowsize, leap, oversampling, scenario_size, observation_length)
    for method in range(5):
        roc_data = MethodComparisons.roc_curve(np.array(taus[0][method]), np.array(taus[1][method]),probe_count=200)
        roc_data = [roc_data[0], roc_data[1], np.array([roc_data[2]])]
        pd.DataFrame(roc_data,index=["tpr", "fpr", "auc"]).to_csv("" + str(method) + "_" + str(i_) + "_" + str(j_) + ".csv")