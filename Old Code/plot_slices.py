import tpr_fpr_auc
import importlib
importlib.reload(tpr_fpr_auc)
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import seaborn as sns
import time
import os
plt.rcParams.update(plt.rcParamsDefault)
plt.rcParams['text.usetex'] = True
labels = ["a)", "b)", "c)", "d)", "e)", "f)", "g)", "h)"]
cols = ["darkblue", "darkred", "darkgoldenrod", "darkgreen"]
markers = ["^", "v", ">", "<"]

number_of_windows = tpr_fpr_auc.number_of_windows
windowsizes = tpr_fpr_auc.windowsizes
observation_lengths = tpr_fpr_auc.observation_lengths

#number_of_windows = 20
#windowsizes = [200,350,500,700,900,1100,1300,1500]
#observation_lengths = [0.3,0.4,0.5,0.6,0.7,0.8,0.9,1]

def plot_slices(window_index,obslen_index):
    method_names = ["Variance", "Lag-1 autocorrelation", "$\lambda$ via ACS", "$\lambda$ via PSD"]

    fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(10,4),sharey=True)
    #fig.tight_layout(pad=5.0)
    props = dict(edgecolor="none", facecolor='white', alpha=0)
    
    dfs = []
    for method in range(4):
        aucs = []
        for i in range(len(windowsizes)):
            aucs.append([])
            for j in range(len(observation_lengths)):
                df = pd.read_csv("tpr_fpr_auc/" + str(method) + "_" + str(i) + "_" + str(j) + ".csv", index_col=0)
                aucs[i].append(df.loc["auc"].iloc[0])
        auc_df = pd.DataFrame(aucs, index=windowsizes, columns=observation_lengths)
        auc_df.columns = np.round(100*auc_df.columns,0).astype(int)
        auc_df.index = number_of_windows * auc_df.index
        dfs.append(auc_df)
        #auc_df = auc_df.iloc[::-1]
    
    axs[0].set_title("AUC values for length " + str(windowsizes[window_index]*number_of_windows) + " time series")
    axs[0].title.set_fontsize(14)
    axs[0].set_xlabel("Fraction of the time series used in estimations [$\%$]",fontsize=12)
    axs[0].set_ylabel("AUC",fontsize=12)
    for method in range(4):
        axs[0].plot(np.round(dfs[method].columns,0).astype(int),dfs[method].iloc[window_index,:],c=cols[method])
    axs[0].legend(["Variance", "AC(1)", "ACS", "PSD"],loc="upper left")
    axs[0].text(-0.15, 0.97, labels[0], transform=axs[0].transAxes,
                            fontsize=20, verticalalignment='top', bbox=props)
    axs[0].tick_params(axis="x",labelsize=14)
    axs[0].tick_params(axis="y",labelsize=14)
    
    axs[1].set_title("AUC values for fraction of " + str(round(observation_lengths[obslen_index]*100)) + "$\%$ used")
    axs[1].title.set_fontsize(14)
    axs[1].set_xlabel("Length of the time series",fontsize=12)
    for method in range(4):
        axs[1].plot(np.round(dfs[method].index,0).astype(int),dfs[method].iloc[:,obslen_index],c=cols[method])
    axs[1].legend(["Variance", "AC(1)", "ACS", "PSD"],loc="upper left")
    axs[1].text(-0.15, 0.97, labels[1], transform=axs[1].transAxes,
                            fontsize=20, verticalalignment='top', bbox=props)
    axs[1].tick_params(axis="x",labelsize=14)
    axs[1].tick_params(axis="y",labelsize=14)
    #plt.savefig("Plots/slices" + time.strftime("%Y%m%d-%H%M%S"), dpi = 300, bbox_inches='tight')
    plt.show()
