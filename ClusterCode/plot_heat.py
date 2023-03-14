import tpr_fpr_auc
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import seaborn as sns
import time
import os
plt.rcParams['text.usetex'] = True
labels = ["a)", "b)", "c)", "d)", "e)", "f)", "g)", "h)"]
cols = ["darkblue", "darkred", "darkgoldenrod", "darkgreen"]
markers = ["^", "v", ">", "<"]

number_of_windows = tpr_fpr_auc.number_of_windows
windowsizes = tpr_fpr_auc.windowsizes
observation_lengths = tpr_fpr_auc.observation_lengths

def plot_heat_auc():
    method_names = ["Variance", "Lag-1 autocorrelation", "$\lambda$ via ACS", "$\lambda$ via PSD"]
    save_names = ["var", "ac1", "acs", "psd"]
    
    plt.rcParams.update({'font.size': 20})
    fig, axs = plt.subplots(nrows=1, ncols=5, gridspec_kw={"width_ratios":[1,1,1,1,0.1]}, figsize=(22,5))
    #fig.tight_layout(pad=2.0)
    props = dict(edgecolor="none", facecolor='white', alpha=0)
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
        auc_df = auc_df.iloc[::-1]
        cmap = sns.light_palette("seagreen",reverse=False, as_cmap=True)
        if method==0:
            sns.heatmap(auc_df, ax=axs[method], yticklabels=True, cbar=False, vmin=0.5, vmax=1, cmap=cmap)
        else:
            sns.heatmap(auc_df, ax=axs[method], yticklabels=False, cbar=False, vmin=0.5, vmax=1, cmap=cmap)
        axs[method].add_patch(Rectangle((4,7),1,1,fill=False,edgecolor="blue"))
        axs[method].add_patch(Rectangle((4,2),1,1,fill=False,edgecolor="red"))
        axs[method].set_title(method_names[method])
        axs[method].set_aspect("equal", adjustable='box')
        axs[method].text(-0.1, 1.1, labels[method], transform=axs[method].transAxes, fontsize=23,
                         verticalalignment='top', bbox=props)
    fig.text(0.5,0.02, "Fraction of the time series used in estimations [\%]", ha="center", va="center")
    fig.colorbar(axs[3].collections[0], cax=axs[4])
    axs[0].set_ylabel("Length of the time series")   
    axs[3].collections[0].colorbar.set_label("AUC")
    plt.savefig("Plots/heat_" + save_names[method] + "_" + time.strftime("%Y%m%d-%H%M%S"), dpi = 300, bbox_inches='tight')
    #plt.show()

plot_heat_auc()