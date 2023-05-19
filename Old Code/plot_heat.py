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
#cmap = sns.light_palette("seagreen",reverse=False, as_cmap=True)
cmap = "winter"

number_of_windows = tpr_fpr_auc.number_of_windows
windowsizes = tpr_fpr_auc.windowsizes
observation_lengths = tpr_fpr_auc.observation_lengths

#number_of_windows = 20
#windowsizes = [200,350,500,700,900,1100,1300,1500]
#observation_lengths = [0.3,0.4,0.5,0.6,0.7,0.8,0.9,1]

def plot_heat_auc(examples = True, slices = True):
    method_names = ["Variance", "Lag-1 autocorrelation", "$\lambda$ via ACS", "$\lambda$ via PSD"]
    save_names = ["var", "ac1", "acs", "psd"]
    
    #plt.rcParams.update({'font.size': 20})
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
        if method==0:
            sns.heatmap(auc_df, ax=axs[method], yticklabels=True, cbar=False, vmin=0.5, vmax=1, cmap=cmap)
        else:
            sns.heatmap(auc_df, ax=axs[method], yticklabels=False, cbar=False, vmin=0.5, vmax=1, cmap=cmap)
        if examples:
            axs[method].add_patch(Rectangle((7,5),1,1,fill=False,edgecolor="red", lw=2))
            axs[method].add_patch(Rectangle((2,5),1,1,fill=False,edgecolor="pink", lw=2))
        if slices: 
            axs[method].add_patch(Rectangle((0,7),9,1,fill=False,edgecolor="black", lw=2, linestyle="dashed"))
            axs[method].add_patch(Rectangle((1,0),1,9,fill=False,edgecolor="black", lw=2, linestyle="dotted"))
        axs[method].set_title(method_names[method],fontsize=20)
        axs[method].set_xlim([-0.1,9.1])
        axs[method].set_ylim([9.1,-0.1])
        axs[method].set_aspect("equal", adjustable='box')
        axs[method].text(-0.1, 1.1, labels[method], transform=axs[method].transAxes, fontsize=23,
                         verticalalignment='top', bbox=props)
        axs[method].tick_params(axis="x",labelsize=20)
        axs[method].tick_params(axis="y",labelsize=16)
    fig.text(0.5,0.02, "Fraction of the time series used in estimations [\%]", ha="center", va="center",fontsize=20)
    fig.colorbar(axs[3].collections[0], cax=axs[4])
    axs[0].set_ylabel("Length of the time series",fontsize=20) 
    axs[0].set_yticklabels(axs[0].get_yticklabels(), rotation = 0)
    axs[3].collections[0].colorbar.set_label("AUC",fontsize=20)
    axs[3].collections[0].colorbar.ax.tick_params(labelsize=20)
    #plt.savefig("Plots/heat_" + time.strftime("%Y%m%d-%H%M%S"), dpi = 300, bbox_inches='tight')
    plt.show()


def plot_heat_auc_slides(examples = True, slices = False):
    method_names = ["Variance", "Lag-1 autocorrelation", "$\lambda$ via ACS", "$\lambda$ via PSD"]
    save_names = ["var", "ac1", "acs", "psd"]
    
    #plt.rcParams.update({'font.size': 20})
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
        if method==0:
            sns.heatmap(auc_df, ax=axs[method], yticklabels=True, cbar=False, vmin=0.5, vmax=1, cmap=cmap)
        else:
            sns.heatmap(auc_df, ax=axs[method], yticklabels=False, cbar=False, vmin=0.5, vmax=1, cmap=cmap)
        if examples:
            axs[method].add_patch(Rectangle((6,3),1,1,fill=False,edgecolor="purple", lw=2))
            axs[method].add_patch(Rectangle((2,6),1,1,fill=False,edgecolor="red", lw=2))
        if slices: 
            axs[method].add_patch(Rectangle((0,7),9,1,fill=False,edgecolor="black", lw=2, linestyle="dashed"))
            axs[method].add_patch(Rectangle((1,0),1,9,fill=False,edgecolor="black", lw=2, linestyle="dotted"))
        axs[method].set_title(method_names[method],fontsize=20)
        axs[method].set_xlim([-0.1,9.1])
        axs[method].set_ylim([9.1,-0.1])
        axs[method].set_aspect("equal", adjustable='box')
        axs[method].text(-0.1, 1.1, labels[method], transform=axs[method].transAxes, fontsize=23,
                         verticalalignment='top', bbox=props)
        axs[method].tick_params(axis="x",labelsize=20)
        axs[method].tick_params(axis="y",labelsize=16)
    fig.text(0.5,0.02, "Fraction of the time series used in estimations [\%]", ha="center", va="center",fontsize=20)
    fig.colorbar(axs[3].collections[0], cax=axs[4])
    axs[0].set_ylabel("Length of the time series",fontsize=20) 
    axs[0].set_yticklabels(axs[0].get_yticklabels(), rotation = 0)
    axs[3].collections[0].colorbar.set_label("AUC",fontsize=20)
    axs[3].collections[0].colorbar.ax.tick_params(labelsize=20)
    #plt.savefig("Plots/heat_" + time.strftime("%Y%m%d-%H%M%S"), dpi = 300, bbox_inches='tight')
    plt.show()


def plot_heat_example_slides(examples = True, slices = False):
    method_names = ["Variance", "Lag-1 autocorrelation", "$\lambda$ via ACS", "$\lambda$ via PSD"]
    save_names = ["var", "ac1", "acs", "psd"]
    
    #plt.rcParams.update({'font.size': 20})
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(22,4))
    #fig.tight_layout(pad=2.0)
    props = dict(edgecolor="none", facecolor='white', alpha=0)
    example1 = []
    example2 = []
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
        example1.append(auc_df.iloc[3,6])
        example2.append(auc_df.iloc[6,3])
    ax.scatter([0,1,2,3],example1,c="purple",s=100)
    ax.scatter([0,1,2,3],example2,c="red",s=100)
    ax.scatter([-0.4,3.5],[0.7,0.7],c="white")
    ax.set_xticks([0,1,2,3])
    ax.set_xticklabels(["","","",""],fontsize=20)
    #ax.set_xticklabels(method_names,fontsize=20)
    ax.yaxis.set_label_position("right")
    ax.yaxis.set_ticks_position("right")
    ax.set_ylabel("AUC",fontsize=20)
    ax.tick_params(axis="y",labelsize=15)
    ax.legend(["using $80\%$ of 18,000 point time series","using $40\%$ of 7,000 point time series"],fontsize=20)
    #ax.spines['top'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    plt.savefig("Plots/example_" + time.strftime("%Y%m%d-%H%M%S"), dpi = 300, bbox_inches='tight')
    plt.show()