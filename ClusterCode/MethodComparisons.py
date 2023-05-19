import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import seaborn as sns
import numpy as np
import pandas as pd
import scipy
from scipy.optimize import curve_fit
from datetime import datetime
import time
import EstimationMethods
import SampleGeneration
import WindowEstimation
#import tpr_fpr_auc
from matplotlib.patches import Rectangle
plt.rcParams['text.usetex'] = True
labels = ["a)", "b)", "c)", "d)", "e)", "f)", "g)", "h)"]
cols = ["darkblue", "darkred", "darkgoldenrod", "darkgreen","darkmagenta"]
markers = ["^", "v", ">", "<","D"]
methods = [EstimationMethods.calculate_var, EstimationMethods.calculate_acor1, EstimationMethods.lambda_acs, EstimationMethods.lambda_psd, EstimationMethods.phi_gls]

def estimator_distributions(sample_size, n, oversampling, lambda_, theta_, kappa_, initial, relevant_lags):
    results = []
    start = datetime.now()
    for i in range(sample_size):
        if i == 0:
            print("Start time: " + str(pd.to_datetime(start).round("1s")) + "; Progress: " +
                  str(round(100 * i / sample_size)) + "%")
        elif i == 10:
            now = datetime.now()
            end = start + (now - start) * sample_size / i
            print("End time: " + str(pd.to_datetime(end).round("1s")) + "; Progress: " +
                  str(round(100 * i / sample_size)) + "%")
        sample = SampleGeneration.generate_path(n=n, lambda_=lambda_, theta_=theta_, kappa_=kappa_,
                                                oversampling=oversampling)
        this_result = []
        for method in methods:
            this_result.append(method(sample, initial=initial, relevant_lags=relevant_lags))
        results.append(this_result)
    return np.transpose(np.array(results))


def plot_estimator_distributions(results, sample_size, lambda_, theta_, kappa_, label_offset=0):
    plt.rcParams.update({'font.size': 20})
    fig, axs = plt.subplots(nrows=1, ncols=5, figsize=(20, 6))
    fig.tight_layout(pad=2.0)
    props = dict(edgecolor="none", facecolor='white', alpha=0)
    for method in range(5):
        if method == 0:
            truev = kappa_ ** 2 / (2 * lambda_ * theta_ * (lambda_ + theta_))
            xlabel = "Variance estimator"
        elif method == 1:
            truev = (lambda_ * np.exp(-theta_) - theta_ * np.exp(-lambda_)) / (lambda_ - theta_)
            xlabel = "AC(1) estimator"
        elif method == 2:
            truev = lambda_
            xlabel = "$\lambda$ estimation via ACS"
        elif method == 3:
            truev = lambda_
            xlabel = "$\lambda$ estimation via PSD"
        else:
            truev = np.exp(-lambda_)
            xlabel = r"$\varphi$ estimator"
        data = results[method]
        p, x = np.histogram(data, bins="auto")
        x = x[:-1] + (x[1] - x[0]) / 2
        i = 0
        est_bin = 0
        while i < len(x) and x[i] < truev:
            est_bin = i
            i += 1
        i = 0
        total = sum([p[j] * (x[j + 1] - x[j]) for j in range(len(p)-1)])
        while sum([p[j] * (x[j + 1] - x[j]) for j in range(max(0, est_bin - i), min(len(p)-1, est_bin + i + 1))]) < 0.68 * total:
            i += 1
        sigma = i*(x[1] - x[0])
        axs[method].plot(x, p/(sample_size*(x[1] - x[0])), color="black")
        axs[method].set_xlabel(xlabel)
        if method == 0:
            axs[method].set_ylabel("Probability density")
        axs[method].axvline(truev, color="red", linestyle="dashed")
        axs[method].axvline(truev + sigma, color="purple", linestyle="dotted", linewidth = 2)
        axs[method].axvline(truev - sigma, color="purple", linestyle="dotted", linewidth = 2)
        axs[method].text(-0.25, 0.95, labels[method + label_offset], transform=axs[method].transAxes, fontsize=23,
                         verticalalignment='top', bbox=props)
    plt.savefig("Plots/distributions" + time.strftime("%Y%m%d-%H%M%S"), dpi = 300, bbox_inches='tight')
    plt.show()


def get_sigma_intervals(results_against_n, lambda_, theta_, kappa_):
    sigmas_against_n = []
    for k in range(len(results_against_n)):
        sigmas = []
        for method in range(5):
            if method == 0:
                truev = kappa_ ** 2 / (2 * lambda_ * theta_ * (lambda_ + theta_))
            elif method == 1:
                truev = (lambda_ * np.exp(-theta_) - theta_ * np.exp(-lambda_)) / (lambda_ - theta_)
            elif method == 2:
                truev = lambda_
            elif method == 3:
                truev = lambda_
            else:
                truev = np.exp(-lambda_)
            data = results_against_n[k][method]
            p, x = np.histogram(data, bins="auto")
            x = x[:-1] + (x[1] - x[0]) / 2
            i = 0
            est_bin = 0
            while i < len(x) and x[i] < truev:
                est_bin = i
                i += 1
            i = 0
            total = sum([p[j] * (x[j + 1] - x[j]) for j in range(len(p)-1)])
            while sum([p[j] * (x[j + 1] - x[j]) for j in range(max(0, est_bin - i), min(len(p)-1, est_bin + i + 1))]) < 0.68 * total:
                i += 1
            sigma = i*(x[1] - x[0])
            sigmas.append(sigma)
        sigmas_against_n.append(sigmas)
    return np.transpose(np.array(sigmas_against_n))


def oneoversqrt(n,a):
    return a/n**0.5

def plot_interval_convergence(ns, sigmas_against_n, label_offset=0):
    omit_beginning = 0
    plt.rcParams.update({'font.size': 20})
    fig, axs = plt.subplots(nrows=1, ncols=4, figsize=(16, 6))
    fig.tight_layout(pad=2.0)
    props = dict(edgecolor="none", facecolor='white', alpha=0)
    for method in range(4):
        sigmas = sigmas_against_n[method]
        popt,pcov = curve_fit(oneoversqrt,ns[omit_beginning:],sigmas[omit_beginning:])
        print(popt)
        if method == 0:
            title = "Variance estimator"
        elif method == 1:
            title = "AC(1) estimator"
        elif method == 2:
            title = "$\lambda$ estimation via ACS"
        elif method == 3:
            title = "$\lambda$ estimation via PSD"
        else:
            title = r"$\varphi$ estimator"
        axs[method].plot(ns[omit_beginning:], sigmas[omit_beginning:], color="black")
        axs[method].plot(ns[omit_beginning:], [oneoversqrt(n,popt[0]) for n in ns][omit_beginning:], color="red")
        axs[method].set_xlabel("Window size $N$")
        axs[method].set_title(title, fontsize=20)
        if method == 0:
            axs[method].set_ylabel("$1\sigma$-interval size")
        axs[method].legend(["1$\sigma$-interval width","$a/\sqrt{N}$ fit, $a=$" + str(round(popt[0],2))], fontsize=15)
        axs[method].text(-0.3, 1, labels[method + label_offset], transform=axs[method].transAxes, fontsize=23,
                         verticalalignment='top', bbox=props)
    plt.savefig("Plots/convergence" + time.strftime("%Y%m%d-%H%M%S"), dpi=300, bbox_inches='tight')
    plt.show()


def roc_curve(pos,neg,probe_count):
    if pos == [] or neg == []:
        return [[],[]]
    minv=-1
    maxv=1
    probes = [maxv*(1-i/probe_count)+minv*i/probe_count for i in range(probe_count+1)]
    tpr = np.array([sum([pos[j]>probes[i] for j in range(len(pos))]) for i in range(probe_count+1)]+[len(pos)])*100/len(pos)
    fpr = np.array([sum([neg[j]>probes[i] for j in range(len(neg))]) for i in range(probe_count+1)]+[len(neg)])*100/len(neg)
    auc = sum([(tpr[i+1]+tpr[i])/2*(fpr[i+1]-fpr[i]) for i in range(len(tpr)-1)])/10000
    return [tpr,fpr,auc]


def comparison_taus(n, windowsize, leap, oversampling, scenario_size, observation_length):
    lambda_pos = np.array([np.sqrt(1-i/n) for i in range(n)])
    lambda_neg = np.array([1 for i in range(n)])
    lambda_scale_min = 0.3
    lambda_scale_max = 0.5
    theta_min = 0.5
    theta_max = 4
    kappa_min = 0.5
    kappa_max = 4
    taus_pos = [[], [], [], [], []]
    taus_neg = [[], [], [], [], []]
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
        for method_number in range(5):
            method = methods[method_number]
            results_pos = WindowEstimation.moving_window(timeseries=sample_pos, method=method, windowsize=windowsize,
                                                         leap=leap, initial=[1,1,1], relevant_lags=3)
            results_neg = WindowEstimation.moving_window(timeseries=sample_neg, method=method, windowsize=windowsize,
                                                         leap=leap, initial=[1, 1, 1], relevant_lags=3)
            if method_number == 2 or method_number == 3:
                results_pos = -1 * results_pos
                results_neg = -1 * results_neg
            taus_pos[method_number].append(scipy.stats.kendalltau(range(len(results_pos)),
                                                                  results_pos)[0])
            taus_neg[method_number].append(scipy.stats.kendalltau(range(len(results_neg)),
                                                                  results_neg)[0])
    return [taus_pos,taus_neg]


def cluster_roc_curves(number_of_windows, windowsizes, oversampling, scenario_size, observation_lengths):
    tau_results=[]
    for i in range(len(windowsizes)):
        tau_results.append([])
        windowsize=windowsizes[i]
        for j in range(len(observation_lengths)):
            observation_length=observation_lengths[j]
            n = number_of_windows*windowsize
            leap = windowsize
            tau_results[i].append(comparison_taus(n, windowsize, leap, oversampling, scenario_size, observation_length))
    taus_df = pd.DataFrame(tau_results, index=windowsizes, columns=observation_lengths)
    return taus_df



def plot_roc_curves_from_cluster_taus(taus_df):
    for windowsize in taus_df.index:
        for observation_length in taus_df.columns:
            taus = taus_df.loc[windowsize,observation_length]
            taus_pos = taus[0]
            taus_neg = taus[1]
            probe_count = 200
            roc_curves = []
            plt.rcParams["figure.figsize"] = (8, 8)
            plt.title("Assessment of Indicator trends after " + str(round(observation_length * 100)) + "\% of CSD")
            plt.xlabel("False Positive Rate [\%]")
            plt.ylabel("True Positive Rate [\%]")
            for method_number in range(5):
                roc_curves.append(roc_curve(taus_pos[method_number], taus_neg[method_number], probe_count))
                plt.plot(roc_curves[method_number][1], roc_curves[method_number][0], c=cols[method_number])
            for method_number in range(5):
                plt.scatter(roc_curves[method_number][1][round(len(roc_curves[method_number][1])/2)],
                            roc_curves[method_number][0][round(len(roc_curves[method_number][1])/2)],c=cols[method_number],
                            marker="*", s=40)
            plt.plot([0, 100], [0, 100], c="black", linestyle="dashed")
            plt.xlim([-0.4, 100])
            plt.ylim([0, 100.4])
            plt.legend(["Variance " + str(round(roc_curves[0][2],2)), "AC(1) " + str(round(roc_curves[1][2],2)),
                        "ACS " + str(round(roc_curves[2][2],2)), "PSD " + str(round(roc_curves[3][2],2))], loc="lower right")
            plt.savefig("Plots_Cluster/roc_curve_w" + str(windowsize) + "_o_" + str(observation_length).replace(".","") + time.strftime("%Y%m%d-%H%M%S"), dpi = 300, bbox_inches='tight')
            plt.show()

            
def get_auc_from_cluster_taus(taus_df):
    auc_dfs = list()
    for method_number in range(4):
        auc_results = []
        for windowsize in taus_df.index:
            this_auc_results = []
            for observation_length in taus_df.columns:
                taus = taus_df.loc[windowsize,observation_length]
                taus_pos = taus[0]
                taus_neg = taus[1]
                probe_count = 200 
                auc = roc_curve(taus_pos[method_number], taus_neg[method_number], probe_count)[2]
                this_auc_results.append(auc)
            auc_results.append(this_auc_results)
        auc_df = pd.DataFrame(auc_results, index=taus_df.index, columns = taus_df.columns)
        auc_dfs.append([auc_df])
    auc_dfs = [item for sublist in auc_dfs for item in sublist]
    return auc_dfs
            
    
def plot_heat_auc(auc_dfs):
    method_names = ["Variance", "Lag-1 autocorrelation", "$\lambda$ via ACS", "$\lambda$ via PSD"]
    save_names = ["var", "ac1", "acs", "psd"]
    for method in range(4):
        df = auc_dfs[method]
        df.columns = np.round(100*df.columns,0).astype(int)
        df = df.iloc[::-1]
        cmap=sns.light_palette("seagreen",reverse=False, as_cmap=True)
        ax = sns.heatmap(df, vmin=0.5, vmax=1, cmap=cmap)
        plt.title(method_names[method])
        plt.xlabel("Fraction of the time series used in estimations [\%]")
        plt.ylabel("Length of each of the $20$ estimation windows in the time series")
        ax.collections[0].colorbar.set_label("AUC")
        #plt.savefig("Plots_Cluster/heat_" + save_names[method] + "_" + time.strftime("%Y%m%d-%H%M%S"), dpi = 300, bbox_inches='tight')
        plt.show()



def plot_mult_roc_curves_from_taus(taus, observation_length):
    number_of_figs = len(taus)
    plt.rcParams.update({'font.size': 15})
    fig, axs = plt.subplots(nrows=1, ncols=number_of_figs, figsize=(12,24))
    fig.tight_layout(pad=5.0)
    props = dict(edgecolor="none", facecolor='white', alpha=0)
    for fig_number in range(number_of_figs):
        taus_pos = taus[fig_number][0]
        taus_neg = taus[fig_number][1]
        probe_count = 200
        roc_curves = []
        axs[fig_number].set_aspect("equal", adjustable='box')
        axs[fig_number].set_title("Assessment of Indicator trends after "
                                  + str(round(observation_length[fig_number] * 100)) + "\% of CSD")
        axs[fig_number].title.set_fontsize(14)
        axs[fig_number].set_xlabel("False Positive Rate [\%]")
        axs[fig_number].set_ylabel("True Positive Rate [\%]")
        for method_number in range(5):
            roc_curves.append(roc_curve(taus_pos[method_number], taus_neg[method_number], probe_count))
        for method_number in range(5):    
            axs[fig_number].plot(roc_curves[method_number][1], roc_curves[method_number][0],c=cols[method_number])
        for method_number in range(5):
            axs[fig_number].scatter(roc_curves[method_number][1][round(len(roc_curves[method_number][1])/2)],
                                    roc_curves[method_number][0][round(len(roc_curves[method_number][1])/2)],
                                    c=cols[method_number], marker=markers[method_number], s=80)
        line_labels = ["Variance [" + str(round(roc_curves[0][2],2)) + "]", "AC(1) [" + str(round(roc_curves[1][2],2)) + "]",
                    "$\lambda^{\mathrm{(ACS)}}$ [" + str(round(roc_curves[2][2],2)) + "]", "$\lambda^{\mathrm{(PSD)}}$ [" + str(round(roc_curves[3][2],2)) + "]",
                    r"$\varphi$ [" + str(round(roc_curves[4][2],2)) + "]"]
        legend_lines = [mlines.Line2D([], [], label=line_labels[method], color=cols[method], marker=markers[method], markersize=8) for method in range(5)]
        axs[fig_number].plot([0, 100], [0, 100], c="black", linestyle="dashed")
        axs[fig_number].set_xlim([-1, 101])
        axs[fig_number].set_ylim([-1, 101])
        axs[fig_number].legend(handles=legend_lines, loc="lower right", fontsize=15, framealpha=1)
        axs[fig_number].text(-0.23, 0.97, labels[fig_number], transform=axs[fig_number].transAxes,
                             fontsize=20, verticalalignment='top', bbox=props)
        axs[fig_number].grid()
    #plt.savefig("Plots/roc_curve" + time.strftime("%Y%m%d-%H%M%S"), dpi = 300, bbox_inches='tight')
    plt.show()


def plot_example_est(n, observation_length, windowsize, leap, lambda_scale, theta_start, theta_end, kappa_start, kappa_end):
    oversampling = 10
    theta_min = 0.5
    theta_max = 4
    kappa_min = 0.5
    kappa_max = 4
    lambda_scale_min = 0.3
    lambda_scale_max = 0.5
    lambda_pos = np.array([np.sqrt(1 - i / n) for i in range(n)])
    lambda_neg = np.array([1 for i in range(n)])
    short_n = round(n * observation_length)
    short_lambda_pos = lambda_pos[:short_n]
    short_lambda_neg = lambda_neg[:short_n]
    theta_ = np.array([theta_start * (1 - i / short_n) + theta_end * i / short_n for i in range(short_n)])
    kappa_ = np.array([kappa_start * (1 - i / short_n) + kappa_end * i / short_n for i in range(short_n)])
    sample_pos = SampleGeneration.generate_path(short_n, lambda_scale * short_lambda_pos, theta_, kappa_,
                                                oversampling=oversampling)
    sample_neg = SampleGeneration.generate_path(short_n, lambda_scale * short_lambda_neg, theta_, kappa_,
                                                oversampling=oversampling)
    plt.rcParams.update({'font.size': 17})
    fig, [[scenA, scenB], [sampA, sampB], [estA, estB]] = plt.subplots(nrows=3, ncols=2, figsize=(14, 10),
                                                         gridspec_kw={'height_ratios': [3, 1, 1]}, sharex=True,tight_layout=True)
    props = dict(edgecolor="none", facecolor='white', alpha=0)
    scenA.plot(lambda_scale * lambda_pos, color="black", linewidth=2)
    scenA.plot(theta_, color="black", linestyle="dashdot", linewidth=2)
    scenA.plot(kappa_, color="black", linestyle="dotted", linewidth=2)
    scenA.legend([r"$\lambda$", r"$\theta$", r"$\kappa$"], loc="center right", fontsize=19)
    scenA.vlines(len(theta_), 0, max(theta_max, kappa_max), color="lightgrey")
    scenA.plot([min(theta_min, kappa_min) for x in theta_], linestyle="dashed", color="red")
    scenA.plot([max(theta_max, kappa_max) for x in theta_], linestyle="dashed", color="red")
    scenA.text(-0.14, 0.95, labels[0], transform=scenA.transAxes, fontsize=20,
               verticalalignment='top', bbox=props)
    scenA.set_ylim([0,max(theta_max, kappa_max)+0.2])
    scenA.plot([lambda_scale_min for x in range(500)], linestyle="dashed", color="red")

    scenB.plot(lambda_scale * lambda_neg, color="black", linewidth=2)
    scenB.plot(theta_, color="black", linestyle="dashdot", linewidth=2)
    scenB.plot(kappa_, color="black", linestyle="dotted", linewidth=2)
    scenB.legend([r"$\lambda$", r"$\theta$", r"$\kappa$"], loc="center right", fontsize=19)
    scenB.vlines(len(theta_), 0, max(theta_max, kappa_max), color="lightgrey")
    scenB.plot([min(theta_min, kappa_min) for x in theta_], linestyle="dashed", color="red")
    scenB.plot([max(theta_max, kappa_max) for x in theta_], linestyle="dashed", color="red")
    scenB.text(-0.14, 0.95, labels[1], transform=scenB.transAxes, fontsize=20,
               verticalalignment='top', bbox=props)
    scenB.set_ylim([0, max(theta_max, kappa_max) + 0.2])
    scenB.plot([lambda_scale_min for x in range(500)], linestyle="dashed", color="red")

    sampA.plot(sample_pos, linewidth=0.5, color="black")
    sampA.axvline(len(theta_), color="lightgrey")
    sampA.set_ylim([-max(abs(np.array(sample_pos))), max(abs(np.array(sample_pos)))])
    sampA.plot([0 for x in range(len(theta_))], color="red")
    for w in range(0, len(theta_) - windowsize, leap):
        sampA.axvline(w, ymin=0.45, ymax=0.55, color="red")
        sampA.axvline(w + windowsize, ymin=0.45, ymax=0.55, color="red")
    sampA.set_xlim([0, n])
    sampA.text(-0.14, 0.95, labels[2], transform=sampA.transAxes, fontsize=20,
               verticalalignment='top', bbox=props)
    sampA.legend(["$X_t$"])

    sampB.plot(sample_neg, linewidth=0.5, color="black")
    sampB.axvline(len(theta_), color="lightgrey")
    sampB.set_ylim([-max(abs(np.array(sample_neg))), max(abs(np.array(sample_neg)))])
    sampB.plot([0 for x in range(len(theta_))], color="red")
    for w in range(0, len(theta_) - windowsize, leap):
        sampB.axvline(w, ymin=0.45, ymax=0.55, color="red")
        sampB.axvline(w + windowsize, ymin=0.45, ymax=0.55, color="red")
    sampB.set_xlim([0, n])
    sampB.text(-0.14, 0.95, labels[3], transform=sampB.transAxes, fontsize=20,
               verticalalignment='top', bbox=props)
    sampB.legend(["$X_t$"])
    
    estA.plot(range(round(windowsize/2),short_n,leap), [EstimationMethods.calculate_var(sample_pos[j:j+windowsize]) for j in range(0,short_n,leap)], linewidth=2, color="black",linestyle = "dashed")
    estA.plot([0],[0], linewidth=2, color="black",linestyle = "dotted")
    estA2 = estA.twinx()
    estA2.plot(range(round(windowsize/2),short_n,leap), [EstimationMethods.calculate_acor1(sample_pos[j:j+windowsize]) for j in range(0,short_n,leap)], linewidth=2, color="black",linestyle = "dotted")
    estA.axvline(len(theta_), color="lightgrey")
    estA.set_xlim([0, n])
    estA.text(-0.14, 0.95, labels[4], transform=estA.transAxes, fontsize=20,
               verticalalignment='top', bbox=props)
    estA.legend(["Var","AC(1)"],fontsize=12)
    estA.set_xlabel("Time $t$")

    estB.plot(range(round(windowsize/2),short_n,leap), [EstimationMethods.calculate_var(sample_neg[j:j+windowsize]) for j in range(0,short_n,leap)], linewidth=2, color="black",linestyle = "dashed")
    estB.plot([0],[0], linewidth=2, color="black",linestyle = "dotted")
    estB2 = estB.twinx()
    estB2.plot(range(round(windowsize/2),short_n,leap), [EstimationMethods.calculate_acor1(sample_neg[j:j+windowsize]) for j in range(0,short_n,leap)], linewidth=2, color="black",linestyle = "dotted")
    estB.axvline(len(theta_), color="lightgrey")
    estB.set_xlim([0, n])
    estB.text(-0.14, 0.95, labels[5], transform=estB.transAxes, fontsize=20,
               verticalalignment='top', bbox=props)
    estB.legend(["Var","AC(1)"],fontsize=12)
    estB.set_xlabel("Time $t$")

    #plt.savefig("Plots/example" + time.strftime("%Y%m%d-%H%M%S"), dpi = 300, bbox_inches='tight')
    plt.show()


def plot_slices(window_index,obslen_index):
    number_of_windows = tpr_fpr_auc.number_of_windows
    windowsizes = tpr_fpr_auc.windowsizes
    observation_lengths = tpr_fpr_auc.observation_lengths

    #number_of_windows = 20
    #windowsizes = [200,350,500,700,900,1100,1300,1500]
    #observation_lengths = [0.3,0.4,0.5,0.6,0.7,0.8,0.9,1]
    
    fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(10,4),sharey=True)
    #fig.tight_layout(pad=5.0)
    props = dict(edgecolor="none", facecolor='white', alpha=0)
    
    dfs = []
    for method in range(5):
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
    axs[0].legend(["Variance", "AC(1)", "$\lambda^{\mathrm{(ACS)}}$", "$\lambda^{\mathrm{(PSD)}}$",r"$\varphi$"],loc="upper left",fontsize=14, framealpha=1)
    axs[0].text(-0.17, 0.97, labels[0], transform=axs[0].transAxes,
                            fontsize=20, verticalalignment='top', bbox=props)
    axs[0].tick_params(axis="x",labelsize=14)
    axs[0].tick_params(axis="y",labelsize=14)
    axs[0].grid()

    axs[1].set_title("AUC values for fraction of " + str(round(observation_lengths[obslen_index]*100)) + "$\%$ used")
    axs[1].title.set_fontsize(14)
    axs[1].set_xlabel("Length of the time series",fontsize=12)
    for method in range(5):
        axs[1].plot(np.round(dfs[method].index,0).astype(int),dfs[method].iloc[:,obslen_index],c=cols[method])
    axs[1].legend(["Variance", "AC(1)", "ACS", "PSD"],loc="upper left",fontsize=14, framealpha=1)
    axs[1].text(-0.17, 0.97, labels[1], transform=axs[1].transAxes,
                            fontsize=20, verticalalignment='top', bbox=props)
    axs[1].tick_params(axis="x",labelsize=14)
    axs[1].tick_params(axis="y",labelsize=14)
    axs[1].grid()
    #plt.savefig("Plots/slices" + time.strftime("%Y%m%d-%H%M%S"), dpi = 300, bbox_inches='tight')
    plt.show()




def plot_heat_auc(examples = True, slices = True):
    number_of_windows = tpr_fpr_auc.number_of_windows
    windowsizes = tpr_fpr_auc.windowsizes
    observation_lengths = tpr_fpr_auc.observation_lengths

    #number_of_windows = 20
    #windowsizes = [200,350,500,700,900,1100,1300,1500]
    #observation_lengths = [0.3,0.4,0.5,0.6,0.7,0.8,0.9,1]
    method_names = ["Variance estimator", "AC(1) estimator", "$\lambda$ via ACS", "$\lambda$ via PSD", r"$\varphi$ estimator"]
    
    #plt.rcParams.update({'font.size': 20})
    fig, axs = plt.subplots(nrows=1, ncols=6, gridspec_kw={"width_ratios":[1,1,1,1,0.1]}, figsize=(27,5))
    #fig.tight_layout(pad=2.0)
    props = dict(edgecolor="none", facecolor='white', alpha=0)
    for method in range(5):
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
    fig.colorbar(axs[4].collections[0], cax=axs[5])
    axs[0].set_ylabel("Length of the time series",fontsize=20) 
    axs[0].set_yticklabels(axs[0].get_yticklabels(), rotation = 0)
    axs[4].collections[0].colorbar.set_label("AUC",fontsize=20)
    axs[4].collections[0].colorbar.ax.tick_params(labelsize=20)
    #plt.savefig("Plots/heat_" + time.strftime("%Y%m%d-%H%M%S"), dpi = 300, bbox_inches='tight')
    plt.show()

