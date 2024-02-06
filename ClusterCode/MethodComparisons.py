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
import tpr_fpr_auc
from matplotlib.patches import Rectangle
plt_scale=2
plt.rcParams['pgf.texsystem'] = 'pdflatex'
plt.rcParams.update({'font.family': 'serif',
                     'text.usetex': True,
                     'font.serif': ['Helvetica'],
                     'font.size': plt_scale*8,
                     'axes.labelsize': plt_scale*10,
                     'axes.titlesize': plt_scale*12,
                     'figure.titlesize': plt_scale*14})
labels = ["(a)", "(b)", "(c)", "(d)", "(e)", "(f)", "(g)", "(h)"]
cols = ["dodgerblue", "darkred", "darkgoldenrod", "darkgreen","darkviolet"]
markers = ["^", "s", "p", "h","o"]
methods = [EstimationMethods.calculate_var, EstimationMethods.calculate_acor1, EstimationMethods.phi_gls, EstimationMethods.lambda_acs, EstimationMethods.lambda_psd]

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
    fig, axs = plt.subplots(nrows=2, ncols=3, figsize=(plt_scale*7.2, plt_scale*4))
    axs = axs.flatten()
    props = dict(edgecolor="none", facecolor='white', alpha=0)
    for method in range(5):
        if method == 0:
            truev = kappa_ ** 2 / (2 * lambda_ * theta_ * (lambda_ + theta_))
            xlabel = "Variance estimator"
        elif method == 1:
            truev = (lambda_ * np.exp(-theta_) - theta_ * np.exp(-lambda_)) / (lambda_ - theta_)
            xlabel = "AC(1) estimator"
        elif method == 2:
            truev = np.exp(-lambda_)
            xlabel = r"$\varphi$ estimator"
        elif method == 3:
            truev = lambda_
            xlabel = "$\lambda$ estimation via ACS"
        else:
            truev = lambda_
            xlabel = "$\lambda$ estimation via PSD"
        data = results[method]
        p, x = np.histogram(data, bins=30)
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
        if method == 0 or method == 3:
            axs[method].set_ylabel("Probability density")
        axs[method].axvline(truev, color="red", linestyle="dashed")
        axs[method].axvline(truev + sigma, color="purple", linestyle="dotted")
        axs[method].axvline(truev - sigma, color="purple", linestyle="dotted")
        axs[method].text(0.05, 0.95, labels[method + label_offset], transform=axs[method].transAxes,
                         verticalalignment='top', bbox=props)
    axs[5].axis('off')
    fig.subplots_adjust(hspace=.35, wspace=.2)
    plt.savefig("Plots/distributions" + time.strftime("%Y%m%d-%H%M%S") + ".pdf", dpi = 300, bbox_inches='tight')
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
                truev = np.exp(-lambda_)
            elif method == 3:
                truev = lambda_
            else:
                truev = lambda_
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
    fig, axs = plt.subplots(nrows=1, ncols=4, figsize=(plt_scale*7.2, plt_scale*2))
    axs = axs.flatten()
    props = dict(edgecolor="none", facecolor='white', alpha=0)
    for method in range(4):
        sigmas = sigmas_against_n[method]
        popt,pcov = curve_fit(oneoversqrt,ns[omit_beginning:],sigmas[omit_beginning:])
        if method == 0:
            title = "Var"
        elif method == 1:
            title = "AC(1)"
        elif method == 2:
            title = "$\lambda^{(\mathrm{ACS})}$"
        else:
            title = "$\lambda^{(\mathrm{PSD})}$"
        axs[method].plot(ns[omit_beginning:], sigmas[omit_beginning:], color="black")
        axs[method].plot(ns[omit_beginning:], [oneoversqrt(n,popt[0]) for n in ns][omit_beginning:], color="red")
        axs[method].set_xlabel("Window size $N$")
        axs[method].set_title(title)
        if method == 0:
            axs[method].set_ylabel("$1\sigma$-interval size")
        axs[method].legend(["1$\sigma$-interval width","$a/\sqrt{N}$ fit, $a=$" + str(round(popt[0],2))],fontsize=12)
        axs[method].text(0.05, 0.1, labels[method + label_offset], transform=axs[method].transAxes,
                         verticalalignment='top', bbox=props)
    
    fig.subplots_adjust(hspace=.35, wspace=.3)
    plt.savefig("Plots/convergence" + time.strftime("%Y%m%d-%H%M%S") + ".pdf", dpi=300, bbox_inches='tight')
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
            if method_number == 3 or method_number == 4:
                results_pos = -1 * results_pos
                results_neg = -1 * results_neg
            taus_pos[method_number].append(scipy.stats.kendalltau(range(len(results_pos)),
                                                                  results_pos)[0])
            taus_neg[method_number].append(scipy.stats.kendalltau(range(len(results_neg)),
                                                                  results_neg)[0])
    return [taus_pos,taus_neg]



def plot_two_roc_curves_from_tpr_fpr():
    number_of_figs = 2
    observation_length = [1,0.6]
    fig, axs = plt.subplots(nrows=1, ncols=number_of_figs, figsize=(plt_scale*3.42,plt_scale*2.5),sharey=True)
    props = dict(edgecolor="none", facecolor='white', alpha=0)
    for fig_number in range(number_of_figs):
        probe_count = 200
        roc_curves = []
        axs[fig_number].set_aspect("equal", adjustable='box')
        axs[fig_number].set_title("Observed CSD: " + str(round(observation_length[fig_number] * 100)) + "\%", fontsize=15)
        axs[fig_number].set_xlabel("False positive rate [\%]")
        if fig_number==0:
            axs[fig_number].set_ylabel("True positive rate [\%]")
        for method in range(5):
            i = 4
            j = [8, 4][fig_number]
            df = pd.read_csv("tpr_fpr_auc/" + str(method) + "_" + str(i) + "_" + str(j) + ".csv", index_col=0)
            roc_curves.append([df.loc["tpr"].values,df.loc["fpr"].values,df.loc["auc"].values[0]])
        for method_number in range(5):    
            axs[fig_number].plot(roc_curves[method_number][1], roc_curves[method_number][0],c=cols[method_number])
            axs[fig_number].scatter(roc_curves[method_number][1][round(len(roc_curves[method_number][1])/2)],
                                    roc_curves[method_number][0][round(len(roc_curves[method_number][1])/2)],
                                    c=cols[method_number], marker=markers[method_number], s=plt_scale*40)
        line_labels = ["Var [" + str(round(roc_curves[0][2],2)) + "]", "AC(1) [" + str(round(roc_curves[1][2],2)) + "]",
                    r"$\varphi$ [" + str(round(roc_curves[2][2],2)) + "]",
                    "$\lambda^{\mathrm{(ACS)}}$ [" + str(round(roc_curves[3][2],2)) + "]", "$\lambda^{\mathrm{(PSD)}}$ [" + str(round(roc_curves[4][2],2)) + "]"]
        legend_lines = [mlines.Line2D([], [], label=line_labels[method], color=cols[method], marker=markers[method], markersize=plt_scale*4) for method in range(5)]
        axs[fig_number].plot([0, 100], [0, 100], c="black", linestyle="dashed")
        axs[fig_number].set_xlim([-1, 101])
        axs[fig_number].set_xticks([0,25,50,75,100])
        axs[fig_number].set_ylim([-1, 101])
        axs[fig_number].set_yticks([0,25,50,75,100])
        axs[fig_number].legend(handles=legend_lines, loc="lower right", framealpha=1, fontsize=10)
        axs[fig_number].text(0.86, 0.62, labels[fig_number], transform=axs[fig_number].transAxes, verticalalignment='top', bbox=props)
        axs[fig_number].grid()
    plt.savefig("Plots/roc_curve" + time.strftime("%Y%m%d-%H%M%S") + ".pdf", dpi = 300, bbox_inches='tight')
    plt.show()


def plot_mult_roc_curves_from_taus(taus, observation_length):
    number_of_figs = len(taus)
    fig, axs = plt.subplots(nrows=1, ncols=number_of_figs, figsize=(plt_scale*3.42,plt_scale*2.5),sharey=True)
    props = dict(edgecolor="none", facecolor='white', alpha=0)
    for fig_number in range(number_of_figs):
        taus_pos = taus[fig_number][0]
        taus_neg = taus[fig_number][1]
        probe_count = 200
        roc_curves = []
        axs[fig_number].set_aspect("equal", adjustable='box')
        axs[fig_number].set_title("Observed CSD: " + str(round(observation_length[fig_number] * 100)) + "\%", fontsize=15)
        axs[fig_number].set_xlabel("False positive rate [\%]")
        if fig_number==0:
            axs[fig_number].set_ylabel("True positive rate [\%]")
        for method_number in range(5):
            roc_curves.append(roc_curve(taus_pos[method_number], taus_neg[method_number], probe_count))
        for method_number in range(5):    
            axs[fig_number].plot(roc_curves[method_number][1], roc_curves[method_number][0],c=cols[method_number])
            axs[fig_number].scatter(roc_curves[method_number][1][round(len(roc_curves[method_number][1])/2)],
                                    roc_curves[method_number][0][round(len(roc_curves[method_number][1])/2)],
                                    c=cols[method_number], marker=markers[method_number], s=plt_scale*40)
        line_labels = ["Var [" + str(round(roc_curves[0][2],2)) + "]", "AC(1) [" + str(round(roc_curves[1][2],2)) + "]",
                    r"$\varphi$ [" + str(round(roc_curves[2][2],2)) + "]",
                    "$\lambda^{\mathrm{(ACS)}}$ [" + str(round(roc_curves[3][2],2)) + "]", "$\lambda^{\mathrm{(PSD)}}$ [" + str(round(roc_curves[4][2],2)) + "]"]
        legend_lines = [mlines.Line2D([], [], label=line_labels[method], color=cols[method], marker=markers[method], markersize=plt_scale*4) for method in range(5)]
        axs[fig_number].plot([0, 100], [0, 100], c="black", linestyle="dashed")
        axs[fig_number].set_xlim([-1, 101])
        axs[fig_number].set_xticks([0,25,50,75,100])
        axs[fig_number].set_ylim([-1, 101])
        axs[fig_number].set_yticks([0,25,50,75,100])
        axs[fig_number].legend(handles=legend_lines, loc="lower right", framealpha=1, fontsize=10)
        axs[fig_number].text(0.86, 0.62, labels[fig_number], transform=axs[fig_number].transAxes, verticalalignment='top', bbox=props)
        axs[fig_number].grid()
    plt.savefig("Plots/roc_curve" + time.strftime("%Y%m%d-%H%M%S") + ".pdf", dpi = 300, bbox_inches='tight')
    plt.show()


def plot_example_est(n, observation_length, windowsize, leap, lambda_scale, theta_start, theta_end, kappa_start, kappa_end, labels_right=False):
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
    fig, [[scenA, scenB], [sampA, sampB], [estA, estB]] = plt.subplots(nrows=3, ncols=2, figsize=(plt_scale*3.42, plt_scale*2.8),
                                                         gridspec_kw={'height_ratios': [2.5, 1, 1]}, sharex=True)
    if labels_right:
        labelx = 0.89
    else:
        labelx = 0.02
    props = dict(edgecolor="none", facecolor='white', alpha=0)
    scenA.plot(lambda_scale * lambda_pos, color="black", linewidth=plt_scale*1)
    scenA.plot(theta_, color="black", linestyle="dashdot", linewidth=plt_scale*1)
    scenA.plot(kappa_, color="black", linestyle="dotted", linewidth=plt_scale*1)
    scenA.legend([r"$\lambda$", r"$\theta$", r"$\kappa$"], loc="center right")
    scenA.vlines(len(theta_), 0, max(theta_max, kappa_max), color="lightgrey")
    scenA.plot([min(theta_min, kappa_min) for x in theta_], linestyle="dashed", color="red")
    scenA.plot([max(theta_max, kappa_max) for x in theta_], linestyle="dashed", color="red")
    scenA.text(labelx, 0.25, labels[0], transform=scenA.transAxes,
               verticalalignment='top', bbox=props)
    scenA.text(0.45, 0.9, "True CSD", transform=scenA.transAxes,
               verticalalignment='top', horizontalalignment="right",bbox=props)
    scenA.set_ylim([0,max(theta_max, kappa_max)+0.2])
    scenA.set_yticks(range(round(max(theta_max, kappa_max)+1)))
    scenA.plot([lambda_scale_min for x in range(500)], linestyle="dashed", color="red")

    scenB.plot(lambda_scale * lambda_neg, color="black", linewidth=plt_scale*1)
    scenB.plot(theta_, color="black", linestyle="dashdot", linewidth=plt_scale*1)
    scenB.plot(kappa_, color="black", linestyle="dotted", linewidth=plt_scale*1)
    scenB.legend([r"$\lambda$", r"$\theta$", r"$\kappa$"], loc="center right")
    scenB.vlines(len(theta_), 0, max(theta_max, kappa_max), color="lightgrey")
    scenB.plot([min(theta_min, kappa_min) for x in theta_], linestyle="dashed", color="red")
    scenB.plot([max(theta_max, kappa_max) for x in theta_], linestyle="dashed", color="red")
    scenB.text(labelx, 0.25, labels[1], transform=scenB.transAxes,
               verticalalignment='top', bbox=props)
    scenB.text(0.45, 0.9, "No CSD", transform=scenB.transAxes,
               verticalalignment='top', horizontalalignment="right", bbox=props)
    scenB.set_ylim([0, max(theta_max, kappa_max) + 0.2])
    scenB.set_yticks(scenA.get_yticks())
    scenB.set_yticklabels([])
    scenB.plot([lambda_scale_min for x in range(500)], linestyle="dashed", color="red")

    sampA.plot(sample_pos, linewidth=plt_scale*0.5, color="black")
    sampA.axvline(len(theta_), color="lightgrey")
    sampA.set_ylim([-max(abs(np.array(sample_pos))), max(abs(np.array(sample_pos)))])
    sampA.plot([0 for x in range(len(theta_))], color="red")
    for w in range(0, len(theta_) - windowsize, leap):
        sampA.axvline(w, ymin=0.45, ymax=0.55, color="red")
        sampA.axvline(w + windowsize, ymin=0.45, ymax=0.55, color="red")
    sampA.set_xlim([0, n])
    sampA.set_yticks([])
    sampA.text(labelx, 0.28, labels[2], transform=sampA.transAxes,
               verticalalignment='top', bbox=props)
    if(observation_length==1):
        sampA.legend(["$X_t$"],fontsize=plt_scale*6, loc="upper left")
    else:
        sampA.legend(["$X_t$"],fontsize=plt_scale*6, loc="upper right")

    sampB.plot(sample_neg, linewidth=plt_scale*0.5, color="black")
    sampB.axvline(len(theta_), color="lightgrey")
    sampB.set_ylim([-max(abs(np.array(sample_neg))), max(abs(np.array(sample_neg)))])
    sampB.plot([0 for x in range(len(theta_))], color="red")
    for w in range(0, len(theta_) - windowsize, leap):
        sampB.axvline(w, ymin=0.45, ymax=0.55, color="red")
        sampB.axvline(w + windowsize, ymin=0.45, ymax=0.55, color="red")
    sampB.set_xlim([0, n])
    sampB.set_yticks([])
    sampB.text(labelx, 0.28, labels[3], transform=sampB.transAxes,
               verticalalignment='top', bbox=props)
    if(observation_length==1):
        sampB.legend(["$X_t$"],fontsize=plt_scale*6, loc="upper left")
    else:
        sampB.legend(["$X_t$"],fontsize=plt_scale*6, loc="upper right")
    
    estA.plot(range(round(windowsize/2),short_n,leap), [EstimationMethods.calculate_var(sample_pos[j:j+windowsize]) for j in range(0,short_n,leap)], linewidth=plt_scale*1, color="black",linestyle = "dashed")
    estA.plot([0],[0], linewidth=plt_scale*1, color="black",linestyle = "dotted")
    estA2 = estA.twinx()
    estA2.plot(range(round(windowsize/2),short_n,leap), [EstimationMethods.calculate_acor1(sample_pos[j:j+windowsize]) for j in range(0,short_n,leap)], linewidth=plt_scale*1, color="black",linestyle = "dotted")
    estA.axvline(len(theta_), color="lightgrey")
    estA.set_xlim([0, n])
    estA.set_xticks(np.linspace(0,n,5)[:-1])
    estA.set_xticklabels((np.linspace(0,n,5)[:-1]/1000))
    estA.text(0.85, -0.3, "x$10^3$", transform=estA.transAxes,
               verticalalignment='top', bbox=props)
    estA.set_yticks([])
    estA2.set_yticks([])
    estA.text(0.89, 0.28, labels[4], transform=estA.transAxes,
               verticalalignment='top', bbox=props)
    estA.legend(["Var","AC(1)"],fontsize=plt_scale*5.5)
    estA.set_xlabel("Time $t$")

    estB.plot(range(round(windowsize/2),short_n,leap), [EstimationMethods.calculate_var(sample_neg[j:j+windowsize]) for j in range(0,short_n,leap)], linewidth=plt_scale*1, color="black",linestyle = "dashed")
    estB.plot([0],[0], linewidth=plt_scale*1, color="black",linestyle = "dotted")
    estB2 = estB.twinx()
    estB2.plot(range(round(windowsize/2),short_n,leap), [EstimationMethods.calculate_acor1(sample_neg[j:j+windowsize]) for j in range(0,short_n,leap)], linewidth=plt_scale*1, color="black",linestyle = "dotted")
    estB.axvline(len(theta_), color="lightgrey")
    estA.set_xticks(np.linspace(0,n,5)[:-1])
    estA.set_xticklabels((np.linspace(0,n,5)[:-1]/1000))
    estB.text(0.85, -0.3, "x$10^3$", transform=estB.transAxes,
               verticalalignment='top', bbox=props)
    estB.set_yticks([])
    estB2.set_yticks([])
    estB.text(0.89, 0.28, labels[5], transform=estB.transAxes,
               verticalalignment='top', bbox=props)
    estB.legend(["Var","AC(1)"],fontsize=plt_scale*5.5)
    estB.set_xlabel("Time $t$")

    fig.subplots_adjust(hspace=.05, wspace=0.05)
    plt.savefig("Plots/example" + time.strftime("%Y%m%d-%H%M%S") + ".pdf", dpi = 300, bbox_inches='tight')
    plt.show()


def plot_slices(window_index,obslen_index):
    number_of_windows = tpr_fpr_auc.number_of_windows
    windowsizes = tpr_fpr_auc.windowsizes
    observation_lengths = tpr_fpr_auc.observation_lengths

    #number_of_windows = 20
    #windowsizes = [200,350,500,700,900,1100,1300,1500]
    #observation_lengths = [0.3,0.4,0.5,0.6,0.7,0.8,0.9,1]
    
    fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(plt_scale*3.42,plt_scale*1.5),sharey=True)
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
    
    axs[0].set_title("Length: " + str(windowsizes[window_index]*number_of_windows), fontsize=15)
    axs[0].set_xlabel("Fraction of CSD observed\n in estimations [$\%$]")
    axs[0].set_xticks([20,40,60,80,100])
    axs[0].set_ylabel("AUC")
    axs[0].set_yticks([0.5,0.6,0.7,0.8,0.9,1])
    for method in range(5):
        axs[0].plot(np.round(dfs[method].columns,0).astype(int),dfs[method].iloc[window_index,:],c=cols[method])
    axs[0].text(0.055, 0.94, labels[0], transform=axs[0].transAxes,
                            verticalalignment='top', bbox=props)
    axs[0].grid()

    axs[1].set_title("Observed CSD: " + str(round(observation_lengths[obslen_index]*100)) + "$\%$", fontsize=15)
    axs[1].set_xlabel("Length of the time series")
    for method in range(5):
        axs[1].plot(np.round(dfs[method].index/1000,0).astype(int),dfs[method].iloc[:,obslen_index],c=cols[method], label=" ")
    handles, legendlabels = axs[1].get_legend_handles_labels()
    legendlabels = ["Var", "AC(1)",r"$\varphi$", "$\lambda^{\mathrm{(ACS)}}$", "$\lambda^{\mathrm{(PSD)}}$"]
    order = [0,3,1,4,2]
    axs[1].legend([handles[idx] for idx in order],[legendlabels[idx] for idx in order],loc="upper right", bbox_to_anchor=(1,0.42), framealpha=0.7,ncol=3, columnspacing=plt_scale*0.1, fontsize=plt_scale*5.5)
    axs[1].text(0.025, 0.94, labels[1], transform=axs[1].transAxes,
                            verticalalignment='top', bbox=props)
    axs[1].text(1.02, 0.08, "x$10^3$", transform=axs[1].transAxes,
                            verticalalignment='top', bbox=props,fontsize = 14)
    axs[1].set_xticks([5,10,15,20,25,30])
    axs[1].grid()
    fig.subplots_adjust(hspace=.15, wspace=.05)
    plt.savefig("Plots/slices" + time.strftime("%Y%m%d-%H%M%S") + ".pdf", dpi = 300, bbox_inches='tight')
    plt.show()




def plot_heat_auc(examples = True, slices = True):
    cmap = "winter"
    number_of_windows = tpr_fpr_auc.number_of_windows
    windowsizes = tpr_fpr_auc.windowsizes
    observation_lengths = tpr_fpr_auc.observation_lengths

    #number_of_windows = 20
    #windowsizes = [200,350,500,700,900,1100,1300,1500]
    #observation_lengths = [0.3,0.4,0.5,0.6,0.7,0.8,0.9,1]
    #method_names = ["Variance estimator", "AC(1) estimator", r"$\varphi$ estimator", "$\lambda$ via ACS", "$\lambda$ via PSD"]
    method_names = ["Var", "AC(1)",r"$\varphi$", "$\lambda^{\mathrm{(ACS)}}$", "$\lambda^{\mathrm{(PSD)}}$"]
    
    
    fig, axs = plt.subplots(nrows=2, ncols=3, figsize=(plt_scale*3.42,plt_scale*2.7))
    axs = axs.flatten()
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
        auc_df.index = np.round(number_of_windows * auc_df.index /1000).astype(int)
        auc_df = auc_df.iloc[::-1]
        xticks = False
        yticks = False
        if method in [0,3]:
            yticks = True
            axs[method].text(0.02, 1.09, "x$10^3$", transform=axs[method].transAxes,
                            verticalalignment='top', bbox=props,fontsize = 14)
        if method in [3,4]:
            xticks = True
        sns.heatmap(auc_df, ax=axs[method], xticklabels=xticks, yticklabels=yticks, cbar=False, vmin=0.5, vmax=1, cmap=cmap)
        if xticks:
            axs[method].set_xticks([0.5,2.5,4.5,6.5,8.5])
        if slices: 
            axs[method].add_patch(Rectangle((0,4),9,1,fill=False,edgecolor="black", lw=plt_scale*1, linestyle="dashed"))
            axs[method].add_patch(Rectangle((4,0),1,9,fill=False,edgecolor="black", lw=plt_scale*1, linestyle="dotted"))
        if examples:
            axs[method].add_patch(Rectangle((8,4),1,1,fill=False,edgecolor="red", lw=plt_scale*1))
            axs[method].add_patch(Rectangle((4,4),1,1,fill=False,edgecolor="pink", lw=plt_scale*1))
        axs[method].set_title(method_names[method],fontsize=plt_scale*10)
        axs[method].set_xlim([-0.1,9.1])
        axs[method].set_ylim([9.1,-0.1])
        axs[method].set_aspect("equal", adjustable='box')
    fig.text(0.5,0, "Fraction of CSD observed in estimations [\%]", ha="center", va="center")
    fig.text(0.05,0.5, "Length of the time series", ha="center", va="center", rotation = 90)
    fig.colorbar(axs[4].collections[0], ax=axs[5], location="right", fraction=0.9, shrink=0.85, aspect=5)
    axs[0].set_yticklabels(axs[0].get_yticklabels(), rotation = 0)
    axs[3].set_yticklabels(axs[3].get_yticklabels(), rotation = 0)
    axs[4].collections[0].colorbar.set_label("AUC",labelpad=15)
    axs[4].collections[0].colorbar.set_ticks([0.5,0.6,0.7,0.8,0.9,1])
    fig.subplots_adjust(hspace=.15, wspace=.05)
    axs[5].axis("off")
    plt.savefig("Plots/heat_" + time.strftime("%Y%m%d-%H%M%S") + ".pdf", dpi = 300, bbox_inches='tight')
    plt.show()

