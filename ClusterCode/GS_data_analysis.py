from scipy import fft, optimize, ndimage
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import time
import SampleGeneration
import EstimationMethods
import WindowEstimation
import MethodComparisons
import importlib
importlib.reload(SampleGeneration)
importlib.reload(EstimationMethods)
importlib.reload(WindowEstimation)
importlib.reload(MethodComparisons)

plt_scale=2
plt.rcParams['pgf.texsystem'] = 'pdflatex'
plt.rcParams.update({'font.family': 'serif',
                     'text.usetex': True,
                     'font.serif': ['Helvetica'],
                     'font.size': plt_scale*8,
                     'axes.labelsize': plt_scale*10,
                     'axes.titlesize': plt_scale*12,
                     'figure.titlesize': plt_scale*14})


lats = np.array(pd.read_csv("Processed/lats").iloc[:,1])
lons = np.array(pd.read_csv("Processed/lons").iloc[:,1])
skip_start = 50
year_ind = range(-10000+skip_start,0)
p_factor = 1e5
window = 500
leap = 300

filter_length = 200



def plot_all_cells():
    for file in os.listdir("Processed/Cell Data"):
        if file[0] == "l":
            plot_cell(file)
            plt.close()

def plot_cell(file):
    k = int(file[3:5])
    l = int(file[9:])
    timeseries_p = pd.read_csv("Processed/Cell Data/"+file, index_col=0).loc[:,"p"].values[skip_start:]*p_factor
    timeseries_v = pd.read_csv("Processed/Cell Data/"+file, index_col=0).loc[:,"v"].values[skip_start:]
    filt_p = ndimage.gaussian_filter1d(timeseries_p,filter_length)
    filt_v = ndimage.gaussian_filter1d(timeseries_v,filter_length)
    d1 = np.diff(1000*filt_v)
    d1 = np.append(d1,d1[-1])
    d2 = np.diff(1000*d1)
    d2 = np.append(d2,d2[-1])
    curvature = np.convolve(d2/(1+d1**2)**(3/2),np.ones(100),mode="same")
    tippp = np.argmin(curvature[:5000])-10000

    leap_ind = range(-10000 + window,tippp,leap)

    var_v = WindowEstimation.moving_window(timeseries_v[:tippp]-filt_v[:tippp],"var",window)[::100]
    ac1_v = WindowEstimation.moving_window(timeseries_v[:tippp]-filt_v[:tippp],"ac1",window)[::100]
    
    [psd_lambda_v,psd_theta_v,psd_kappa_v] = WindowEstimation.moving_window(timeseries_v[:tippp]-filt_v[:tippp],"psd",window,leap,return_all_params=True,initial=[1,1,1]).transpose()
    psd_theta_p = WindowEstimation.moving_window(timeseries_p[:tippp]-filt_p[:tippp],"psd",window,leap,return_all_params=True,initial=[1,1,1]).transpose()[0]
    [acs_lambda_v,acs_theta_v] = WindowEstimation.moving_window(timeseries_v[:tippp]-filt_v[:tippp],"acs",window,leap,return_all_params=True,initial=[1,1],relevant_lags=3).transpose()
    acs_theta_p = WindowEstimation.moving_window(timeseries_p[:tippp]-filt_p[:tippp],"acs",window,leap,return_all_params=True,initial=[1,1],relevant_lags=3).transpose()[0]
    
    fig, axs = plt.subplots(nrows=5, ncols=1,figsize=(plt_scale*3.42, plt_scale*5),
                                                    gridspec_kw={'height_ratios': [1, 1, 1, 1, 1]}, 
                                                    sharex=True)
    v_color = "darkgreen"
    p_color = "darkblue"
    theocolor = "red"
    samplelinewidth = plt_scale*0.5
    theolinewidth = plt_scale*1
    v_marker = "D"
    p_marker = "s"
    marker_size = plt_scale*12
    props = dict(edgecolor="none", facecolor='white', alpha=0)
    axs[0].plot(year_ind,timeseries_v, color = v_color, linewidth = samplelinewidth)
    axs[0].plot(year_ind,filt_v, color = theocolor, linewidth = theolinewidth)
    axs[0].axvline(tippp, color = "grey")
    axs[0].set_ylabel("Vegetation fraction")
    axs[0].text(0.013, 0.17, "(a)", transform=axs[0].transAxes, verticalalignment='top', bbox=props)
    axs[0].legend(["Vegetation $V_t$",r"Equil. vegetation $\bar V(t)$"], loc = "upper right",fontsize=14)
    axs[1].plot(year_ind,timeseries_p, color = p_color, linewidth = samplelinewidth)
    axs[1].plot(year_ind,filt_p, color = theocolor, linewidth = theolinewidth)
    axs[1].axvline(tippp, color = "grey")
    axs[1].set_ylabel("Precipitation")
    axs[1].text(0.013, 0.17, "(b)", transform=axs[1].transAxes, verticalalignment='top', bbox=props)
    axs[1].legend(["Precipitation $P_t$",r"Equil. precipitation $\bar P(t)$"], loc = "upper right",fontsize=14)
    axs[2].plot(range(-10000 + window,tippp,100)[:len(var_v)],var_v, color = v_color, linestyle= "dashed",linewidth=theolinewidth)
    axs[2].plot([0],[0],linestyle="dotted", color = v_color)
    axs[2].axvline(tippp, color = "grey")
    axs[2].set_ylabel("Variance")
    ac1plt = axs[2].twinx()
    ac1plt.plot(range(-10000 + window,tippp,100)[:len(ac1_v)],ac1_v, color = v_color, linestyle= "dotted",linewidth=theolinewidth)
    ac1plt.set_ylabel("AC(1)")
    axs[2].text(0.013, 0.17, "(c)", transform=axs[2].transAxes, verticalalignment='top', bbox=props)
    axs[2].legend([r"Variance of $V_t-\bar V(t)$",r"AC(1) of $V_t-\bar V(t)$"], loc = "upper right",fontsize=14)
    axs[3].scatter(leap_ind[:len(acs_lambda_v)],acs_lambda_v, color = v_color,s=marker_size, marker = v_marker)
    axs[3].scatter(leap_ind[:len(psd_lambda_v)],psd_lambda_v, color = v_color,s=marker_size, facecolors='none', marker = v_marker)
    axs[3].axvline(tippp, color = "grey")
    axs[3].set_ylabel("Stability of $V$")
    axs[3].yaxis.set_label_coords(-0.1,.5)
    axs[3].text(0.015, 0.17, "(d)", transform=axs[3].transAxes, verticalalignment='top', bbox=props)
    axs[3].legend([r"$\lambda^\mathrm{(ACS)}$ on $V_t-\bar V(t)$",r"$\lambda^\mathrm{(PSD)}$ on $V_t-\bar V(t)$"], loc = "upper right",fontsize=14)
    axs[4].scatter(leap_ind[:len(acs_theta_v)],acs_theta_v, color = v_color,s=marker_size, marker = v_marker)
    axs[4].scatter(leap_ind[:len(acs_theta_p)],acs_theta_p, color = p_color,s=marker_size, marker = p_marker)
    axs[4].scatter(leap_ind[:len(psd_theta_v)],psd_theta_v, color = v_color,s=marker_size, facecolors='none', marker = v_marker)
    axs[4].scatter(leap_ind[:len(psd_theta_p)],psd_theta_p, color = p_color,s=marker_size, facecolors='none', marker = p_marker)
    axs[4].axvline(tippp, color = "grey")
    axs[4].set_ylabel("Correlation of $P$")
    axs[4].text(0.013, 0.17, "(e)", transform=axs[4].transAxes, verticalalignment='top', bbox=props)
    axs[4].legend([r"$\theta^\mathrm{(ACS)}$ on $V_t-\bar V(t)$",r"$\theta^\mathrm{(ACS)}$ on $P_t-\bar P(t)$",r"$\theta^\mathrm{(PSD)}$ on $V_t-\bar V(t)$",r"$\theta^\mathrm{(PSD)}$ on $P_t-\bar P(t)$"], loc = "upper right",fontsize=14)
    axs[4].set_xlabel("Years before present")
    fig.text(0.5,0.9, "Cell with coordinates " + str(round(lats[k],1)) + "$^\circ$lat and " + str(round(lons[l],2)) + "$^\circ$lon", ha="center", va="center")

    fig.subplots_adjust(hspace=.05, wspace=0.05)
    plt.savefig("Plots/Assessment_lat" + str(k) + "_lon" + str(l) + ".pdf", dpi = 300, bbox_inches='tight')
    plt.show()

### Only for the purpose of analysing the relative runtimes of the estimation methods
def plot_cell_time(file):
    k = int(file[3:5])
    l = int(file[9:])
    timeseries_p = pd.read_csv("Processed/Cell Data/"+file, index_col=0).loc[:,"p"].values[skip_start:]*p_factor
    timeseries_v = pd.read_csv("Processed/Cell Data/"+file, index_col=0).loc[:,"v"].values[skip_start:]
    filt_p = ndimage.gaussian_filter1d(timeseries_p,filter_length)
    filt_v = ndimage.gaussian_filter1d(timeseries_v,filter_length)
    d1 = np.diff(1000*filt_v)
    d1 = np.append(d1,d1[-1])
    d2 = np.diff(1000*d1)
    d2 = np.append(d2,d2[-1])
    curvature = np.convolve(d2/(1+d1**2)**(3/2),np.ones(100),mode="same")
    tippp = np.argmin(curvature[:5000])-10000

    leap_ind = range(-10000 + window,tippp,leap)



    print("Time elapsed for each estimation routine:")
    start_all = time.time()
    start = time.time()
    var_v = WindowEstimation.moving_window(timeseries_v[:tippp]-filt_v[:tippp],"var",window)[::100]
    end = time.time()
    print("Variance V total: " + str(end-start) + "s")
    print("Variance V per window: " + str((end-start)/len(var_v)/100) + "s")

    start = time.time()
    ac1_v = WindowEstimation.moving_window(timeseries_v[:tippp]-filt_v[:tippp],"ac1",window)[::100]
    end = time.time()
    print("AC(1) V total: " + str(end-start) + "s")
    print("AC(1) V per window: " + str((end-start)/len(ac1_v)/100) + "s")
    
    start = time.time()
    [psd_lambda_v,psd_theta_v,psd_kappa_v] = WindowEstimation.moving_window(timeseries_v[:tippp]-filt_v[:tippp],"psd",window,leap,return_all_params=True,initial=[1,1,1]).transpose()
    end = time.time()
    print("PSD V total: " + str(end-start) + "s")
    print("PSD V per window: " + str((end-start)/len(psd_lambda_v)) + "s")
    
    start = time.time()
    psd_theta_p = WindowEstimation.moving_window(timeseries_p[:tippp]-filt_p[:tippp],"psd",window,leap,return_all_params=True,initial=[1,1,1]).transpose()[0]
    end = time.time()
    print("PSD P total: " + str(end-start) + "s")
    print("PSD P per window: " + str((end-start)/len(psd_theta_p)) + "s")
    
    start = time.time()
    [acs_lambda_v,acs_theta_v] = WindowEstimation.moving_window(timeseries_v[:tippp]-filt_v[:tippp],"acs",window,leap,return_all_params=True,initial=[1,1],relevant_lags=3).transpose()
    end = time.time()
    print("ACS V total: " + str(end-start) + "s")
    print("ACS V per window: " + str((end-start)/len(acs_lambda_v)) + "s")
    
    start = time.time()
    acs_theta_p = WindowEstimation.moving_window(timeseries_p[:tippp]-filt_p[:tippp],"acs",window,leap,return_all_params=True,initial=[1,1],relevant_lags=3).transpose()[0]
    end = time.time()
    print("ACS P: " + str(end-start) + "s")
    print("ACS P per window: " + str((end-start)/len(acs_theta_p)) + "s")
    end_all = time.time()
    print("Total time elapsed for the estimation routines: " + str(end_all-start_all) + "s")
    
    fig, axs = plt.subplots(nrows=5, ncols=1,figsize=(plt_scale*3.42, plt_scale*5),
                                                    gridspec_kw={'height_ratios': [1, 1, 1, 1, 1]}, 
                                                    sharex=True)
    v_color = "darkgreen"
    p_color = "darkblue"
    theocolor = "red"
    samplelinewidth = plt_scale*0.5
    theolinewidth = plt_scale*1
    v_marker = "D"
    p_marker = "s"
    marker_size = plt_scale*12
    props = dict(edgecolor="none", facecolor='white', alpha=0)
    axs[0].plot(year_ind,timeseries_v, color = v_color, linewidth = samplelinewidth)
    axs[0].plot(year_ind,filt_v, color = theocolor, linewidth = theolinewidth)
    axs[0].axvline(tippp, color = "grey")
    axs[0].set_ylabel("Vegetation fraction")
    axs[0].text(0.013, 0.17, "(a)", transform=axs[0].transAxes, verticalalignment='top', bbox=props)
    axs[0].legend(["Vegetation $V_t$",r"Equil. vegetation $\bar V(t)$"], loc = "upper right",fontsize=14)
    axs[1].plot(year_ind,timeseries_p, color = p_color, linewidth = samplelinewidth)
    axs[1].plot(year_ind,filt_p, color = theocolor, linewidth = theolinewidth)
    axs[1].axvline(tippp, color = "grey")
    axs[1].set_ylabel("Precipitation")
    axs[1].text(0.013, 0.17, "(b)", transform=axs[1].transAxes, verticalalignment='top', bbox=props)
    axs[1].legend(["Precipitation $P_t$",r"Equil. precipitation $\bar P(t)$"], loc = "upper right",fontsize=14)
    axs[2].plot(range(-10000 + window,tippp,100)[:len(var_v)],var_v, color = v_color, linestyle= "dashed",linewidth=theolinewidth)
    axs[2].plot([0],[0],linestyle="dotted", color = v_color)
    axs[2].axvline(tippp, color = "grey")
    axs[2].set_ylabel("Variance")
    ac1plt = axs[2].twinx()
    ac1plt.plot(range(-10000 + window,tippp,100)[:len(ac1_v)],ac1_v, color = v_color, linestyle= "dotted",linewidth=theolinewidth)
    ac1plt.set_ylabel("AC(1)")
    axs[2].text(0.013, 0.17, "(c)", transform=axs[2].transAxes, verticalalignment='top', bbox=props)
    axs[2].legend([r"Variance of $V_t-\bar V(t)$",r"AC(1) of $V_t-\bar V(t)$"], loc = "upper right",fontsize=14)
    axs[3].scatter(leap_ind[:len(acs_lambda_v)],acs_lambda_v, color = v_color,s=marker_size, marker = v_marker)
    axs[3].scatter(leap_ind[:len(psd_lambda_v)],psd_lambda_v, color = v_color,s=marker_size, facecolors='none', marker = v_marker)
    axs[3].axvline(tippp, color = "grey")
    axs[3].set_ylabel("Stability of $V$")
    axs[3].yaxis.set_label_coords(-0.1,.5)
    axs[3].text(0.015, 0.17, "(d)", transform=axs[3].transAxes, verticalalignment='top', bbox=props)
    axs[3].legend([r"$\lambda^\mathrm{(ACS)}$ on $V_t-\bar V(t)$",r"$\lambda^\mathrm{(PSD)}$ on $V_t-\bar V(t)$"], loc = "upper right",fontsize=14)
    axs[4].scatter(leap_ind[:len(acs_theta_v)],acs_theta_v, color = v_color,s=marker_size, marker = v_marker)
    axs[4].scatter(leap_ind[:len(acs_theta_p)],acs_theta_p, color = p_color,s=marker_size, marker = p_marker)
    axs[4].scatter(leap_ind[:len(psd_theta_v)],psd_theta_v, color = v_color,s=marker_size, facecolors='none', marker = v_marker)
    axs[4].scatter(leap_ind[:len(psd_theta_p)],psd_theta_p, color = p_color,s=marker_size, facecolors='none', marker = p_marker)
    axs[4].axvline(tippp, color = "grey")
    axs[4].set_ylabel("Correlation of $P$")
    axs[4].text(0.013, 0.17, "(e)", transform=axs[4].transAxes, verticalalignment='top', bbox=props)
    axs[4].legend([r"$\theta^\mathrm{(ACS)}$ on $V_t-\bar V(t)$",r"$\theta^\mathrm{(ACS)}$ on $P_t-\bar P(t)$",r"$\theta^\mathrm{(PSD)}$ on $V_t-\bar V(t)$",r"$\theta^\mathrm{(PSD)}$ on $P_t-\bar P(t)$"], loc = "upper right",fontsize=14)
    axs[4].set_xlabel("Years before present")
    fig.text(0.5,0.9, "Cell with coordinates " + str(round(lats[k],1)) + "$^\circ$lat and " + str(round(lons[l],2)) + "$^\circ$lon", ha="center", va="center")

    fig.subplots_adjust(hspace=.05, wspace=0.05)
    #plt.savefig("Plots/Assessment_lat" + str(k) + "_lon" + str(l) + ".pdf", dpi = 300, bbox_inches='tight')
    plt.show()
