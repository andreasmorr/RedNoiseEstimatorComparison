from scipy import fft, optimize, ndimage
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import SampleGeneration
import EstimationMethods
import WindowEstimation
import MethodComparisons
import importlib
importlib.reload(SampleGeneration)
importlib.reload(EstimationMethods)
importlib.reload(WindowEstimation)
importlib.reload(MethodComparisons)

lats = np.array(pd.read_csv("Processed/lats").iloc[:,1])
lons = np.array(pd.read_csv("Processed/lons").iloc[:,1])
skip_start = 50
year_ind = range(-10000+skip_start,0)
p_factor = 1e7
window = 500
leap = 100



def plot_all_cells():
    for file in os.listdir("Processed/Cell Data"):
        if file[0] == "l":
            k = int(file[3:5])
            l = int(file[9:])
            timeseries_p = pd.read_csv("Processed/Cell Data/"+file, index_col=0).loc[:,"p"].values[skip_start:]*p_factor
            timeseries_v = pd.read_csv("Processed/Cell Data/"+file, index_col=0).loc[:,"v"].values[skip_start:]
            filt_p = ndimage.gaussian_filter1d(timeseries_p,300)
            filt_v = ndimage.gaussian_filter1d(timeseries_v,300)
            d1 = np.diff(filt_v)
            d1 = np.append(d1,d1[-1])
            d2 = np.diff(d1)
            d2 = np.append(d2,d2[-1])
            curvature = d2/(1+d1**2)**(3/2)
            tippp = np.argmin(curvature)-10000

            leap_ind = range(-10000 + window,tippp,leap)

            var_v = WindowEstimation.moving_window(timeseries_v[:tippp]-filt_v[:tippp],"var",500)
            ac1_v = WindowEstimation.moving_window(timeseries_v[:tippp]-filt_v[:tippp],"ac1",500)
            [lambda_v,theta_v,sigma_v] = WindowEstimation.moving_window(timeseries_v[:tippp]-filt_v[:tippp],"psd",500,100,return_all_params=True,initial=[1,1,1]).transpose()
            theta_p = WindowEstimation.moving_window(timeseries_p[:tippp]-filt_p[:tippp],"psd",500,100,return_all_params=True,initial=[1,1,1]).transpose()[0]

            fig, axs = plt.subplots(nrows=5, ncols=1,figsize=(8, 14),
                                                            gridspec_kw={'height_ratios': [1, 1, 1, 1, 1]}, 
                                                            sharex=True)
            v_color = "darkgreen"
            p_color = "darkblue"
            theocolor = "red"
            samplelinewidth = 0.5
            theolinewidth = 2
            props = dict(edgecolor="none", facecolor='white', alpha=0)
            axs[0].plot(year_ind,timeseries_v, color = v_color, linewidth = samplelinewidth)
            axs[0].plot(year_ind,filt_v, color = theocolor, linewidth = theolinewidth)
            axs[0].axvline(tippp, color = "grey")
            axs[0].set_ylabel("Vegetation fraction $V$")
            axs[0].text(-0.15, 0.95, "a)", transform=axs[0].transAxes, fontsize=15, verticalalignment='top', bbox=props)
            axs[1].plot(year_ind,timeseries_p, color = p_color, linewidth = samplelinewidth)
            axs[1].plot(year_ind,filt_p, color = theocolor, linewidth = theolinewidth)
            axs[1].axvline(tippp, color = "grey")
            axs[1].set_ylabel("Precipitation $P$")
            axs[1].text(-0.15, 0.95, "b)", transform=axs[1].transAxes, fontsize=15, verticalalignment='top', bbox=props)
            axs[2].plot(year_ind[window-1:tippp][:len(var_v)],var_v, color = v_color, linestyle= "dashed")
            axs[2].axvline(tippp, color = "grey")
            axs[2].set_ylabel("Variance of $V$")
            ac1plt = axs[2].twinx()
            ac1plt.plot(year_ind[window-1:tippp][:len(ac1_v)],ac1_v, color = v_color, linestyle= "dotted")
            ac1plt.set_ylabel("AC(1) of $V$")
            axs[2].text(-0.15, 0.95, "c)", transform=axs[2].transAxes, fontsize=15, verticalalignment='top', bbox=props)
            axs[3].scatter(leap_ind[:len(lambda_v)],lambda_v, color = v_color)
            axs[3].axvline(tippp, color = "grey")
            axs[3].set_ylabel("Vegetation stability")
            axs[3].text(-0.15, 0.95, "d)", transform=axs[3].transAxes, fontsize=15, verticalalignment='top', bbox=props)
            axs[4].scatter(leap_ind[:len(theta_v)],theta_v, color = v_color)
            axs[4].scatter(leap_ind[:len(theta_p)],theta_p, color = p_color, marker="D")
            axs[4].axvline(tippp, color = "grey")
            axs[4].text(-0.15, 0.95, "e)", transform=axs[4].transAxes, fontsize=15, verticalalignment='top', bbox=props)
            axs[4].set_ylabel("Precipitation correlation")
            axs[4].set_xlabel("Years before present")
            fig.suptitle("Cell with coordinates " + str(round(lats[k],1)) + "$^\circ$ lat and " + str(round(lons[l],1)) + "$^\circ$ lat")
            plt.savefig("Plots/Assessment_lat" + str(k) + "_lon" + str(l), dpi = 300, bbox_inches='tight')
            #plt.show()
            plt.close()

plot_all_cells()
