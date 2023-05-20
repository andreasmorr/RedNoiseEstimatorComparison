from netCDF4 import Dataset
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import math

descriptions = ["precip_mm_srf", "fracPFTs_mm_srf", "sm_mm_soil", "temp_mm_1_5m"]
for i in [0,1]:
    data = Dataset("Holocene data/xokm." + descriptions[i] + ".monthly.nc")
    print(data.variables)
    lats = data.variables['latitude'][:]
    lons = data.variables['longitude'][:]
    month_ind = [float(j) / 12 for j in range(-120000, 0)]
    year_ind = range(-12000,0)
    season = range(12)
    time_values = np.zeros(len(ind))
    for k in range(24,29):
        for l in range(-4,2):
            if not ((k == 24 and (l == -4 or l == -3)) or (k == 25 and l == -4) or (k == 26 and l == -4)):
                mask_count = 0
                if i == 0:
                    this_data = data.variables[descriptions[i]][:, k, l]
                else:
                    this_data = data.variables[descriptions[i]][:, 0, k, l] #precip
                for j in range(this_data.size):
                    if np.ma.getmask(this_data)[j]:
                        this_data[j] = this_data[j - 12]
                        mask_count += 1
                time_values += this_data.data
                if i==0:
                    #pd.Series(1-this_data.data, ind).to_csv("Cell Data/west_data_" + descriptions[i] + "lat" + str(k) + "lon" + str(l))
                    pd.Series(1-np.array([sum([this_data.data[12 * i + j] for j in season]) / len(season) for i in range(math.floor(len(this_data.data) / 12))]), year_ind).to_csv("Processed/Cell Data/seasonal_west_data_" + descriptions[i] + "lat" + str(k) + "lon" + str(l))
                else:
                    #pd.Series(this_data.data, ind).to_csv("Cell Data/west_data_" + descriptions[i] + "lat" + str(k) + "lon" + str(l))
                    pd.Series(np.array([sum([this_data.data[12 * i + j] for j in season])/len(season) for i in range(math.floor(len(this_data.data)/12))]), year_ind).to_csv("Processed/Cell Data/seasonal_west_data_" + descriptions[i] + "lat" + str(k) + "lon" + str(l))
                print(mask_count)
    time_values = time_values/26
    if i == 1:
        time_values = 1 - time_values #noprecip
    time_series = pd.Series(time_values, month_ind)
    time_series.to_csv("Processed/west_data_" + descriptions[i])
    seasonal_values = [sum([time_values[12*i + j] for j in season])/len(season) for i in year_ind]
    seasonal_series = pd.Series(seasonal_values, year_ind)
    seasonal_series.to_csv("Processed/seasonal_west_data_" + descriptions[i])