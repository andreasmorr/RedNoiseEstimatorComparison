from netCDF4 import Dataset
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import math

cells = [[26,-3],[26,-2],[26,-1],[26,0],[26,1],[27,-4],[27,-3],[27,-2],[27,-1],[27,0],[27,1],[28,-4],[28,-3],[28,-2],[28,-1],[28,0],[28,1]]


month_ind = [float(j) / 12 for j in range(-120000, 0)]
year_ind = range(-10000,0)
season = range(12)


for [k, l] in cells:
    df = pd.DataFrame(0,index = year_ind, columns = ["p","v"])

    description = "precip_mm_srf"
    data = Dataset("Holocene data/xokm." + description + ".monthly.nc")
    lats = data.variables['latitude'][:]
    lons = data.variables['longitude'][:]
    print("Lat: "+str(lats[k])+ ", Lon: " + str(lons[l]))
    mask_count = 0
    this_data = data.variables[description][:, 0, k, l]
    for j in range(this_data.size):
        if np.ma.getmask(this_data)[j]:
            this_data[j] = this_data[j - 12]
            mask_count += 1
    df.loc[:,"p"] = np.array([sum([this_data.data[12 * i + j] for j in season])/len(season) for i in range(math.floor(len(this_data.data)/12))])
    print("Precip mask count: "+str(mask_count))
    description = "fracPFTs_mm_srf"
    data = Dataset("Holocene data/xokm." + description + ".monthly.nc")
    mask_count = 0
    this_data = data.variables[description][:, k, l]
    for j in range(this_data.size):
        if np.ma.getmask(this_data)[j]:
            this_data[j] = this_data[j - 12]
            mask_count += 1
    df.loc[:,"v"] = 1-np.array([sum([this_data.data[12 * i + j] for j in season]) / len(season) for i in range(math.floor(len(this_data.data) / 12))])
    print("Veg mask count: "+str(mask_count))
    df.to_csv("Processed/Cell Data/lat" + str(k) + "_lon" + str(l))

pd.Series(lats).to_csv("Processed/lats")
pd.Series(lons).to_csv("Processed/lons")


