import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import math
import os

plt.rcParams['figure.dpi'] = 200
cumulative = None
for file in os.listdir("Compare fracPFT"):
    if file[0] == "l":
        current = pd.read_csv("Compare fracPFT/"+file, index_col=0).values.flatten()
        if cumulative is None:
            cumulative = current
        else:
            cumulative += current
        #print(file)
        fig, ax = plt.subplots()
        lat = int(file[3:6])/10

        if file[9] == "+":
            lon = int(file[10:])/100
        else:
            lon = -int(file[10:]) / 100
        ax.set_title("Latitude:" + str(lat) + ", Longitude: " + str(lon))
        ax.plot(current,linewidth=0.4)
        plt.savefig("Compare fracPFT/Plots/" + file)
        plt.show()


cumulative = cumulative/(len(os.listdir("Compare fracPFT"))-2)
fig, ax = plt.subplots()
ax.set_title("Average over Western Sahara")
ax.plot(cumulative,linewidth=0.4)
plt.savefig("Compare fracPFT/Plots/cumulative")
plt.show()

