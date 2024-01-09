#!/usr/bin/env python3

from netCDF4 import Dataset
import glob
import os
import matplotlib.pyplot as plt
import numpy as np

output_loc = 'OUTPUT'

g = glob.glob(os.path.join(output_loc,'*.nc'))

fig,ax = plt.subplots(1,1)
val = []
for i in g:
    print(i)
    c = Dataset(i,'r')
    time = np.array(c['month_time'])
    co2 = np.array(c['fluxengine_OF_in_areaday_cumulative'])
    co2_o = np.array(c['fluxengine_OF_out_areaday_cumulative'])

    co2_p = (co2-co2_o)/co2_o
    val.append(co2_p[-1]*100)
    ax.plot(time,co2_p)
ax.set_ylim([-1,1])
print(val)
print(np.nanmedian(val))
fig,ax = plt.subplots(1,1)
ax.boxplot(val)
ax.plot([0.5,1.5],[0,0])
ax.set_ylim([-20,20])
plt.show()
