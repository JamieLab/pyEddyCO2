#!/usr/bin/env python3

from netCDF4 import Dataset
import glob
import os
import matplotlib.pyplot as plt
import numpy as np
import pyEddy_main as pyEddy_m
import pyEddy_plot as eplt

output_loc = 'OUTPUT'

g = glob.glob(os.path.join(output_loc,'*.nc'))
fig,ax = plt.subplots(2,2)
m = eplt.base_tracks_start(ax[0,0],latb=[-60,10],lonb = [-75,40])
val = []
for i in g:
    print(i)
    c = Dataset(i,'r')
    time = np.array(c['month_time'])
    time2 = []
    for j in range(len(time)):
        time2.append(pyEddy_m.date_con(int(time[j])))
    co2 = np.array(c['fluxengine_OF_in_areaday_cumulative'])
    co2_o = np.array(c['fluxengine_OF_out_areaday_cumulative'])
    lat = np.array(c['latitude'])
    lon = np.array(c['longitude'])
    x,y = m(lon,lat)
    ax[0,0].plot(x,y)

    co2_p = (co2-co2_o)/co2_o
    val.append(co2_p[-1]*100)
    ax[0,1].plot(time2,co2)
eplt.base_tracks_end(ax[0],m)
ax[0,1].set_ylim([-3,0.5])
ax[0,1].set_ylabel('Net Eddy CO${_2}$ flux (Tg C)')
ax[0,1].set_xlabel('Year')
print(val)
print(np.nanmedian(val))
ax[1,0].boxplot(val)
ax[1,0].plot([0.5,1.5],[0,0])
ax[1,0].set_ylim([-20,20])
#ax.set_title('N='+str(len(g)))
ax[1,0].set_ylabel('Change in eddy CO${_2}$ flux compared to the surrounding \n water CO${_2}$ flux (%)')
ax[1,0].set_xticks(ticks=[1],labels =[f'Anticylonic (N = {len(g)})'])
fig.tight_layout()
plt.show()
