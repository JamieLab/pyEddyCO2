#!/usr/bin/env python3

from netCDF4 import Dataset
import glob
import os
import matplotlib.pyplot as plt
import numpy as np
import pyEddy_main as pyEddy_m
import pyEddy_plot as eplt
import datetime
import matplotlib.transforms
import cmocean
font = {'weight' : 'normal',
        'size'   :12}
matplotlib.rc('font', **font)

output_loc = 'F:/eddy/anticyclonic'

no_load = ['cost_association','speed_contour_height','speed_contour_latitude','speed_contour_longitude','speed_contour_shape_error',
    'speed_average','speed_area','inner_contour_height','latitude_max','longitude_max','num_point_s','uavg_profile']
# File where our eddy data is located
file = 'F:/Data/AVISO_EDDIES/META3.2_DT_allsat_Anticyclonic_long_19930101_20220209.nc'
#file = 'D:/Data/AVISO_EDDIES/META3.2_DT_allsat_Cyclonic_long_19930101_20220209.nc'
# Loading the eddy netcdf data into a dictionary that corresponds to the netcdf file variable names.
# This will not load anything that is defined in "no_load". If you want to load all variables leave
# no_load empty (i.e [])
eddy_v,desc = pyEddy_m.AVISO_load(file,no_load)
print(desc)
# # Here we split the eddy dict down to a set spatial region, and temporal window. This will
# # include all eddies even if they appear in the domain for 1 day. We can perform checks later that
# # they remain in the domain for a period. If you want all the data, then you don't need to use
# # this function.
#
# eddy_v = pyEddy_m.box_split(eddy_v,[-34,-30],[5,25],[datetime.datetime(2002,7,1),datetime.datetime(2018,12,31)],strict_time=True)
eddy_v = pyEddy_m.box_split(eddy_v,[-60,60],[-180,180],[datetime.datetime(1998,1,1),datetime.datetime(2022,12,31)],strict_time=False)
# eddy_v = pyEddy_m.box_split(eddy_v,[-35,-15],[5,25],[datetime.datetime(2002,7,1),datetime.datetime(2018,12,31)],strict_time=True)
print(len(np.unique(eddy_v['track'])))
#
eddy_v = pyEddy_m.eddy_length_min(eddy_v,365)
print(len(np.unique(eddy_v['track'])))
#
# eddy_v = pyEddy_m.eddy_cross_lon_lat(eddy_v,0)
# print(len(np.unique(eddy_v['track'])))


# # g = glob.glob(os.path.join(output_loc,'*.nc'))
# # #g = np.unique(eddy_v['track'])
fig,ax = plt.subplots(2,2, figsize=(12, 12))
m = eplt.base_tracks_start(ax[0,0],latb=[-80,80],lonb = [-180,180])
m2 = eplt.base_tracks_start(ax[1,1],latb=[-80,80],lonb = [-180,180])
val = []
cum = []
#print(np.unique(eddy_v['track']))

for i in  np.unique(eddy_v['track']):
    print(i)
    if os.path.exists(output_loc+'/'+str(i)+'.nc') == True:
        c = Dataset(output_loc+'/'+str(i)+'.nc','r')
        #c = Dataset(i,'r')
        time = np.array(c['month_time'])
        time2 = []
        for j in range(len(time)):
            time2.append(pyEddy_m.date_con(int(time[j])))
        if time2[0].year >=1998:
            co2 = np.array(c['flux_in_areaday_cumulative'])
            co2_o = np.array(c['flux_out_areaday_cumulative'])
            co2_unc = np.array(c['flux_unc_tot_in_areaday_cumulative'])/2
            #co2_o_unc = np.array(c['flux_unc_total_out_areaday_cumulative'])
            lat = np.array(c['latitude'])
            lon = np.array(c['longitude'])
            x,y = m(lon,lat)


            co2_p = (co2-co2_o)/np.abs(co2_o)
            if np.isnan(co2[-1]-co2_o[-1]) == 0:
                print(co2_p[-1]*100)
                print(co2[-1])
                print(co2_o[-1])
                print(co2[-1]-co2_o[-1])
                cum.append(co2[-1]-co2_o[-1])
                val.append(co2_p[-1]*100)
                if np.sign(co2_p[-1]) == 1:
                    col = '#D41159'
                else:
                    col = '#1A85FF'

                p = ax[0,1].plot(time2,co2,col)

                ax[0,0].scatter(x,y,s=1,color=col)
                ax[0,1].fill_between(time2,co2-co2_unc,co2+co2_unc,color=col,alpha=0.4)
                ax[1,1].scatter(x[0],y[0],s=3,color=col)
        c.close()
val = np.array(val)
cum = np.array(cum)
eplt.base_tracks_end(ax[0],m)
eplt.base_tracks_end(ax[1,1],m2)
ax[0,1].set_ylim([-4,4])
ax[0,1].set_ylabel('Net Eddy CO${_2}$ flux (Tg C)')
ax[0,1].set_xlabel('Year')
print(val)
print(np.nanmedian(val))
print(np.nansum(cum))
print(np.nanmedian(cum))
ax[1,0].boxplot(val[np.isnan(val) == False])
ax[1,0].plot([0.5,1.5],[0,0])
ax[1,0].set_ylim([-40,40])
#ax.set_title('N='+str(len(g)))
ax[1,0].set_ylabel('Change in eddy CO${_2}$ flux \n compared to the surrounding \n water CO${_2}$ flux (%)')
ax[1,0].set_xticks(ticks=[1],labels =['(N = '+str(len(val[np.isnan(val) == False])) + ')'])
#ax[1,0].set_xticks(ticks=[1],labels =['(N = '+str(len(g)) + ')'])
fig.tight_layout()
fig.savefig('figs/anti_new_global_eddy_flux.png',dpi=300)

"""
"""
fig2,ax = plt.subplots(1,1, figsize=(16, 6))

for i in np.unique(eddy_v['track']):
    print(i)
    c = Dataset(output_loc+'/'+str(i)+'.nc','r')
    #c = Dataset(i,'r')
    time = np.array(c['month_time'])
    time2 = []
    for j in range(len(time)):
        time2.append(pyEddy_m.date_con(int(time[j])))
    if time2[0].year >=1998:
        co2 = np.array(c['flux_in_areaday_cumulative'])
        co2_o = np.array(c['flux_out_areaday_cumulative'])
        co2_unc = np.array(c['flux_unc_tot_in_areaday_cumulative'])
        #co2_o_unc = np.array(c['flux_unc_total_out_areaday_cumulative'])
        lat = np.array(c['latitude'])
        lon = np.array(c['longitude'])
        #x,y = m(lon,lat)


        co2_p = (co2-co2_o)/co2_o
        #val.append(co2_p[-1]*100)
        col = 'k'
        p = ax.plot(time2,co2,col)


        #ax.plot(x,y,color=a)
        #ax.fill_between(time2,co2-co2_unc,co2+co2_unc,color=col,alpha=0.4)
    c.close()
ax.set_ylim([-4,4])
ax.set_ylabel('Net Eddy CO${_2}$ flux (Tg C)')
ax.set_xlabel('Year')
fig2.savefig('figs/anti_global_eddy_flux_test_netfluxexpand.png',dpi=300)

"""
"""
fig,ax = plt.subplots(1,1, figsize=(14,10))
m = eplt.base_tracks_start(ax,latb=[-80,80],lonb = [-180,180])
# m2 = eplt.base_tracks_start(ax[1,1],latb=[-80,80],lonb = [-180,180])
val = []
cum = []
#print(np.unique(eddy_v['track']))

for i in  np.unique(eddy_v['track']):
    print(i)
    if os.path.exists(output_loc+'/'+str(i)+'.nc') == True:
        c = Dataset(output_loc+'/'+str(i)+'.nc','r')
        #c = Dataset(i,'r')
        time = np.array(c['month_time'])
        time2 = []
        for j in range(len(time)):
            time2.append(pyEddy_m.date_con(int(time[j])))
        if time2[0].year >=1998:
            co2 = np.array(c['flux_in_areaday_cumulative'])
            co2_o = np.array(c['flux_out_areaday_cumulative'])
            co2_unc = np.array(c['flux_unc_tot_in_areaday_cumulative'])/2
            #co2_o_unc = np.array(c['flux_unc_total_out_areaday_cumulative'])
            lat = np.array(c['month_latitude'])
            lon = np.array(c['month_longitude'])
            x,y = m(lon,lat)

            a = ax.scatter(x[0],y[0],s=6,c = co2[-1],vmin = -1,vmax=1,cmap=cmocean.cm.balance)
        c.close()
c2 = plt.colorbar(a)
c2.set_label('Cumulative net CO$_{2}$ flux (Tg C)')
eplt.base_tracks_end(ax,m)
fig.savefig('figs/anti_global_eddy_flux_uptake.png',dpi=300)
