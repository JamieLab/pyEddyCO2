#!/usr/bin/env python3

from netCDF4 import Dataset
import glob
import os
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import numpy as np
import pyEddy_main as pyEddy_m
import pyEddy_plot as eplt
import datetime
import matplotlib.transforms
import cmocean
from matplotlib import patches
import datetime
import weight_stats as ws
import sys
oceaniculoc = 'C:\\Users\\df391\\OneDrive - University of Exeter\\Post_Doc_ESA_Contract\\OceanICU'
sys.path.append(os.path.join(oceaniculoc,'Data_Loading'))
sys.path.append(os.path.join(oceaniculoc))
import data_utils as du

font = {'weight' : 'normal',
        'size'   :12}
matplotlib.rc('font', **font)

output_loc = 'F:/eddy/v0-3/n_anticyclonic'
output_loc_cy = 'F:/eddy/v0-3/n_cyclonic'

no_load = ['cost_association','speed_contour_height','speed_contour_latitude','speed_contour_longitude','speed_contour_shape_error',
    'speed_average','speed_area','inner_contour_height','latitude_max','longitude_max','num_point_s','uavg_profile']
# File where our eddy data is located
anti_file = 'F:/Data/AVISO_EDDIES/META3.2_DT_allsat_Anticyclonic_long_19930101_20220209.nc'
cycl_file = 'F:/Data/AVISO_EDDIES/META3.2_DT_allsat_Cyclonic_long_19930101_20220209.nc'

"""
"""
load = True
hold_all = False
if load:
    # Loading the eddy netcdf data into a dictionary that corresponds to the netcdf file variable names.
    # This will not load anything that is defined in "no_load". If you want to load all variables leave
    # no_load empty (i.e [])
    eddy_an,desc = pyEddy_m.AVISO_load(anti_file,no_load)
    print(desc)
    # # Here we split the eddy dict down to a set spatial region, and temporal window. This will
    # # include all eddies even if they appear in the domain for 1 day. We can perform checks later that
    # # they remain in the domain for a period. If you want all the data, then you don't need to use
    # # this function.
    #
    # eddy_v = pyEddy_m.box_split(eddy_v,[-34,-30],[5,25],[datetime.datetime(2002,7,1),datetime.datetime(2018,12,31)],strict_time=True)
    eddy_an = pyEddy_m.box_split(eddy_an,[-80,80],[-180,180],[datetime.datetime(1993,1,1),datetime.datetime(2022,12,31)],strict_time=False)
    if hold_all:
        eddy_an_all = eddy_an
    # # eddy_v = pyEddy_m.box_split(eddy_v,[-35,-15],[5,25],[datetime.datetime(2002,7,1),datetime.datetime(2018,12,31)],strict_time=True)
    # #
    eddy_an = pyEddy_m.eddy_length_min(eddy_an,365)
    #
    eddy_cy,desc = pyEddy_m.AVISO_load(cycl_file,no_load)
    print(desc)
    # # Here we split the eddy dict down to a set spatial region, and temporal window. This will
    # # include all eddies even if they appear in the domain for 1 day. We can perform checks later that
    # # they remain in the domain for a period. If you want all the data, then you don't need to use
    # # this function.
    #
    # eddy_v = pyEddy_m.box_split(eddy_v,[-34,-30],[5,25],[datetime.datetime(2002,7,1),datetime.datetime(2018,12,31)],strict_time=True)
    eddy_cy = pyEddy_m.box_split(eddy_cy,[-80,80],[-180,180],[datetime.datetime(1993,1,1),datetime.datetime(2022,12,31)],strict_time=False)
    if hold_all:
        eddy_cy_all = eddy_cy
    # eddy_v = pyEddy_m.box_split(eddy_v,[-35,-15],[5,25],[datetime.datetime(2002,7,1),datetime.datetime(2018,12,31)],strict_time=True)
    #
    eddy_cy = pyEddy_m.eddy_length_min(eddy_cy,365)
    # eddy_v = pyEddy_m.eddy_cross_lon_lat(eddy_v,0)
    # print(len(np.unique(eddy_v['track'])))

let = ['a','b','c','d','e','f','g','h','i','j','k','l']
# # g = glob.glob(os.path.join(output_loc,'*.nc'))
# # #g = np.unique(eddy_v['track'])#

plot_figure_1 = False
plot_figure_2 = False
plot_figure_3 = False
plot_figure_3_norm = False
plot_figure_4 =False
plot_figure_4_bio = False
plot_figure_5 = False
plot_figure_5_bio = False
plot_figure_5_compare=False
estimate_cumulative =False
plot_figure_socat = False
plot_figure_socat_bio = False
decadal_stats = False
assess_missing = True
"""
Figure 1
"""
if plot_figure_1:
    ref_time = datetime.datetime(1950,1,1)
    font = {'weight' : 'normal',
            'size'   :20}
    matplotlib.rc('font', **font)
    fig = plt.figure(figsize=(28,16))
    row = 2;col=3
    gs = GridSpec(row,col, figure=fig, wspace=0.25,hspace=0.15,bottom=0.1,top=0.95,left=0.05,right=0.95)
    axs = [[fig.add_subplot(gs[i, j]) for j in range(col)] for i in range(row)]
    flatList = [element for innerList in axs for element in innerList]
    ax = flatList

    c = Dataset(output_loc+'/'+'194465'+'.nc','r')
    lat = np.array(c['latitude'])
    lon = np.array(c['longitude'])
    time = np.array(c['time'])
    m = eplt.base_tracks_start(ax[2],latb=[45,60],lonb = [170,190])
    x,y = m(lon,lat)
    ax[2].scatter(x,y,s=12,c = time,cmap=cmocean.cm.thermal)
    eplt.base_tracks_end(ax[2],m)
    parallels = np.arange(50.,62,5)
    meridion = np.arange(170,191,10)
    m.drawparallels(parallels,labels=[True,False,False,False])
    m.drawmeridians(meridion,labels=[False,False,False,True])
    ax[2].set_xlabel('Longitude',labelpad=30)
    ax[2].set_ylabel('Latitude',labelpad=60)

    mon_time = np.array(c['month_time'])
    time_da = []
    time_mon = []
    for i in range(len(mon_time)):
        time_mon.append(ref_time+datetime.timedelta(days=int(mon_time[i])))
    for i in range(len(time)):
        time_da.append(ref_time+datetime.timedelta(days=int(time[i])))
    time_mon=np.array(time_mon)
    time_da = np.array(time_da)

    sst = np.array(c['cci_sst_in_median'])
    f = np.where(np.isnan(sst) == 0)
    sst= sst[f]
    sst_unc = np.array(c['cci_sst_in_unc_mean'][f])
    ax[0].plot(time_da[f],sst,color='k')
    ax[0].fill_between(time_da[f],sst-sst_unc/2,sst+sst_unc/2,alpha=0.6,color='k')
    ax[0].fill_between(time_da[f],sst-sst_unc,sst+sst_unc,alpha=0.4,color='k')
    sst_month = np.array(c['month_cci_sst_in_median'])
    ax[0].plot(time_mon,sst_month,color='r',linewidth =2)
    ax[0].set_ylabel('Sea Surface Temperature (Kelvin)')


    sss = np.array(c['cmems_so_in_median'])
    f = np.where(np.isnan(sss) == 0)
    sss = sss[f]
    sss_unc = 0.3*2
    ax[1].plot(time_da[f],sss,color='k')
    ax[1].fill_between(time_da[f],sss-sss_unc/2,sss+sss_unc/2,alpha=0.6,color='k')
    ax[1].fill_between(time_da[f],sss-sss_unc,sss+sss_unc,alpha=0.4,color='k')
    sss_month = np.array(c['month_cmems_so_in_median'])
    ax[1].plot(time_mon,sss_month,color='r',linewidth =2)
    ax[1].set_ylabel('Sea Surface Salinity (psu)')

    # ws = np.array(c['ccmp_wind_in_median'])
    # f = np.where(np.isnan(ws) == 0)
    # ws = ws[f]
    # ws_unc = 3.8
    # ax[3].plot(time_da[f],ws,color='k')
    # ax[3].fill_between(time_da[f],ws-ws_unc/2,ws+ws_unc/2,alpha=0.6,color='k')
    # ax[3].fill_between(time_da[f],ws-ws_unc,ws+ws_unc,alpha=0.4,color='k')
    # ax[3].set_ylim([0,25])
    # ws_month = np.array(c['month_ccmp_wind_in_median'])
    # ax[3].plot(time_mon,ws_month,color='b')

    fco2 = np.array(c['month_fco2_sw_in_physics'])
    xco2 = np.array(c['month_xco2'])
    fco2_unc = np.array(c['month_fco2_tot_unc_in_physics'])
    socat_fco2 = np.array(c['socat_mean_fco2'])
    socat_fco2[socat_fco2==0] = np.nan
    ax[3].plot(time_mon,fco2,color='k')
    ax[3].scatter(time_da,socat_fco2,72,color='r',zorder=6)
    ax[3].fill_between(time_mon,fco2-fco2_unc/2,fco2+fco2_unc/2,alpha=0.6,color='k')
    ax[3].fill_between(time_mon,fco2-fco2_unc,fco2+fco2_unc,alpha=0.4,color='k')
    ax[3].plot(time_mon,xco2,'k--')
    ax[3].set_ylabel('fCO$_{2 (sw)}$ or xCO$_{2 (atm)}$ ($\mu$atm or ppm)')
    ax[3].set_ylim([200,550])
    fco2 = np.array(c['flux_in_physics'])
    fco2_unc = np.array(c['flux_unc_in_physics'])*np.abs(fco2)

    ax[4].plot(time_mon,fco2,color='k',zorder=6)
    ax[4].fill_between(time_mon,fco2-fco2_unc/2,fco2+fco2_unc/2,alpha=0.6,color='k')
    ax[4].fill_between(time_mon,fco2-fco2_unc,fco2+fco2_unc,alpha=0.4,color='k')
    ax[4].plot([time_da[0],time_da[-1]],[0,0],'k--')
    ax[4].set_ylim([-0.6,0.6])
    ax[4].set_ylabel('Air-sea CO$_2$ flux (g C m$^{-2}$ d$^{-1}$)\n(-ve indicates atmosphere to ocean exchange)')

    fco2 = np.array(c['flux_in_physics_areaday_cumulative'])
    fco2_unc = np.array(c['flux_unc_tot_in_physics_areaday_cumulative'])

    fco2_out = np.array(c['flux_out_physics_areaday_cumulative'])
    fco2_unc_out = np.array(c['flux_unc_tot_out_physics_areaday_cumulative'])

    ax[5].plot(time_mon,fco2,color='k',zorder=6,linewidth=2)
    ax[5].fill_between(time_mon,fco2-fco2_unc/2,fco2+fco2_unc/2,alpha=0.6,color='k',zorder=5)
    ax[5].fill_between(time_mon,fco2-fco2_unc,fco2+fco2_unc,alpha=0.4,color='k',zorder=5)

    ax[5].plot(time_mon,fco2_out,color='r',zorder=6,linewidth=2)
    ax[5].fill_between(time_mon,fco2_out-fco2_unc_out/2,fco2_out+fco2_unc_out/2,alpha=0.6,color='r')
    ax[5].fill_between(time_mon,fco2_out-fco2_unc_out,fco2_out+fco2_unc_out,alpha=0.4,color='r')
    ax[5].set_ylabel('Cumulative air-sea CO$_2$ flux (Tg C)\n(-ve indicates atmosphere to ocean exchange)')
    ax[5].plot([time_da[0],time_da[-1]],[0,0],'k--')

    c.close()
    ax2 = [0,1]
    for i in ax2:
        ax[i].set_xticklabels([])
    ax2 = [0,1,3,4,5]
    for i in ax2:
        ax[i].tick_params(axis='x', labelrotation=25)
        ax[i].set_xlim([time_da[0],time_da[-1]])
    ax2 = [3,4,5]
    for i in ax2:
        ax[i].set_xlabel('Date (Year - Month)')
    for i in range(6):
        ax[i].text(0.05,0.96,f'({let[i]})',transform=ax[i].transAxes,va='top',fontweight='bold',fontsize = 24)
    fig.savefig('figs/manuscript/figure_1.png',dpi=300)

"""
Figure 2
"""
if plot_figure_2:
    font = {'weight' : 'normal',
            'size'   : 14}
    matplotlib.rc('font', **font)
    fold = output_loc

    files = ['496.nc','194465.nc']
    fig = plt.figure(figsize=(15,15))
    gs = GridSpec(2,2, figure=fig, wspace=0.25,hspace=0.15,bottom=0.1,top=0.93,left=0.075,right=0.95)
    ax = [fig.add_subplot(gs[0,0]),fig.add_subplot(gs[0,1]),fig.add_subplot(gs[1,0]),fig.add_subplot(gs[1,1])]
    cols = ['#332288','#44AA99','#882255','#DDCC77', '#117733', '#88CCEE','#999933','#CC6677']
    cou = 0
    for file in files:
        c = Dataset(fold + '/' + file,'r')
        keys = c.variables.keys()

        l = []
        for key in keys:
            if ('unc' in key) & ('cumulative' in key) & ('in_physics' in key):
                l.append(key)

        print(l)
        label=['Gas Transfer','Wind','fCO$_{2 (sw)}$','Schmidt','Solubility skin','Solubility subskin','fCO$_{2 (atm)}$']
        uncs = ['k','wind','xco2atm','fco2sw']
        uncs_comp= ['ph2o','schmidt','solskin_unc','solsubskin_unc']

        combined = np.zeros((len(uncs)+len(uncs_comp),len(c[l[0]])))
        print(combined.shape)
        t = 0
        for i in range(len(uncs)):
            combined[t,:] = np.array(c['flux_unc_'+uncs[i]+'_in_physics_areaday_cumulative'])
            t=t+1

        for i in range(len(uncs_comp)):
            combined[t,:] = np.sqrt(np.array(c['flux_unc_'+uncs_comp[i]+'_in_physics_areaday_cumulative'])**2 + np.array(c['flux_unc_'+uncs_comp[i]+'_fixed_in_physics_areaday_cumulative'])**2)
            t = t+1

        print(combined)
        data_atm = combined[[2,4],:]
        combined = combined[[0,1,3,5,6,7],:]
        print(combined.shape)
        atm = np.sqrt(np.sum(data_atm**2,axis=0))
        atm = atm[np.newaxis,:]
        combined = np.append(combined,atm,axis=0)
        print(combined.shape)
        print(combined)
        totals = []
        for i in range(combined.shape[1]):
            totals.append(np.sum(combined[:,i]))


        for i in range(combined.shape[1]):
            bottom = 0
            for j in range(combined.shape[0]):
                if i == 1:
                    p = ax[cou].bar(i+1,(combined[j,i]/totals[i])*100,bottom=bottom,color=cols[j],label=label[j])
                else:
                    p = ax[cou].bar(i+1,(combined[j,i]/totals[i])*100,bottom=bottom,color=cols[j])
                bottom = bottom + (combined[j,i]/totals[i])*100

        print(totals)
        ax[cou].set_xlabel('Month since formation')
        ax[cou].set_ylabel('Relative contribution to uncertainty (%)')
        cou = cou+1
        for i in range(combined.shape[0]):
            ax[cou].plot(np.array(range(1,combined.shape[1]+1)),combined[i,:],color=cols[i],label=label[i])
        ax[cou].legend()
        ax[cou].set_xlabel('Month since formation')
        ax[cou].set_ylabel('Absolute contribution to uncertainty (Tg C)')
        cou=cou+1
    let = ['a','b','c','d']
    for i in range(4):
        #worldmap.plot(color="lightgrey", ax=ax[i])
        ax[i].text(0.92,1.06,f'({let[i]})',transform=ax[i].transAxes,va='top',fontweight='bold',fontsize = 24)
    fig.savefig('figs/manuscript/Figure_2.png')

"""
Figure 3
"""
if plot_figure_3:
    # Generating the percentage of eddies spawned per 1 degree pixel
    if not du.checkfileexist(os.path.join('data','anti_percent_eddies.nc')):
        lon,lat = du.reg_grid()
        all_eddies = np.zeros((len(lon),len(lat)))
        long_eddies = np.copy(all_eddies)

        f = np.where((eddy_an_all['observation_number'] == 1))[0]
        eddy_an_all['latitude'] = eddy_an_all['latitude'][f]; eddy_an_all['longitude'] = eddy_an_all['longitude'][f]

        for i in range(len(lon)):
            print(i)
            for j in range(len(lat)):
                f = np.where((eddy_an_all['latitude'] > lat[j]-0.5) & (eddy_an_all['latitude'] <= lat[j]+0.5) & (eddy_an_all['longitude'] > lon[i]-0.5) & (eddy_an_all['longitude'] <= lon[i]+0.5))[0]
                all_eddies[i,j] = len(f)
                f = np.where((eddy_an['observation_number'] == 1)&(eddy_an['latitude'] > lat[j]-0.5) & (eddy_an['latitude'] <= lat[j]+0.5) & (eddy_an['longitude'] > lon[i]-0.5) & (eddy_an['longitude'] <= lon[i]+0.5))[0]
                long_eddies[i,j] = len(f)


        du.netcdf_create_basic(os.path.join('data','anti_percent_eddies.nc'),all_eddies,'all_eddies',lat,lon)
        du.netcdf_append_basic(os.path.join('data','anti_percent_eddies.nc'),long_eddies,'long_eddies')

        f = np.where((eddy_cy_all['observation_number'] == 1))[0]
        eddy_cy_all['latitude'] = eddy_cy_all['latitude'][f]; eddy_cy_all['longitude'] = eddy_cy_all['longitude'][f]
        all_eddies = np.zeros((len(lon),len(lat)))
        long_eddies = np.copy(all_eddies)
        for i in range(len(lon)):
            print(i)
            for j in range(len(lat)):
                f = np.where((eddy_cy_all['latitude'] > lat[j]-0.5) & (eddy_cy_all['latitude'] <= lat[j]+0.5) & (eddy_cy_all['longitude'] > lon[i]-0.5) & (eddy_cy_all['longitude'] <= lon[i]+0.5))[0]
                all_eddies[i,j] = len(f)
                f = np.where((eddy_cy['observation_number'] == 1) & (eddy_cy['latitude'] > lat[j]-0.5) & (eddy_cy['latitude'] <= lat[j]+0.5) & (eddy_cy['longitude'] > lon[i]-0.5) & (eddy_cy['longitude'] <= lon[i]+0.5))[0]
                long_eddies[i,j] = len(f)


        du.netcdf_create_basic(os.path.join('data','cycl_percent_eddies.nc'),all_eddies,'all_eddies',lat,lon)
        du.netcdf_append_basic(os.path.join('data','cycl_percent_eddies.nc'),long_eddies,'long_eddies')

    font = {'weight' : 'normal',
            'size'   :20}
    matplotlib.rc('font', **font)
    fig = plt.figure(figsize=(36,20))
    gs = GridSpec(2,2, figure=fig, wspace=0.35,hspace=0.15,bottom=0.05,top=0.95,left=0.05,right=0.93)
    ax = [fig.add_subplot(gs[0,0]),fig.add_subplot(gs[1,0]),fig.add_subplot(gs[0,1]),fig.add_subplot(gs[1,1])]
    m = eplt.base_tracks_start(ax[0],latb=[-70,70],lonb = [-180,180])
    m2 = eplt.base_tracks_start(ax[1],latb=[-70,70],lonb = [-180,180])
    m3 = eplt.base_tracks_start(ax[2],latb=[-70,70],lonb = [-180,180])
    m4 = eplt.base_tracks_start(ax[3],latb=[-70,70],lonb = [-180,180])
    val = []
    cum = []
    #print(np.unique(eddy_v['track']))
    cmap = cmocean.tools.crop_by_percent(cmocean.cm.balance, 20, which='both', N=None)
    for i in  np.unique(eddy_an['track']):
        print(i)
        if os.path.exists(output_loc+'/'+str(i)+'.nc') == True:
            c = Dataset(output_loc+'/'+str(i)+'.nc','r')
            #c = Dataset(i,'r')
            time = np.array(c['month_time'])
            time2 = []
            for j in range(len(time)):
                time2.append(pyEddy_m.date_con(int(time[j])))

            co2 = np.array(c['flux_in_physics_areaday_cumulative'])
            lat = np.array(c['month_latitude'])
            lon = np.array(c['month_longitude'])
            x,y = m(lon,lat)

            a = ax[0].scatter(x[0],y[0],s=12,c = co2[-1],vmin = -1,vmax=1,cmap=cmap)
            c.close()
    c = Dataset(os.path.join('data','anti_percent_eddies.nc'),'r')
    lat = np.array(c['latitude'])
    lon = np.array(c['longitude'])
    eddies = np.array(c['long_eddies'])/np.array(c['all_eddies']) *100
    c.close()
    long,latg = np.meshgrid(lon,lat)
    x,y = m3(long,latg)
    ax[2].pcolor(x,y,np.transpose(eddies),vmin=0,vmax=30,cmap=cmocean.cm.dense)
    # #Cyclonic
    for i in  np.unique(eddy_cy['track']):
        print(i)
        if os.path.exists(output_loc_cy+'/'+str(i)+'.nc') == True:
            c = Dataset(output_loc_cy+'/'+str(i)+'.nc','r')
            #c = Dataset(i,'r')
            time = np.array(c['month_time'])
            time2 = []
            for j in range(len(time)):
                time2.append(pyEddy_m.date_con(int(time[j])))

            co2 = np.array(c['flux_in_physics_areaday_cumulative'])
            lat = np.array(c['month_latitude'])
            lon = np.array(c['month_longitude'])
            x,y = m2(lon,lat)

            a = ax[1].scatter(x[0],y[0],s=12,c = co2[-1],vmin = -1,vmax=1,cmap=cmap)
            c.close()
    #
    c = Dataset(os.path.join('data','cycl_percent_eddies.nc'),'r')
    lat = np.array(c['latitude'])
    lon = np.array(c['longitude'])
    eddies = np.array(c['long_eddies'])/np.array(c['all_eddies']) *100
    c.close()
    long,latg = np.meshgrid(lon,lat)
    x,y = m3(long,latg)
    b = ax[3].pcolor(x,y,np.transpose(eddies),vmin=0,vmax=30,cmap=cmocean.cm.dense)

    ax[1].set_xlabel('Longitude',labelpad=30)
    ax[1].set_ylabel('Latitude',labelpad=45)
    ax[3].set_xlabel('Longitude',labelpad=30)
    ax[0].set_ylabel('Latitude',labelpad=45)
    eplt.base_tracks_end(ax[0],m)
    eplt.base_tracks_end(ax[1],m2)
    eplt.base_tracks_end(ax[2],m3)
    eplt.base_tracks_end(ax[3],m4)
    cbar = fig.add_axes([0.95,0.2,0.02,0.6])
    cbar2 = fig.add_axes([0.44,0.2,0.02,0.6])
    cba = fig.colorbar(a,cax=cbar2)
    cba2 = fig.colorbar(b,cax=cbar)
    for i in range(4):
        ax[i].text(0.92,1.06,f'({let[i]})',transform=ax[i].transAxes,va='top',fontweight='bold',fontsize = 24)
    cba.set_label('Cumulative net CO$_{2}$ flux (Tg C)')
    cba2.set_label('Percentage of eddy trajectories that are long lived (%)')
    fig.savefig('figs/manuscript/figure_3.png',dpi=300)


if plot_figure_3_norm:

    font = {'weight' : 'normal',
            'size'   :20}
    matplotlib.rc('font', **font)
    fig = plt.figure(figsize=(18,20))
    gs = GridSpec(2,1, figure=fig, wspace=0.35,hspace=0.15,bottom=0.05,top=0.95,left=0.05,right=0.8)
    ax = [fig.add_subplot(gs[0,0]),fig.add_subplot(gs[1,0])]
    m = eplt.base_tracks_start(ax[0],latb=[-70,70],lonb = [-180,180])
    m2 = eplt.base_tracks_start(ax[1],latb=[-70,70],lonb = [-180,180])

    val = []
    cum = []
    #print(np.unique(eddy_v['track']))
    cmap = cmocean.tools.crop_by_percent(cmocean.cm.balance, 20, which='both', N=None)
    for i in  np.unique(eddy_an['track']):
        print(i)
        if os.path.exists(output_loc+'/'+str(i)+'.nc') == True:
            c = Dataset(output_loc+'/'+str(i)+'.nc','r')
            #c = Dataset(i,'r')
            time = np.array(c['month_time'])
            time2 = []
            for j in range(len(time)):
                time2.append(pyEddy_m.date_con(int(time[j])))

            co2 = np.array(c['flux_in_physics_areaday_cumulative']) / len(np.array(c['time']))
            lat = np.array(c['month_latitude'])
            lon = np.array(c['month_longitude'])
            x,y = m(lon,lat)

            a = ax[0].scatter(x[0],y[0],s=12,c = co2[-1],vmin = -0.002,vmax=0.002,cmap=cmap)
            c.close()

    # #Cyclonic
    for i in  np.unique(eddy_cy['track']):
        print(i)
        if os.path.exists(output_loc_cy+'/'+str(i)+'.nc') == True:
            c = Dataset(output_loc_cy+'/'+str(i)+'.nc','r')
            #c = Dataset(i,'r')
            time = np.array(c['month_time'])
            time2 = []
            for j in range(len(time)):
                time2.append(pyEddy_m.date_con(int(time[j])))

            co2 = np.array(c['flux_in_physics_areaday_cumulative']) / len(np.array(c['time']))
            lat = np.array(c['month_latitude'])
            lon = np.array(c['month_longitude'])
            x,y = m2(lon,lat)

            a = ax[1].scatter(x[0],y[0],s=12,c = co2[-1],vmin = -0.002,vmax=0.002,cmap=cmap)
            c.close()
    #


    ax[1].set_xlabel('Longitude',labelpad=30)
    ax[1].set_ylabel('Latitude',labelpad=45)

    ax[0].set_ylabel('Latitude',labelpad=45)
    eplt.base_tracks_end(ax[0],m)
    eplt.base_tracks_end(ax[1],m2)

    cbar = fig.add_axes([0.85,0.2,0.02,0.6])

    cba = fig.colorbar(a,cax=cbar)
    for i in range(2):
        ax[i].text(0.92,1.06,f'({let[i]})',transform=ax[i].transAxes,va='top',fontweight='bold',fontsize = 24)
    cba.set_label('Cumulative net CO$_{2}$ flux (Tg C d$^{-1}$)')
    fig.savefig('figs/manuscript/figure_3_normalised.png',dpi=300)
"""
Figure 4
"""
def unc_prop(inps,ens = 1000):
    out = []
    for i in range(ens):
        in_flux = inps[:,0] + (inps[:,1] * np.random.normal(0, 0.5, len(inps[:,1])))
        out_flux = inps[:,2] + (inps[:,3] * np.random.normal(0, 0.5, len(inps[:,3])))
        out.append(np.nanmedian(((in_flux-out_flux)/np.abs(out_flux))*100))
    out = np.array(out)
    print(f'Mean = {np.mean(out)}')
    print(f'Median = {np.median(out)}')
    print(f'2 standard dev = {np.std(out)*2}')
    return [np.mean(out),np.median(out),np.std(out)*2]

if plot_figure_4:
    font = {'weight' : 'normal',
            'size'   :20}
    matplotlib.rc('font', **font)
    line_widths = 3
    xlims = [0.85,1.15]
    fig = plt.figure(figsize=(25,20))
    gs = GridSpec(2,3, figure=fig, wspace=0.95,hspace=0.15,bottom=0.1,top=0.93,left=0.075,right=0.97)
    ax = [fig.add_subplot(gs[0,0:2]),fig.add_subplot(gs[1,0:2]),fig.add_subplot(gs[0,2]),fig.add_subplot(gs[1,2])]
    m = eplt.base_tracks_start(ax[0],latb=[-70,70],lonb = [-180,180])
    m2 = eplt.base_tracks_start(ax[1],latb=[-70,70],lonb = [-180,180])
    """
    Anticyclonic Start
    """
    val = []
    cum = []
    #print(np.unique(eddy_v['track']))
    cmap = cmocean.tools.crop_by_percent(cmocean.cm.balance, 20, which='both', N=None)
    unc_cal = []
    cou = 0
    for i in  np.unique(eddy_an['track']):
        #print(i)
        if os.path.exists(output_loc+'/'+str(i)+'.nc') == True:
            c = Dataset(output_loc+'/'+str(i)+'.nc','r')
            #c = Dataset(i,'r')
            time = np.array(c['month_time'])
            time2 = []
            for j in range(len(time)):
                time2.append(pyEddy_m.date_con(int(time[j])))

            co2 = np.array(c['flux_in_physics_areaday_cumulative'])
            co2_unc = np.array(c['flux_unc_tot_in_physics_areaday_cumulative'])
            co2_o = np.array(c['flux_out_physics_areaday_cumulative'])
            co2_o_unc = np.array(c['flux_unc_tot_out_physics_areaday_cumulative'])
            co2_p = ((co2-co2_o)/np.abs(co2_o))*100
            cum.append((co2-co2_o)[-1])
            unc_cal.append([co2[-1],co2_unc[-1],co2_o[-1],co2_o_unc[-1]])
            val.append(co2_p[-1])
            lat = np.array(c['month_latitude'])
            lon = np.array(c['month_longitude'])
            x,y = m(lon,lat)

            a = ax[0].scatter(x[0],y[0],s=12,c = co2_p[-1],vmin = -50,vmax=50,cmap=cmap)
            c.close()
            # if cou == 40:
            #     break
            # cou=cou+1
    val = np.array(val)
    print(f'Cumulative = {np.sum(np.array(cum))}')
    unc_cal = np.array(unc_cal)
    unc_anti = unc_prop(unc_cal)
    ax[2].boxplot(val[np.isnan(val) == False],labels=[f'Anticyclonic (N = {len(val[np.isnan(val) == False])})\nMedian = {str(np.round(np.median(val[np.isnan(val) == False]),1))} $\pm$ {str(np.round(unc_anti[2],1))} %'],boxprops = dict(linewidth=line_widths),
        medianprops = dict(linewidth=line_widths,color='r'),whiskerprops= dict(linewidth=line_widths),capprops=dict(linewidth=line_widths))
    med = np.median(val[np.isnan(val) == False])
    ax[2].fill_between(xlims,med-unc_anti[2],med+unc_anti[2],color='r',alpha=0.2)
    ax[2].fill_between(xlims,med-(unc_anti[2]/2),med+(unc_anti[2]/2),color='r',alpha=0.4)
    ax[2].set_ylim([-50,50])
    # ax[2].annotate('Median\n' + str(np.round(med,1))+'%', xy=(1.07,med), xytext=(1.115,med+10),
    #         arrowprops=dict(facecolor='black'),horizontalalignment='center')
    """
    Cyclonic Start
    """
    val_cy = []
    unc_cal_cy = []
    cum_cy = []
    # #Cyclonic
    cou = 0
    for i in  np.unique(eddy_cy['track']):
        #print(i)
        if os.path.exists(output_loc_cy+'/'+str(i)+'.nc') == True:
            c = Dataset(output_loc_cy+'/'+str(i)+'.nc','r')
            #c = Dataset(i,'r')
            time = np.array(c['month_time'])
            time2 = []
            for j in range(len(time)):
                time2.append(pyEddy_m.date_con(int(time[j])))

            co2 = np.array(c['flux_in_physics_areaday_cumulative'])
            co2_unc = np.array(c['flux_unc_tot_in_physics_areaday_cumulative'])
            co2_o = np.array(c['flux_out_physics_areaday_cumulative'])
            co2_o_unc = np.array(c['flux_unc_tot_out_physics_areaday_cumulative'])
            co2_p = ((co2-co2_o)/np.abs(co2_o))*100
            unc_cal_cy.append([co2[-1],co2_unc[-1],co2_o[-1],co2_o_unc[-1]])
            val_cy.append(co2_p[-1])
            cum_cy.append((co2-co2_o)[-1])
            lat = np.array(c['month_latitude'])
            lon = np.array(c['month_longitude'])
            x,y = m2(lon,lat)

            a = ax[1].scatter(x[0],y[0],s=12,c = co2_p[-1],vmin = -50,vmax=50,cmap=cmap)
            c.close()
            # if cou == 40:
            #     break
            # cou=cou+1
    val_cy = np.array(val_cy)
    unc_cal_cy = np.array(unc_cal_cy)
    unc_cy = unc_prop(unc_cal_cy)
    print(f'Cumulative = {np.nansum(np.array(cum_cy))}')
    ax[3].boxplot(val_cy[np.isnan(val_cy) == False],labels=[f'Cyclonic (N = {len(val_cy[np.isnan(val_cy) == False])})\nMedian = {str(np.round(np.median(val_cy[np.isnan(val_cy) == False]),1))} $\pm$ {str(np.round(unc_cy[2],1))} %'],boxprops = dict(linewidth=line_widths),
        medianprops = dict(linewidth=line_widths,color='r'),whiskerprops= dict(linewidth=line_widths),capprops=dict(linewidth=line_widths))
    med = np.median(val_cy[np.isnan(val_cy) == False])
    ax[3].fill_between(xlims,med-unc_cy[2],med+unc_cy[2],color='r',alpha=0.2)
    ax[3].fill_between(xlims,med-(unc_cy[2]/2),med+(unc_cy[2]/2),color='r',alpha=0.4)
    # ax[3].annotate('Median\n' + str(np.round(med,1))+'%', xy=(1.07,med), xytext=(1.115,med+10),
    #         arrowprops=dict(facecolor='black'),horizontalalignment='center')
    ax[3].set_ylim([-50,50])

    ax[1].set_xlabel('Longitude',labelpad=30)
    ax[1].set_ylabel('Latitude',labelpad=45)
    # # ax[0].set_xlabel('Longitude',labelpad=30)
    ax[0].set_ylabel('Latitude',labelpad=45)
    eplt.base_tracks_end(ax[0],m)
    eplt.base_tracks_end(ax[1],m2)
    cbar = fig.add_axes([0.65,0.2,0.02,0.6])
    cba = fig.colorbar(a,cax=cbar)
    cba.set_label('Change in eddy CO${_2}$ flux compared to the surrounding water CO${_2}$ flux (%)')
    for i in range(4):
        ax[i].text(0.92,1.06,f'({let[i]})',transform=ax[i].transAxes,va='top',fontweight='bold',fontsize = 24)
    lab= 'Change in eddy CO${_2}$ flux compared \nto the surrounding water CO${_2}$ flux (%)'
    ax[2].set_ylabel(lab); ax[3].set_ylabel(lab)

    ax[2].set_xlim(xlims); ax[3].set_xlim(xlims)
    ax[2].plot(xlims,[0,0],'k--');ax[3].plot(xlims,[0,0],'k--')
    fig.savefig('figs/manuscript/figure_4.png',dpi=300)


if plot_figure_4_bio:
    font = {'weight' : 'normal',
            'size'   :20}
    matplotlib.rc('font', **font)
    line_widths = 3
    xlims = [0.85,1.15]
    fig = plt.figure(figsize=(25,20))
    gs = GridSpec(2,3, figure=fig, wspace=0.95,hspace=0.15,bottom=0.1,top=0.93,left=0.075,right=0.97)
    ax = [fig.add_subplot(gs[0,0:2]),fig.add_subplot(gs[1,0:2]),fig.add_subplot(gs[0,2]),fig.add_subplot(gs[1,2])]
    m = eplt.base_tracks_start(ax[0],latb=[-70,70],lonb = [-180,180])
    m2 = eplt.base_tracks_start(ax[1],latb=[-70,70],lonb = [-180,180])
    """
    Anticyclonic Start
    """
    val = []
    cum = []
    #print(np.unique(eddy_v['track']))
    cmap = cmocean.tools.crop_by_percent(cmocean.cm.balance, 20, which='both', N=None)
    unc_cal = []
    cou = 0
    for i in  np.unique(eddy_an['track']):
        #print(i)
        if os.path.exists(output_loc+'/'+str(i)+'.nc') == True:
            c = Dataset(output_loc+'/'+str(i)+'.nc','r')
            #c = Dataset(i,'r')
            time = np.array(c['month_time'])
            time2 = []
            for j in range(len(time)):
                time2.append(pyEddy_m.date_con(int(time[j])))

            co2 = np.array(c['flux_in_bio_areaday_cumulative'])
            co2_unc = np.array(c['flux_unc_tot_in_bio_areaday_cumulative'])
            co2_o = np.array(c['flux_out_bio_areaday_cumulative'])
            co2_o_unc = np.array(c['flux_unc_tot_out_bio_areaday_cumulative'])
            co2_p = ((co2-co2_o)/np.abs(co2_o))*100
            cum.append((co2-co2_o)[-1])
            unc_cal.append([co2[-1],co2_unc[-1],co2_o[-1],co2_o_unc[-1]])
            val.append(co2_p[-1])
            lat = np.array(c['month_latitude'])
            lon = np.array(c['month_longitude'])
            x,y = m(lon,lat)

            a = ax[0].scatter(x[0],y[0],s=12,c = co2_p[-1],vmin = -50,vmax=50,cmap=cmap)
            c.close()
            # if cou == 40:
            #     break
            # cou=cou+1
    val = np.array(val)
    print(f'Cumulative = {np.sum(np.array(cum))}')
    unc_cal = np.array(unc_cal)
    unc_anti = unc_prop(unc_cal)
    ax[2].boxplot(val[np.isnan(val) == False],labels=[f'Anticyclonic (N = {len(val[np.isnan(val) == False])})\nMedian = {str(np.round(np.median(val[np.isnan(val) == False]),1))} $\pm$ {str(np.round(unc_anti[2],1))} %'],boxprops = dict(linewidth=line_widths),
        medianprops = dict(linewidth=line_widths,color='r'),whiskerprops= dict(linewidth=line_widths),capprops=dict(linewidth=line_widths))
    med = np.median(val[np.isnan(val) == False])
    ax[2].fill_between(xlims,med-unc_anti[2],med+unc_anti[2],color='r',alpha=0.2)
    ax[2].fill_between(xlims,med-(unc_anti[2]/2),med+(unc_anti[2]/2),color='r',alpha=0.4)
    ax[2].set_ylim([-50,50])
    # ax[2].annotate('Median\n' + str(np.round(med,1))+'%', xy=(1.07,med), xytext=(1.115,med+10),
    #         arrowprops=dict(facecolor='black'),horizontalalignment='center')
    """
    Cyclonic Start
    """
    val_cy = []
    unc_cal_cy = []
    cum_cy = []
    # #Cyclonic
    cou = 0
    for i in  np.unique(eddy_cy['track']):
        #print(i)
        if os.path.exists(output_loc_cy+'/'+str(i)+'.nc') == True:
            c = Dataset(output_loc_cy+'/'+str(i)+'.nc','r')
            #c = Dataset(i,'r')
            time = np.array(c['month_time'])
            time2 = []
            for j in range(len(time)):
                time2.append(pyEddy_m.date_con(int(time[j])))

            co2 = np.array(c['flux_in_bio_areaday_cumulative'])
            co2_unc = np.array(c['flux_unc_tot_in_bio_areaday_cumulative'])
            co2_o = np.array(c['flux_out_bio_areaday_cumulative'])
            co2_o_unc = np.array(c['flux_unc_tot_out_bio_areaday_cumulative'])
            co2_p = ((co2-co2_o)/np.abs(co2_o))*100
            unc_cal_cy.append([co2[-1],co2_unc[-1],co2_o[-1],co2_o_unc[-1]])
            val_cy.append(co2_p[-1])
            cum_cy.append((co2-co2_o)[-1])
            lat = np.array(c['month_latitude'])
            lon = np.array(c['month_longitude'])
            x,y = m2(lon,lat)

            a = ax[1].scatter(x[0],y[0],s=12,c = co2_p[-1],vmin = -50,vmax=50,cmap=cmap)
            c.close()
            # if cou == 40:
            #     break
            # cou=cou+1
    val_cy = np.array(val_cy)
    unc_cal_cy = np.array(unc_cal_cy)
    unc_cy = unc_prop(unc_cal_cy)
    print(f'Cumulative = {np.nansum(np.array(cum_cy))}')
    ax[3].boxplot(val_cy[np.isnan(val_cy) == False],labels=[f'Cyclonic (N = {len(val_cy[np.isnan(val_cy) == False])})\nMedian = {str(np.round(np.median(val_cy[np.isnan(val_cy) == False]),1))} $\pm$ {str(np.round(unc_cy[2],1))} %'],boxprops = dict(linewidth=line_widths),
        medianprops = dict(linewidth=line_widths,color='r'),whiskerprops= dict(linewidth=line_widths),capprops=dict(linewidth=line_widths))
    med = np.median(val_cy[np.isnan(val_cy) == False])
    ax[3].fill_between(xlims,med-unc_cy[2],med+unc_cy[2],color='r',alpha=0.2)
    ax[3].fill_between(xlims,med-(unc_cy[2]/2),med+(unc_cy[2]/2),color='r',alpha=0.4)
    # ax[3].annotate('Median\n' + str(np.round(med,1))+'%', xy=(1.07,med), xytext=(1.115,med+10),
    #         arrowprops=dict(facecolor='black'),horizontalalignment='center')
    ax[3].set_ylim([-50,50])

    ax[1].set_xlabel('Longitude',labelpad=30)
    ax[1].set_ylabel('Latitude',labelpad=45)
    # # ax[0].set_xlabel('Longitude',labelpad=30)
    ax[0].set_ylabel('Latitude',labelpad=45)
    eplt.base_tracks_end(ax[0],m)
    eplt.base_tracks_end(ax[1],m2)
    cbar = fig.add_axes([0.65,0.2,0.02,0.6])
    cba = fig.colorbar(a,cax=cbar)
    cba.set_label('Change in eddy CO${_2}$ flux compared to the surrounding water CO${_2}$ flux (%)')
    for i in range(4):
        ax[i].text(0.92,1.06,f'({let[i]})',transform=ax[i].transAxes,va='top',fontweight='bold',fontsize = 24)
    lab= 'Change in eddy CO${_2}$ flux compared \nto the surrounding water CO${_2}$ flux (%)'
    ax[2].set_ylabel(lab); ax[3].set_ylabel(lab)

    ax[2].set_xlim(xlims); ax[3].set_xlim(xlims)
    ax[2].plot(xlims,[0,0],'k--');ax[3].plot(xlims,[0,0],'k--')
    fig.savefig('figs/manuscript/figure_4_bio.png',dpi=300)

"""
Figure 5
"""
if plot_figure_5:
    font = {'weight' : 'normal',
            'size'   :10}
    matplotlib.rc('font', **font)
    line_widths = 1

    recaap_file = 'data/RECCAP2_region_masks_all_v20221025.nc'
    # loading the RECAAP ocean definitons
    c = Dataset(recaap_file,'r')
    rec_lon = np.array(c['lon'])-180
    rec_lat = np.array(c['lat'])
    long,latg = np.meshgrid(rec_lon,rec_lat)
    oceans = np.array(c['open_ocean'])
    oceans2 = np.zeros((oceans.shape))
    oceans2[:,180:] = oceans[:,0:180]
    oceans2[:,0:180] = oceans[:,180:]
    print(oceans.shape)
    oceans2[oceans2==0] = np.nan
    for i in range(1,4):
        f = np.where((oceans2 == i) & (latg <0))
        oceans2[f] = i+0.5
    oceans2[oceans2 ==4] = np.nan
    oceans2[oceans2 == 3] = np.nan
    oceans2[oceans2 == 5] = 4
    c.close()
    locas = [2.0,1.0,2.5,1.5,3.5,4.0]
    print(locas)
    fig = plt.figure(figsize=(21,10))
    gs = GridSpec(3,4, figure=fig, wspace=0.3,hspace=0.25,bottom=0.1,top=0.99,left=0.05,right=0.975)
    ax = [fig.add_subplot(gs[:,1:3]),fig.add_subplot(gs[0,0]),fig.add_subplot(gs[1,0]),fig.add_subplot(gs[2,0]),fig.add_subplot(gs[0,3]),fig.add_subplot(gs[1,3]),fig.add_subplot(gs[2,3])]
    m = eplt.base_tracks_start(ax[0],latb=[-70,70],lonb = [-180,180])
    x,y = m(long,latg)
    ax[0].pcolor(x,y,oceans2)

    ax[0].set_xlabel('Longitude',labelpad=30)
    ax[0].set_ylabel('Latitude',labelpad=45)
    val= []
    unc_cal = []
    cum = []
    anti_loc = []
    cou = 1
    for i in  np.unique(eddy_an['track']):
        #print(i)
        if os.path.exists(output_loc+'/'+str(i)+'.nc') == True:
            c = Dataset(output_loc+'/'+str(i)+'.nc','r')
            #c = Dataset(i,'r')
            time = np.array(c['month_time'])
            time2 = []
            for j in range(len(time)):
                time2.append(pyEddy_m.date_con(int(time[j])))

            co2 = np.array(c['flux_in_physics_areaday_cumulative'])
            co2_unc = np.array(c['flux_unc_tot_in_physics_areaday_cumulative'])
            co2_o = np.array(c['flux_out_physics_areaday_cumulative'])
            co2_o_unc = np.array(c['flux_unc_tot_out_physics_areaday_cumulative'])
            co2_p = ((co2-co2_o)/np.abs(co2_o))*100

            cum.append((co2-co2_o)[-1])
            unc_cal.append([co2[-1],co2_unc[-1],co2_o[-1],co2_o_unc[-1]])
            val.append(co2_p[-1])
            lat = np.array(c['month_latitude'])
            lon = np.array(c['month_longitude'])

            f = np.where(np.abs(lat[0] - rec_lat) == np.min(np.abs(lat[0] - rec_lat)))[0]
            g = np.where(np.abs(lon[0] - rec_lon) == np.min(np.abs(lon[0] - rec_lon)))[0]
            anti_loc.append(oceans2[f,g])
            print(oceans2[f,g])
            # cou= cou+1
            # if cou == 60:
            #     break
    anti_loc = np.array(anti_loc)
    unc_cal = np.array(unc_cal)
    val = np.array(val)

    val_cy = []
    unc_cal_cy = []
    cum_cy = []
    cy_loc = []
    cou= 1
    for i in np.unique(eddy_cy['track']):
        #print(i)
        if os.path.exists(output_loc_cy+'/'+str(i)+'.nc') == True:
            c = Dataset(output_loc_cy+'/'+str(i)+'.nc','r')
            #c = Dataset(i,'r')
            time = np.array(c['month_time'])
            time2 = []
            for j in range(len(time)):
                time2.append(pyEddy_m.date_con(int(time[j])))

            co2 = np.array(c['flux_in_physics_areaday_cumulative'])
            co2_unc = np.array(c['flux_unc_tot_in_physics_areaday_cumulative'])
            co2_o = np.array(c['flux_out_physics_areaday_cumulative'])
            co2_o_unc = np.array(c['flux_unc_tot_out_physics_areaday_cumulative'])
            co2_p = ((co2-co2_o)/np.abs(co2_o))*100

            cum_cy.append((co2-co2_o)[-1])
            unc_cal_cy.append([co2[-1],co2_unc[-1],co2_o[-1],co2_o_unc[-1]])
            val_cy.append(co2_p[-1])
            lat = np.array(c['month_latitude'])
            lon = np.array(c['month_longitude'])

            f = np.where(np.abs(lat[0] - rec_lat) == np.min(np.abs(lat[0] - rec_lat)))[0]
            g = np.where(np.abs(lon[0] - rec_lon) == np.min(np.abs(lon[0] - rec_lon)))[0]
            cy_loc.append(oceans2[f,g])
            print(oceans2[f,g])
            # cou= cou+1
            # if cou == 60:
            #     break
    cy_loc = np.array(cy_loc)
    unc_cal_cy = np.array(unc_cal_cy)
    val_cy = np.array(val_cy)

    cou = 1
    xlims = [0.85,1.45]
    lab= 'Change in eddy CO${_2}$ flux compared \nto the surrounding water CO${_2}$ flux (%)'
    ax[0].text(0.92,1.06,f'({let[0]})',transform=ax[0].transAxes,va='top',fontweight='bold',fontsize = 14)
    for i in locas:
        print(cou)
        f = np.where((anti_loc == i))[0]
        f = f[np.where(np.isnan(val[f]) == 0)]
        g = np.where((cy_loc == i))[0]
        g = g[np.where(np.isnan(val_cy[g]) == 0)]
        # print(f)

        unc_anti = unc_prop(unc_cal[f])
        unc_cy = unc_prop(unc_cal_cy[g])
        ax[cou].boxplot([val[f],val_cy[g]],boxprops = dict(linewidth=line_widths),labels=[f'Anticyclonic (N = {len(f)})\nMedian = {str(np.round(np.median(val[f]),1))} $\pm$ {str(np.round(unc_anti[2],1))} %',f'Cyclonic (N = {len(g)})\nMedian = {str(np.round(np.median(val_cy[g]),1))} $\pm$ {str(np.round(unc_cy[2],1))} %'],
            medianprops = dict(linewidth=line_widths,color='r'),whiskerprops= dict(linewidth=line_widths),capprops=dict(linewidth=line_widths),positions=[1,1.3],widths=0.2)
        med = np.median(val[f])
        ax[cou].fill_between([0.88,1.12],med-unc_anti[2],med+unc_anti[2],color='r',alpha=0.2)
        ax[cou].fill_between([0.88,1.12],med-(unc_anti[2]/2),med+(unc_anti[2]/2),color='r',alpha=0.4)

        med = np.median(val_cy[g])
        ax[cou].fill_between([1.18,1.42],med-unc_cy[2],med+unc_cy[2],color='r',alpha=0.2)
        ax[cou].fill_between([1.18,1.42],med-(unc_cy[2]/2),med+(unc_cy[2]/2),color='r',alpha=0.4)
        ax[cou].set_ylim([-50,50])
        ax[cou].set_xlim([0.85,1.45])
        ax[cou].plot(xlims,[0,0],'k--')
        ax[cou].set_ylabel(lab)
        print(let[cou])
        print(ax[cou])
        ax[cou].text(1.03,0.92,f'({let[cou]})',transform=ax[cou].transAxes,va='top',fontweight='bold',fontsize = 14)
        cou=cou+1

    #North Pacific
    x,y = m(-145,35)
    arrow = patches.ConnectionPatch([1.45,-20],[x,y],coordsA=ax[1].transData,coordsB=ax[0].transData,color="black",arrowstyle="-|>",
        mutation_scale=30, linewidth=3,)
    fig.patches.append(arrow)

    #North Atlantic
    x,y = m(-30,35)
    arrow = patches.ConnectionPatch([1.45,-20],[x,y],coordsA=ax[2].transData,coordsB=ax[0].transData,color="black",arrowstyle="-|>",
        mutation_scale=30, linewidth=3,)
    fig.patches.append(arrow)

    #South Pacific
    x,y = m(-145,-25)
    arrow = patches.ConnectionPatch([1.45,-20],[x,y],coordsA=ax[3].transData,coordsB=ax[0].transData,color="black",arrowstyle="-|>",
        mutation_scale=30, linewidth=3,)
    fig.patches.append(arrow)

    #South Atlantic
    x,y = m(-15,-20)
    arrow = patches.ConnectionPatch([0.7,-20],[x,y],coordsA=ax[4].transData,coordsB=ax[0].transData,color="black",arrowstyle="-|>",
        mutation_scale=30, linewidth=3,)
    fig.patches.append(arrow)

    #South Indian
    x,y = m(80,-20)
    arrow = patches.ConnectionPatch([0.7,-20],[x,y],coordsA=ax[5].transData,coordsB=ax[0].transData,color="black",arrowstyle="-|>",
        mutation_scale=30, linewidth=3,)
    fig.patches.append(arrow)

    #Southern Ocean
    x,y = m(0,-60)
    arrow = patches.ConnectionPatch([0.7,-20],[x,y],coordsA=ax[6].transData,coordsB=ax[0].transData,color="black",arrowstyle="-|>",
        mutation_scale=30, linewidth=3,)
    fig.patches.append(arrow)



    eplt.base_tracks_end(ax[0],m)



    fig.savefig('figs/manuscript/figure_5.png',dpi=300)

if plot_figure_5_bio:
    font = {'weight' : 'normal',
            'size'   :10}
    matplotlib.rc('font', **font)
    line_widths = 1

    recaap_file = 'data/RECCAP2_region_masks_all_v20221025.nc'
    # loading the RECAAP ocean definitons
    c = Dataset(recaap_file,'r')
    rec_lon = np.array(c['lon'])-180
    rec_lat = np.array(c['lat'])
    long,latg = np.meshgrid(rec_lon,rec_lat)
    oceans = np.array(c['open_ocean'])
    oceans2 = np.zeros((oceans.shape))
    oceans2[:,180:] = oceans[:,0:180]
    oceans2[:,0:180] = oceans[:,180:]
    print(oceans.shape)
    oceans2[oceans2==0] = np.nan
    for i in range(1,4):
        f = np.where((oceans2 == i) & (latg <0))
        oceans2[f] = i+0.5
    oceans2[oceans2 ==4] = np.nan
    oceans2[oceans2 == 3] = np.nan
    oceans2[oceans2 == 5] = 4
    c.close()
    locas = [2.0,1.0,2.5,1.5,3.5,4.0]
    print(locas)
    fig = plt.figure(figsize=(21,10))
    gs = GridSpec(3,4, figure=fig, wspace=0.3,hspace=0.25,bottom=0.1,top=0.99,left=0.05,right=0.975)
    ax = [fig.add_subplot(gs[:,1:3]),fig.add_subplot(gs[0,0]),fig.add_subplot(gs[1,0]),fig.add_subplot(gs[2,0]),fig.add_subplot(gs[0,3]),fig.add_subplot(gs[1,3]),fig.add_subplot(gs[2,3])]
    m = eplt.base_tracks_start(ax[0],latb=[-70,70],lonb = [-180,180])
    x,y = m(long,latg)
    ax[0].pcolor(x,y,oceans2)

    ax[0].set_xlabel('Longitude',labelpad=30)
    ax[0].set_ylabel('Latitude',labelpad=45)
    val= []
    unc_cal = []
    cum = []
    anti_loc = []
    cou = 0
    for i in  np.unique(eddy_an['track']):
        #print(i)
        if os.path.exists(output_loc+'/'+str(i)+'.nc') == True:
            c = Dataset(output_loc+'/'+str(i)+'.nc','r')
            #c = Dataset(i,'r')
            time = np.array(c['month_time'])
            time2 = []
            for j in range(len(time)):
                time2.append(pyEddy_m.date_con(int(time[j])))

            co2 = np.array(c['flux_in_bio_areaday_cumulative'])
            co2_unc = np.array(c['flux_unc_tot_in_bio_areaday_cumulative'])
            co2_o = np.array(c['flux_out_bio_areaday_cumulative'])
            co2_o_unc = np.array(c['flux_unc_tot_out_bio_areaday_cumulative'])
            co2_p = ((co2-co2_o)/np.abs(co2_o))*100

            cum.append((co2-co2_o)[-1])
            unc_cal.append([co2[-1],co2_unc[-1],co2_o[-1],co2_o_unc[-1]])
            val.append(co2_p[-1])
            lat = np.array(c['month_latitude'])
            lon = np.array(c['month_longitude'])

            f = np.where(np.abs(lat[0] - rec_lat) == np.min(np.abs(lat[0] - rec_lat)))[0]
            g = np.where(np.abs(lon[0] - rec_lon) == np.min(np.abs(lon[0] - rec_lon)))[0]
            anti_loc.append(oceans2[f,g])
            print(oceans2[f,g])
            # cou= cou+1
            # if cou == 60:
            #     break
    anti_loc = np.array(anti_loc)
    unc_cal = np.array(unc_cal)
    val = np.array(val)

    val_cy = []
    unc_cal_cy = []
    cum_cy = []
    cy_loc = []
    cou= 0
    for i in np.unique(eddy_cy['track']):
        #print(i)
        if os.path.exists(output_loc_cy+'/'+str(i)+'.nc') == True:
            c = Dataset(output_loc_cy+'/'+str(i)+'.nc','r')
            #c = Dataset(i,'r')
            time = np.array(c['month_time'])
            time2 = []
            for j in range(len(time)):
                time2.append(pyEddy_m.date_con(int(time[j])))

            co2 = np.array(c['flux_in_bio_areaday_cumulative'])
            co2_unc = np.array(c['flux_unc_tot_in_bio_areaday_cumulative'])
            co2_o = np.array(c['flux_out_bio_areaday_cumulative'])
            co2_o_unc = np.array(c['flux_unc_tot_out_bio_areaday_cumulative'])
            co2_p = ((co2-co2_o)/np.abs(co2_o))*100

            cum_cy.append((co2-co2_o)[-1])
            unc_cal_cy.append([co2[-1],co2_unc[-1],co2_o[-1],co2_o_unc[-1]])
            val_cy.append(co2_p[-1])
            lat = np.array(c['month_latitude'])
            lon = np.array(c['month_longitude'])

            f = np.where(np.abs(lat[0] - rec_lat) == np.min(np.abs(lat[0] - rec_lat)))[0]
            g = np.where(np.abs(lon[0] - rec_lon) == np.min(np.abs(lon[0] - rec_lon)))[0]
            cy_loc.append(oceans2[f,g])
            print(oceans2[f,g])
            # cou= cou+1
            # if cou == 60:
            #     break
    cy_loc = np.array(cy_loc)
    unc_cal_cy = np.array(unc_cal_cy)
    val_cy = np.array(val_cy)

    cou = 1
    xlims = [0.85,1.45]
    lab= 'Change in eddy CO${_2}$ flux compared \nto the surrounding water CO${_2}$ flux (%)'
    ax[0].text(0.92,1.06,f'({let[0]})',transform=ax[0].transAxes,va='top',fontweight='bold',fontsize = 14)
    for i in locas:
        f = np.where((anti_loc == i))[0]
        f = f[np.where(np.isnan(val[f]) == 0)]
        g = np.where((cy_loc == i))[0]
        g = g[np.where(np.isnan(val_cy[g]) == 0)]
        # print(f)

        unc_anti = unc_prop(unc_cal[f])
        unc_cy = unc_prop(unc_cal_cy[g])
        ax[cou].boxplot([val[f],val_cy[g]],boxprops = dict(linewidth=line_widths),labels=[f'Anticyclonic (N = {len(f)})\nMedian = {str(np.round(np.median(val[f]),1))} $\pm$ {str(np.round(unc_anti[2],1))} %',f'Cyclonic (N = {len(g)})\nMedian = {str(np.round(np.median(val_cy[g]),1))} $\pm$ {str(np.round(unc_cy[2],1))} %'],
            medianprops = dict(linewidth=line_widths,color='r'),whiskerprops= dict(linewidth=line_widths),capprops=dict(linewidth=line_widths),positions=[1,1.3],widths=0.2)
        med = np.median(val[f])
        ax[cou].fill_between([0.88,1.12],med-unc_anti[2],med+unc_anti[2],color='r',alpha=0.2)
        ax[cou].fill_between([0.88,1.12],med-(unc_anti[2]/2),med+(unc_anti[2]/2),color='r',alpha=0.4)

        med = np.median(val_cy[g])
        ax[cou].fill_between([1.18,1.42],med-unc_cy[2],med+unc_cy[2],color='r',alpha=0.2)
        ax[cou].fill_between([1.18,1.42],med-(unc_cy[2]/2),med+(unc_cy[2]/2),color='r',alpha=0.4)
        ax[cou].set_ylim([-50,50])
        ax[cou].set_xlim([0.85,1.45])
        ax[cou].plot(xlims,[0,0],'k--')
        ax[cou].set_ylabel(lab)
        ax[cou].text(1.03,0.92,f'({let[cou]})',transform=ax[cou].transAxes,va='top',fontweight='bold',fontsize = 14)
        cou=cou+1

    #North Pacific
    x,y = m(-145,35)
    arrow = patches.ConnectionPatch([1.45,-20],[x,y],coordsA=ax[1].transData,coordsB=ax[0].transData,color="black",arrowstyle="-|>",
        mutation_scale=30, linewidth=3,)
    fig.patches.append(arrow)

    #North Atlantic
    x,y = m(-30,35)
    arrow = patches.ConnectionPatch([1.45,-20],[x,y],coordsA=ax[2].transData,coordsB=ax[0].transData,color="black",arrowstyle="-|>",
        mutation_scale=30, linewidth=3,)
    fig.patches.append(arrow)

    #South Pacific
    x,y = m(-145,-25)
    arrow = patches.ConnectionPatch([1.45,-20],[x,y],coordsA=ax[3].transData,coordsB=ax[0].transData,color="black",arrowstyle="-|>",
        mutation_scale=30, linewidth=3,)
    fig.patches.append(arrow)

    #South Atlantic
    x,y = m(-15,-20)
    arrow = patches.ConnectionPatch([0.7,-20],[x,y],coordsA=ax[4].transData,coordsB=ax[0].transData,color="black",arrowstyle="-|>",
        mutation_scale=30, linewidth=3,)
    fig.patches.append(arrow)

    #South Indian
    x,y = m(80,-20)
    arrow = patches.ConnectionPatch([0.7,-20],[x,y],coordsA=ax[5].transData,coordsB=ax[0].transData,color="black",arrowstyle="-|>",
        mutation_scale=30, linewidth=3,)
    fig.patches.append(arrow)

    #Southern Ocean
    x,y = m(0,-60)
    arrow = patches.ConnectionPatch([0.7,-20],[x,y],coordsA=ax[6].transData,coordsB=ax[0].transData,color="black",arrowstyle="-|>",
        mutation_scale=30, linewidth=3,)
    fig.patches.append(arrow)



    eplt.base_tracks_end(ax[0],m)



    fig.savefig('figs/manuscript/figure_5_bio.png',dpi=300)

if plot_figure_5_compare:
    font = {'weight' : 'normal',
            'size'   :10}
    matplotlib.rc('font', **font)
    line_widths = 1

    recaap_file = 'data/RECCAP2_region_masks_all_v20221025.nc'
    # loading the RECAAP ocean definitons
    c = Dataset(recaap_file,'r')
    rec_lon = np.array(c['lon'])-180
    rec_lat = np.array(c['lat'])
    long,latg = np.meshgrid(rec_lon,rec_lat)
    oceans = np.array(c['open_ocean'])
    oceans2 = np.zeros((oceans.shape))
    oceans2[:,180:] = oceans[:,0:180]
    oceans2[:,0:180] = oceans[:,180:]
    print(oceans.shape)
    oceans2[oceans2==0] = np.nan
    for i in range(1,4):
        f = np.where((oceans2 == i) & (latg <0))
        oceans2[f] = i+0.5
    oceans2[oceans2 ==4] = np.nan
    oceans2[oceans2 == 3] = np.nan
    oceans2[oceans2 == 5] = 4
    c.close()
    locas = [2.0,1.0,2.5,1.5,3.5,4.0]
    print(locas)
    fig = plt.figure(figsize=(21,10))
    gs = GridSpec(3,4, figure=fig, wspace=0.3,hspace=0.25,bottom=0.1,top=0.99,left=0.05,right=0.975)
    ax = [fig.add_subplot(gs[:,1:3]),fig.add_subplot(gs[0,0]),fig.add_subplot(gs[1,0]),fig.add_subplot(gs[2,0]),fig.add_subplot(gs[0,3]),fig.add_subplot(gs[1,3]),fig.add_subplot(gs[2,3])]
    m = eplt.base_tracks_start(ax[0],latb=[-70,70],lonb = [-180,180])
    x,y = m(long,latg)
    ax[0].pcolor(x,y,oceans2)

    ax[0].set_xlabel('Longitude',labelpad=30)
    ax[0].set_ylabel('Latitude',labelpad=45)
    val= []
    val_ph = []
    unc_cal = []
    unc_cal_ph = []
    anti_loc = []
    cou = 0
    for i in  np.unique(eddy_an['track']):
        #print(i)
        if os.path.exists(output_loc+'/'+str(i)+'.nc') == True:
            c = Dataset(output_loc+'/'+str(i)+'.nc','r')
            #c = Dataset(i,'r')
            time = np.array(c['month_time'])
            time2 = []
            for j in range(len(time)):
                time2.append(pyEddy_m.date_con(int(time[j])))

            co2 = np.array(c['flux_in_bio_areaday_cumulative'])
            co2_unc = np.array(c['flux_unc_tot_in_bio_areaday_cumulative'])
            co2_o = np.array(c['flux_out_bio_areaday_cumulative'])
            co2_o_unc = np.array(c['flux_unc_tot_out_bio_areaday_cumulative'])
            co2_p = ((co2-co2_o)/np.abs(co2_o))*100

            unc_cal.append([co2[-1],co2_unc[-1],co2_o[-1],co2_o_unc[-1]])
            val.append(co2_p[-1])
            lat = np.array(c['month_latitude'])
            lon = np.array(c['month_longitude'])

            f = np.where(np.abs(lat[0] - rec_lat) == np.min(np.abs(lat[0] - rec_lat)))[0]
            g = np.where(np.abs(lon[0] - rec_lon) == np.min(np.abs(lon[0] - rec_lon)))[0]
            anti_loc.append(oceans2[f,g])
            print(oceans2[f,g])

            co2 = np.array(c['flux_in_physics_areaday_cumulative'])
            co2_unc = np.array(c['flux_unc_tot_in_physics_areaday_cumulative'])
            co2_o = np.array(c['flux_out_physics_areaday_cumulative'])
            co2_o_unc = np.array(c['flux_unc_tot_out_physics_areaday_cumulative'])
            co2_p = ((co2-co2_o)/np.abs(co2_o))*100
            unc_cal_ph.append([co2[-1],co2_unc[-1],co2_o[-1],co2_o_unc[-1]])
            val_ph.append(co2_p[-1])
            c.close()
            # cou= cou+1
            # if cou == 60:
            #     break
    anti_loc = np.array(anti_loc)
    unc_cal = np.array(unc_cal)
    unc_cal_ph = np.array(unc_cal_ph)
    val = np.array(val)
    val_ph = np.array(val_ph)

    val_cy = []
    unc_cal_cy = []
    val_cy_ph = []
    unc_cal_cy_ph = []
    cy_loc = []
    cou= 0
    for i in np.unique(eddy_cy['track']):
        #print(i)
        if os.path.exists(output_loc_cy+'/'+str(i)+'.nc') == True:
            c = Dataset(output_loc_cy+'/'+str(i)+'.nc','r')
            #c = Dataset(i,'r')
            time = np.array(c['month_time'])
            time2 = []
            for j in range(len(time)):
                time2.append(pyEddy_m.date_con(int(time[j])))

            co2 = np.array(c['flux_in_bio_areaday_cumulative'])
            co2_unc = np.array(c['flux_unc_tot_in_bio_areaday_cumulative'])
            co2_o = np.array(c['flux_out_bio_areaday_cumulative'])
            co2_o_unc = np.array(c['flux_unc_tot_out_bio_areaday_cumulative'])
            co2_p = ((co2-co2_o)/np.abs(co2_o))*100


            unc_cal_cy.append([co2[-1],co2_unc[-1],co2_o[-1],co2_o_unc[-1]])
            val_cy.append(co2_p[-1])
            lat = np.array(c['month_latitude'])
            lon = np.array(c['month_longitude'])

            f = np.where(np.abs(lat[0] - rec_lat) == np.min(np.abs(lat[0] - rec_lat)))[0]
            g = np.where(np.abs(lon[0] - rec_lon) == np.min(np.abs(lon[0] - rec_lon)))[0]
            cy_loc.append(oceans2[f,g])
            print(oceans2[f,g])

            co2 = np.array(c['flux_in_physics_areaday_cumulative'])
            co2_unc = np.array(c['flux_unc_tot_in_physics_areaday_cumulative'])
            co2_o = np.array(c['flux_out_physics_areaday_cumulative'])
            co2_o_unc = np.array(c['flux_unc_tot_out_physics_areaday_cumulative'])
            co2_p = ((co2-co2_o)/np.abs(co2_o))*100
            unc_cal_cy_ph.append([co2[-1],co2_unc[-1],co2_o[-1],co2_o_unc[-1]])
            val_cy_ph.append(co2_p[-1])
            # cou= cou+1
            # if cou == 60:
            #     break
    cy_loc = np.array(cy_loc)
    unc_cal_cy = np.array(unc_cal_cy)
    val_cy = np.array(val_cy)
    unc_cal_cy_ph = np.array(unc_cal_cy_ph)
    val_cy_ph = np.array(val_cy_ph)

    cou = 1
    xlims = [0.85,1.45]
    lab= 'Change in eddy CO${_2}$ flux compared \nto the surrounding water CO${_2}$ flux (%)'
    ax[0].text(0.92,1.06,f'({let[0]})',transform=ax[0].transAxes,va='top',fontweight='bold',fontsize = 14)
    for i in locas:
        f = np.where((anti_loc == i))[0]
        f = f[np.where(np.isnan(val[f]) == 0)]
        g = np.where((cy_loc == i))[0]
        g = g[np.where(np.isnan(val_cy[g]) == 0)]
        # print(f)

        unc_anti = unc_prop(unc_cal[f])
        unc_cy = unc_prop(unc_cal_cy[g])

        unc_anti_ph = unc_prop(unc_cal_ph[f])
        unc_cy_ph = unc_prop(unc_cal_cy_ph[g])

        ax[cou].boxplot([val[f],val_cy[g]],boxprops = dict(linewidth=line_widths),
            labels=[f'Anticyclonic (N = {len(f)})\nC Median = {str(np.round(np.median(val[f]),1))} $\pm$ {str(np.round(unc_anti[2],1))} %\nP Median = {str(np.round(np.median(val_ph[f]),1))} $\pm$ {str(np.round(unc_anti_ph[2],1))} %',
            f'Cyclonic (N = {len(g)})\nC Median = {str(np.round(np.median(val_cy[g]),1))} $\pm$ {str(np.round(unc_cy[2],1))} %\nP Median = {str(np.round(np.median(val_cy_ph[g]),1))} $\pm$ {str(np.round(unc_cy_ph[2],1))} %'],
            medianprops = dict(linewidth=line_widths,color='r'),whiskerprops= dict(linewidth=line_widths),capprops=dict(linewidth=line_widths),positions=[1,1.3],widths=0.2)
        med = np.median(val[f])
        ax[cou].plot([0.88,1.12],[med,med],color='r')
        ax[cou].fill_between([0.88,1.12],med-unc_anti[2],med+unc_anti[2],color='r',alpha=0.2)
        ax[cou].fill_between([0.88,1.12],med-(unc_anti[2]/2),med+(unc_anti[2]/2),color='r',alpha=0.4)

        med = np.median(val_ph[f])
        ax[cou].plot([0.88,1.12],[med,med],color='b')
        ax[cou].fill_between([0.88,1.12],med-unc_anti_ph[2],med+unc_anti_ph[2],color='b',alpha=0.2)
        ax[cou].fill_between([0.88,1.12],med-(unc_anti_ph[2]/2),med+(unc_anti_ph[2]/2),color='b',alpha=0.4)

        med = np.median(val_cy[g])
        ax[cou].plot([1.18,1.42],[med,med],color='r')
        ax[cou].fill_between([1.18,1.42],med-unc_cy[2],med+unc_cy[2],color='r',alpha=0.2)
        ax[cou].fill_between([1.18,1.42],med-(unc_cy[2]/2),med+(unc_cy[2]/2),color='r',alpha=0.4)

        med = np.median(val_cy_ph[g])
        ax[cou].plot([1.18,1.42],[med,med],color='b')
        ax[cou].fill_between([1.18,1.42],med-unc_cy_ph[2],med+unc_cy_ph[2],color='b',alpha=0.2)
        ax[cou].fill_between([1.18,1.42],med-(unc_cy_ph[2]/2),med+(unc_cy_ph[2]/2),color='b',alpha=0.4)
        ax[cou].set_ylim([-50,50])
        ax[cou].set_xlim([0.85,1.45])
        ax[cou].plot(xlims,[0,0],'k--')
        ax[cou].set_ylabel(lab)
        ax[cou].text(1.03,0.92,f'({let[cou]})',transform=ax[cou].transAxes,va='top',fontweight='bold',fontsize = 14)
        cou=cou+1

    #North Pacific
    x,y = m(-145,35)
    arrow = patches.ConnectionPatch([1.45,-20],[x,y],coordsA=ax[1].transData,coordsB=ax[0].transData,color="black",arrowstyle="-|>",
        mutation_scale=30, linewidth=3,)
    fig.patches.append(arrow)

    #North Atlantic
    x,y = m(-30,35)
    arrow = patches.ConnectionPatch([1.45,-20],[x,y],coordsA=ax[2].transData,coordsB=ax[0].transData,color="black",arrowstyle="-|>",
        mutation_scale=30, linewidth=3,)
    fig.patches.append(arrow)

    #South Pacific
    x,y = m(-145,-25)
    arrow = patches.ConnectionPatch([1.45,-20],[x,y],coordsA=ax[3].transData,coordsB=ax[0].transData,color="black",arrowstyle="-|>",
        mutation_scale=30, linewidth=3,)
    fig.patches.append(arrow)

    #South Atlantic
    x,y = m(-15,-20)
    arrow = patches.ConnectionPatch([0.7,-20],[x,y],coordsA=ax[4].transData,coordsB=ax[0].transData,color="black",arrowstyle="-|>",
        mutation_scale=30, linewidth=3,)
    fig.patches.append(arrow)

    #South Indian
    x,y = m(80,-20)
    arrow = patches.ConnectionPatch([0.7,-20],[x,y],coordsA=ax[5].transData,coordsB=ax[0].transData,color="black",arrowstyle="-|>",
        mutation_scale=30, linewidth=3,)
    fig.patches.append(arrow)

    #Southern Ocean
    x,y = m(0,-60)
    arrow = patches.ConnectionPatch([0.7,-20],[x,y],coordsA=ax[6].transData,coordsB=ax[0].transData,color="black",arrowstyle="-|>",
        mutation_scale=30, linewidth=3,)
    fig.patches.append(arrow)



    eplt.base_tracks_end(ax[0],m)



    fig.savefig('figs/manuscript/figure_5_compare.png',dpi=300)

def unc_prop_sum(inps,ens = 1000):
    out = []
    for i in range(ens):
        in_flux = inps[:,0] + (inps[:,1] * np.random.normal(0, 0.5, len(inps[:,1])))
        out_flux = inps[:,2] + (inps[:,3] * np.random.normal(0, 0.5, len(inps[:,3])))
        out.append(np.nansum(in_flux-out_flux))
    out = np.array(out)
    print(f'Mean = {np.mean(out)}')
    print(f'Median = {np.median(out)}')
    print(f'2 standard dev = {np.std(out)*2}')
    return [np.mean(out),np.median(out),np.std(out)*2]

if estimate_cumulative:

    recaap_file = 'data/RECCAP2_region_masks_all_v20221025.nc'
    # loading the RECAAP ocean definitons
    c = Dataset(recaap_file,'r')
    rec_lon = np.array(c['lon'])-180
    rec_lat = np.array(c['lat'])
    long,latg = np.meshgrid(rec_lon,rec_lat)
    oceans = np.array(c['open_ocean'])
    oceans2 = np.zeros((oceans.shape))
    oceans2[:,180:] = oceans[:,0:180]
    oceans2[:,0:180] = oceans[:,180:]
    print(oceans.shape)
    oceans2[oceans2==0] = np.nan
    for i in range(1,4):
        f = np.where((oceans2 == i) & (latg <0))
        oceans2[f] = i+0.5
    oceans2[oceans2 ==4] = np.nan
    oceans2[oceans2 == 3] = np.nan
    oceans2[oceans2 == 5] = 4
    c.close()
    """
    Anticyclonic Eddies
    """

    val= []
    unc_cal = []
    cum = []
    anti_loc = []
    cou = 0
    for i in  np.unique(eddy_an['track']):
        #print(i)
        if os.path.exists(output_loc+'/'+str(i)+'.nc') == True:
            c = Dataset(output_loc+'/'+str(i)+'.nc','r')
            #c = Dataset(i,'r')
            time = np.array(c['month_time'])
            time2 = []
            for j in range(len(time)):
                time2.append(pyEddy_m.date_con(int(time[j])))

            co2 = np.array(c['flux_in_physics_areaday_cumulative'])
            co2_unc = np.array(c['flux_unc_tot_in_physics_areaday_cumulative'])
            co2_o = np.array(c['flux_out_physics_areaday_cumulative'])
            co2_o_unc = np.array(c['flux_unc_tot_out_physics_areaday_cumulative'])
            co2_p = ((co2-co2_o)/np.abs(co2_o))*100

            cum.append((co2-co2_o)[-1])
            unc_cal.append([co2[-1],co2_unc[-1],co2_o[-1],co2_o_unc[-1]])
            val.append(co2_p[-1])
            lat = np.array(c['month_latitude'])
            lon = np.array(c['month_longitude'])

            f = np.where(np.abs(lat[0] - rec_lat) == np.min(np.abs(lat[0] - rec_lat)))[0]
            g = np.where(np.abs(lon[0] - rec_lon) == np.min(np.abs(lon[0] - rec_lon)))[0]
            anti_loc.append(oceans2[f,g])
            #print(oceans2[f,g])
            c.close()
            # cou= cou+1
            # if cou == 60:
            #     break
    anti_loc = np.array(anti_loc)
    unc_cal = np.array(unc_cal)
    val = np.array(val)

    val_cy = []
    unc_cal_cy = []
    cum_cy = []
    cy_loc = []
    cou= 0
    for i in np.unique(eddy_cy['track']):
        #print(i)
        if os.path.exists(output_loc_cy+'/'+str(i)+'.nc') == True:
            c = Dataset(output_loc_cy+'/'+str(i)+'.nc','r')
            #c = Dataset(i,'r')
            time = np.array(c['month_time'])
            time2 = []
            for j in range(len(time)):
                time2.append(pyEddy_m.date_con(int(time[j])))

            co2 = np.array(c['flux_in_physics_areaday_cumulative'])
            co2_unc = np.array(c['flux_unc_tot_in_physics_areaday_cumulative'])
            co2_o = np.array(c['flux_out_physics_areaday_cumulative'])
            co2_o_unc = np.array(c['flux_unc_tot_out_physics_areaday_cumulative'])
            co2_p = ((co2-co2_o)/np.abs(co2_o))*100

            cum_cy.append((co2-co2_o)[-1])
            unc_cal_cy.append([co2[-1],co2_unc[-1],co2_o[-1],co2_o_unc[-1]])
            val_cy.append(co2_p[-1])
            f = np.where(np.abs(lat[0] - rec_lat) == np.min(np.abs(lat[0] - rec_lat)))[0]
            g = np.where(np.abs(lon[0] - rec_lon) == np.min(np.abs(lon[0] - rec_lon)))[0]
            cy_loc.append(oceans2[f,g])
            c.close()
    cy_loc = np.array(cy_loc)
    unc_cal_cy = np.array(unc_cal_cy)
    val_cy = np.array(val_cy)

    print(unc_prop_sum(unc_cal))
    print(unc_prop_sum(unc_cal_cy))
    print(unc_prop_sum(np.concatenate((unc_cal,unc_cal_cy))))

def scatter_socat_data(file,ax,c=[200,600],unc = False,text=True,lab='Neural network fCO$_{2 (sw)}$ ($\mu$atm)',month_split=False,months=[]):
    c = np.array(c)
    data = np.loadtxt(file,delimiter=',',skiprows=1)
    if month_split:
        t=0
        for j in range(2):
            for i in range(len(months[0])):
                if j ==0:
                    f = np.where((data[:,2]>0) & (data[:,1] == months[j][i]))[0]
                else:
                    f = np.where((data[:,2]<0) & (data[:,1] == months[j][i]))[0]
                #print(f)
                if t==0:
                    v = f
                    t=1
                else:
                    v = np.concatenate((v,f))
                #print(v)
        data = data[v,:]

    ax.scatter(data[:,4], data[:,5],s=4)
    if unc:
        ax.scatter(data[:,4], data[:,5],s=4,color='r',zorder=2)
        ax.errorbar(data[:,4],data[:,5],yerr=data[:,6],color='tab:blue',linestyle='None',zorder=1)
    ax.plot(np.array(c),np.array(c),'k-')
    ax.set_xlim(c)
    ax.set_ylim(c)
    ax.set_xlabel('SOCAT fCO$_{2 (sw)}$ ($\mu$atm)')
    ax.set_ylabel(lab)
    stats = unweight(data[:,4], data[:,5],ax,c=c,text=text)
    #ax.plot(c,c*stats[1]+stats[2],'k--')
    weighted(data[:,4], data[:,5],1/data[:,6],ax,c=c,text=text)

def residual_spatial_plot(file,ax,c=[-40,40],lab=''):
    c = np.array(c)
    data = np.loadtxt(file,delimiter=',',skiprows=1)
    m = eplt.base_tracks_start(ax,latb=[-70,70],lonb = [-180,180])
    x,y = m(data[:,3],data[:,2])
    sc_out = ax.scatter(x,y,c=data[:,4]-data[:,5],cmap = cmocean.cm.balance,vmin=c[0],vmax=c[1])
    eplt.base_tracks_end(ax,m)
    return sc_out

def unweight(x,y,ax,c,unit = '$\mu$atm',plot=False,loc = [0.54,0.35],text = True):
    """
    Function to calculate the unweighted statistics and add them to a scatter plot (well any plot really, in the bottom right corner)
    """
    stats_un = ws.unweighted_stats(x,y,'val')
    if plot:
        h2 = ax.plot(c,c*stats_un['slope']+stats_un['intercept'],'k--',zorder=5, label = 'Unweighted')
    rmsd = '%.2f' %np.round(stats_un['rmsd'],2)
    bias = '%.2f' %np.round(stats_un['rel_bias'],2)
    sl = '%.2f' %np.round(stats_un['slope'],2)
    ip = '%.2f' %np.round(stats_un['intercept'],2)
    n = stats_un['n']
    if text:
        ax.text(loc[0],loc[1],f'Unweighted Stats\nRMSD = {rmsd} {unit}\nBias = {bias} {unit}\nSlope = {sl}\nIntercept = {ip}\nN = {n}',transform=ax.transAxes,va='top')
    return [stats_un['rmsd'],stats_un['slope'],stats_un['intercept']]

def weighted(x,y,weights,ax,c,unit = '$\mu$atm',plot=True,text=True):
    """
    Function to calculate the weighted statistics and add them to a scatter plot (well any plot really, in the top left corner)
    """
    #weights = 1/(np.sqrt(y_test_all[:,1]**2 + y_test_preds_all[:,1]**2))
    stats = ws.weighted_stats(x,y,weights,'val')
    if plot:
        h1 = ax.plot(c,c*stats['slope']+stats['intercept'],'k--',zorder=5, label = 'Weighted')
    rmsd = '%.2f' % np.round(stats['rmsd'],2)
    bias = '%.2f' %np.round(stats['rel_bias'],2)
    sl = '%.2f' %np.round(stats['slope'],2)
    ip = '%.2f' %np.round(stats['intercept'],2)
    n = stats['n']
    if text:
        ax.text(0.02,0.95,f'Weighted Stats\nRMSD = {rmsd} {unit}\nBias = {bias} {unit}\nSlope = {sl}\nIntercept = {ip}\nN = {n}',transform=ax.transAxes,va='top')
    return h1

if plot_figure_socat:

    anti_file = 'data/v0-3/fco2_out.csv'
    cy_file = 'data/v0-3/fco2_out_cy.csv'

    font = {'weight' : 'normal',
            'size'   :20}
    matplotlib.rc('font', **font)
    fig = plt.figure(figsize=(20,14))
    gs = GridSpec(2,2, figure=fig, wspace=0.25,hspace=0.15,bottom=0.1,top=0.95,left=0.1,right=0.95)
    ax = [fig.add_subplot(gs[0,0]), fig.add_subplot(gs[0,1]),fig.add_subplot(gs[1,0]),fig.add_subplot(gs[1,1])]

    scatter_socat_data(anti_file,ax[0])
    scatter_socat_data(cy_file,ax[2])
    scatter_socat_data(anti_file,ax[1],unc=True,text=False)
    scatter_socat_data(cy_file,ax[3],unc=True,text=False)
    for i in range(4):
        ax[i].text(0.90,0.96,f'({let[i]})',transform=ax[i].transAxes,va='top',fontweight='bold',fontsize = 24)
    fig.savefig('figs/manuscript/socat_figure.png',dpi=300)


    font = {'weight' : 'normal',
            'size'   :20}
    matplotlib.rc('font', **font)
    fig = plt.figure(figsize=(30,14))
    gs = GridSpec(2,4, figure=fig, wspace=0.25,hspace=0.15,bottom=0.1,top=0.95,left=0.05,right=1.05)
    ax = [fig.add_subplot(gs[0,0]), fig.add_subplot(gs[0,1]),fig.add_subplot(gs[0,2:4]),fig.add_subplot(gs[1,0]),fig.add_subplot(gs[1,1]),fig.add_subplot(gs[1,2:4])]

    scatter_socat_data(anti_file,ax[0])
    scatter_socat_data(cy_file,ax[3])
    scatter_socat_data(anti_file,ax[1],unc=True,text=False)
    scatter_socat_data(cy_file,ax[4],unc=True,text=False)
    sc_out = residual_spatial_plot(anti_file,ax[2])
    residual_spatial_plot(cy_file,ax[5])

    cba_ax = fig.add_axes([0.59,0.2,0.02,0.6])
    cba = fig.colorbar(sc_out,cax=cba_ax)
    cba.set_label('Difference between nerual netowrk and SOCAT fCO$_{2 (sw)}$ ($\mu$atm)\n(SOCAT - nerual network)')
    cba_ax.yaxis.set_ticks_position('left')
    cba_ax.yaxis.set_label_position('left')
    for i in range(6):
        ax[i].text(0.90,0.96,f'({let[i]})',transform=ax[i].transAxes,va='top',fontweight='bold',fontsize = 24)
    fig.savefig('figs/manuscript/socat_figure_extra.png',dpi=300)

    font = {'weight' : 'normal',
            'size'   :20}
    matplotlib.rc('font', **font)
    fig = plt.figure(figsize=(34,14))
    gs = GridSpec(2,4, figure=fig, wspace=0.35,hspace=0.2,bottom=0.05,top=0.97,left=0.07,right=0.97)
    ax = [fig.add_subplot(gs[0,0]), fig.add_subplot(gs[0,1]),fig.add_subplot(gs[0,2]),fig.add_subplot(gs[0,3]),fig.add_subplot(gs[1,0]),fig.add_subplot(gs[1,1]),fig.add_subplot(gs[1,2]),fig.add_subplot(gs[1,3])]

    scatter_socat_data(anti_file,ax[0],month_split=True,months=[[12,1,2],[6,7,8]])
    scatter_socat_data(cy_file,ax[1],month_split=True,months=[[12,1,2],[6,7,8]])

    ax[0].text(-0.25,0.58,'Winter',transform=ax[0].transAxes,va='top',fontweight='bold',fontsize = 32,horizontalalignment='center', verticalalignment='center',rotation=90)
    scatter_socat_data(anti_file,ax[2],month_split=True,months=[[3,4,5],[9,10,11]])
    scatter_socat_data(cy_file,ax[3],month_split=True,months=[[3,4,5],[9,10,11]])

    ax[2].text(-0.25,0.58,'Spring',transform=ax[2].transAxes,va='top',fontweight='bold',fontsize = 32,horizontalalignment='center', verticalalignment='center',rotation=90)

    scatter_socat_data(anti_file,ax[4],month_split=True,months=[[6,7,8],[12,1,2]])
    scatter_socat_data(cy_file,ax[5],month_split=True,months=[[6,7,8],[12,1,2]])

    ax[4].text(-0.25,0.58,'Summer',transform=ax[4].transAxes,va='top',fontweight='bold',fontsize = 32,horizontalalignment='center', verticalalignment='center',rotation=90)
    scatter_socat_data(anti_file,ax[6],month_split=True,months=[[9,10,11],[3,4,5]])
    scatter_socat_data(cy_file,ax[7],month_split=True,months=[[9,10,11],[3,4,5]])

    ax[6].text(-0.25,0.58,'Autumn',transform=ax[6].transAxes,va='top',fontweight='bold',fontsize = 32,horizontalalignment='center', verticalalignment='center',rotation=90)
    for i in range(8):
        ax[i].text(0.90,0.96,f'({let[i]})',transform=ax[i].transAxes,va='top',fontweight='bold',fontsize = 24)
    fig.savefig('figs/manuscript/socat_figure_extra_seasonal.png',dpi=300)

if plot_figure_socat_bio:

    anti_file = 'data/v0-3/fco2_out_bio.csv'
    cy_file = 'data/v0-3/fco2_out_bio_cy.csv'
    font = {'weight' : 'normal',
            'size'   :20}
    matplotlib.rc('font', **font)
    fig = plt.figure(figsize=(20,14))
    gs = GridSpec(2,2, figure=fig, wspace=0.25,hspace=0.15,bottom=0.1,top=0.95,left=0.1,right=0.95)
    ax = [fig.add_subplot(gs[0,0]), fig.add_subplot(gs[0,1]),fig.add_subplot(gs[1,0]),fig.add_subplot(gs[1,1])]

    scatter_socat_data(anti_file,ax[0],lab = 'Neural network fCO$_{2 (sw)}$ ($\mu$atm)')
    scatter_socat_data(cy_file,ax[2],lab = 'Neural network fCO$_{2 (sw)}$ ($\mu$atm)')
    scatter_socat_data(anti_file,ax[1],unc=True,text=False, lab = 'Neural network fCO$_{2 (sw)}$ ($\mu$atm)')
    scatter_socat_data(cy_file,ax[3],unc=True,text=False, lab = 'Neural network fCO$_{2 (sw)}$ ($\mu$atm)')
    for i in range(4):
        ax[i].text(0.90,0.96,f'({let[i]})',transform=ax[i].transAxes,va='top',fontweight='bold',fontsize = 24)
    fig.savefig('figs/manuscript/socat_figure_bio.png',dpi=300)

    font = {'weight' : 'normal',
            'size'   :20}
    matplotlib.rc('font', **font)
    fig = plt.figure(figsize=(34,14))
    gs = GridSpec(2,4, figure=fig, wspace=0.35,hspace=0.2,bottom=0.05,top=0.97,left=0.07,right=0.97)
    ax = [fig.add_subplot(gs[0,0]), fig.add_subplot(gs[0,1]),fig.add_subplot(gs[0,2]),fig.add_subplot(gs[0,3]),fig.add_subplot(gs[1,0]),fig.add_subplot(gs[1,1]),fig.add_subplot(gs[1,2]),fig.add_subplot(gs[1,3])]

    scatter_socat_data(anti_file,ax[0],month_split=True,months=[[12,1,2],[6,7,8]])
    scatter_socat_data(cy_file,ax[1],month_split=True,months=[[12,1,2],[6,7,8]])

    ax[0].text(-0.25,0.58,'Winter',transform=ax[0].transAxes,va='top',fontweight='bold',fontsize = 32,horizontalalignment='center', verticalalignment='center',rotation=90)
    scatter_socat_data(anti_file,ax[2],month_split=True,months=[[3,4,5],[9,10,11]])
    scatter_socat_data(cy_file,ax[3],month_split=True,months=[[3,4,5],[9,10,11]])

    ax[2].text(-0.25,0.58,'Spring',transform=ax[2].transAxes,va='top',fontweight='bold',fontsize = 32,horizontalalignment='center', verticalalignment='center',rotation=90)

    scatter_socat_data(anti_file,ax[4],month_split=True,months=[[6,7,8],[12,1,2]])
    scatter_socat_data(cy_file,ax[5],month_split=True,months=[[6,7,8],[12,1,2]])

    ax[4].text(-0.25,0.58,'Summer',transform=ax[4].transAxes,va='top',fontweight='bold',fontsize = 32,horizontalalignment='center', verticalalignment='center',rotation=90)
    scatter_socat_data(anti_file,ax[6],month_split=True,months=[[9,10,11],[3,4,5]])
    scatter_socat_data(cy_file,ax[7],month_split=True,months=[[9,10,11],[3,4,5]])

    ax[6].text(-0.25,0.58,'Autumn',transform=ax[6].transAxes,va='top',fontweight='bold',fontsize = 32,horizontalalignment='center', verticalalignment='center',rotation=90)
    for i in range(8):
        ax[i].text(0.90,0.96,f'({let[i]})',transform=ax[i].transAxes,va='top',fontweight='bold',fontsize = 24)
    fig.savefig('figs/manuscript/socat_figure_bio_extra_seasonal.png',dpi=300)

if decadal_stats:
    timeperiods = [[datetime.datetime(1993,1,1),datetime.datetime(1999,12,31)],
        [datetime.datetime(2000,1,1),datetime.datetime(2009,12,31)],
        [datetime.datetime(2010,1,1),datetime.datetime(2020,12,31)]]
    out = []
    for t in range(len(timeperiods)):
        eddy_an2 = pyEddy_m.box_split(eddy_an,[-80,80],[-180,180],timeperiods[t],strict_time=False)
        eddy_cy2 = pyEddy_m.box_split(eddy_cy,[-80,80],[-180,180],timeperiods[t],strict_time=False)

        val= []
        unc_cal = []
        for i in  np.unique(eddy_an2['track']):
            #print(i)
            if os.path.exists(output_loc+'/'+str(i)+'.nc') == True:
                c = Dataset(output_loc+'/'+str(i)+'.nc','r')
                #c = Dataset(i,'r')

                co2 = np.array(c['flux_in_physics_areaday_cumulative'])
                co2_unc = np.array(c['flux_unc_tot_in_physics_areaday_cumulative'])
                co2_o = np.array(c['flux_out_physics_areaday_cumulative'])
                co2_o_unc = np.array(c['flux_unc_tot_out_physics_areaday_cumulative'])
                co2_p = ((co2-co2_o)/np.abs(co2_o))*100

                unc_cal.append([co2[-1],co2_unc[-1],co2_o[-1],co2_o_unc[-1]])
                val.append(co2_p[-1])
                c.close()

        unc_cal = np.array(unc_cal)
        val = np.array(val)

        unc_anti = unc_prop(unc_cal)
        med_anti = np.median(val)
        len_anti = len(val)

        val= []
        unc_cal = []
        for i in  np.unique(eddy_cy2['track']):
            #print(i)
            if os.path.exists(output_loc_cy+'/'+str(i)+'.nc') == True:
                c = Dataset(output_loc_cy+'/'+str(i)+'.nc','r')
                #c = Dataset(i,'r')

                co2 = np.array(c['flux_in_physics_areaday_cumulative'])
                co2_unc = np.array(c['flux_unc_tot_in_physics_areaday_cumulative'])
                co2_o = np.array(c['flux_out_physics_areaday_cumulative'])
                co2_o_unc = np.array(c['flux_unc_tot_out_physics_areaday_cumulative'])
                co2_p = ((co2-co2_o)/np.abs(co2_o))*100

                unc_cal.append([co2[-1],co2_unc[-1],co2_o[-1],co2_o_unc[-1]])
                val.append(co2_p[-1])
                c.close()
        unc_cal = np.array(unc_cal)
        val = np.array(val)

        unc_cy = unc_prop(unc_cal)
        med_cy = np.median(val)
        len_cy = len(val)
        out.append([timeperiods[t][0].year,timeperiods[t][1].year,med_anti,unc_anti[2],len_anti,med_cy,unc_cy[2],len_cy])
    out = np.array(out)
    np.savetxt('data/decadal.csv',out,delimiter=',')

if assess_missing:
    outs = []
    for i in np.unique(eddy_an['track']):
        print(i)
        if os.path.exists(output_loc+'/'+str(i)+'.nc') == True:
            c = Dataset(output_loc+'/'+str(i)+'.nc','r')
            sst = np.array(c['cci_sst_in_mean'])
            c.close()
            f = np.where(np.isnan(sst) == 1)[0]
            outs.append(len(f)/len(sst))

    for i in  np.unique(eddy_cy['track']):
        print(i)
        if os.path.exists(output_loc_cy+'/'+str(i)+'.nc') == True:
            c = Dataset(output_loc_cy+'/'+str(i)+'.nc','r')
            sst = np.array(c['cci_sst_in_mean'])
            c.close()
            f = np.where(np.isnan(sst) == 1)[0]
            outs.append(len(f)/len(sst))

    outs = np.array(outs)*100
    print(np.mean(outs))
    print(np.median(outs))
    print(np.max(outs))
    print(np.min(outs))
    print(np.std(outs))
