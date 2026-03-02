#!/usr/bin/env python3
from netCDF4 import Dataset
import numpy as np
import matplotlib.pyplot as plt
import datetime
import cmocean
import cartopy
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import geopandas as gpd
from matplotlib.gridspec import GridSpec
import matplotlib as mpl
from matplotlib.animation import FuncAnimation, PillowWriter,FFMpegWriter
from matplotlib.colors import ListedColormap,LinearSegmentedColormap
import matplotlib.colors as colors
import os
import pyEddy_main as pyEddy_m
import pyEddy_plot as eplt

mpl.rcParams['animation.ffmpeg_path'] = r'C:\\Users\\df391\\OneDrive - University of Exeter\\Python\ffmpeg\\bin\\ffmpeg.exe'
import weight_stats as ws
import sys
oceaniculoc = 'C:\\Users\\df391\\OneDrive - University of Exeter\\Post_Doc_ESA_Contract\\OceanICU'
sys.path.append(os.path.join(oceaniculoc,'Data_Loading'))
sys.path.append(os.path.join(oceaniculoc))
import data_utils as du

output_loc = 'F:/eddy/v0-3/n_anticyclonic'
ref_time = datetime.datetime(1950,1,1)
# font = {'weight' : 'normal',
#         'size'   :20}
# matplotlib.rc('font', **font)
def animate(r,t):
    print(r)
    r=r+1
    fig.clear()
    row = 2;col=3
    gs = GridSpec(row,col, figure=fig, wspace=0.25,hspace=0.15,bottom=0.1,top=0.95,left=0.05,right=0.95)
    axs = [[fig.add_subplot(gs[i, j]) for j in range(col)] for i in range(row)]
    flatList = [element for innerList in axs for element in innerList]
    ax = flatList

    #c = Dataset(output_loc+'/'+'679331'+'.nc','r')#
    c = Dataset(output_loc+'/'+'611053'+'.nc','r')

    lat = np.array(c['latitude'])
    lon = np.array(c['longitude'])
    time = np.array(c['time'])
    m = eplt.base_tracks_start(ax[2],latb=[-60,-20],lonb = [40,100])
    x,y = m(lon,lat)
    ax[2].scatter(x,y,s=12,c = time,cmap=cmocean.cm.thermal)
    x,y = m(lon[r],lat[r])
    ax[2].scatter(x,y,s=12,color='r')
    eplt.base_tracks_end(ax[2],m)
    parallels = np.arange(50.,62,5)
    meridion = np.arange(170,191,10)
    m.drawparallels(parallels,labels=[True,False,False,False])
    m.drawmeridians(meridion,labels=[False,False,False,True])
    ax[2].set_xlabel('Longitude',labelpad=15)
    ax[2].set_ylabel('Latitude',labelpad=30)

    mon_time = np.array(c['month_time'])
    time_da = []
    time_mon = []
    for i in range(len(mon_time)):
        time_mon.append(ref_time+datetime.timedelta(days=int(mon_time[i])))
    for i in range(len(time)):
        time_da.append(ref_time+datetime.timedelta(days=int(time[i])))
    time_mon=np.array(time_mon)
    time_da = np.array(time_da)
    g = np.where(time[r] > mon_time+15)[0]
    t= len(g)

    sst = np.array(c['cci_sst_in_median'])
    sst_unc = np.array(c['cci_sst_in_unc_mean'])
    ax[0].plot(time_da[0:r],sst[0:r],color='k')
    # print(r)
    # print(sst[0:r])
    # print(sst_unc[0:r])
    ax[0].fill_between(time_da[0:r],sst[0:r]-sst_unc[0:r]/2,sst[0:r]+sst_unc[0:r]/2,alpha=0.6,color='k')
    ax[0].fill_between(time_da[0:r],sst[0:r]-sst_unc[0:r],sst[0:r]+sst_unc[0:r],alpha=0.4,color='k')
    sst_month = np.array(c['month_cci_sst_in_median'])[0:t]
    ax[0].plot(time_mon[0:t],sst_month[0:t],color='r',linewidth =2)
    ax[0].set_ylabel('Sea Surface Temperature (Kelvin)')


    sss = np.array(c['cmems_so_in_median'])
    sss_unc = 0.3*2
    ax[1].plot(time_da[0:r],sss[0:r],color='k')
    ax[1].fill_between(time_da[0:r],sss[0:r]-sss_unc/2,sss[0:r]+sss_unc/2,alpha=0.6,color='k')
    ax[1].fill_between(time_da[0:r],sss[0:r]-sss_unc,sss[0:r]+sss_unc,alpha=0.4,color='k')
    sss_month = np.array(c['month_cmems_so_in_median'])
    ax[1].plot(time_mon[0:t],sss_month[0:t],color='r',linewidth =2)
    ax[1].set_ylabel('Sea Surface Salinity (psu)')

    ax[3].set_ylabel('fCO$_{2 (sw)}$ or xCO$_{2 (atm)}$ ($\mu$atm or ppm)')
    ax[3].set_ylim([200,550])
    ax[4].plot([time_da[0],time_da[-1]],[0,0],'k--')
    ax[4].set_ylim([-0.2,0.2])
    ax[4].set_ylabel('Air-sea CO$_2$ flux (g C m$^{-2}$ d$^{-1}$)\n(-ve indicates atmosphere to ocean exchange)')
    ax[5].set_ylabel('Cumulative air-sea CO$_2$ flux (Tg C)\n(-ve indicates atmosphere to ocean exchange)')
    ax[5].plot([time_da[0],time_da[-1]],[0,0],'k--')

    if t>0:
        fco2 = np.array(c['month_fco2_sw_in_physics'])
        xco2 = np.array(c['month_xco2'])
        fco2_unc = np.array(c['month_fco2_tot_unc_in_physics'])
        socat_fco2 = np.array(c['socat_mean_fco2'])
        socat_fco2[socat_fco2==0] = np.nan
        ax[3].plot(time_mon[0:t],fco2[0:t],color='k')
        # ax[3].scatter(time_da,socat_fco2,72,color='r',zorder=6)
        ax[3].fill_between(time_mon[0:t],fco2[0:t]-fco2_unc[0:t]/2,fco2[0:t]+fco2_unc[0:t]/2,alpha=0.6,color='k')
        ax[3].fill_between(time_mon[0:t],fco2[0:t]-fco2_unc[0:t],fco2[0:t]+fco2_unc[0:t],alpha=0.4,color='k')
        ax[3].plot(time_mon[0:t],xco2[0:t],'k--')


        fco2 = np.array(c['flux_in_physics'])
        fco2_unc = np.array(c['flux_unc_in_physics'])*np.abs(fco2)
        #
        ax[4].plot(time_mon[0:t],fco2[0:t],color='k',zorder=6)
        ax[4].fill_between(time_mon[0:t],fco2[0:t]-fco2_unc[0:t]/2,fco2[0:t]+fco2_unc[0:t]/2,alpha=0.6,color='k')
        ax[4].fill_between(time_mon[0:t],fco2[0:t]-fco2_unc[0:t],fco2[0:t]+fco2_unc[0:t],alpha=0.4,color='k')

        #
        fco2 = np.array(c['flux_in_physics_areaday_cumulative'])
        fco2_unc = np.array(c['flux_unc_tot_in_physics_areaday_cumulative'])

        fco2_out = np.array(c['flux_out_physics_areaday_cumulative'])
        fco2_unc_out = np.array(c['flux_unc_tot_out_physics_areaday_cumulative'])

        ax[5].plot(time_mon[0:t],fco2[0:t],color='k',zorder=6,linewidth=2)
        ax[5].fill_between(time_mon[0:t],fco2[0:t]-fco2_unc[0:t]/2,fco2[0:t]+fco2_unc[0:t]/2,alpha=0.6,color='k',zorder=5)
        ax[5].fill_between(time_mon[0:t],fco2[0:t]-fco2_unc[0:t],fco2[0:t]+fco2_unc[0:t],alpha=0.4,color='k',zorder=5)

        ax[5].plot(time_mon[0:t],fco2_out[0:t],color='r',zorder=6,linewidth=2)
        ax[5].fill_between(time_mon[0:t],fco2_out[0:t]-fco2_unc_out[0:t]/2,fco2_out[0:t]+fco2_unc_out[0:t]/2,alpha=0.6,color='r')
        ax[5].fill_between(time_mon[0:t],fco2_out[0:t]-fco2_unc_out[0:t],fco2_out[0:t]+fco2_unc_out[0:t],alpha=0.4,color='r')


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

fps = 30
t = 0
fig = plt.figure(figsize=(14,8))
ani = FuncAnimation(fig, animate, interval=200, blit=False, repeat=True,frames=858,fargs=(t,))#937)
ani.save('animated2.mp4', dpi=300, writer=FFMpegWriter(fps=fps))
