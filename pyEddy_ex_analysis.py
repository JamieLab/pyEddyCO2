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
font = {'weight' : 'normal',
        'size'   :12}
matplotlib.rc('font', **font)

output_loc = 'F:/eddy/n_anticyclonic'
output_loc_cy = 'F:/eddy/n_cyclonic'

no_load = ['cost_association','speed_contour_height','speed_contour_latitude','speed_contour_longitude','speed_contour_shape_error',
    'speed_average','speed_area','inner_contour_height','latitude_max','longitude_max','num_point_s','uavg_profile']
# File where our eddy data is located
anti_file = 'F:/Data/AVISO_EDDIES/META3.2_DT_allsat_Anticyclonic_long_19930101_20220209.nc'
cycl_file = 'F:/Data/AVISO_EDDIES/META3.2_DT_allsat_Cyclonic_long_19930101_20220209.nc'

"""
"""
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
# eddy_v = pyEddy_m.box_split(eddy_v,[-35,-15],[5,25],[datetime.datetime(2002,7,1),datetime.datetime(2018,12,31)],strict_time=True)
#
eddy_cy = pyEddy_m.eddy_length_min(eddy_cy,365)
# eddy_v = pyEddy_m.eddy_cross_lon_lat(eddy_v,0)
# print(len(np.unique(eddy_v['track'])))

let = ['a','b','c','d','e','f','g']
# # g = glob.glob(os.path.join(output_loc,'*.nc'))
# # #g = np.unique(eddy_v['track'])
plot_figure_2 = False
plot_figure_4 = False
plot_figure_4_bio = True
plot_figure_5 = False
plot_figure_5_bio = False
"""
Figure 2
"""
if plot_figure_2:
    font = {'weight' : 'normal',
            'size'   :20}
    matplotlib.rc('font', **font)
    fig = plt.figure(figsize=(18,20))
    gs = GridSpec(2,1, figure=fig, wspace=0.25,hspace=0.15,bottom=0.1,top=0.93,left=0.075,right=0.875)
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

            co2 = np.array(c['flux_in_physics_areaday_cumulative'])
            lat = np.array(c['month_latitude'])
            lon = np.array(c['month_longitude'])
            x,y = m(lon,lat)

            a = ax[0].scatter(x[0],y[0],s=12,c = co2[-1],vmin = -1,vmax=1,cmap=cmap)
            c.close()
    #
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

    ax[1].set_xlabel('Longitude',labelpad=30)
    ax[1].set_ylabel('Latitude',labelpad=45)
    # ax[0].set_xlabel('Longitude',labelpad=30)
    ax[0].set_ylabel('Latitude',labelpad=45)
    eplt.base_tracks_end(ax[0],m)
    eplt.base_tracks_end(ax[1],m2)
    cbar = fig.add_axes([0.90,0.2,0.02,0.6])
    cba = fig.colorbar(a,cax=cbar)
    for i in range(2):
        ax[i].text(0.92,1.06,f'({let[i]})',transform=ax[i].transAxes,va='top',fontweight='bold',fontsize = 24)
    cba.set_label('Cumulative net CO$_{2}$ flux (Tg C)')
    fig.savefig('figs/manuscript/figure_2.png',dpi=300)

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

# fig = plt.figure(figsize=(21,15))
# gs = GridSpec(2,3, figure=fig, wspace=0.25,hspace=0.15,bottom=0.1,top=0.93,left=0.075,right=0.95)
# ax = [fig.add_subplot(gs[0,0:2]),fig.add_subplot(gs[0,2]),fig.add_subplot(gs[1,0:2]),fig.add_subplot(gs[1,2])]
# m = eplt.base_tracks_start(ax[0],latb=[-80,80],lonb = [-180,180])
# m2 = eplt.base_tracks_start(ax[2],latb=[-80,80],lonb = [-180,180])
# val = []
# cum = []
# #print(np.unique(eddy_v['track']))
#
# for i in  np.unique(eddy_v['track']):
#     print(i)
#     if os.path.exists(output_loc+'/'+str(i)+'.nc') == True:
#         c = Dataset(output_loc+'/'+str(i)+'.nc','r')
#         #c = Dataset(i,'r')
#         time = np.array(c['month_time'])
#         time2 = []
#         for j in range(len(time)):
#             time2.append(pyEddy_m.date_con(int(time[j])))
#         if time2[0].year >=1993:
#             co2 = np.array(c['flux_in_bio_areaday_cumulative'])
#             co2_o = np.array(c['flux_out_bio_areaday_cumulative'])
#             co2_unc = np.array(c['flux_unc_tot_in_bio_areaday_cumulative'])/2
#             #co2_o_unc = np.array(c['flux_unc_total_out_areaday_cumulative'])
#             lat = np.array(c['latitude'])
#             lon = np.array(c['longitude'])
#             x,y = m(lon,lat)
#
#
#             co2_p = (co2-co2_o)/np.abs(co2_o)
#             if np.isnan(co2[-1]-co2_o[-1]) == 0:
#                 print(co2_p[-1]*100)
#                 print(co2[-1])
#                 print(co2_o[-1])
#                 print(co2[-1]-co2_o[-1])
#                 cum.append(co2[-1]-co2_o[-1])
#                 val.append(co2_p[-1]*100)
#                 if np.sign(co2_p[-1]) == 1:
#                     col = '#D41159'
#                 else:
#                     col = '#1A85FF'
#
#                 # p = ax[0,1].plot(time2,co2,col)
#
#                 # ax[0,0].scatter(x,y,s=1,color=col)
#                 # ax[0,1].fill_between(time2,co2-co2_unc,co2+co2_unc,color=col,alpha=0.4)
#                 ax[0].scatter(x[0],y[0],s=3,color=col)
#         c.close()
#         if i == 40:
#             break
# val = np.array(val)
# cum = np.array(cum)
# eplt.base_tracks_end(ax[0],m)
# # eplt.base_tracks_end(ax[1,1],m2)
# # ax[0,1].set_ylim([-4,4])
# # ax[0,1].set_ylabel('Net Eddy CO${_2}$ flux (Tg C)')
# # ax[0,1].set_xlabel('Year')
# print(val)
# print(np.nanmedian(val))
# print(np.nansum(cum))
# print(np.nanmedian(cum))
# ax[1].boxplot(val[np.isnan(val) == False])
# ax[1].plot([0.5,1.5],[0,0])
# ax[1].set_ylim([-40,40])
# #ax.set_title('N='+str(len(g)))
# ax[1].set_ylabel('Change in eddy CO${_2}$ flux \n compared to the surrounding \n water CO${_2}$ flux (%)')
# ax[1].set_xticks(ticks=[1],labels =['(N = '+str(len(val[np.isnan(val) == False])) + ')'])
# #ax[1,0].set_xticks(ticks=[1],labels =['(N = '+str(len(g)) + ')'])
# # fig.tight_layout()
#
# for i in range(4):
#     #worldmap.plot(color="lightgrey", ax=ax[i])
#     ax[i].text(0.92,1.06,f'({let[i]})',transform=ax[i].transAxes,va='top',fontweight='bold',fontsize = 24)
# fig.savefig('figs/anti_new_global_eddy_flux.png',dpi=300)

# """
# """
# fig2,ax = plt.subplots(1,1, figsize=(16, 6))
#
# for i in np.unique(eddy_v['track']):
#     print(i)
#     c = Dataset(output_loc+'/'+str(i)+'.nc','r')
#     #c = Dataset(i,'r')
#     time = np.array(c['month_time'])
#     time2 = []
#     for j in range(len(time)):
#         time2.append(pyEddy_m.date_con(int(time[j])))
#     if time2[0].year >=1998:
#         co2 = np.array(c['flux_in_areaday_cumulative'])
#         co2_o = np.array(c['flux_out_areaday_cumulative'])
#         co2_unc = np.array(c['flux_unc_tot_in_areaday_cumulative'])
#         #co2_o_unc = np.array(c['flux_unc_total_out_areaday_cumulative'])
#         lat = np.array(c['latitude'])
#         lon = np.array(c['longitude'])
#         #x,y = m(lon,lat)
#
#
#         co2_p = (co2-co2_o)/co2_o
#         #val.append(co2_p[-1]*100)
#         col = 'k'
#         p = ax.plot(time2,co2,col)
#
#
#         #ax.plot(x,y,color=a)
#         #ax.fill_between(time2,co2-co2_unc,co2+co2_unc,color=col,alpha=0.4)
#     c.close()
# ax.set_ylim([-4,4])
# ax.set_ylabel('Net Eddy CO${_2}$ flux (Tg C)')
# ax.set_xlabel('Year')
# fig2.savefig('figs/anti_global_eddy_flux_test_netfluxexpand.png',dpi=300)
#
