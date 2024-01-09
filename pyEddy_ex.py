#!/usr/bin/env python3

import pyEddy_main as pyEddy_m
import pyEddy_plot as EdPlt
import pyEddy_earthobs as Edeobs
import datetime
import numpy as np
from matplotlib.gridspec import GridSpec
import matplotlib.pyplot as plt

output_loc = 'OUTPUT'
fluxengine_input_file = 'D:/eddy/input.nc'
# To save speed and memory these are the variables in the netcdf we don't want to load. This can be modified,
# depending on what is required
no_load = ['cost_association','speed_contour_height','speed_contour_latitude','speed_contour_longitude','speed_contour_shape_error',
    'speed_average','speed_area','inner_contour_height','latitude_max','longitude_max','num_point_s','uavg_profile']
# File where our eddy data is located
file = 'D:/Data/AVISO_EDDIES/META3.2_DT_allsat_Anticyclonic_long_19930101_20220209.nc'
# Loading the eddy netcdf data into a dictionary that corresponds to the netcdf file variable names.
# This will not load anything that is defined in "no_load". If you want to load all variables leave
# no_load empty (i.e [])
eddy_v,desc = pyEddy_m.AVISO_load(file,no_load)
print(desc)
# # Here we split the eddy dict down to a set spatial region, and temporal window. This will
# # include all eddies even if they appear in the domain for 1 day. We can perform checks later that
# # they remain in the domain for a period. If you want all the data, then you don't need to use
# # this function.
print(len(np.unique(eddy_v['track'])))

eddy_v = pyEddy_m.box_split(eddy_v,[-34,-30],[5,25],[datetime.datetime(2000,1,1),datetime.datetime(2018,12,31)])
print(len(np.unique(eddy_v['track'])))

eddy_v = pyEddy_m.eddy_length_min(eddy_v,365)
print(len(np.unique(eddy_v['track'])))

#
# fig, ax = plt.subplots()
# m=EdPlt.base_tracks_start(ax,latb=[-60,10],lonb = [-75,40])
# #EdPlt.plot_eddy_shape(ax,m,eddy,val=0,step=60)
# EdPlt.base_tracks_plt(ax,eddy,m)
# EdPlt.base_tracks_end(ax,m)
# plt.show()
# uni =
# eddy_track = uni[2]
for eddy_track in np.unique(eddy_v['track'])[2:]:
    print(eddy_track)
    eddy_t = Edeobs.timeseries_start(eddy_v,eddy_track)
    eddy_t,desc = Edeobs.add_sstCCI('D:\Data\SST-CCI',eddy_t,desc,radm = 3,plot=0)
    eddy_t,desc = Edeobs.add_cmems('D:\Data\CMEMS\SSS\DAILY',eddy_t,desc,var = 'so',radm = 3,plot=0)
    eddy_t,desc = Edeobs.add_cmems('D:\Data\CMEMS\MLD\DAILY',eddy_t,desc,var = 'mlotst',radm = 3,plot=0,units='log10(m)',log10=True,depth=False)
    eddy_t,desc = Edeobs.add_wind('D:\Data\CCMP\V3.1',eddy_t,desc,radm=3)
    Edeobs.timeseries_save(eddy_t,desc,output_loc,eddy_track)
    Edeobs.produce_monthly(eddy_track,output_loc,vars=['cmems_mlotst_in_median','cmems_mlotst_out_median','cmems_so_in_median','cmems_so_out_median','cci_sst_in_median','cci_sst_out_median','ccmp_wind_in_median','ccmp_wind_out_median','effective_area'])
    Edeobs.add_noaa('D:/Data/NOAA_ERSL/DATA/MONTHLY',output_loc,eddy_track)
    Edeobs.add_province('D:/ESA_CONTRACT/NN/GCB_TESTING/inputs/neural_network_input.nc',output_loc,eddy_track,prov_var ='prov')
    Edeobs.add_variable_zeros(output_loc,eddy_track,'month_time',['cci_sst_anom','xco2_atm_anom','cmems_sss_anom','cmems_mld_anom'],'month_time')
    Edeobs.add_era5('D:/Data/ERA5/MONTHLY/DATA',output_loc,eddy_track,var='msl')
    Edeobs.calc_fco2('D:/ESA_CONTRACT/NN/GCB_TESTING',output_loc,eddy_track,province_var = 'month_province',
        input_var=['month_cci_sst_in_median','month_xco2','month_cmems_so_in_median','month_cmems_mlotst_in_median','cci_sst_anom','xco2_atm_anom','cmems_sss_anom','cmems_mld_anom'])
    Edeobs.calc_fco2('D:/ESA_CONTRACT/NN/GCB_TESTING',output_loc,eddy_track,province_var = 'month_province',
        input_var=['month_cci_sst_out_median','month_xco2','month_cmems_so_out_median','month_cmems_mlotst_out_median','cci_sst_anom','xco2_atm_anom','cmems_sss_anom','cmems_mld_anom'],
        add_text='_out')
    Edeobs.fluxengine_file_generate(output_loc,eddy_track,sub_sst = 'month_cci_sst_in_median',sss = 'month_cmems_so_in_median', ws = 'month_ccmp_wind_in_median',press = 'month_msl',
        xco2atm = 'month_xco2',fco2 = 'month_fco2_sw_in',fluxengine_file = fluxengine_input_file)
    Edeobs.fluxengine_run(output_loc,eddy_track,fluxengine_input_file = fluxengine_input_file)
    Edeobs.fluxengine_file_generate(output_loc,eddy_track,sub_sst = 'month_cci_sst_out_median',sss = 'month_cmems_so_out_median', ws = 'month_ccmp_wind_out_median',press = 'month_msl',
        xco2atm = 'month_xco2',fco2 = 'month_fco2_sw_out',fluxengine_file = fluxengine_input_file)
    Edeobs.fluxengine_run(output_loc,eddy_track,add_text='_out',fluxengine_input_file = fluxengine_input_file)
    Edeobs.eddy_co2_flux(output_loc,eddy_track)
    Edeobs.eddy_co2_flux(output_loc,eddy_track,inout='_out')
