#!/usr/bin/env python3

import pyEddy_main as pyEddy_m
import pyEddy_plot as EdPlt
import pyEddy_earthobs as Edeobs
import datetime
import numpy as np
from matplotlib.gridspec import GridSpec
import matplotlib.pyplot as plt

output_loc = 'OUTPUT'
# To save speed and memory these are the variables in the netcdf we don't want to load. This can be modified,
# depending on what is required
# no_load = ['cost_association','speed_contour_height','speed_contour_latitude','speed_contour_longitude','speed_contour_shape_error',
#     'speed_average','speed_area','inner_contour_height','latitude_max','longitude_max','num_point_s','uavg_profile']
# # File where our eddy data is located
# file = 'D:/Data/AVISO_EDDIES/META3.2_DT_allsat_Anticyclonic_long_19930101_20220209.nc'
# # Loading the eddy netcdf data into a dictionary that corresponds to the netcdf file variable names.
# # This will not load anything that is defined in "no_load". If you want to load all variables leave
# # no_load empty (i.e [])
# eddy,desc = pyEddy_m.AVISO_load(file,no_load)
# print(desc)
# # # Here we split the eddy dict down to a set spatial region, and temporal window. This will
# # # include all eddies even if they appear in the domain for 1 day. We can perform checks later that
# # # they remain in the domain for a period. If you want all the data, then you don't need to use
# # # this function.
# print(len(np.unique(eddy['track'])))
#
# eddy = pyEddy_m.box_split(eddy,[-60,10],[-70,25],[datetime.datetime(2002,7,1),datetime.datetime(2018,12,31)])
# print(len(np.unique(eddy['track'])))
#
# eddy = pyEddy_m.eddy_length_min(eddy,365)
# print(len(np.unique(eddy['track'])))
#
# #
# # fig, ax = plt.subplots()
# # m=EdPlt.base_tracks_start(ax,latb=[-60,10],lonb = [-75,40])
# # EdPlt.plot_eddy_shape(ax,m,eddy,val=0,step=60)
# # EdPlt.base_tracks_end(ax,m)
# # plt.show()
# eddy_t = Edeobs.timeseries_start(eddy,eddy['track'][0])
# eddy_t,desc = Edeobs.add_sstCCI('D:\Data\SST-CCI',eddy_t,desc,radm = 3,plot=0)
# eddy_t,desc = Edeobs.add_cmems('D:\Data\CMEMS\SSS\DAILY',eddy_t,desc,var = 'so',radm = 3,plot=0)
# eddy_t,desc = Edeobs.add_cmems('D:\Data\CMEMS\MLD\DAILY',eddy_t,desc,var = 'mlotst',radm = 3,plot=0,units='log10(m)',log10=True,depth=False)
# eddy_t,desc = Edeobs.add_wind('D:\Data\CCMP\V3.1',eddy_t,desc,radm=3)
# Edeobs.timeseries_save(eddy_t,desc,output_loc)
# Edeobs.produce_monthly(218589,output_loc,vars=['cmems_mlotst_in_median','cmems_mlotst_out_median','cmems_so_in_median','cmems_so_out_median','cci_sst_in_median','cci_sst_out_median','ccmp_wind_in_median','ccmp_wind_out_median'])
Edeobs.add_noaa('D:/Data/NOAA_ERSL/DATA/MONTHLY',output_loc,218589)
#
