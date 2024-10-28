#!/usr/bin/env python3

import pyEddy_main as pyEddy_m
import pyEddy_plot as EdPlt
import pyEddy_earthobs as Edeobs
import datetime
import numpy as np
from matplotlib.gridspec import GridSpec
import matplotlib.pyplot as plt
import os

def analysis(eddy_v,desc,output_loc):
    for eddy_track in np.unique(eddy_v['track']):
        print(eddy_track)
        if os.path.exists(output_loc+'/'+str(eddy_track)+'.nc') == False:
            eddy_t = Edeobs.timeseries_start(eddy_v,desc,output_loc,eddy_track) # This creates the netcdf_file

            Edeobs.add_sstCCI('E:/Data/SST-CCI/v301',radm = 3,plot=0,output_loc = output_loc,track = eddy_track,bias=0.05)
            Edeobs.add_OCCCI('E:/Data/OC-CCI/v6.0/daily',radm = 3,plot=0,output_loc = output_loc,track = eddy_track,bias=0)
            Edeobs.add_cmems('F:/Data/CMEMS/SSS/DAILY',track = eddy_track,output_loc=output_loc,var = 'so',radm = 3,plot=0,name = 'CMEMS Salinity ')
            Edeobs.add_cmems('F:/Data/CMEMS/MLD/DAILY',track = eddy_track,output_loc=output_loc,var = 'mlotst',radm = 3,plot=0,units='log10(m)',log10=True,depth=False, name = 'CMEMS Mixed Layer Depth ')
            Edeobs.add_wind('F:/Data/CCMP/V3.1',track = eddy_track,output_loc=output_loc,radm=3,plot=0)


            Edeobs.produce_monthly(eddy_track,output_loc,vars=['cmems_mlotst_in_median','cmems_mlotst_out_median','cmems_so_in_median','cmems_so_out_median',
                'cci_sst_in_median','cci_sst_out_median','ccmp_wind_in_median','ccmp_wind_out_median','ccmp_wind^2_in_median','ccmp_wind^2_out_median','effective_area','cci_oc_chla_in_median','cci_oc_chla_out_median'])

            Edeobs.produce_monthly(eddy_track,output_loc,vars=['cci_sst_in_unc_mean','cci_sst_out_unc_mean'],unc=True,unc_days=3)

            Edeobs.add_noaa('F:/Data/NOAA_ERSL/DATA/MONTHLY',output_loc,eddy_track)
            Edeobs.add_era5('F:/Data/ERA5/MONTHLY/DATA',output_loc,eddy_track,var='msl')
            Edeobs.add_variable_zeros(output_loc,eddy_track,'month_xco2',['ice'],'month_time')
            Edeobs.add_province('F:/OceanCarbon4Climate/NN/GCB2024_full_version_biascorrected/inputs/neural_network_input.nc',output_loc,eddy_track,prov_var ='prov_smoothed',add_text='physics')
            Edeobs.calc_anomaly_wrt_fco2net('E:/SCOPE/NN/Ford_et_al_SOM_chla/inputs/neural_network_input.nc',output_loc,[eddy_track],
                ['month_cci_sst_in_median','month_cci_sst_out_median','month_cmems_so_in_median','month_cmems_so_out_median','month_cmems_mlotst_in_median','month_cmems_mlotst_out_median','month_xco2','month_cci_oc_chla_in_median','month_cci_oc_chla_out_median'],
                ['CCI_SST_analysed_sst','CCI_SST_analysed_sst','CMEMS_so','CMEMS_so','CMEMS_mlotst','CMEMS_mlotst','NOAA_ERSL_xCO2','OC-CCI_chl','OC-CCI_chl'])
            Edeobs.calc_fco2('F:/OceanCarbon4Climate/NN/GCB2024_full_version_biascorrected',output_loc,eddy_track,province_var = 'month_province_physics',
                    input_var=['month_cci_sst_in_median','month_xco2','month_cmems_so_in_median','month_cmems_mlotst_in_median','month_cci_sst_in_median_anom','month_xco2_anom','month_cmems_so_in_median_anom','month_cmems_mlotst_in_median_anom'],
                    add_text='_in_physics')
            Edeobs.calc_fco2('F:/OceanCarbon4Climate/NN/GCB2024_full_version_biascorrected',output_loc,eddy_track,province_var = 'month_province_physics',
                input_var=['month_cci_sst_out_median','month_xco2','month_cmems_so_out_median','month_cmems_mlotst_out_median','month_cci_sst_out_median_anom','month_xco2_anom','month_cmems_so_out_median_anom','month_cmems_mlotst_out_median_anom'],
                add_text='_out_physics')


            Edeobs.fluxengine_file_generate(output_loc,eddy_track,sub_sst = 'month_cci_sst_in_median',sss = 'month_cmems_so_in_median', ws = 'month_ccmp_wind_in_median',ws2='month_ccmp_wind^2_in_median',press = 'month_msl',
                xco2atm = 'month_xco2',fco2 = 'month_fco2_sw_in_physics',fco2_net = 'month_fco2_net_unc_in_physics',fco2_para='month_fco2_para_unc_in_physics',fco2_val = 'month_fco2_val_unc_in_physics',fco2_tot = 'month_fco2_tot_unc_in_physics',fluxengine_file = fluxengine_input_file
                ,sst_unc = 'month_cci_sst_in_unc_mean')

            Edeobs.fluxengine_run(output_loc,eddy_track,config_file=config_file,fluxengine_input_file = fluxengine_input_file,fluxengine_path=fluxengine_loc,add_text='_in_physics')

            Edeobs.fluxengine_file_generate(output_loc,eddy_track,sub_sst = 'month_cci_sst_out_median',sss = 'month_cmems_so_out_median', ws = 'month_ccmp_wind_out_median',ws2='month_ccmp_wind^2_out_median',press = 'month_msl',
                xco2atm = 'month_xco2',fco2 = 'month_fco2_sw_out_physics',fco2_net = 'month_fco2_net_unc_out_physics',fco2_para='month_fco2_para_unc_out_physics',fco2_val = 'month_fco2_val_unc_out_physics',fco2_tot = 'month_fco2_tot_unc_out_physics',
                fluxengine_file = fluxengine_input_file,sst_unc = 'month_cci_sst_out_unc_mean')
            Edeobs.fluxengine_run(output_loc,eddy_track,config_file=config_file,add_text='_out_physics',fluxengine_input_file = fluxengine_input_file,fluxengine_path=fluxengine_loc)
            Edeobs.eddy_co2_flux(output_loc,eddy_track,inout = '_in_physics')
            Edeobs.eddy_co2_flux(output_loc,eddy_track,inout='_out_physics')

            """
            """
            Edeobs.add_province('E:/SCOPE/NN/Ford_et_al_SOM_chla/inputs/neural_network_input.nc',output_loc,eddy_track,prov_var ='prov_smoothed',add_text='bio')
            Edeobs.calc_fco2('E:/SCOPE/NN/Ford_et_al_SOM_chla',output_loc,eddy_track,province_var = 'month_province_bio',
                    input_var=['month_cci_sst_in_median','month_xco2','month_cmems_so_in_median','month_cmems_mlotst_in_median','month_cci_oc_chla_in_median','month_cci_sst_in_median_anom','month_xco2_anom','month_cmems_so_in_median_anom','month_cmems_mlotst_in_median_anom','month_cci_oc_chla_in_median_anom'],
                    add_text='_in_bio')
            Edeobs.calc_fco2('E:/SCOPE/NN/Ford_et_al_SOM_chla',output_loc,eddy_track,province_var = 'month_province_bio',
                input_var=['month_cci_sst_out_median','month_xco2','month_cmems_so_out_median','month_cmems_mlotst_out_median','month_cci_oc_chla_out_median','month_cci_sst_out_median_anom','month_xco2_anom','month_cmems_so_out_median_anom','month_cmems_mlotst_out_median_anom','month_cci_oc_chla_out_median_anom'],
                add_text='_out_bio')


            Edeobs.fluxengine_file_generate(output_loc,eddy_track,sub_sst = 'month_cci_sst_in_median',sss = 'month_cmems_so_in_median', ws = 'month_ccmp_wind_in_median',ws2='month_ccmp_wind^2_in_median',press = 'month_msl',
                xco2atm = 'month_xco2',fco2 = 'month_fco2_sw_in_bio',fco2_net = 'month_fco2_net_unc_in_bio',fco2_para='month_fco2_para_unc_in_bio',fco2_val = 'month_fco2_val_unc_in_bio',fco2_tot = 'month_fco2_tot_unc_in_bio',fluxengine_file = fluxengine_input_file
                ,sst_unc = 'month_cci_sst_in_unc_mean')

            Edeobs.fluxengine_run(output_loc,eddy_track,config_file=config_file,fluxengine_input_file = fluxengine_input_file,fluxengine_path=fluxengine_loc,add_text='_in_bio')

            Edeobs.fluxengine_file_generate(output_loc,eddy_track,sub_sst = 'month_cci_sst_out_median',sss = 'month_cmems_so_out_median', ws = 'month_ccmp_wind_out_median',ws2='month_ccmp_wind^2_out_median',press = 'month_msl',
                xco2atm = 'month_xco2',fco2 = 'month_fco2_sw_out_bio',fco2_net = 'month_fco2_net_unc_out_bio',fco2_para='month_fco2_para_unc_out_bio',fco2_val = 'month_fco2_val_unc_out_bio',fco2_tot = 'month_fco2_tot_unc_out_bio',
                fluxengine_file = fluxengine_input_file,sst_unc = 'month_cci_sst_out_unc_mean')
            Edeobs.fluxengine_run(output_loc,eddy_track,config_file=config_file,add_text='_out_bio',fluxengine_input_file = fluxengine_input_file,fluxengine_path=fluxengine_loc)
            Edeobs.eddy_co2_flux(output_loc,eddy_track,inout = '_in_bio')
            Edeobs.eddy_co2_flux(output_loc,eddy_track,inout='_out_bio')

            #break



fluxengine_loc = 'F:/eddy/fluxengine_c'
fluxengine_input_file = fluxengine_loc+'/output.nc'
config_file = 'fluxengine_config_night_cy.conf'
# To save speed and memory these are the variables in the netcdf we don't want to load. This can be modified,
# depending on what is required
no_load = ['cost_association','speed_contour_height','speed_contour_latitude','speed_contour_longitude','speed_contour_shape_error',
    'speed_average','speed_area','inner_contour_height','latitude_max','longitude_max','num_point_s','uavg_profile']

"""
Running the anticyclonic eddies...
"""
# File where our eddy data is located
file = 'F:/Data/AVISO_EDDIES/META3.2_DT_allsat_Cyclonic_long_19930101_20220209.nc'
output_loc = 'F:/eddy/n_cyclonic'
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
print(len(np.unique(eddy_v['track'])))

eddy_v = pyEddy_m.box_split(eddy_v,[-80,80],[-180,180],[datetime.datetime(1993,1,1),datetime.datetime(2022,12,31)],strict_time=False)
#eddy_v = pyEddy_m.box_split(eddy_v,[-35,-15],[5,25],[datetime.datetime(2000,1,1),datetime.datetime(2018,12,31)])
print(len(np.unique(eddy_v['track'])))

eddy_v = pyEddy_m.eddy_length_min(eddy_v,365)
print(len(np.unique(eddy_v['track'])))

# eddy_v = pyEddy_m.eddy_cross_lon_lat(eddy_v,0)
# print(len(np.unique(eddy_v['track'])))
analysis(eddy_v,desc,output_loc)
# calc_anomalies(output_loc,eddy_v,fluxengine_loc,fluxengine_input_file,config_file = config_file)
# """
# Running the cyclonic eddies...
# """
# file = 'D:/Data/AVISO_EDDIES/META3.2_DT_allsat_Cyclonic_long_19930101_20220209.nc'
# output_loc = 'D:/eddy/cyclonic'
# #file = 'D:/Data/AVISO_EDDIES/META3.2_DT_allsat_Cyclonic_long_19930101_20220209.nc'
# # Loading the eddy netcdf data into a dictionary that corresponds to the netcdf file variable names.
# # This will not load anything that is defined in "no_load". If you want to load all variables leave
# # no_load empty (i.e [])
# eddy_v,desc = pyEddy_m.AVISO_load(file,no_load)
# print(desc)
# # # Here we split the eddy dict down to a set spatial region, and temporal window. This will
# # # include all eddies even if they appear in the domain for 1 day. We can perform checks later that
# # # they remain in the domain for a period. If you want all the data, then you don't need to use
# # # this function.
# print(len(np.unique(eddy_v['track'])))
#
# #eddy_v = pyEddy_m.box_split(eddy_v,[-34,-30],[5,25],[datetime.datetime(2000,1,1),datetime.datetime(2018,12,31)])
# eddy_v = pyEddy_m.box_split(eddy_v,[-35,-15],[5,25],[datetime.datetime(2000,1,1),datetime.datetime(2018,12,31)],strict_time=True)
# print(len(np.unique(eddy_v['track'])))
#
# eddy_v = pyEddy_m.eddy_length_min(eddy_v,365)
# print(len(np.unique(eddy_v['track'])))
#
# eddy_v = pyEddy_m.eddy_cross_lon_lat(eddy_v,0)
# print(len(np.unique(eddy_v['track'])))
# analysis(eddy_v,output_loc)
