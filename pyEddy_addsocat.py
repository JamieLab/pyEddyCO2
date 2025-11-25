#!/usr/bin/env python3

from netCDF4 import Dataset
import numpy as np
import pandas as pd
import datetime
import os
import glob
import matplotlib.path as mpltPath
import matplotlib.pyplot as plt
import pyEddy_main as pyEddy_m
import pyEddy_earthobs as Edeobs
import calendar
import sys

def add_socat(file,socat_data,socat_file,rean = True,plot=False):
    c = Dataset(file,'r')
    keys = c.variables.keys()

    # if 'socat_mean_fco2' in keys:
    #     print('Done')
    #     c.close()
    # else:
    lat = np.array(c['latitude'])
    lon = np.array(c['longitude'])
    time = np.array(c['time'])
    con_lat = np.array(c['effective_contour_latitude'])
    con_lon = np.array(c['effective_contour_longitude'])
    c.close()

    fco2_out = np.zeros((len(time),9)); fco2_out[:] = np.nan
    for i in range(len(time)):
        print(i)
        date = (datetime.datetime(1950,1,1) + datetime.timedelta(days=int(time[i])))
        day = date.day
        mon = date.month
        yr = date.year

        lonc = lon[i]
        latc = lat[i]
        lons = con_lon[i,:]
        lats = con_lat[i,:]
        if (lats[0] != 0.0) & (lons[0] != 180.0):

            f = np.where((socat_data['yr'] == yr) & (socat_data['mon'] == mon) & (socat_data['day'] == day))[0]
            #print(f)
            if len(f)>0:
                radmax = Edeobs.radius_check(lonc,latc,lons,lats)
                if (lonc - (radmax*2) < -180) or (lonc + (radmax*2) > 180):
                    print('Near the dateline')

                    path = mpltPath.Path(np.transpose([lons,lats]))

                    lon_socat_temp = np.array(socat_data['longitude [dec.deg.E]'][f])
                    h = np.where(np.sign(lon_socat_temp) != np.sign(lonc))
                    lon_socat_temp[h] = lon_socat_temp[h] + (360 * np.sign(lonc))
                    l = np.transpose([lon_socat_temp,np.array(socat_data['latitude [dec.deg.N]'][f])])
                else:
                    path = mpltPath.Path(np.transpose([lons,lats]))
                    l = np.transpose([np.array(socat_data['longitude [dec.deg.E]'][f]),np.array(socat_data['latitude [dec.deg.N]'][f])])
                #print(l)
                inp = path.contains_points(l)
                #print(inp)
                if plot:
                    plt.figure()
                    plt.scatter(lons,lats)
                    plt.scatter(l[:,0],l[:,1])
                    plt.show()
                g = np.where(inp == True)[0]
                if len(g) > 0:
                    print('Yes')
                    if rean == False:
                        fco2 = socat_data['fCO2rec [uatm]'][f[g]]
                        sst = socat_data['SST [deg C]'][f[g]]
                    else:
                        fco2 = socat_data['fCO2_reanalysed [uatm]'][f[g]]
                        sst = socat_data['T_subskin [C]'][f[g]]
                    sal = socat_data['sal'][f[g]]
                    fco2_out[i,0] = np.nanmean(fco2)
                    fco2_out[i,1] = np.nanstd(fco2)
                    fco2_out[i,2] = len(np.where(np.isnan(fco2)==0)[0])
                    fco2_out[i,3] = np.nanmean(sst)
                    fco2_out[i,4] = np.nanstd(sst)
                    fco2_out[i,5] = len(np.where(np.isnan(sst)==0)[0])
                    fco2_out[i,6] = np.nanmean(sal)
                    fco2_out[i,7] = np.nanstd(sal)
                    fco2_out[i,8] = len(np.where(np.isnan(sal)==0)[0])
            else:
                print('Broken eddy polygon...')
    if rean:
        rean = 'True'
    else:
        rean = 'Falses'
    c = Dataset(file,'a')
    keys = c.variables.keys()
    if 'socat_mean_fco2' in keys:
        c['socat_mean_fco2'][:] = fco2_out[:,0]
    else:
        var_o = c.createVariable('socat_mean_fco2','f4','d_time')
        var_o.units='uatm'
        var_o[:] = fco2_out[:,0]
    c['socat_mean_fco2'].last_modified = datetime.datetime.now().strftime(('%d/%m/%Y %H:%M'))
    c['socat_mean_fco2'].long_name = 'in situ mean fCO2(sw) from SOCAT observations'
    c['socat_mean_fco2'].recalculated = rean
    c['socat_mean_fco2'].socat_file = socat_file

    if 'socat_std_fco2' in keys:
        c['socat_std_fco2'][:] = fco2_out[:,1]
    else:
        var_o = c.createVariable('socat_std_fco2','f4','d_time')
        var_o.units='uatm'
        var_o[:] = fco2_out[:,1]
    c['socat_std_fco2'].last_modified = datetime.datetime.now().strftime(('%d/%m/%Y %H:%M'))
    c['socat_std_fco2'].long_name = 'in situ standard deviation fCO2(sw) from SOCAT observations'
    c['socat_std_fco2'].recalculated = rean
    c['socat_std_fco2'].socat_file = socat_file

    if 'socat_n_fco2' in keys:
        c['socat_n_fco2'][:] = fco2_out[:,2]
    else:
        var_o = c.createVariable('socat_n_fco2','f4','d_time')
        var_o.units='count'
        var_o[:] = fco2_out[:,2]
    c['socat_n_fco2'].last_modified = datetime.datetime.now().strftime(('%d/%m/%Y %H:%M'))
    c['socat_n_fco2'].long_name = 'in situ SOCAT observations sample number'
    c['socat_n_fco2'].recalculated = rean
    c['socat_n_fco2'].socat_file = socat_file

    if 'socat_mean_sst' in keys:
        c['socat_mean_sst'][:] = fco2_out[:,3]
    else:
        var_o = c.createVariable('socat_mean_sst','f4','d_time')
        var_o.units='degC'
        var_o[:] = fco2_out[:,3]
    c['socat_mean_sst'].last_modified = datetime.datetime.now().strftime(('%d/%m/%Y %H:%M'))
    c['socat_mean_sst'].long_name = 'in situ mean SST from SOCAT observations'
    c['socat_mean_sst'].recalculated = rean
    c['socat_mean_sst'].socat_file = socat_file

    if 'socat_std_sst' in keys:
        c['socat_std_sst'][:] = fco2_out[:,4]
    else:
        var_o = c.createVariable('socat_std_sst','f4','d_time')
        var_o.units='degC'
        var_o[:] = fco2_out[:,4]
    c['socat_std_sst'].last_modified = datetime.datetime.now().strftime(('%d/%m/%Y %H:%M'))
    c['socat_std_sst'].long_name = 'in situ standard deviation SST from SOCAT observations'
    c['socat_std_sst'].recalculated = rean
    c['socat_std_sst'].socat_file = socat_file

    if 'socat_n_sst' in keys:
        c['socat_n_sst'][:] = fco2_out[:,5]
    else:
        var_o = c.createVariable('socat_n_sst','f4','d_time')
        var_o.units='count'
        var_o[:] = fco2_out[:,5]
    c['socat_n_sst'].last_modified = datetime.datetime.now().strftime(('%d/%m/%Y %H:%M'))
    c['socat_n_sst'].long_name = 'in situ SST from SOCAT observations sample number'
    c['socat_n_sst'].recalculated = rean
    c['socat_n_sst'].socat_file = socat_file

    if 'socat_mean_sal' in keys:
        c['socat_mean_sst'][:] = fco2_out[:,6]
    else:
        var_o = c.createVariable('socat_mean_sal','f4','d_time')
        var_o.units='psu'
        var_o[:] = fco2_out[:,6]
    c['socat_mean_sal'].last_modified = datetime.datetime.now().strftime(('%d/%m/%Y %H:%M'))
    c['socat_mean_sal'].long_name = 'in situ mean salinity from SOCAT observations'
    c['socat_mean_sal'].recalculated = rean
    c['socat_mean_sal'].socat_file = socat_file

    if 'socat_std_sal' in keys:
        c['socat_std_sal'][:] = fco2_out[:,7]
    else:
        var_o = c.createVariable('socat_std_sal','f4','d_time')
        var_o.units='psu'
        var_o[:] = fco2_out[:,7]
    c['socat_std_sal'].last_modified = datetime.datetime.now().strftime(('%d/%m/%Y %H:%M'))
    c['socat_std_sal'].long_name = 'in situ standard deviation salinity from SOCAT observations'
    c['socat_std_sal'].recalculated = rean
    c['socat_std_sal'].socat_file = socat_file

    if 'socat_n_sal' in keys:
        c['socat_n_sal'][:] = fco2_out[:,8]
    else:
        var_o = c.createVariable('socat_n_sal','f4','d_time')
        var_o.units='count'
        var_o[:] = fco2_out[:,8]
    c['socat_n_sal'].last_modified = datetime.datetime.now().strftime(('%d/%m/%Y %H:%M'))
    c['socat_n_sal'].long_name = 'in situ salinity from SOCAT observations sample number'
    c['socat_n_sal'].recalculated = rean
    c['socat_n_sal'].socat_file = socat_file


    c.close()

def load_socat(file,skiprows=7577):
    data = pd.read_table(file,sep='\t',skiprows=skiprows)
    # Making sure data are in -180 to 180 limits
    data['longitude [dec.deg.E]'][data['longitude [dec.deg.E]'] > 180] = data['longitude [dec.deg.E]'][data['longitude [dec.deg.E]'] > 180] -360
    data['longitude [dec.deg.E]'][data['longitude [dec.deg.E]'] < -180] = data['longitude [dec.deg.E]'][data['longitude [dec.deg.E]'] < -180] +360
    return data


loc = 'F:/eddy/v0-3/n_anticyclonic/'
socat_file = 'E:/Data/_Datasets/SOCAT/v2024/testing/Fordetal_SOCATv2024_ESACCIv3_biascorrected_Humpherys_daily_unc_withheader_v2.tsv'
files = glob.glob(loc+'*.nc')
print(files)
# files = ['F:/eddy/n_anticyclonic\\10098.nc']
# files = ['F:/eddy/v0-3/n_anticyclonic/640581.nc']
socat_data = load_socat(socat_file)
# socat_data=0
for file in files:
    print(file)
    file_s = file.split('\\')[-1].split('.')
    print(file_s)
    if int(file_s[0][0:3]) >= 605:
        add_socat(file,socat_data,socat_file)
        Edeobs.produce_monthly(int(file_s[0]),loc,vars=['socat_mean_fco2','socat_mean_sst','socat_mean_sal'],average = 'mean')

t = 0
for file in files:
    print(file)
    c = Dataset(file,'r')
    socat_fco2 = np.array(c['month_socat_mean_fco2'])
    f = np.where(np.isnan(socat_fco2) == 0)[0]
    if len(f) > 0:
        nn_fco2 = np.array(c['month_fco2_sw_in_physics'])
        unc = np.array(c['month_fco2_tot_unc_in_physics'])
        time = np.array(c['month_time'])
        lat = np.array(c['month_latitude'])
        lon = np.array(c['month_longitude'])
        yr = []; mon =[]; day = [];
        for i in range(len(time)):
            date = (datetime.datetime(1950,1,1) + datetime.timedelta(days=int(time[i])))
            mon.append(date.month)
            yr.append(date.year)
        yr = np.array(yr); mon = np.array(mon);
        if t == 0:
            out = np.array([yr[f],mon[f],lat[f],lon[f],socat_fco2[f],nn_fco2[f],unc[f]])
            if out.shape[0] > out.shape[1]:
                out=np.transpose(out)
            t=1
        else:
            out = np.concatenate((out,np.transpose(np.array([yr[f],mon[f],lat[f],lon[f],socat_fco2[f],nn_fco2[f],unc[f]]))),axis=0)
        print(out.shape)
    c.close()
np.savetxt('data/v0-3/fco2_out.csv',out,delimiter=',',header='Year,Month,Latitude (deg N),Longitude (Deg E),SOCAT fCO2sw (uatm),UExP-FNN-U fCO2sw (uatm),UExP-FNN-U fCO2sw unc (uatm)')
#
t = 0
for file in files:
    print(file)
    c = Dataset(file,'r')
    socat_fco2 = np.array(c['month_socat_mean_fco2'])
    f = np.where(np.isnan(socat_fco2) == 0)[0]
    if len(f) > 0:
        nn_fco2 = np.array(c['month_fco2_sw_in_bio'])
        unc = np.array(c['month_fco2_tot_unc_in_bio'])
        time = np.array(c['month_time'])
        lat = np.array(c['month_latitude'])
        lon = np.array(c['month_longitude'])
        yr = []; mon =[]; day = [];
        for i in range(len(time)):
            date = (datetime.datetime(1950,1,1) + datetime.timedelta(days=int(time[i])))
            mon.append(date.month)
            yr.append(date.year)
        yr = np.array(yr); mon = np.array(mon);
        if t == 0:
            out = np.array([yr[f],mon[f],lat[f],lon[f],socat_fco2[f],nn_fco2[f],unc[f]])
            if out.shape[0] > out.shape[1]:
                out=np.transpose(out)
            t=1
        else:
            out = np.concatenate((out,np.transpose(np.array([yr[f],mon[f],lat[f],lon[f],socat_fco2[f],nn_fco2[f],unc[f]]))),axis=0)
        print(out.shape)
    c.close()
np.savetxt('data/v0-3/fco2_out_bio.csv',out,delimiter=',',header='Year,Month,Latitude (deg N),Longitude (Deg E),SOCAT fCO2sw (uatm),UExP-FNN-U fCO2sw (uatm),UExP-FNN-U fCO2sw unc (uatm)')
