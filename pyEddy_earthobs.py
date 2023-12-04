#!/usr/bin/env python3

from netCDF4 import Dataset
import numpy as np
import datetime
import os
import glob
import matplotlib.path as mpltPath
import matplotlib.pyplot as plt
import pyEddy_main as pyEddy_m
import morel91 as pp

def timeseries_start(eddy, track_no):
    f = np.squeeze(np.argwhere(eddy['track'] == track_no))
    eddy = pyEddy_m.dict_split(eddy,f)
    return eddy

def timeseries_save(eddy,desc,s_loc):
    out = Dataset(os.path.join(s_loc,str(eddy['track'][0]) +'.nc'),mode='w',format='NETCDF4_CLASSIC')
    time_dim = out.createDimension('d_time', len(eddy['time']))
    n_sample = out.createDimension('n_sample', eddy['effective_contour_latitude'].shape[1])

    for key in eddy.keys():
        if len(eddy[key].shape) > 1:
            var = out.createVariable(key,np.float32,('d_time','n_sample'))
            var[:] = eddy[key]
        else:
            var = out.createVariable(key,np.float32,('d_time'))
            var[:] = eddy[key]
        for val in desc[key].keys():
            setattr(var,val,desc[key][val])
    out.close()

def circle(x,y,r):
    # Function to generate a circle with radius, r, and centre at x, y.
    # This function can create a patch accepted by pointineddy().
    th = np.arange(0,2*np.pi,np.pi/20)
    xunit = r * np.cos(th) + x
    yunit = r * np.sin(th) + y
    return xunit,yunit

def ellipse(x,y,rlon,rlat):
    # Function to generate a ellipse with different radii for the x and y with a centre at x, y.
    # This function can create a patch accepted by pointineddy().
    th = np.arange(0,2*np.pi,np.pi/20)
    xunit = rlat * np.cos(th) + x
    yunit = rlon * np.sin(th) + y
    return xunit,yunit

def radius_check(lonc,latc,lons,lats):
    # Function checks the maximum radius of the eddy contour patch. This informs the
    # load_file() function of the eddy radius, and this is used to compute an additional
    # area to retirved from the file. Takes the mean of the x and y directions incases
    # where the eddy is more ellipse like then circular.
    x = np.max(np.abs(lonc-lons))
    y = np.max(np.abs(latc-lats))
    return np.mean([x,y])

def me_mtodeg(rad,lat):
    # Function to estimate the radii of the circle. r is the radii of the longitude axis
    # at the latitude of the eddy, and the r2 is the radii on the latitude axis.
    #print(rad)
    r = lon_mtodeg(rad,lat)
    #print(r)
    r2 = lon_mtodeg(rad,0)
    #print(r2)
    return [r,r2]

def lon_mtodeg(rad,lat):
    # Function to roughly convert a radii value in m to deg.
    r = (rad/111320) * np.cos(np.radians(lat))
    return r

def julian(date):
    # Function to get the julian date number. This function will include leap years etc.
    # This is mainly used for older NASA data which used julian day.
    d = datetime.datetime(date.year-1,12,31)
    return (date-d).days

def load_file(set,latv,lonv,latc,lonc,radmax,radm,lat=[],lon=[]):
    # Function which loads the latitude and longitude values from a netcdf variable, set.
    # Then calculates on the grid, where the eddy is with a buffer region around the eddy,
    # for example, find latitude and longitudes in 3x the radius of the eddy centre.
    # This allows us to extract only the data required from the data netcdf, instead of
    # loading the whole global grid - not a issue for coarse resolution data, but intensive
    # for higher resolution data.
    if (len(lat) == 0) & (len(lon) == 0):
        lat = np.squeeze(set.variables[latv])
        lon = np.squeeze(set.variables[lonv])
    if np.max(lon) > 180:
        f = np.argwhere(lon > 180)
        lon = lon - 360
    f = np.squeeze(np.argwhere((lat <= latc + radmax*radm) & (lat >= latc - radmax*radm)))
    g = np.squeeze(np.argwhere((lon <= lonc + radmax*radm) & (lon >= lonc - radmax*radm)))
    #var = np.squeeze(set.variables[datav][0,f,g])
    lat = lat[f]
    lon = lon[g]
    return lat,lon,f,g

def pointsineddy(lon,lat,lons,lats):
    # Function to assess which EarthObservation data are within the bounds
    # of the eddy, which allows extract of these data. Also able to use this mask
    # to extract the data surrounding the eddy (environmental conditions).
    long,latg = np.meshgrid(lon,lat)
    path = mpltPath.Path(np.transpose([lons,lats]))
    l = np.transpose(np.squeeze([np.reshape(long,[-1,1]),np.reshape(latg,[-1,1])]))
    inp = path.contains_points(l)
    inp = np.reshape(inp,long.shape)
    return inp

def inouteddy(data,inp):
    # Function to produce median, iqr, mean, std and number of values for data within the
    # eddy and outside the eddy seperately. This provides the statistics required for the daily
    # timeseries.
    ineddy = [np.nanmedian(data[inp==True]),np.subtract(*np.nanpercentile(data[inp==True], [75, 25])),
        np.nanmean(data[inp==True]),np.nanstd(data[inp==True]),len(data[inp==True]) - np.sum(np.isnan(data[inp==True])),len(data[inp==True])]
    outeddy =[np.nanmedian(data[inp==False]),np.subtract(*np.nanpercentile(data[inp==False], [75, 25])),
        np.nanmean(data[inp==False]),np.nanstd(data[inp==False]),len(data[inp==False]) - np.sum(np.isnan(data[inp==False])),len(data[inp==False])]
    #print(ineddy)
    return ineddy,outeddy

def inoutsplit(eddy,out,var,desc,units):
    vals = ['_median','_iqr','_mean','_std','_valid_n','_total_n']
    for i in range(0,len(vals)):
        eddy[var+vals[i]] = out[:,i]
        desc[var+vals[i]] = {}
        if (i == 5) or (i ==4):
            desc[var+vals[i]]['units'] = ''
        else:
            desc[var+vals[i]]['units'] = units
    return eddy,desc

def add_sstCCI(loc,eddy,desc,radm = 3,plot=0):
    # Function to extract sstCCI data for the eddy and the surrounding environment.
    sstin = np.empty((len(eddy['time']),6))
    sstin[:] = np.nan
    sstout = np.empty((len(eddy['time']),6))
    sstout[:] = np.nan
    for t in range(0,len(eddy['time'])):
        print(t)
        date = pyEddy_m.date_con(eddy['time'][t])
        lonc = eddy['longitude'][t]
        latc = eddy['latitude'][t]
        lons = eddy['effective_contour_longitude'][t,:]
        lats = eddy['effective_contour_latitude'][t,:]
        if (lats[0] != 0.0) & (lons[0] != 180.0):
            file = glob.glob(os.path.join(loc,date.strftime("%Y"),date.strftime("%m"),date.strftime("%Y%m%d*.nc")))
            # if radius_check(lonc,latc,lons,lats) > rad[1]:
            radmax = radius_check(lonc,latc,lons,lats)
            # else:
            #     radmax = rad[1]
            c = Dataset(file[0])
            lat,lon,f,g = load_file(c,'lat','lon',latc,lonc,radmax,radm)
            sst = np.squeeze(c.variables['analysed_sst'][0,f,g])
            sst[sst.mask==True] = np.nan
            sst = sst.data
            lat,lon,f,g = load_file(c,'lat','lon',latc,lonc,radmax,radm)
            sst_u = np.squeeze(c.variables['analysed_sst_uncertainty'][0,f,g])
            c.close()

            if plot == 1:
                f,ax1 = plt.subplots()
                m = ax1.pcolor(lon,lat,sst)
                plt.colorbar(m)
                ax1.plot(lons,lats,'b--')
                # cix,ciy = ellipse(lonc,latc,rad[0],rad[1])
                # ax1.plot(cix,ciy,'r--')
                plt.show()
            #lons,lats = ellipse(lonc,latc,rad[0],rad[1])
            inp = pointsineddy(lon,lat,lons,lats)
            sstin[t,:],sstout[t,:] = inouteddy(sst,inp)
    eddy,desc = inoutsplit(eddy,sstin,'cci_sst_in',desc,'kelvin')
    eddy,desc = inoutsplit(eddy,sstout,'cci_sst_out',desc,'kelvin')
    return eddy,desc

def add_MODchl(loc,eddy,desc,radm = 3,plot=0):
    # Add chl-a values from MODIS-A downloaded from NASA OceanColour data website
    #print(os.path.join(loc,date.strftime("%Y"),str(julian(date)),date.strftime("A%Y*.nc")))

    sstin = np.empty((len(eddy['time']),6))
    sstin[:] = np.nan
    sstout = np.empty((len(eddy['time']),6))
    sstout[:] = np.nan
    for t in range(0,len(eddy['time'])):
        print(t)
        date = pyEddy_m.date_con(eddy['time'][t])
        lonc = eddy['longitude'][t]
        latc = eddy['latitude'][t]
        lons = eddy['effective_contour_longitude'][t,:]
        lats = eddy['effective_contour_latitude'][t,:]
        if (lats[0] != 0.0) & (lons[0] != 180.0):
            file = glob.glob(os.path.join(loc,date.strftime("%Y"),str(julian(date)),date.strftime("A%Y*.nc")))
            # if radius_check(lonc,latc,lons,lats) > rad[1]:
            radmax = radius_check(lonc,latc,lons,lats)
            # else:
            #     radmax = rad[1]
            c = Dataset(file[0])
            lat,lon,f,g = load_file(c,'lat','lon',latc,lonc,radmax,radm)
            sst = np.log10(np.squeeze(c.variables['chlor_a'][f,g]))
            sst[sst.mask==True] = np.nan
            sst = sst.data
            c.close()

            if plot == 1:
                f,ax1 = plt.subplots()
                m = ax1.pcolor(lon,lat,sst)
                plt.colorbar(m)
                ax1.plot(lons,lats,'b--')
                # cix,ciy = ellipse(lonc,latc,rad[0],rad[1])
                # ax1.plot(cix,ciy,'r--')
                plt.show()
            #lons,lats = ellipse(lonc,latc,rad[0],rad[1])
            inp = pointsineddy(lon,lat,lons,lats)
            sstin[t,:],sstout[t,:] = inouteddy(sst,inp)
    #print(sstout)
    eddy,desc = inoutsplit(eddy,sstin,'modis_chla_in',desc,'log10(mg/m3)')
    eddy,desc = inoutsplit(eddy,sstout,'modis_chla_out',desc,'log10(mg/m3)')
    return eddy,desc

def add_wind(loc,eddy,desc,radm = 3,plot=0):
    #
    sstin = np.empty((len(eddy['time']),6))
    sstin[:] = np.nan
    sstout = np.empty((len(eddy['time']),6))
    sstout[:] = np.nan
    for t in range(0,len(eddy['time'])):
        print(t)
        date = pyEddy_m.date_con(eddy['time'][t])
        lonc = eddy['longitude'][t]
        latc = eddy['latitude'][t]
        lons = eddy['effective_contour_longitude'][t,:]
        lats = eddy['effective_contour_latitude'][t,:]
        if (lats[0] != 0.0) & (lons[0] != 180.0):
            file = glob.glob(os.path.join(loc,date.strftime("Y%Y"),date.strftime("M%m"),date.strftime("CCMP_Wind_Analysis_%Y%m%d*.nc")))
            # if radius_check(lonc,latc,lons,lats) > rad[1]:
            radmax = radius_check(lonc,latc,lons,lats)
            # else:
            #     radmax = rad[1]
            c = Dataset(file[0])
            lat,lon,f,g = load_file(c,'latitude','longitude',latc,lonc,radmax,radm)
            uwnd = np.squeeze(c.variables['uwnd'][:,f,g])
            print(uwnd.shape)
            vwnd = np.squeeze(c.variables['vwnd'][:,f,g])
            sst = np.squeeze(np.nanmean(np.sqrt(uwnd**2 + vwnd**2),axis=0))
            sst[sst.mask==True] = np.nan
            sst = sst.data
            c.close()

            if plot == 1:
                f,ax1 = plt.subplots()
                m = ax1.pcolor(lon,lat,sst)
                plt.colorbar(m)
                #plt.colorbar()

                #cix,ciy = ellipse(lonc,latc,rad[0],rad[1])
                #ax1.plot(cix,ciy,'r--')
                ax1.plot(lons,lats,'b--')
                plt.show()
            #lons,lats = ellipse(lonc,latc,rad[0],rad[1])
            inp = pointsineddy(lon,lat,lons,lats)
            sstin[t,:],sstout[t,:] = inouteddy(sst,inp)
    eddy,desc = inoutsplit(eddy,sstin,'ccmp_wind_in',desc,'ms-1')
    eddy,desc = inoutsplit(eddy,sstout,'ccmp_wind_out',desc,'ms-1')
    return eddy,desc

def add_GlobPAR(loc,eddy,desc,radm = 3,plot=0):
    sstin = np.empty((len(eddy['time']),6))
    sstin[:] = np.nan
    sstout = np.empty((len(eddy['time']),6))
    sstout[:] = np.nan
    for t in range(0,len(eddy['time'])):
        print(t)
        date = pyEddy_m.date_con(eddy['time'][t])
        lonc = eddy['longitude'][t]
        latc = eddy['latitude'][t]
        lons = eddy['effective_contour_longitude'][t,:]
        lats = eddy['effective_contour_latitude'][t,:]
        if (lats[0] != 0.0) & (lons[0] != 180.0):
            file = glob.glob(os.path.join(loc,date.strftime("L3m_%Y%m%d*.nc")))
            # if radius_check(lonc,latc,lons,lats) > rad[1]:
            radmax = radius_check(lonc,latc,lons,lats)
            # else:
            #     radmax = rad[1]
            c = Dataset(file[0])
            lat,lon,f,g = load_file(c,'lat','lon',latc,lonc,radmax,radm)
            sst = np.squeeze(c.variables['PAR_mean'][f,g])
            sst[sst.mask==True] = np.nan
            sst = sst.data
            c.close()

            if plot == 1:
                f,ax1 = plt.subplots()
                m = ax1.pcolor(lon,lat,sst)
                plt.colorbar(m)
                ax1.plot(lons,lats,'b--')
                # cix,ciy = ellipse(lonc,latc,rad[0],rad[1])
                # ax1.plot(cix,ciy,'r--')
                plt.show()
            #lons,lats = ellipse(lonc,latc,rad[0],rad[1])
            inp = pointsineddy(lon,lat,lons,lats)
            sstin[t,:],sstout[t,:] = inouteddy(sst,inp)
    #print(sstout)
    eddy,desc = inoutsplit(eddy,sstin,'glob_par_in',desc,'einstein/m2/day')
    eddy,desc = inoutsplit(eddy,sstout,'glob_par_out',desc,'einstein/m2/day')
    return eddy,desc
