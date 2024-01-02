#!/usr/bin/env python3

from netCDF4 import Dataset
import numpy as np
import datetime
import os
import glob
import matplotlib.path as mpltPath
import matplotlib.pyplot as plt
import pyEddy_main as pyEddy_m
import calendar

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

    # if np.max(lon) > 180:
    #     f = np.argwhere(lon > 180)
    #     lon = lon - 360
    lon_len = len(lon)
    lat,lon,f,g = find_f_g(lat,lon,latc,lonc,radmax,radm)

    return lat,lon,f,g,lon_len

def find_f_g(lat,lon,latc,lonc,radmax,radm):
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
            lon =c.variables['lon'][:]; lat = c.variables['lat'][:]
            #lat,lon,f,g = load_file(c,'lat','lon',latc,lonc,radmax,radm)
            if (lonc - (radmax*radm) < lon[0]) or (lonc + (radmax*radm) > lon[-1]):
                print('First')
                sst = np.squeeze(c.variables['analysed_sst'][0,:,:])
                sst[sst.mask==True] = np.nan
                sst_u = np.squeeze(c.variables['analysed_sst_uncertainty'][0,:,:])
                sst_u[sst_u.mask==True] = np.nan
                lon2 = np.copy(lon)
                #print(sst.shape)
                #lon =c.variables['longitude'][:]; lat = c.variables['latitude'][:]
                lon,sst = grid_switch(lon,sst)
                lon2,sst_u = grid_switch(lon,sst_u)
                lat,lon,f,g = find_f_g(lat,lon,latc,lonc,radmax,radm)
                sst=sst[np.ix_(f,g)]
                sst_u=sst_u[np.ix_(f,g)]
            else:
                print('Second')
                lat,lon,f,g = find_f_g(lat,lon,latc,lonc,radmax,radm)
                sst = np.squeeze(c.variables['analysed_sst'][0,f,g])
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

def add_cmems(loc,eddy,desc,var='so',units='psu',radm = 3,plot=0,log10=False,depth=True):
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
            file = glob.glob(os.path.join(loc,date.strftime("%Y"),date.strftime("%Y_%m*.nc")))
            day_val = date.day-1 # -1 as index starts at 0
            print(day_val)
            # if radius_check(lonc,latc,lons,lats) > rad[1]:
            radmax = radius_check(lonc,latc,lons,lats)
            # else:
            #     radmax = rad[1]
            c = Dataset(file[0])
            lon =c.variables['longitude'][:]; lat = c.variables['latitude'][:]
            #lat,lon,f,g = load_file(c,'lat','lon',latc,lonc,radmax,radm)
            if (lonc - (radmax*radm) < lon[0]) or (lonc + (radmax*radm) > lon[-1]):
                print('First')
                if depth:
                    sst = np.squeeze(c.variables[var][day_val,0,:,:])
                else:
                    sst = np.squeeze(c.variables[var][day_val,:,:])
                sst[sst.mask==True] = np.nan
                if log10:
                    sst = np.log10(sst)

                lon,sst = grid_switch(lon,sst)

                lat,lon,f,g = find_f_g(lat,lon,latc,lonc,radmax,radm)
                sst=sst[np.ix_(f,g)]
            else:
                print('Second')
                lat,lon,f,g = find_f_g(lat,lon,latc,lonc,radmax,radm)
                if depth:
                    sst = np.squeeze(c.variables[var][day_val,0,f,g])
                else:
                    sst = np.squeeze(c.variables[var][day_val,f,g])
                if log10:
                    sst = np.log10(sst)
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
    eddy,desc = inoutsplit(eddy,sstin,f'cmems_{var}_in',desc,units)
    eddy,desc = inoutsplit(eddy,sstout,f'cmems_{var}_out',desc,units)
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
        # if lonc<0:
        #     lonc = lonc+360
        latc = eddy['latitude'][t]
        lons = eddy['effective_contour_longitude'][t,:]
        #lons[lons<0] = lons[lons<0] + 360
        lats = eddy['effective_contour_latitude'][t,:]
        if (lats[0] != 0.0) & (lons[0] != 180.0):
            print(os.path.join(loc,date.strftime("Y%Y"),date.strftime("M%m"),date.strftime("CCMP_Wind_Analysis_%Y%m%d*.nc")))
            file = glob.glob(os.path.join(loc,date.strftime("Y%Y"),date.strftime("M%m"),date.strftime("CCMP_Wind_Analysis_%Y%m%d*.nc")))
            # if radius_check(lonc,latc,lons,lats) > rad[1]:
            radmax = radius_check(lonc,latc,lons,lats)
            # else:
            #     radmax = rad[1]
            c = Dataset(file[0])
            lon =c.variables['longitude'][:]; lat = c.variables['latitude'][:]
            #lat,lon,f,g,lon_len = load_file(c,'latitude','longitude',latc,lonc,radmax,radm)

            # print(lon_len)
            # print(len(g))
            print(lon[0])
            print(lonc - (radmax*radm))
            print(lon[-1])
            print(lonc + (radmax*radm))
            if (lonc - (radmax*radm) < lon[0]) or (lonc + (radmax*radm) > lon[-1]):
                print('First')
                sst = np.squeeze(np.nanmean(c.variables['ws'][:,:,:],axis=2))
                #print(sst.shape)
                #lon =c.variables['longitude'][:]; lat = c.variables['latitude'][:]
                lon,sst = grid_switch(lon,sst)
                lat,lon,f,g = find_f_g(lat,lon,latc,lonc,radmax,radm)
                # print(f)
                # print(g)
                sst=sst[np.ix_(f,g)]
            else:
                print('Second')
                lat,lon,f,g = find_f_g(lat,lon,latc,lonc,radmax,radm)
                sst = np.squeeze(np.nanmean(c.variables['ws'][f,g,:],axis=2))
            #print(f)
            #print(g)
            c.close()
            print(sst.shape)
            # sst[sst.mask==True] = np.nan
            # sst = sst.data


            if plot == 1:
                f,ax1 = plt.subplots()
                m = ax1.pcolor(lon,lat,sst)
                plt.colorbar(m)
                #plt.colorbar()

                #cix,ciy = ellipse(lonc,latc,rad[0],rad[1])
                #ax1.plot(cix,ciy,'r--')
                ax1.plot(lons,lats,'b--')
                plt.show()

            inp = pointsineddy(lon,lat,lons,lats)
            sstin[t,:],sstout[t,:] = inouteddy(sst,inp)
    eddy,desc = inoutsplit(eddy,sstin,'ccmp_wind_in',desc,'ms-1')
    eddy,desc = inoutsplit(eddy,sstout,'ccmp_wind_out',desc,'ms-1')
    return eddy,desc

def grid_switch(lon,var):
    """
    Function to switch a global grid from 0-360E to -180 to 180E. This requires the
    array to be manually modified.
    """

    #print(var_sh)

    if var.shape[0] == len(lon):
        var_sh = int(var.shape[0]/2)
        var_temp = np.empty((var.shape))
        var_temp[0:var_sh,:] = var[var_sh:,:]
        var_temp[var_sh:,:] = var[0:var_sh,:]
    else:
        var_sh = int(var.shape[1]/2)
        var_temp = np.empty((var.shape))
        var_temp[:,0:var_sh] = var[:,var_sh:]
        var_temp[:,var_sh:] = var[:,0:var_sh]
    #lon_temp = np.empty((var_sh*2))
    lon = lon-180
    # lon_temp[var_sh:] = lon[0:var_sh]
    # lon_temp[0:var_sh] = lon[var_sh:]
    return lon,var_temp

def produce_monthly(track,s_loc,vars=[]):
    vars.append('latitude')
    vars.append('longitude')
    c = Dataset(os.path.join(s_loc,str(track) +'.nc'),'r')
    time = np.array(c['time'])

    time_2 = np.zeros((len(time),2))
    for i in range(len(time)):
        time_2[i,0] = pyEddy_m.date_con(int(time[i])).month
        time_2[i,1] = pyEddy_m.date_con(int(time[i])).year

    max_year = np.max(time_2[:,1])
    min_year = np.min(time_2[:,1])

    time_m = []
    out_time = []
    ye = min_year
    mon = 1
    while ye < max_year+1:
        f = np.where((time_2[:,0] == mon) & (time_2[:,1] == ye))[0]
        month_r = calendar.monthrange(int(ye),int(mon))[1]
        if len(f) == month_r:
            time_m.append([f])
            out_time.append(pyEddy_m.date_convert(datetime.datetime(int(ye),int(mon),15)))
        mon=mon+1
        if mon == 13:
            ye=ye+1
            mon=1
    out = {}
    for v in vars:
        data = np.array(c[v])
        data_o = np.zeros((len(out_time))); data_o[:] = np.nan
        for i in range(len(out_time)):
            data_o[i] = np.nanmean(data[time_m[i]])
        out[v] = data_o
    c.close()

    c = Dataset(os.path.join(s_loc,str(track) +'.nc'),'a')
    try:
        c.createDimension('month_time',len(time_m))
    except:
        print('Dimension exists?')
    if 'month_time' in c.variables.keys():
        c['month_time'][:] = np.array(out_time)
    else:
        m = c.createVariable('month_time',np.float32,('month_time'))
        m[:] = np.array(out_time)
    c['month_time'].units = "days since 1950-01-01 00:00:00"
    for v in vars:
        if 'month_'+v in c.variables.keys():
            c['month_'+v][:] = out[v]
        else:
            m = c.createVariable('month_'+v,np.float32,('month_time'))
            m[:] = out[v]
    c.close()

def add_noaa(loc,s_loc,track):
    c = Dataset(os.path.join(s_loc,str(track) +'.nc'),'r')
    time = np.array(c['month_time'])
    lat = np.array(c['month_latitude'])
    lon = np.array(c['month_longitude'])
    c.close()
    xco2 = np.zeros((len(time))); xco2[:] = np.nan
    for i in range(len(time)):
        timed = pyEddy_m.date_con(int(time[i]))
        file = glob.glob(os.path.join(loc,timed.strftime("%Y_%m_*.nc")))
        c = Dataset(file[0],'r')
        latg = np.abs(np.array(c['latitude']) - lat[i])
        long = np.abs(np.array(c['longitude']) - lon[i])

        f = np.where(latg == np.min(latg))[0]
        #print(f)
        g = np.where(long == np.min(long))[0]
        #print(g)
        xco2[i] = np.array(c.variables['xCO2'][g,f])
        c.close()

    c = Dataset(os.path.join(s_loc,str(track) +'.nc'),'a')
    if 'month_xco2' in c.variables.keys():
        c['month_xco2'][:] = np.array(xco2)
    else:
        m = c.createVariable('month_xco2',np.float32,('month_time'))
        m[:] = np.array(xco2)
    c['month_xco2'].units = 'uatm'
