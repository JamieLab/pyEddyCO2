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

def load_argofile(file,skiprows=8):
    """
    Function to load the Argo Datafile.
    Skiprows should be manually set to the header length of the file.
    """
    data = pd.read_table(file,sep=',',skiprows=skiprows)
    return data

def convert_time(argo,headername='date',temp_file=False):
    """
    Function to add year, month, day columns to the Pandas table for the Argo data.
    """
    date = argo[headername]
    day = np.zeros(len(data)); day[:] = np.nan
    month = np.copy(day)
    year = np.copy(day)
    for i in range(len(data)):
        print(str(i/len(data)*100))
        d = date[i]#
        if np.isnan(d) == 0:
            dt = datetime.datetime.strptime(str(int(d)),'%Y%m%d%H%M%S')
            day[i] = dt.day
            month[i] = dt.month
            year[i] = dt.year

    data['Year'] = year
    data['Month'] = month
    data['Day'] = day
    if temp_file:
        data.to_csv(temp_file,sep=',',index = False)
    return data

def check_argo(file,argo_data,argo_out):
    """
    Function to check whether an eddy colocates with Argo profilers
    """
    c = Dataset(file,'r')
    keys = c.variables.keys()

    lat = np.array(c['latitude'])
    lon = np.array(c['longitude'])
    time = np.array(c['time'])
    con_lat = np.array(c['effective_contour_latitude'])
    con_lon = np.array(c['effective_contour_longitude'])
    c.close()

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
            f = np.where((argo_data['Year'] == yr) & (argo_data['Month'] == mon) & (argo_data['Day'] == day))[0]
            if len(f)>0:
                radmax = Edeobs.radius_check(lonc,latc,lons,lats)
                if (lonc - (radmax*2) < -180) or (lonc + (radmax*2) > 180):
                    print('Near the dateline')
                    #lons[lons<-180] = lons[lons<-180] + 360
                    path = mpltPath.Path(np.transpose([lons,lats]))

                    lon_socat_temp = np.array(argo_data['longitude'][f])
                    h = np.where(np.sign(lon_socat_temp) != np.sign(lonc))
                    lon_socat_temp[h] = lon_socat_temp[h] + (360 * np.sign(lonc))
                    l = np.transpose([lon_socat_temp,np.array(argo_data['latitude'][f])])
                else:
                    path = mpltPath.Path(np.transpose([lons,lats]))
                    l = np.transpose([np.array(argo_data['longitude'][f]),np.array(argo_data['latitude'][f])])
                #print(l)
                inp = path.contains_points(l)
                #print(inp)
                if (lonc - (radmax*2) < -180) or (lonc + (radmax*2) > 180):
                    plt.figure()
                    plt.scatter(lons,lats)
                    plt.scatter(lon_socat_temp,argo_data['latitude'][f])
                    plt.show()
                g = np.where(inp == True)[0]
                if len(g) > 0:
                    print('Yes')

                    for j in range(len(g)):
                        a = pd.DataFrame([{'argo_file': argo_data['file'][f[g[j]]],
                            'eddy_file': file,
                            'year':yr,
                            'month': mon,
                            'day': day,
                            'latitude': argo_data['latitude'][f[g[j]]],
                            'longitude': argo_data['longitude'][f[g[j]]],
                            'eddy_index': i
                            }])
                        argo_out = pd.concat([argo_out,a],ignore_index=True,axis=0)
        else:
            print('Broken eddy polygon...')

    return argo_out

file = 'F:/Data/Argo/ar_index_global_prof_12112025.txt'
f = file.split('.')

# data = load_argofile(file)
# data = convert_time(data,temp_file = f[0]+'_DJF.txt')
# print(data)

data = load_argofile(f[0]+'_DJF.txt',skiprows=0)
argo_out = pd.DataFrame(columns=['argo_file','eddy_file','year','month','day','latitude','longitude','eddy_index'])

loc = 'F:/eddy/n_anticyclonic/'
files = glob.glob(loc+'6*.nc')
print(files)
files = ['F:/eddy/n_anticyclonic/322158.nc']
# files = ['F:/eddy/n_anticyclonic/666884.nc']

for file in files:
    print(file)
    file_s = file.split('\\')[-1].split('.')
    print(file_s)
    argo_out = check_argo(file,data,argo_out)

# argo_out.to_csv('argo_matched.csv',sep=',')

data = load_argofile('argo_matched.csv',skiprows=0)
print(data)
uni = np.unique(data['eddy_file'])
print(uni)

t = 0
for i in range(len(uni)):
    f = np.where(data['eddy_file'] == uni[i])[0]
    print(uni[i] + ' = ' + str(len(f)))
    if t < len(f):
        t = len(f)
        a = uni[i]

a = 'F:/eddy/n_anticyclonic\\679331.nc'
print(t)
print(a)

c = Dataset(a,'r')
lon = np.array(c['longitude'])
lat = np.array(c['latitude'])
c.close()

plt.figure()
f = np.where(data['eddy_file'] == a)[0]
print(f)
plt.scatter(lon,lat)
plt.scatter(data['longitude'][f],data['latitude'][f])
plt.show()
