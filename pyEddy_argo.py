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
from ftplib import FTP

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

def check_argo(file,argo_data,argo_out,plot=False):
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
                if plot:
                    plt.figure()
                    plt.scatter(lons,lats)
                    plt.scatter(l[:,0],l[:,1])
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

def makefolder(fold):
    if not os.path.exists(fold):
        os.makedirs(fold)

def checkfileexist(file):
    #print(file)
    g = glob.glob(file)
    #print(g)
    if not g:
        return False
    else:
        return True

def argo_download(argo_loc,argo_list):
    ftp_loc = 'ftp.ifremer.fr' # FTP Location
    loc = '/ifremer/argo/dac' # Location within the FTP that the Argo files are
    #Make the local argo folder

    makefolder(argo_loc)
    # Load the argo profiles list

    for i in range(0,len(argo_list)):
        print(str(i) + ': ' +argo_list['argo_file'].iloc[i])
        s = argo_list['argo_file'].iloc[i].split('/')
        makefolder(os.path.join(argo_loc,s[0]))
        makefolder(os.path.join(argo_loc,s[0],s[1]))
        makefolder(os.path.join(argo_loc,s[0],s[1],s[2]))
        if (checkfileexist(os.path.join(argo_loc,argo_list['argo_file'].iloc[i])) == False) & (argo_list['argo_file'].iloc[i].split('/')[-1][0] != 'R'): # Second term to only get delayed (so corrected) files....
            print('Downloading: ' + argo_list['argo_file'].iloc[i])
            ftp = FTP(ftp_loc)
            ftp.login()
            l = argo_list['argo_file'].iloc[i].split('/')
            #print(loc+'/'+'/'.join(l))
            ftp.cwd(loc+'/'+'/'.join(l[0:-1]))
            #print(l[-1])
            ftp.retrbinary("RETR " + loc+'/'+'/'.join(l) ,open(os.path.join(argo_loc,argo_list['argo_file'].iloc[i]), 'wb').write)
            ftp.close()

if __name__ == '__main__':
    """
    Setup bits and file paths
    """

    file = 'F:/Data/Argo/ar_index_global_prof_12112025.txt'# This is the path to your Argo file.
    argo_save_loc = 'F:/Data/ARGO/core_data'
    loc = 'F:/eddy/v0-3/n_anticyclonic/' #This is the directory with the eddies files
    f = file.split('.') # Here I split the file name so I can append extra bits to the file name

    # Comment these lines if you have already converted the times
    data = load_argofile(file)
    # This function converts the Argo date string into year, mon, day columns, and then saves a new file with these columns
    data = convert_time(data,temp_file = f[0]+'_DJF.txt')
    print(data)

    # Load the new generated file (I do this so we dont have to keep rerunning the date conversion).
    data = load_argofile(f[0]+'_DJF.txt',skiprows=0)
    argo_out = pd.DataFrame(columns=['argo_file','eddy_file','year','month','day','latitude','longitude','eddy_index']) # Here I setup the output pandas table


    # files = glob.glob(loc+'6*.nc') # This bit of code is looking for all the eddy files that start with a '6' (so subsetting to something more manageable)
    files = [os.path.join(loc,'679331.nc')] # But here you can see you can also just provide files manually (example of of the South Atlantic eddy)

    print(files)
    # This loop cycles through each eddy file and runs the matching script (and then outputs all the Argo matches to the argo_out table)
    for file in files:
        print(file)
        file_s = file.split('\\')[-1].split('.')
        print(file_s)
        argo_out = check_argo(file,data,argo_out,plot=False) # Toggle the plot = True if you want to see the matching for each timestep

    argo_out.to_csv('argo_matched.csv',sep=',') # This saves that argo_out table to a file

    data = load_argofile('argo_matched.csv',skiprows=0) # Load the argo_out table, so we dont have to run the matching again.
    print(data)
    uni = np.unique(data['eddy_file']) # Looking for the unique eddies in the file
    print(uni)

    # Cycling through the argo_out table to find the number of argo profiles in each eddy and prints it out.
    # This finds the eddy with the most argo matches.
    t = 0
    for i in range(len(uni)):
        f = np.where(data['eddy_file'] == uni[i])[0]
        print(uni[i] + ' = ' + str(len(f)))
        if t < len(f):
            t = len(f)
            a = uni[i]

    print(t)
    print(a)

    # Quick bit of plotting to show you the eddy track with the most argo profilers matched.
    c = Dataset(a,'r')
    lon = np.array(c['longitude'])
    lat = np.array(c['latitude'])
    c.close()

    plt.figure()
    f = np.where(data['eddy_file'] == a)[0]
    print(f)
    plt.scatter(lon,lat)
    plt.scatter(data['longitude'][f],data['latitude'][f])


    #Lets download the Argo data for this specific eddy
    argo_download(argo_save_loc,data) # This takes the data variable (that is saved as 'argo_matched.csv', and downloads the netcdf files.

    #Example basic plotting that Dan showed
    col = 3
    row= int(np.ceil(len(data)/col/5))
    fig,ax = plt.subplots(row,col,figsize=(col*7,row*7))
    ax = ax.ravel()
    t = 0
    for i in range(len(data)):
        c = Dataset(os.path.join(argo_save_loc,data['argo_file'][i]),'r')
        depth = np.array(c['PRES_ADJUSTED'])
        temp = np.array(c['TEMP_ADJUSTED'])
        temp[temp>40] = np.nan
        c.close()


        if len(depth.shape) > 1:
            for j in [0]:#range(depth.shape[0]):
                ax[t].plot(temp[j,:],depth[j,:],label=str(data['year'][i]) + '/' +str(data['month'][i])+ '/' +str(data['day'][i]))
        else:
            ax[t].plot(temp,depth,str(data['year'][i]) + '/' +str(data['month'][i])+ '/' +str(data['day'][i]))

        ax[t].set_ylim([0,2000])
        ax[t].invert_yaxis()
        if i !=0:
            if i % 5 == 0:
                ax[t].legend(fontsize=8)
                t=t+1
    plt.show()
