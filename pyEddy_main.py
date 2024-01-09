#!/usr/bin/env python3

from netCDF4 import Dataset
import numpy as np
import datetime

def AVISO_load(file,no_load):
    no_att = ['scale_factor','add_offset','min','max']
    # Setup output dictionary
    out = {}
    desc = {}
    c = Dataset(file,'r')
    # Cycle through the variables within the netcdf file
    for v in c.variables:
        print(v)
        # If this variable does not appear on the no_load list, then the variable
        # is loaded into the output dictionary.
        if any(x in v for x in no_load) == False:
            print('Load!')
            out[v] = np.squeeze(np.array(c.variables[v]))
            t = {}
            for key in c.variables[v].ncattrs():
                print(key)
                if any(x in key for x in no_att) == False:
                    print('Load Att!')
                    t[key] = c.variables[v].getncattr(key)
            desc[v] = t

    c.close()
    #Longitude correction, so all longitudes are in degrees east with negative west values
    f = np.argwhere(out['longitude'] > 180)
    out['longitude'][f] = out['longitude'][f] - 360
    f = np.argwhere(out['effective_contour_longitude'] > 180)
    out['effective_contour_longitude'][f[:,0],f[:,1]] = out['effective_contour_longitude'][f[:,0],f[:,1]] - 360
    return out,desc

def box_split(eddy,latb,lonb,dateb):
    # Here we find any eddy that strays into our spatial box defined by latb,lonb and the time
    # window dateb.
    f = np.argwhere((eddy['time'] < date_convert(dateb[1])) & (eddy['time']>=date_convert(dateb[0])) &
        (eddy['latitude'] < latb[1]) & (eddy['latitude']>= latb[0]) &
        (eddy['longitude'] < lonb[1]) & (eddy['longitude'] >= lonb[0]))
    # We extract all the eddy track numbers even if the eddy falls within our box for a single day
    # We can perform checks later if the eddy stays in the domain or leaves after 1 day
    uni = np.unique(eddy['track'][f])
    # Here we find all the values that correspond to our unique eddy track numbers in our domain.
    # And we cut our eddy directory to this. If we set our spatial box to the globe and the eddy time
    # window to the full time length, then no need to call this function.
    f = np.argwhere(np.isin(eddy['track'],uni) == True)
    eddy = dict_split(eddy,f)
    return eddy

def dict_split(eddy,f):
    # Function to split the eddy database based on indexes of values. For example used by eddy_length_min to
    # split full tracks that meet the timelength critera, or box_split for eddy tracks that fall within the bounds.
    eddy2 = {}
    for v in eddy.keys():
        eddy2[v] = np.squeeze(eddy[v][f])
    return eddy2

def date_convert(dates,d = datetime.datetime(1950,1,1,0,0,0)):
    out = (dates - d).days
    return out

def date_con(dates,d = datetime.datetime(1950,1,1,0,0,0)):
    out= d + datetime.timedelta(days=dates)
    return out

def eddy_length_min(eddy,mindays,maxdays = 100000):
    #Function to strip out eddies with a lifetime greater than maxdays, and less than min days
    uni,count = np.unique(eddy['track'],return_counts=True)
    f = np.argwhere((count < maxdays) & (count >= mindays))
    f = np.argwhere(np.isin(eddy['track'],uni[f]) == True)
    eddy = dict_split(eddy,f)
    return eddy
