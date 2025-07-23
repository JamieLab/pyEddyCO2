#!/usr/bin/env python3
import numpy as np
import datetime
import matplotlib.pyplot as plt
import matplotlib.transforms
from mpl_toolkits.basemap import Basemap
import matplotlib.transforms

def base_tracks_start(ax,latb,lonb):
    m = Basemap(projection='merc',llcrnrlat=latb[0],urcrnrlat=latb[1],\
                llcrnrlon=lonb[0],urcrnrlon=lonb[1],lat_ts=20,resolution='l',ax=ax)
    return m

def base_tracks_end(ax,m,draw_parrells = True):
    m.drawmapboundary(fill_color='#99ffff')
    m.fillcontinents(color='#C2c3c3')
    m.drawcoastlines()
    parallels = np.arange(-90.,91,45.)
    meridion = np.arange(-180,181,60.)
    if draw_parrells:
        m.drawparallels(parallels,labels=[True,False,False,False])
        m.drawmeridians(meridion,labels=[False,False,False,True])

def base_tracks_plt(ax,eddy,m):
    f = np.unique(eddy['track'])
    for i in f:
        #print(i)
        g = np.argwhere(i == eddy['track'])
        x,y = m(eddy['longitude'][g],eddy['latitude'][g])
        m.plot(x,y)

def plot_eddy_shape(ax,m,eddy,val=0,step=30):
    f = np.unique(eddy['track'])
    print(f)
    print(f[val])
    f = np.argwhere(eddy['track'] == f[val])
    le = len(f)
    #print(le)
    x,y = m(eddy['longitude'][f],eddy['latitude'][f])
    m.plot(x,y,'r-')
    for i in range(0,le,step):
        print(i)
        x,y = m(np.squeeze(eddy['effective_contour_longitude'][f[i],:]),np.squeeze(eddy['effective_contour_latitude'][f[i],:]))
        m.plot(x,y,'b--')
