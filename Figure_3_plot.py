#!/usr/bin/env python3
from netCDF4 import Dataset
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import matplotlib.transforms
font = {'weight' : 'normal',
        'size'   : 14}
matplotlib.rc('font', **font)
fold = 'F:/eddy/n_anticyclonic'

files = ['496.nc','194465.nc']
fig = plt.figure(figsize=(15,15))
gs = GridSpec(2,2, figure=fig, wspace=0.25,hspace=0.15,bottom=0.1,top=0.93,left=0.075,right=0.95)
ax = [fig.add_subplot(gs[0,0]),fig.add_subplot(gs[0,1]),fig.add_subplot(gs[1,0]),fig.add_subplot(gs[1,1])]
cols = ['#332288','#44AA99','#882255','#DDCC77', '#117733', '#88CCEE','#999933','#CC6677']
cou = 0
for file in files:
    c = Dataset(fold + '/' + file,'r')
    keys = c.variables.keys()

    l = []
    for key in keys:
        if ('unc' in key) & ('cumulative' in key) & ('in_physics' in key):
            l.append(key)

    print(l)
    label=['Gas Transfer','Wind','fCO$_{2 (sw)}$','Schmidt','Solubility skin','Solubility subskin','fCO$_{2 (atm)}$']
    uncs = ['k','wind','xco2atm','fco2sw']
    uncs_comp= ['ph2o','schmidt','solskin_unc','solsubskin_unc']

    combined = np.zeros((len(uncs)+len(uncs_comp),len(c[l[0]])))
    print(combined.shape)
    t = 0
    for i in range(len(uncs)):
        combined[t,:] = np.array(c['flux_unc_'+uncs[i]+'_in_physics_areaday_cumulative'])
        t=t+1

    for i in range(len(uncs_comp)):
        combined[t,:] = np.sqrt(np.array(c['flux_unc_'+uncs_comp[i]+'_in_physics_areaday_cumulative'])**2 + np.array(c['flux_unc_'+uncs_comp[i]+'_fixed_in_physics_areaday_cumulative'])**2)
        t = t+1

    print(combined)
    data_atm = combined[[2,4],:]
    combined = combined[[0,1,3,5,6,7],:]
    print(combined.shape)
    atm = np.sqrt(np.sum(data_atm**2,axis=0))
    atm = atm[np.newaxis,:]
    combined = np.append(combined,atm,axis=0)
    print(combined.shape)
    print(combined)
    totals = []
    for i in range(combined.shape[1]):
        totals.append(np.sum(combined[:,i]))


    for i in range(combined.shape[1]):
        bottom = 0
        for j in range(combined.shape[0]):
            if i == 1:
                p = ax[cou].bar(i+1,(combined[j,i]/totals[i])*100,bottom=bottom,color=cols[j],label=label[j])
            else:
                p = ax[cou].bar(i+1,(combined[j,i]/totals[i])*100,bottom=bottom,color=cols[j])
            bottom = bottom + (combined[j,i]/totals[i])*100

    print(totals)
    ax[cou].set_xlabel('Month since formation')
    ax[cou].set_ylabel('Relative contribution to uncertainty (%)')
    cou = cou+1
    for i in range(combined.shape[0]):
        ax[cou].plot(np.array(range(1,combined.shape[1]+1)),combined[i,:],color=cols[i],label=label[i])
    ax[cou].legend()
    ax[cou].set_xlabel('Month since formation')
    ax[cou].set_ylabel('Absolute contribution to uncertainty (Tg C)')
    cou=cou+1
let = ['a','b','c','d']
for i in range(4):
    #worldmap.plot(color="lightgrey", ax=ax[i])
    ax[i].text(0.92,1.06,f'({let[i]})',transform=ax[i].transAxes,va='top',fontweight='bold',fontsize = 24)
fig.savefig('figs/manuscript/Figure_3.png')
