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


# loc = 'F:/eddy/v0-3/n_anticyclonic/'
loc = 'F:/eddy/v0-3/n_cyclonic/'
files = glob.glob(loc+'*.nc')

for i in files:
    print(i)
    c = Dataset(i,'a')
    c.contact = 'Daniel J. Ford (d.ford@exeter.ac.uk)'
    c.dataset_repository_doi = 'https://doi.org/10.5281/zenodo.15689876'
    c.code_location = 'https://github.com/JamieLab/pyEddyCO2'
    c.version = 'v0-3'
    c.publication = 'Daniel J. Ford, Jamie D. Shutler, Katy L. Sheen, Gavin H. Tilstone, and Vassilis Kitidis. UEx-L-Eddies: Decadal and global long-lived mesoscale eddy trajectories with coincident air-sea CO2 fluxes and environmental conditions (in review)'
    c.eddy_trajectories = 'META3.2_DT_allsat'
    c.close()
