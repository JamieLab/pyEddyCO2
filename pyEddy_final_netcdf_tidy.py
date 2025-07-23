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


loc = 'F:/eddy/n_anticyclonic/'
files = glob.glob(loc+'*.nc')

for i in files:
    c = Dataset(i,'a')
    c.contact = 'Daniel J. Ford (d.ford@exeter.ac.uk)'
    c.dataset_repository_doi = '10.5281/zenodo.15689877'
    c.code_location = 'https://github.com/JamieLab/pyEddyCO2'
    c.version = 'v0-1'
    c.publication = 'Daniel J. Ford, Jamie D. Shutler, Katy L. Sheen1, Gavin H. Tilstone, and Vassilis Kitidis. UEx-Eddies: a biogeochemical long lived mesoscale eddy trajectories for studying air-sea CO2 fluxes (in prep)'
    c.eddy_trajectories = 'META3.2_DT_allsat'
    c.close()
