#!/usr/bin/env python3
"""
Created by Daniel J. Ford (d.ford@exeter.ac.uk)
Date: 03/2023

"""
#This is needed or the code crashes with the reanalysis step...
if __name__ == '__main__':
    import os
    import sys
    os.chdir('C:\\Users\\df391\\OneDrive - University of Exeter\\Post_Doc_ESA_Contract\\OceanICU')

    print(os.getcwd())
    print(os.path.join(os.getcwd(),'Data_Loading'))

    sys.path.append(os.path.join(os.getcwd(),'Data_Loading'))
    sys.path.append(os.path.join(os.getcwd()))
    import data_utils as du
    create_inp =False
    run_neural =False
    run_flux = True

    fluxengine_config = 'C:/Users/df391/OneDrive - University of Exeter/Post_Doc_ESA_Contract/OceanICU/fluxengine_config/fluxengine_config_night.conf'

    model_save_loc = 'D:/Eddies/NN/eddy_nn'
    inps = os.path.join(model_save_loc,'inputs')
    data_file = os.path.join(inps,'neural_network_input.nc')
    start_yr = 1985
    end_yr = 2023
    log,lag = du.reg_grid(lat=1,lon=1)

    if create_inp:
        from neural_network_train import make_save_tree
        make_save_tree(model_save_loc)
        # #
        # from Data_Loading.cmems_glorysv12_download import cmems_average
        #
        # cmems_average('D:/Data/CMEMS/SSS/MONTHLY',os.path.join(inps,'SSS'),start_yr=start_yr,end_yr=end_yr,log=log,lag=lag,variable='so')
        # cmems_average('D:/Data/CMEMS/MLD/MONTHLY',os.path.join(inps,'MLD'),start_yr=start_yr,end_yr=end_yr,log=log,lag=lag,variable='mlotst',log_av = True)
        # # #
        # from Data_Loading.cci_sstv2 import cci_sst_spatial_average
        # cci_sst_spatial_average(data='D:/Data/SST-CCI/V301/monthly',out_loc=os.path.join(inps,'SST'),start_yr=start_yr,end_yr=end_yr,log=log,lag=lag)
        # # #
        # from Data_Loading.interpolate_noaa_ersl import interpolate_noaa
        # interpolate_noaa('D:/Data/NOAA_ERSL/2024_download.txt',grid_lon = log,grid_lat = lag,out_dir = os.path.join(inps,'xco2atm'),start_yr=start_yr,end_yr = end_yr)
        # #
        # from Data_Loading.ERA5_data_download import era5_average
        # era5_average(loc = "D:/Data/ERA5/MONTHLY/DATA", outloc=os.path.join(inps,'msl'),log=log,lag=lag,var='msl',start_yr = start_yr,end_yr =end_yr)
        # era5_average(loc = "D:/Data/ERA5/MONTHLY/DATA", outloc=os.path.join(inps,'blh'),log=log,lag=lag,var='blh',start_yr = start_yr,end_yr =end_yr)
        # era5_average(loc = "D:/Data/ERA5/MONTHLY/DATA", outloc=os.path.join(inps,'d2m'),log=log,lag=lag,var='d2m',start_yr = start_yr,end_yr =end_yr)
        # era5_average(loc = "D:/Data/ERA5/MONTHLY/DATA", outloc=os.path.join(inps,'t2m'),log=log,lag=lag,var='t2m',start_yr = start_yr,end_yr =end_yr)
        # era5_average(loc = "D:/Data/ERA5/MONTHLY/DATA", outloc=os.path.join(inps,'msdwlwrf'),log=log,lag=lag,var='msdwlwrf',start_yr = start_yr,end_yr =end_yr)
        # era5_average(loc = "D:/Data/ERA5/MONTHLY/DATA", outloc=os.path.join(inps,'msdwswrf'),log=log,lag=lag,var='msdwswrf',start_yr = start_yr,end_yr =end_yr)
        #
        # era5_average(loc = "D:/Data/ERA5/MONTHLY/DATA", outloc=os.path.join(inps,'si10'),log=log,lag=lag,var='si10',start_yr = start_yr,end_yr =end_yr)
        # #
        # import Data_Loading.gebco_resample as ge
        # ge.gebco_resample('D:/Data/Bathymetry/GEBCO_2023.nc',log,lag,save_loc = os.path.join(inps,'bath.nc'),save_loc_fluxengine = os.path.join(inps,'fluxengine_bath.nc'))
        # # #
        # from Data_Loading.OSISAF_download import OSISAF_spatial_average
        # from Data_Loading.OSISAF_download import OSISAF_merge_hemisphere
        # OSISAF_spatial_average(data='D:/Data/OSISAF/monthly',out_loc=os.path.join(inps,'OSISAF'),start_yr=start_yr,end_yr=end_yr,log=log,lag=lag,hemi = 'NH')
        # OSISAF_spatial_average(data='D:/Data/OSISAF/monthly',out_loc=os.path.join(inps,'OSISAF'),start_yr=start_yr,end_yr=end_yr,log=log,lag=lag,hemi = 'SH')
        # OSISAF_merge_hemisphere(os.path.join(inps,'OSISAF'),os.path.join(inps,'bath.nc'),start_yr=start_yr,end_yr=end_yr,log=log,lag=lag)
        #
        # import Data_Loading.ccmp_average as cc
        # cc.ccmp_average('D:/Data/CCMP/v3.1/monthly',outloc=os.path.join(inps,'ccmpv3.1'),start_yr=start_yr,end_yr=end_yr,log=log,lag=lag,v =3.1,geb_file='D:/Data/Bathymetry/GEBCO_2023.nc',var='ws')


        #rean.regrid_fco2_data(socat_file,latg=lag,long=log,save_loc=inps)
        import construct_input_netcdf as cinp
        #Vars should have each entry as [Extra_Name, netcdf_variable_name,data_location,produce_anomaly]
        # vars = [['CCI_SST','analysed_sst',os.path.join(inps,'SST','%Y','%Y%m*.nc'),1]
        # ,['CCI_SST','sea_ice_fraction',os.path.join(inps,'SST','%Y','%Y%m*.nc'),1]
        # ,['CCI_SST','analysed_sst_uncertainty',os.path.join(inps,'SST','%Y','%Y%m*.nc'),0]
        # ,['NOAA_ERSL','xCO2',os.path.join(inps,'xco2atm','%Y','%Y_%m*.nc'),1]
        # ,['ERA5','blh',os.path.join(inps,'blh','%Y','%Y_%m*.nc'),0]
        # ,['ERA5','d2m',os.path.join(inps,'d2m','%Y','%Y_%m*.nc'),0]
        # ,['ERA5','msdwlwrf',os.path.join(inps,'msdwlwrf','%Y','%Y_%m*.nc'),0]
        # ,['ERA5','msdwswrf',os.path.join(inps,'msdwswrf','%Y','%Y_%m*.nc'),0]
        # ,['ERA5','msl',os.path.join(inps,'msl','%Y','%Y_%m*.nc'),0]
        # ,['ERA5','t2m',os.path.join(inps,'t2m','%Y','%Y_%m*.nc'),0]
        # ,['ERA5','si10',os.path.join(inps,'si10','%Y','%Y_%m*.nc'),0]
        # ,['CMEMS','so',os.path.join(inps,'SSS','%Y','%Y_%m*.nc'),1]
        # ,['CMEMS','mlotst',os.path.join(inps,'MLD','%Y','%Y_%m*.nc'),1]
        # ,['CCMP','ws',os.path.join(inps,'ccmpv3.1','%Y','CCMP_3.1_ws_%Y%m*.nc'),0]
        # ,['CCMP','ws^2',os.path.join(inps,'ccmpv3.1','%Y','CCMP_3.1_ws_%Y%m*.nc'),0]
        # ,['OSISAF','total_standard_uncertainty',os.path.join(inps,'OSISAF','%Y','%Y%m_*_COM.nc'),0]
        # ,['OSISAF','ice_conc',os.path.join(inps,'OSISAF','%Y','%Y%m_*_COM.nc'),0]
        # ,['Longhurst','longhurst','D:/Data/Longhurst/Longhurst_1_deg_old.nc',0]
        # ]
        # cinp.driver(data_file,vars,start_yr = start_yr,end_yr = end_yr,lon = log,lat = lag)
        # import run_reanalysis as rean
        # socat_file = 'D:/Data/_DataSets/SOCAT/v2024/SOCATv2024_ESACCIv3.nc'
        # rean.load_prereanalysed(socat_file,data_file,start_yr = start_yr,end_yr=end_yr,name = 'CCI_SST')
        # #
        # #
        # cinp.fill_with_var(model_save_loc,'CCMP_ws','ERA5_si10',log=log,lag=lag)
        # cinp.fill_with_var(model_save_loc,'CCMP_ws^2','ERA5_si10',log=log,lag=lag,mod ='power2')
        # cinp.land_clear(model_save_loc)

        cinp.convert_prov(model_save_loc,'Longhurst_longhurst',13.0,9.0)
        cinp.convert_prov(model_save_loc,'Longhurst_longhurst',25.0,16.0)
        cinp.convert_prov(model_save_loc,'Longhurst_longhurst',27.0,22.0)
        cinp.convert_prov(model_save_loc,'Longhurst_longhurst',28.0,22.0)
        cinp.convert_prov(model_save_loc,'Longhurst_longhurst',47.0,33.0)

    if run_neural:
        import neural_network_train as nnt
        nnt.driver(data_file,fco2_sst = 'CCI_SST', prov = 'Longhurst_longhurst',var = ['CCI_SST_analysed_sst','NOAA_ERSL_xCO2','CMEMS_so','CMEMS_mlotst'],
           model_save_loc = model_save_loc +'/',unc =[0.3,1,0.2,0.05],bath = 'GEBCO_elevation',bath_cutoff = None,fco2_cutoff_low = 50,fco2_cutoff_high = 750,sea_ice = None,
           tot_lut_val=3000,socat_sst=True)
        nnt.plot_total_validation_unc(fco2_sst = 'CCI_SST',model_save_loc = model_save_loc,ice = None,prov='Longhurst_longhurst')
        nnt.plot_mapped(model_save_loc)
    if run_flux:
        import fluxengine_driver as fl
        print('Running flux calculations....')
        fl.fluxengine_netcdf_create(model_save_loc,input_file = data_file,tsub='CCI_SST_analysed_sst',ws = 'CCMP_ws_ERA5_si10',ws2 = 'CCMP_ws^2_ERA5_si10',seaice = 'OSISAF_ice_conc',
             sal='CMEMS_so',msl = 'ERA5_msl',xCO2 = 'NOAA_ERSL_xCO2',start_yr=start_yr,end_yr=end_yr, coare_out = os.path.join(inps,'coare'), tair = 'ERA5_t2m', dewair = 'ERA5_d2m',
             rs = 'ERA5_msdwswrf', rl = 'ERA5_msdwlwrf', zi = 'ERA5_blh',coolskin = 'COARE3.5')
        fl.fluxengine_run(model_save_loc,fluxengine_config,start_yr,end_yr)
        fl.flux_uncertainty_calc(model_save_loc,start_yr = start_yr,end_yr=end_yr,fco2_tot_unc = -1,k_perunc=0.2,unc_input_file=data_file,sst_unc='CCI_SST_analysed_sst_uncertainty',wind_unc=0.9)
        fl.calc_annual_flux(model_save_loc,lon=log,lat=lag,start_yr=start_yr,end_yr=end_yr)
        # fl.fixed_uncertainty_append(model_save_loc,lon=log,lat=lag)
        #fl.variogram_evaluation(model_save_loc,output_file='sst_decorrelation',input_array='CCI_SST_analysed_sst_uncertainty',input_datafile=data_file,start_yr = start_yr,end_yr=end_yr)
        #fl.variogram_evaluation(model_save_loc,output_file='ice_decorrelation',input_array='OSISAF_total_standard_uncertainty',hemisphere=True,input_datafile=data_file,start_yr = start_yr,end_yr=end_yr)
        # fl.variogram_evaluation(model_save_loc,output_file='fco2_net_decorrelation',input_datafile =os.path.join(model_save_loc,'output.nc'),input_array='fco2_net_unc',start_yr = start_yr,end_yr=end_yr)
        # fl.variogram_evaluation(model_save_loc,output_file='fco2_decorrelation',start_yr = start_yr,end_yr=end_yr)
        # fl.variogram_evaluation(model_save_loc,output_file='wind_decorrelation',input_array=['CCMP_ws','ERA5_si10'],input_datafile=[data_file,data_file],start_yr = start_yr,end_yr=end_yr)

        # fl.montecarlo_flux_testing(model_save_loc,decor='fco2_net_decorrelation.csv',flux_var = 'flux_unc_fco2sw_net',start_yr = start_yr,end_yr=end_yr)
        # fl.montecarlo_flux_testing(model_save_loc,decor='fco2_decorrelation.csv',flux_var = 'flux_unc_fco2sw_para',start_yr = start_yr,end_yr=end_yr)
        # fl.montecarlo_flux_testing(model_save_loc,decor='fco2_decorrelation.csv',flux_var = 'flux_unc_fco2sw_val',start_yr = start_yr,end_yr=end_yr)
        # fl.montecarlo_flux_testing(model_save_loc,decor='ice_decorrelation.csv',seaice=True,seaice_var='OSISAF_total_standard_uncertainty',start_yr = start_yr,end_yr=end_yr)
        # fl.montecarlo_flux_testing(model_save_loc,decor='sst_decorrelation.csv',flux_var = 'flux_unc_ph2o',start_yr = start_yr,end_yr=end_yr)
        # fl.montecarlo_flux_testing(model_save_loc,decor='sst_decorrelation.csv',flux_var = 'flux_unc_schmidt',start_yr = start_yr,end_yr=end_yr)
        # fl.montecarlo_flux_testing(model_save_loc,decor='sst_decorrelation.csv',flux_var = 'flux_unc_solskin_unc',start_yr = start_yr,end_yr=end_yr)
        # fl.montecarlo_flux_testing(model_save_loc,decor='sst_decorrelation.csv',flux_var = 'flux_unc_solsubskin_unc',start_yr = start_yr,end_yr=end_yr)
        # fl.montecarlo_flux_testing(model_save_loc,decor='wind_decorrelation.csv',flux_var = 'flux_unc_wind',start_yr = start_yr,end_yr=end_yr)
        # fl.montecarlo_flux_testing(model_save_loc,decor=[2000,1500],flux_var = 'flux_unc_xco2atm',start_yr = start_yr,end_yr=end_yr)
        # fl.plot_relative_contribution(model_save_loc,model_plot='C:/Users/df391/OneDrive - University of Exeter/Post_Doc_ESA_Contract/Watsonetal2023.csv',model_plot_label='UoEx-Watson')
