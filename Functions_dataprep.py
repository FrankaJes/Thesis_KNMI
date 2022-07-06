"""
This file contains functions used in notebooks for calculating relations between:
    dynamic sea level (zos)
    steric sea level (zostoga) 
    global mean surface air temperature (gsat) 
    ocean temperature (tos)

@author: Franka Jesse
"""

import sys
import os
import glob

import numpy as np
import xarray as xr
import pandas as pd
import matplotlib.pyplot as plt

from scipy import interpolate
from scipy.interpolate import InterpolatedUnivariateSpline
from scipy import stats 
from scipy.optimize import curve_fit 

import sklearn
from sklearn import linear_model
from sklearn.linear_model import LinearRegression
from sklearn import metrics
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score



#=========================================================
#%% Functions to read data GSAT, zos, zostoga
#=========================================================

#%% needed to read CMIP6
def select_tglob_cmip6_files(data_dir, sce, verbose=False):
    '''The file name of cmip6 tglob files from the climate explorer contains 
    some information about the variant as well. Only one needs to be chosen 
    and the name of the model needs to be cleaned.'''
    
    path = f'{data_dir}global_tas_mon_*_{sce}_000.dat'
    files  = glob.glob(path)

    model_names = []
    for f in files:
        file_name_no_path = f.split('/')[-1]
        model_name = file_name_no_path.split('_')[3]
        model_names.append(model_name)

    model_names.sort()

    clean_model_names = []
    lp_list = []
    for m in model_names:
        lp = m.split('-')[-1]
        lp_list.append(lp)
        if lp in ['f2', 'p2', 'p1', 'p3', 'f3']:
            clean_model_names.append(m[:-3])
        else:
            clean_model_names.append(m)

    df = pd.DataFrame({'model_names': model_names,
                       'clean_model_names': clean_model_names,
                       'fp': lp_list})

    for m in df['clean_model_names']:
        if (list(df['clean_model_names']).count(m) > 1):
            fp_choice = df[df.clean_model_names==m]['fp']
            ind = df[(df.fp!=fp_choice.iloc[0]) & (df.clean_model_names==m)].index
            
            if verbose:
                print(f'Multiple variants of {m}')
                print('Available variants')
                print(fp_choice)
                print(f'Choosing {fp_choice.iloc[0]}')
                
            df.drop(ind, inplace=True)

    file_list = []
    for m in df.model_names:
        file_list.append(f'{data_dir}global_tas_mon_{m}_{sce}_000.dat')
    
    df['file_names'] = file_list
    
    return df


#%% Read GSAT
def tglob_cmip(data_dir, mip, sce, start_date, ye, LowPass=False):
    '''Read the text files of monthly temperature for each CMIP5 model and store
    yearly averged values in an xarray DataArray.
    Output data is in degree Kelvin'''
    
    nb_y = ye-start_date+1
    
    if mip == 'CMIP5':
        temp_data_dir = f'{data_dir}Tglobal_CMIP5/'
        path = f'{temp_data_dir}global_tas_Amon_*_{sce}_r1i1p1.dat'
        files = glob.glob(path)
        
        model_names = [f[52:-17] for f in files]
        df = pd.DataFrame({'clean_model_names': model_names, 'file_names': files})
        
    elif mip =='CMIP6':
        temp_data_dir = f'{data_dir}Tglobal_CMIP6/'
        df = select_tglob_cmip6_files(temp_data_dir, sce)
        
    else:
        print(f'ERROR: Value of TEMP: {mip} not recognized')

    
    col_names = ['Year', 'Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug',
                 'Sep', 'Oct', 'Nov', 'Dec']

    for i in range(len(df)):
        TEMP = pd.read_csv(df.file_names.iloc[i], 
                           comment='#', 
                           delim_whitespace=True,
                           names=col_names)
        TEMP = TEMP.set_index('Year')
        TGLOBi = xr.DataArray(TEMP.mean(axis=1))
        mod = df.clean_model_names.iloc[i]
        TGLOBi = TGLOBi.expand_dims({'model':[mod]})

        if i==0:
            TGLOB = TGLOBi
        else:
            TGLOB = xr.concat([TGLOB, TGLOBi], dim='model')

    TGLOB = TGLOB.rename({'Year':'time'})
    TGLOB = TGLOB.sel(time=slice(start_date,ye))

    if LowPass:
        new_time = xr.DataArray( np.arange(start_date,ye+1), dims='time', 
                coords=[np.arange(start_date,ye+1)], name='time' )
        fit_coeff = TGLOB.polyfit('time', 2)
        TGLOB = xr.polyval(coord=new_time, coeffs=fit_coeff.polyfit_coefficients) 

    TGLOB.name = 'GSAT'
    
    return TGLOB


#%% zostoga
def read_zostoga_ds(data_dir, mip, sce):
    '''Read both historical and scenario datasets, select the intersecting 
    models and concatenate the two datasets'''
    
    hist_ds = xr.open_mfdataset(
        f'{data_dir}/{mip}_zostoga/{mip}_zostoga_historical_*.nc')*100
    sce_ds = xr.open_mfdataset(
        f'{data_dir}/{mip}_zostoga/{mip}_zostoga_{sce}_*.nc')*100

    model_intersection = list(set(hist_ds.model.values) & 
                              set(sce_ds.model.values))
    model_intersection.sort()
    tot_ds = xr.concat([hist_ds,sce_ds],'time').sel(model=model_intersection)
    
    return tot_ds

#%% zos
def read_zos_ds(data_dir, mip, sce):
    '''Read both historical and scenario datasets, select the intersecting 
    models and concatenate the two datasets'''
    
    hist_ds = xr.open_mfdataset(
        f'{data_dir}/{mip}_zos_historical/{mip}_zos_historical_*.nc')
    sce_ds = xr.open_mfdataset(
        f'{data_dir}/{mip}_zos_{sce}/{mip}_zos_{sce}_*.nc')

    model_intersection = list(set(hist_ds.model.values) & 
                              set(sce_ds.model.values))
    model_intersection.sort()
    tot_ds = xr.concat([hist_ds,sce_ds],'time').sel(model=model_intersection)

    return tot_ds

#%% AMOC
def read_amoc_ds(data_dir, mip, sce):
    '''
    Read both historical and scenario datasets, select the intersecting 
    models and concatenate the two datasets. 
    '''
    # Conversion to Sv (density of water and factor of 10^6 for Sv) via Devision Factor:
    DF = 1026*10**6 
    
    tot_ds = []
    for var in ['msftmz','msftyz']:
        for reg in ['26N', '35N']:
            hist_ds = xr.open_mfdataset(
                f'{data_dir}/{mip}_amoc/{mip}_{var}_{reg}_historical_*.nc')/DF
            sce_ds = xr.open_mfdataset(
                f'{data_dir}/{mip}_amoc/{mip}_{var}_{reg}_{sce}_*.nc')/DF

            model_intersection = list(set(hist_ds.model.values) & 
                                      set(sce_ds.model.values))
            model_intersection.sort()
            
            amoc_ds = xr.concat([hist_ds,sce_ds],'time').sel(model=model_intersection)
            amoc_ds = amoc_ds.rename({var:f'AMOC_{reg}'})
            tot_ds.append(amoc_ds)
    
    m26N, m35N, y26N, y35N = tot_ds
    
    # concatenate m26N and y26N - without model overlap! so first select right models
    overlap = list(set(m26N.model.values)&set(y26N.model.values))
    
    mods_m = m26N.model.values
    mods_y = np.array([ele for ele in y26N.model.values if ele not in overlap])
    
    print(mods_m)
    print(mods_y)
    AMOC_26N = xr.concat([m26N.sel(model=mods_m), y26N.sel(model=mods_y)],'model')
    AMOC_35N = xr.concat([m35N.sel(model=mods_m), y35N.sel(model=mods_y)],'model')
    
    mods_alph = np.sort(AMOC_26N.model.values)

    AMOC_26N = AMOC_26N.sel(model=mods_alph)
    AMOC_35N = AMOC_35N.sel(model=mods_alph)

    return AMOC_26N, AMOC_35N


def zostoga_ranges(ds_zostoga, percentiles, yr_start, yr_end):
    
    zostogas = []
    for perc in percentiles:
        ds = ds_zostoga.sel(quantiles=perc, years=slice(yr_start,yr_end))
        vals = ds.sea_level_change.values/10
        zostogas.append(vals)
    return zostogas

