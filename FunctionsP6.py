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
#%% Regression functions (linear and multi-linear)
#=========================================================

#%% Linear regression
def lin_reg(varx,vary):
    regr = linear_model.LinearRegression()
    
    varx = np.array(varx).reshape(-1,1)
    vary = np.array(vary).reshape(-1,1)

    regr.fit(varx, vary)

    vary_pred = regr.predict(varx)

    mse = mean_squared_error(vary, vary_pred)
    r2 = r2_score(vary, vary_pred)
    slope = regr.coef_[0,0]
    intercept = regr.intercept_

    return vary_pred, mse, r2, slope, intercept



#%% Multi-linear regression
def lin_reg_multi(varx, vary):
    regr = linear_model.LinearRegression()

    regr.fit(varx, vary)
    
    vary_pred = regr.predict(varx)

    mse = mean_squared_error(vary, vary_pred)
    r2 = r2_score(vary, vary_pred)
    slope = regr.coef_
    intercept = regr.intercept_
    
    return vary_pred, mse, r2, slope, intercept


#=========================================================
#%% Model selection
#=========================================================
def check_mods(ds, varx1, varx2, vary):
    '''
    Find models for which both zos and gsat are available
    '''
    x1mods = ds[varx1].dropna('model','all').model.values
    x2mods = ds[varx2].dropna('model','all').model.values
    ymods = ds[vary].dropna('model','all').model.values
    
    model_list = np.sort(list(set(x1mods)&set(x2mods)&set(ymods)))
    
    ds_new = ds.sel(model=model_list)
    
    return model_list, ds_new


#=========================================================
#%% Name change functions
#=========================================================

#%% Scenario name edit
def edit_scen_name(sce,correction):
    '''
    Convert name from rcp85 to RCP8.5 and ssp585 to SSP5-8.5
    '''
    
    if correction == 'No':
        if 'rcp' in sce:
            sce_out = f'{sce[0:3].upper()}{sce[3]}.{sce[4]}'
        elif 'ssp' in sce:
            sce_out = f'{sce[0:3].upper()}{sce[3]}-{sce[4]}.{sce[5]}'
        else:
            print('Scenario name not recognised')
    
    elif correction == 'Yes':
        if 'rcp' in sce:
            sce_out = f'{sce[0:3].upper()}{sce[3]}.{sce[4]}-C'
        elif 'ssp' in sce:
            sce_out = f'{sce[0:3].upper()}{sce[3]}-{sce[4]}.{sce[5]}-C'
        else:
            print('Scenario name not recognised')
    
    else:
        print('Correction is not recognised')
    
    return sce_out




#=========================================================
#%% Function to plot individual models DSL and fits
#=========================================================
def checks_2(ds, CMIP5_params, CMIP6_params, mip, sce, av_mods):
    
    if sce == 'ssp126' or sce =='rcp26':
        sce_index = 0
    elif sce == 'ssp245' or sce =='rcp45':
        sce_index = 1
    elif sce == 'ssp585' or sce =='rcp85':
        sce_index = 2
    
    if mip == 'cmip5':
        paramsets = CMIP5_params
    elif mip == 'cmip6':
        paramsets = CMIP6_params
        
        
    # Compute fit results
    params = paramsets[sce_index]
    ds_new = ds.sel(model=params.index, time=slice(1950,2100))

    rows = np.round(av_mods/4+.499)
    
    fig, ax = plt.subplots(int(rows),4,figsize=(24,20),sharex=True,sharey=True)
    fig.suptitle(f'{mip} - Scenario: {sce}', fontsize=16)

    fig.subplots_adjust(hspace = 0.5, wspace=0.2)

    ax = ax.ravel()
    i = 0
    for mods in params.index:
    
        mod_params = params.loc[mods]
        DSL_fit = mod_params['alpha'] + mod_params['beta']*ds_new.GSAT.sel(model=mods)+ mod_params['gamma']*ds_new.zostoga.sel(model=mods)
        r2 = mod_params['r2-score']
        mse = mod_params['mse']
        
        ax[i].plot(ds_new.time.values, ds_new.zos_KNMI14.sel(model=mods).values, label='Data')
        ax[i].plot(ds_new.time.values, DSL_fit, c='r',label='Regression Model')
        ax[i].tick_params(axis='x', labelsize=15)
        ax[i].tick_params(axis='y', labelsize=15)
        ax[i].set_xlabel('Yr', fontsize=15)
        ax[i].set_ylabel('DSL', fontsize = 15)
        ax[i].set_ylim([-10,40])
        ax[i].set_title(f'Model: {mods} \n R2-score: {r2:.2f}, RMS: {mse:.2f} cm', fontsize=16)
        ax[i].grid(True, alpha=0.3)
        
        if i == 0:
            ax[i].legend()
            
        i = i+1
    #fig.savefig(savepath+f'{mip}_{sce}',dpi=250)


#=========================================================
#%% Functions that compute fit to percentile scores to compute distributions for zostoga and GSAT from AR6 ranges
#=========================================================

def func(x, a, scale):
    func_cdf = stats.lognorm
    iloc = -10

    return func_cdf.cdf(x, a, iloc, scale)

def range_to_dist(df, var, n):
    '''
    INPUT:
        - df: dataframe as above
        - var: either 'GSAT' or 'zostoga'
        - pc: percentiles: ea [5, 50, 95]
        - n: number of samples you want to generate
    OUTPUT:
        - random distribution with n values
    '''
    distrs = []
    func_cdf = stats.lognorm

    for i in range(2020,2101):
        yr = i
        
        if var == 'GSAT':
            data_sel = df.loc[yr][0:3]
            pc = np.array([5,50,95])*0.01
            
        elif var == 'zostoga':
            data_sel = df.loc[yr][3:6]
            pc = np.array([17,50,83])*0.01

        iloc = -10
        popt, pcov = curve_fit(func, data_sel, pc)
        err = np.square(func(data_sel.astype(float), popt[0], popt[1]) - pc).mean()
    
        # Calculate mean and std for probability function
        mean = func_cdf.mean(popt[0], iloc, popt[1])
        std = func_cdf.std(popt[0], iloc, popt[1])
        
        # Translate to mu and sigma
        mu = np.log(mean**2/np.sqrt(mean**2+std**2))
        sigma = np.sqrt(np.log(1+(std**2/mean**2)))
        
        # Sample n values from distribution and store in array
        distr = np.random.lognormal(mu, sigma, n)
        distr = np.sort(distr)
        
        # Save distribution to array
        distrs.append(distr)
        
    return distrs


#=========================================================
#%% Functions that computes DSL projection using Method 2
#=========================================================

def compute_DSLs(mip, sce, CMIP5_params, CMIP6_params, samples, G_distrs, zt_distrs, dep):
    
    if sce == 'ssp126' or sce =='rcp26':
        sce_index = 0
    elif sce == 'ssp245' or sce =='rcp45':
        sce_index = 1
    elif sce == 'ssp585' or sce =='rcp85':
        sce_index = 2
    
    if mip == 'cmip5':
        paramsets = CMIP5_params[sce_index]
    elif mip == 'cmip6':
        paramsets = CMIP6_params[sce_index]
        
    # Create arrays in which we store random_params and 
    random_params = np.zeros([samples,5])
    DSL = np.zeros(81)
    DSLs = np.zeros([samples,81])
    
    # Loop over the number of samples you want to take
    for j in range(0, samples):
        # Select a random set of alpha, beta, gamma, linear dependence: zostoga(GSAT) = a1 + a0 * GSAT
        random = paramsets.sample()
        random_params[j] = random['alpha'], random['beta'], random['gamma'], random['a0'], random['a1']

        dist_GS = np.random.choice(np.arange(0,len(G_distrs[0])))

        if dep == 'complete':
            dist_zt = dist_GS

            for i in range(0,81):
                GSAT_val, zt_val = G_distrs[i][dist_GS], zt_distrs[i][dist_zt]
                DSL[i] = random_params[j][0] + GSAT_val * random_params[j][1] + zt_val * random_params[j][2]
            
        elif dep == 'none':
            dist_zt = np.random.choice(np.arange(0,len(zt_distrs[0])))

            for i in range(0,81):
                GSAT_val, zt_val = G_distrs[i][dist_GS], zt_distrs[i][dist_zt]
                DSL[i] = random_params[j][0] + GSAT_val * random_params[j][1] + zt_val * random_params[j][2]
            
        elif dep == 'linear':
            for i in range(0,81):
                GSAT_val = G_distrs[i][dist_GS]
                zt_val = random_params[j][4] + random_params[j][3] * GSAT_val
                DSL[i] = random_params[j][0] + GSAT_val * random_params[j][1] + zt_val * random_params[j][2]
        # Loop over all time steps using these parameter sets and GSAT-zostoga paths, start in 2020
        DSLs[j] = DSL
    
    #print(np.median(zt_vals))
    
    return DSLs

# Function to plot median and 5-95% range from method 2
def plot_Method2(CMIP5_DSLs, CMIP6_DSLs, dep, savepath, savenameCMIP5, savenameCMIP6):
    
    sces = ['*2.6', '*4.5', '*8.5']
    k = ['yellow','orange', 'red']
    yrs = np.linspace(2020,2101,81)

    fig, ax = plt.subplots(figsize=(10,6),dpi=100)
    for i in range(3):
        ax.plot(yrs, np.median(CMIP5_DSLs[i],axis=0),c=k[i],label=f'CMIP5 Sce {sces[i]}')
        ax.fill_between(yrs, np.median(CMIP5_DSLs[i],axis=0) + 1.64*CMIP5_DSLs[i].std(axis=0),\
                        CMIP5_DSLs[i].mean(axis=0)-1.64*CMIP5_DSLs[i].std(axis=0),alpha=0.1, color=k[i],label=f'CMIP5 Sce {sces[i]}, 5-95%')
    
    ax.set_title(f'CMIP5 - Method 2 - {dep} GSAT&zostoga')
    ax.set_xlabel('Time (yr)')
    ax.set_ylabel('DSL anomaly (cm)')
    ax.set_xlim([2020,2100])
    ax.set_ylim([-5,50])
    plt.legend()
    ax.grid(True)
    plt.savefig(savepath+f'/{savenameCMIP5}',dpi=150)

    fig, ax = plt.subplots(figsize=(10,6),dpi=100)
    for i in range(3):
        ax.plot(yrs, np.median(CMIP6_DSLs[i],axis=0),c=k[i],label=f'CMIP6 Sce {sces[i]}')
        ax.fill_between(yrs, np.median(CMIP6_DSLs[i],axis=0) + 1.64*CMIP6_DSLs[i].std(axis=0),\
                        np.median(CMIP5_DSLs[i],axis=0)-1.64*CMIP5_DSLs[i].std(axis=0),alpha=0.1, color=k[i],label=f'CMIP6 Sce {sces[i]}, 5-95%')
    
    ax.set_title(f'CMIP6 - Method 2 - {dep} GSAT&zostoga')
    ax.set_xlabel('Time (yr)')
    ax.set_ylabel('DSL anomaly (cm)')
    ax.set_xlim([2020,2100])
    ax.set_ylim([-5,50])
    plt.legend()
    ax.grid(True)
    plt.savefig(savepath+f'/{savenameCMIP6}',dpi=150)
    


def plot_Method2_compare_onlymedian(ds5, ds6, CMIP5_DSLs, CMIP6_DSLs, dep):
    
    sces = ['*2.6', '*4.5', '*8.5']
    k = ['yellow','orange', 'red']
    yrs = np.linspace(2020,2101,81)
    
    for i in range(3):
        
        ds_new5 = check_mods(ds5[i], 'GSAT', 'zostoga', 'zos_KNMI14')
        ds_new6 = check_mods(ds6[i], 'GSAT', 'zostoga', 'zos_KNMI14')

        ds_new5 = ds_new5.sel(time=slice(2020,2101))
        ds_new6 = ds_new6.sel(time=slice(2020,2101))

        fig, ax = plt.subplots(figsize=(10,6))
  
        ax.plot(yrs, np.median(CMIP5_DSLs[i],axis=0),color='blue',label=f'CMIP5 Method Sce {sces[i]}')
        ax.plot(yrs, np.median(CMIP6_DSLs[i],axis=0),c='red',label=f'CMIP6 Method Sce {sces[i]}')
        
        ax.plot(yrs, ds_new5.zos_KNMI14.median(dim='model').values, color='blue',ls='dashed',label=f'CMIP5 models Sce {sces[i]}')
        ax.plot(yrs, ds_new6.zos_KNMI14.median(dim='model').values, color='red',ls='dashed',label=f'CMIP6 models Sce {sces[i]}')

        ax.set_title(f'Results scenario {sces[i]} - {dep}')
        ax.set_xlabel('Time (yr)')
        ax.set_ylabel('DSL anomaly (cm)')
        ax.grid(True)
        plt.legend()
        ax.set_xlim([2020,2100])
        ax.set_ylim([-10,30])


def plot_Method2_compare(ds5, ds6, CMIP5_DSLs, CMIP6_DSLs, dep):
    
    sces = ['*2.6', '*4.5', '*8.5']
    k = ['yellow','orange', 'red']
    yrs = np.linspace(2020,2101,81)
    
    for i in range(3):
        
        ds_new5 = check_mods(ds5[i], 'GSAT', 'zostoga', 'zos_KNMI14')
        ds_new6 = check_mods(ds6[i], 'GSAT', 'zostoga', 'zos_KNMI14')

        ds_new5 = ds_new5.sel(time=slice(2020,2101))
        ds_new6 = ds_new6.sel(time=slice(2020,2101))

        # CMIP5
        fig, ax = plt.subplots(figsize=(10,6))
  
        ax.plot(yrs, np.median(CMIP5_DSLs[i],axis=0),color='blue',label=f'CMIP5 Method Sce {sces[i]}')
        ax.fill_between(yrs, np.median(CMIP5_DSLs[i],axis=0) + 1.64*CMIP5_DSLs[i].std(axis=0),\
                        CMIP5_DSLs[i].mean(axis=0)-1.64*CMIP5_DSLs[i].std(axis=0),alpha=0.3, color='blue',label=f'CMIP5 Method Sce {sces[i]}, 5-95%')
    
        
        ax.plot(yrs, ds_new5.zos_KNMI14.median(dim='model').values, color='k',ls='dashed',label=f'CMIP5 models Sce {sces[i]}')
        ax.fill_between(yrs, ds_new5.zos_KNMI14.median(dim='model').values + 1.64*ds_new5.zos_KNMI14.std(dim='model').values,\
                        ds_new5.zos_KNMI14.median(dim='model').values-1.64*ds_new5.zos_KNMI14.std(dim='model').values,alpha=0.3, color='grey',label=f'CMIP5 models Sce {sces[i]}, 5-95%')
        
        ax.set_title(f'CMIP5 - Results scenario {sces[i]} - {dep}')
        ax.set_xlabel('Time (yr)')
        ax.set_ylabel('DSL anomaly (cm)')
        ax.grid(True)
        plt.legend()
        ax.set_xlim([2020,2100])
        ax.set_ylim([-10,30])
        
        
        # CMIP6
        fig, ax = plt.subplots(figsize=(10,6))
        
        ax.plot(yrs, np.median(CMIP6_DSLs[i],axis=0),c='red',label=f'CMIP6 Method Sce {sces[i]}')
        ax.fill_between(yrs, np.median(CMIP6_DSLs[i],axis=0) + 1.64*CMIP6_DSLs[i].std(axis=0),\
                        CMIP6_DSLs[i].mean(axis=0)-1.64*CMIP6_DSLs[i].std(axis=0),alpha=0.3, color='red',label=f'CMIP6 Method Sce {sces[i]}, 5-95%')
    
        ax.plot(yrs, ds_new6.zos_KNMI14.median(dim='model').values, color='k',ls='dashed',label=f'CMIP6 models Sce {sces[i]}')
        ax.fill_between(yrs, ds_new6.zos_KNMI14.median(dim='model').values + 1.64*ds_new6.zos_KNMI14.std(dim='model').values,\
                        ds_new6.zos_KNMI14.median(dim='model').values-1.64*ds_new6.zos_KNMI14.std(dim='model').values,alpha=0.3, color='grey',label=f'CMIP6 models Sce {sces[i]}, 5-95%')
    
        ax.set_title(f'CMIP6 - Results scenario {sces[i]} - {dep}')
        ax.set_xlabel('Time (yr)')
        ax.set_ylabel('DSL anomaly (cm)')
        ax.grid(True)
        plt.legend()
        ax.set_xlim([2020,2100])
        ax.set_ylim([-10,30])


