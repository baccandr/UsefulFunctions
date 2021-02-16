# -*- coding: utf-8 -*-
"""
Created on Wed Feb  3 10:52:15 2021

@author: baccarini_a
"""
import pandas as pd
import numpy as np
from datetime import datetime
import re

import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import matplotlib.dates as date

#%% Load Data Function

def LoadSTAP(fname):
    STAP_df = pd.read_csv(fname, skiprows=29, sep = '\t')    
    # Set datetime index
    dateindex = '20'+STAP_df.iloc[:,0]+' '+STAP_df.iloc[:,1]
    STAP_df.index = pd.to_datetime(dateindex)
    STAP_df = STAP_df.iloc[:,2:]
    STAP_df[['invmm_r','invmm_g','invmm_b']] = STAP_df[['invmm_r','invmm_g','invmm_b']].apply(pd.to_numeric,errors='coerce')
    
    return(STAP_df)

def LoadPOPS(fname):
    POPS = pd.read_csv(fname)
    POPS.index =  pd.to_datetime(POPS['DateTime'],unit='s')
    POPS.index = POPS.index.round('s')
    
    # Create PSD
    logdelta = (POPS[' logmax'][0]-POPS[' logmin'][0])/POPS[' nbins'][0]
    bins = np.array([115, 125, 135, 150, 165, 185, 210, 250, 350, 475, 575, 855, 1220, 1530, 1990, 2585])
    POPS_PSD = POPS.iloc[:,-17:-1].div(POPS[' POPS_Flow']*logdelta,axis=0)
    POPS_PSD.columns = bins
    
    return(POPS, POPS_PSD)

def LoadOzone(fname):
    Ozone = pd.read_csv(fname)
    Ozone.iloc[:,0] = pd.to_numeric(Ozone.iloc[:,0], errors = 'coerce')
    Ozone.index = pd.to_datetime(Ozone.iloc[:,4]+' '+Ozone.iloc[:,5], dayfirst = True)
    
    # remove date columns
    Ozone_sel = Ozone.iloc[:,:4]
    Ozone_sel.columns = [['Ozone [ppb]', 'Temp [C]','Pres [hPa]', 'Flow [ml/min]']]
    return(Ozone_sel)

def LoadMCPC(fname):
    MCPC_df = pd.read_csv(fname, skiprows=13, sep = '\t')    
    # Set datetime index
    dateindex = '20'+MCPC_df.iloc[:,0]+' '+MCPC_df.iloc[:,1]
    MCPC_df.index = pd.to_datetime(dateindex)
    MCPC_df = MCPC_df.iloc[:,2:]
    return(MCPC_df)

def LoadOnboardPC(fname, start_idx, TimeDel, TimeFreq='1480ms'):
    PC = pd.read_csv(fname)
    PC.iloc[:,:] = PC[['Temperature', ' RH', ' CO2', ' Pressure', ' Altitude', ' Temp_box']]\
            .apply(pd.to_numeric, errors = 'coerce')
            
    PC.index = pd.date_range(start = start_idx + TimeDel, periods = PC.shape[0], freq=TimeFreq)
    return(PC)

def LoadOnboardPC02(fname):
    '''
    This function is for the new onboard PC data format
    '''
    s = str(fname)
    date = re.search('Sembrancher_Jan2020(.*)onboard_computer', s).group(1)[1:-1]
    df = pd.read_csv(fname,sep=';',engine='python')
    # create datetimeindex
    time = df[' UTC time'].apply(pd.to_numeric,errors='coerce').astype(int).astype(str)
    # remove wrong entries
    df = df[(time.str.len()>4)&(time.str.len()<7)]
    time = time[(time.str.len()>4)&(time.str.len()<7)]
    
    idx = pd.to_datetime(date+' '+time, format='%Y_%m_%d %H%M%S')
    df.index = idx
    # remove jumps in time
    df2 = df[df[' UTC time'].diff().fillna(0) <= 1]
    
    # adjust numeric values
    numcols = [2, 4, 6, 7, 8, 9, 11, 14, 15, 16, 17, 18, 19]
    df2.iloc[:,numcols] = df2.iloc[:,numcols].apply(pd.to_numeric,errors='coerce')
    df_rs = df2.resample('1s').interpolate(method='time', limit = 60)
    
    return(df_rs)

def LoadMSEMS(fname):
    MSEMS_df = pd.read_csv(fname, skiprows=56, sep = '\t')    
    # Set datetime index
    dateindex = '20'+MSEMS_df.iloc[:,0]+' '+MSEMS_df.iloc[:,1]
    MSEMS_df.index = pd.to_datetime(dateindex)
    
    # Get diameter bins
    Dp = MSEMS_df.loc[:,'bin_dia1':'bin_conc1'].iloc[0,:-1].astype(float)
    MSEMS_PSD = MSEMS_df.loc[:,'bin_conc1':].iloc[:,:-1]
    MSEMS_PSD.columns = Dp
    MSEMS_PSD = MSEMS_PSD.astype(float,errors='ignore')
    MSEMS_PSD.index = MSEMS_df.index
    return(MSEMS_PSD, MSEMS_df)
#%% Plotting Functions

#%% General atmospheric functions 
def Air_density(T,P,RH=0):
    ''' This function returns air density based on ideal gas law.
    Input:
        T in Kelvin
        P in hPa
        RH in %
    Output:
        dry air density in kg/m3
        wet air density in kg/m3'''
    rho0=1.29
    P0=1013.25
    T0=273.15
    rho=rho0*(P/P0)*(T0/T)
    rhow=rho*(1-0.378*waterpressure(RH,T,P)/P)
    return rho,rhow

def EstimateAltitude(P0,Pb, T0):
    Rho0 = Air_density(T0,P0,RH=50)[1]
    g = 9.8
    H = 100*P0/(Rho0*g)
    Elevation = -H*np.log(Pb/P0)
    return(Elevation)

def Watersatpress(press,temp):
    """This function calculates water saturation vapir pressure for moist air.
    The equation is based on WMO CIMO guide and should be valid from -45 to 60C
    ref link: https://www.wmo.int/pages/prog/www/IMOP/CIMO-Guide.html.
    input: 
        P in hPa
        T in kelvin
    output:
        H2O saturation vapour pressure in hPa"""
    
    temp=temp-273.16 #conversion to centigrade temperature
    ew=6.112*np.exp(17.62*temp/(243.12+temp)) #calculate saturation pressure for pure water vapour
    f=1.0016+3.15*10**(-6)*press-0.0074/press
    
    WsatP=ew*f
    return WsatP

def absolutehum(RH,press,temp):
    """ Calculate absolute humidity.
    Input:
        RH
        press in hPa
        T in kelvin
    output:
        Rhov in kg/m3"""
        
    RH=np.float(RH)
    Rv=8.314/18
    ew=Watersatpress(press,temp)*100 #convert to pascal
    rhov=ew*RH/100*1/(Rv*temp)
    return rhov

def watercontent(RH,press,temp):
    """ Calculate absolute water content:
    Input:
        RH
        press in hPa
        T in kelvin
    output:
        water content in molecules/cc"""
        
    #RH=np.float(RH)
    R=8.314 #gas constant
    Na=6.022e23 #avogadro number
    
    ew=Watersatpress(press,temp)*100 #convert to pascal
    Wcont=RH/100*ew/(R*temp)*Na/1000000
    return Wcont

def waterpressure(RH,T,P):
    ''' This function return water vapour pressure at given P and T.
    Input:
        RH in %
        T in Kelvin
        P in hPa
    Output:
        Partial pressure of water vapour in hPa'''
    Pw=Watersatpress(P,T)*RH/100.
    return Pw