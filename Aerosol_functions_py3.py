# -*- coding: utf-8 -*-
"""
Created on Mon Jan 21 10:32:10 2019

@author: baccarini_a
"""
import numpy as np
import pandas as pd
import datetime
from scipy import interpolate
from scipy.optimize import curve_fit
from scipy.interpolate import interp1d
import statsmodels.api as sm

import matplotlib.pyplot as plt
import matplotlib.dates as date
from matplotlib.colors import LogNorm

#%% More basic atmospheric functions
def ppbconversion(ppb,T,P): #temperature in kelvin and pressure in Pascal
    volmol=8.3144*T/P*10**(6) #volume of 1 mol in cc
    molecules=ppb/10**(9)*6.022*10**(23)/volmol
    return(molecules)
    
def Air_viscosity(T):
    ''' This function return air viscosity based on Sutherland equation.
    Input:
        T in Kelvin
    Output:
        viscosity in Pa*s'''
    #viscosity calculation
    eta=1.458e-6*T**1.5/(T+110.4)   
    return eta

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

def meanfreepath(P,T):
    """This function calculates air mean free path at a certain T and P.
    the scaling is based on Willeke's relation. Lambda0 is based on 
    Allen and Raabe.
    Input:
        P in hPa
        T in kelvin
    Output:
        mean free path in nanometer"""
    T0=296.15
    P0=1013.25
    lambd_zero=67.3 #reference mean free path at std conditions
    
    lambd=lambd_zero*(T/T0)*(P0/P)*((1+110.4/T0)/(1+110.4/T))
    return lambd

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

def Tdewpoint(RH,T,P):
    ''' This function calculates dewpoint temperature. It is based on 
    Vaisala suggestion and should hold between -20 and 50 C.
    Input:
        RH in %
        T in Kelvin
        P in hPa
    Output:
        Dewpoint temperature in Kelvin'''
    m=7.591386
    A=6.116441
    Tn=240.7263
    Pw=waterpressure(RH,T,P)
    Td=Tn/(m/np.log10(Pw/A)-1)
    return Td
    
#%% Aerosol functions
    
def CalculateMassPSD(PSD, Dp, rho, limit = [0,1e3]):
    ''' Calculate mass of a given number particle
    size distribution. It is possible to specify
    integration limits if needed.

    Input:
        PSD in dN/dlogDp
        Dp in nm
        rho in g/cm^3
        limit in nm
    Output: 
        Mass concentration in ug/m^3
    '''
    # Convert into a volume distribution
    VolPSD = PSD*np.pi/6*Dp**3
    IntegVol = VolPSD.apply(lambda x: np.trapz(x[(Dp>limit[0])&(Dp<limit[1])],
                                               np.log10(Dp[(Dp>limit[0])&(Dp<limit[1])])),axis=1)
    IntegMass=IntegVol*rho*1e-9 
    return IntegMass

def CalculateMass(Dp, rho):
    ''' Calculate mass of a particle given its diameter
    and density. 
    Input:
        Dp in nm
        rho in g/cm^3
    Output: 
        mass in ng
    '''
    volume = np.pi/6*(Dp)**3
    mass = rho*volume*1e-12
    return mass

def DiffusionLoss(flow,D,L):
    """ This function calculates diffusion losses 
    in a circular tube and returns the transmission.
    Input parameters must have the following units:
    flow-> liters per minute
    D-> cm2/s [diffusion coefficient]
    L-> cm [tube length]
    The code is based on Bowen[1976] and Ingham[1975],
    for a synthesis Kulkarni and Baron in Aerosol
    measurement, third edition, page 366-367."""
    Pleng=np.zeros(len(D))
    F=flow*100./6 #conversion of flow from LPM to cm3/s
    mu=np.pi*D*L/F #deposition parameter
    muindex=mu<0.02 #boolean array to be used as index for Plen
    Pleng[muindex]=1-2.5638*mu[muindex]**(2./3)+1.2*mu[muindex]+0.1767*mu[muindex]**(4./3)
    Pleng[Pleng==0]=0.81905*np.exp(-3.6568*mu[Pleng==0])+0.09753*np.exp(-22.305*mu[Pleng==0])+0.0325*np.exp(-56.961*mu[Pleng==0])+0.01544*np.exp(-107.62*mu[Pleng==0])
        
    return Pleng
	
def DiffusionLossOLD(flow,D,L):
    """
	Old diffusion loss function that works with single values
	but not with array as input.
	This function calculates diffusion losses 
    in a circular tube and returns the transmission.
    Input parameters must have the following units:
    flow-> liters per minute
    D-> cm2/s [diffusion coefficient]
    L-> cm [tube length]
    The code is based on Bowen[1976] and Ingham[1975],
    for a synthesis Kulkarni and Baron in Aerosol
    measurement, third edition, page 366-367."""
    
	
    F=flow*100./6 #conversion of flow from LPM to cm3/s
    
    mu=np.pi*D*L/F #deposition parameter
    
    if mu <0.02:
        Pleng=1-2.5638*mu**(2./3)+1.2*mu+0.1767*mu**(4./3)
    else:
        Pleng=0.81905*np.exp(-3.6568*mu)+0.09753*np.exp(-22.305*mu)+0.0325*np.exp(-56.961*mu)+0.01544*np.exp(-107.62*mu)
        
    return Pleng
        

def SA_DiffCoeff(T,P):
    """This function scales sulfuric acid diffusion coefficient
    to the real sampling temperature and pressure. This script
    is based on Hanson and Eisele 2000. 
    Input:
        P in hPa
        T in Kelvin
    Output:
        D in cm2/s
    Note: This scaling can be used for RH around 40%, in case of 
    significant variations in RH refer to Hanson and Eisele 
    original paper."""
    
    Dref=0.0786
    Patm=1013.25
    Tref=298.
    D=Dref*(Patm/P)*(T/Tref)**1.75
    
    return D

def HIO3_density(nwat):
    '''
    Input: number of water molecules per iodic acid molecule
    Based on the paramterization from Kumar et al 2010.
    '''
    molality=1/(18*nwat*1e-3)
    Fit_Coeff=np.array([3.05,2.03,-2.41e-1,8.12e-3,1.35e-6])*10**(-2)
    rho_HIO3=0.997+np.sum([molality**i for i in range(1,6)]*Fit_Coeff)
    
    return(rho_HIO3)

def SA_density(nwat):
    '''
    Input: number of water molecules per sulfuric acid molecules.
    Sulfuric acid density based on Myhre 1998. This is for 273 K
    and it should be extended to other temperatures.
    '''
    mass_SA=98.079
    massfrac=mass_SA/(mass_SA+nwat*18)
    Fit_Coeff=np.array([999.8426,547.2659,526.9250e1,-621.3958e2,409.0293e3,-159.6989e4,
              385.7411e4,-580.8064e4,530.1976e4,-268.2616e4,576.4288e3])
    rho_SA=np.sum([massfrac**i for i in range(len(Fit_Coeff))]*Fit_Coeff)
    
    return(rho_SA*1e-3)

def Gas_collision(T,Da,Db,Ma,Mb):
    '''
    insert description
    '''
    Kb=1.381e-23
    #reduced mass
    mu=Ma*Mb/(Ma+Mb)
    # collision cross section
    sigma=np.pi/4*(Da+Db)**2
    
    Kcoll=sigma*np.sqrt(8*Kb*T/(np.pi*mu))
    return Kcoll

def Particle_DiffCoeff(T,Cc,Dp):
    """This function calculates the diffusion coefficient for an
    aerosol particle with a diameter Dp. This script is based on
    the Stokes-Einstein equation. Viscosity is scaled based on the 
    Sutherland equation.
    Input:
        T in kelvin
        Cc is the Cunningham slip correction factor
        Dp in nanometer
    Output:
        Diffusion coefficient in cm2/s"""
    #viscosity calculation
    eta=1.458e-6*T**1.5/(T+110.4)
    
    Dp=Dp*10**(-9) #convert particle diameter in meters
    Kb=1.3806e-23
    D=Kb*T*Cc/(3*np.pi*eta*Dp)*10**4
    
    return D

def Condensation_Sink(T,P,alpha,Dpinp,PSD):
    '''This function calculates the Sulfuric acid condensation sink based on
    Dal Maso et al. (2002) original equation. It is designed to work also with
    PSDs that are not evenly spaced in diameter, to get the number of particles
    per size bin as required from the original equation the following operations
    are performed:
        - The midpoint between each diameter bin Dp is calculated in a log-space
          scale: ... Dp_mid(i-1), Dp(i), Dp_mid(i), Dp(i+1), ...
        - The PSD is interpolated over these mid points
        - The PSD is integrated for each Dp bin from Dp_mid(i-1) to Dp_mid(i),
          this returns the number of particles per size bin
        - Dal Maso et al. equation is applied to calculate the condensation sink.
    Input:
        T in Kelvin
        P in hPa
        alpha accomodation coefficient
        Dpinp Dp in nm, it is assumed to be logspaced
        PSD dN/dlogDp, it has to be a pandas df with datetime as index
    Output:
        Condensation sink in s^(-1)'''
    
    # Nan check
    if PSD.isnull().values.any():
        print('PSD contains nan(s) they will be replaced with zeros')
        PSD.fillna(0,inplace=True)
    
    # Calculate variables needed for CS
    Dp=np.array(Dpinp,dtype=float)
    DSA=SA_DiffCoeff(T,P)
    Kn=Knudsen_num(Dp,P,T)
    tran_coeff=(1+Kn)/(1+(4./(3*alpha)+0.337)*Kn+4./(3*alpha)*Kn**2)
    
    # Create mid diameter for integration
    Dplog=np.log10(Dp)
    Dplogmid=np.zeros(len(Dplog)+1)
    Dplogmid[1:-1]=((Dplog[:-1]+Dplog[1:])/2)
    Dplogmid[0]=2*Dplog[0]-Dplogmid[1]
    Dplogmid[-1]=2*Dplog[-1]-Dplogmid[-2]
    Dpfinal=np.sort(np.concatenate((Dplog,Dplogmid)))
    
    # Define interpolating function
    interp_func=interpolate.interp1d(Dplog,PSD,axis=1,fill_value='extrapolate')
    PSD_interp=interp_func(Dpfinal)
    
    # Count the number of particles in each size bin
    Concbins=np.zeros(PSD.shape)
    for j in range(len(Dp)):
        Concbins[:,j]=np.trapz(PSD_interp[:,(2*j):(2*j)+3],Dpfinal[(2*j):(2*j)+3])
        
    # Calculate Condensation sink
    CondSink=pd.Series(2*np.pi*DSA*np.sum(1e-7*Concbins*Dp*tran_coeff,axis=1),index=PSD.index)
    return CondSink

def Knudsen_num(Dp,P,T):
    """This function calculates the Knudsen number at given T and P
    Input:
        Dp particle diameter in nm
        P in hPa
        T in kelvin
    Output:
        Kn"""
    #Air mean free path calculation
    T0=296.15
    P0=1013.25
    lambd_zero=67.3 #reference mean free path at std conditions
            
    lambd=lambd_zero*(T/T0)*(P0/P)*((1+110.4/T0)/(1+110.4/T))
    
    # Check if temperature is a series
    if hasattr(T, "__len__"):
        Kn=2*np.outer(lambd,1/Dp) #Knudsen number
    else:   
        Kn=2*lambd/Dp #Knudsen number
    return Kn
       
    
def Cunningham(Dp,P,T):
    """This function calculates the Cunningham slip correction factor
    based on the parametrization from Kim et al 2005.
    Input:
        Dp particle diameter in nm
        P in hPa
        T in kelvin
    Output:
        Cc"""
    #Air mean free path calculation
    T0=296.15
    P0=1013.25
    lambd_zero=67.3 #reference mean free path at std conditions
    
    lambd=lambd_zero*(T/T0)*(P0/P)*((1+110.4/T0)/(1+110.4/T))
    
    Kn=2*lambd/Dp #Knudsen number
    
    Cc=1+Kn*(1.165+0.483*np.exp(-0.997/Kn))
    
    return Cc

def Reynoldnum(T,P,RH,v,L):
    '''This function calculates the reynold number as the ratio of inertial 
    to viscous forces. For inlet calculation L should be the diameter of the tubegas
    whereas for an aerosol particle this should be the diameter.
    Input:
        T in Kelvin
        P in hPa
        RH in %
        v velocity of the gas in m/s
        L characteristic length scale in m
    Output: Reynold number
    '''
    rho=Air_density(T,P,RH)
    eta=Air_viscosity(T)
    Re=rho[0]*v*L/eta
    return Re
    
def SettlingVelocity(T,P,Dp,rhoP=1000.,chi=1.):
    '''This function calculates the settling velocity of an aerosol particle,
    is defined in the Stokes regime (Re<0.1).
    Input:
        T in Kelvin
        P in hPa
        Dp particle diameter in nanometer
        RhoP particle density in kg/m3
        chi is the shape factor
    Return:
        Vs in m/s
    '''
    eta=Air_viscosity(T)
    Cc=Cunningham(Dp,P,T)
    g=9.807 #gravity acceleration
    Vs=rhoP*Dp**2*g*Cc/(18*eta*chi)*10**(-18) #the final factor is to account for the diameter in nm
    return Vs

def StokesInlet(T,P,Dp,d,U,rhoP=1000.):
    ''' This function calculates the Stokes number of the sampling probe, refer
    to Willeke and Baron 2005 for details.
    Input:
        T in Kelvin
        P in hPa
        Dp particle diameter in nanometer
        d is the inner tube diameter in meter
        U is the flow velocity in m/s
        rhoP is the particle density in kg/m3
    '''
    Cc=Cunningham(Dp,P,T)#Cc takes Dp in nm
    Dp=Dp*10**(-9) #conversion to meters

    eta=Air_viscosity(T)
    Stk=Dp**2*rhoP*Cc*U/(18*eta*d)
    tau=(Dp**2)*rhoP*Cc/(18*eta)
    return (Stk, tau)

def Sedimentation(T,P,Dp,d,L,U,theta=0,rhoP=1000.,chi=1.):
    ''' This function calculates losses by sedimentation for a tube of certain
    length, diameter and inclination. It is based on the experimental work
    of Heyder and Gebhart (1977). Only tested quickly against Particle loss calculator.
    Input:
        T in Kelvin
        P in hPa
        Dp particle diameter in nanometer
        d is the inner tube diameter in centimeter
        L is the tube length in centimeter
        U is the flow velocity in lpm
        theta is the angle of the inlet with respect to the horizontal
        rhoP is the particle density in kg/m3
        chi is the particle shape factor
    Output:
        Transmission efficiency due to sedimentation'''
    U=U/(60*np.pi/4*d**2)*10 #Conversion from LPM to m/s
    L=L*0.01 #convert into meters
    d=d*0.01 #convert into meters
    theta=theta*2*np.pi/360 #convert to rad
    Vs=SettlingVelocity(T,P,Dp,rhoP,chi)
    
    Z=L*Vs/(d*U) #gravitational deposition parameter
    eps=3./4*Z
    k=eps*np.cos(theta)
    
    if np.any(Vs*np.sin(theta)/U>0.1):
        print('flow is too low and the equation may not be valid for your case')
    Trans_sed=1-2./np.pi*(2*k*np.sqrt(1-k**(2./3))-k**(1./3)*np.sqrt(1-k**(2./3))+np.arcsin(k**(1./3)))
    return Trans_sed

def InertialDeposition_bend(T,P,Dp,d,U,theta=0,rhoP=1000.):
    '''This function calculates losses by inertial deposition in a bend with a
    certain angle. It is the same used in the particle loss calculator from
    von der Weiden. However I'm not sure about its accuracy because it doesn't
    take into account the curvature radius but just the angle of the bend.
    NOT FULLY TESTED
    Input:
        T in Kelvin
        P in hPa
        Dp particle diameter in nanometer
        d is the inner tube diameter in centimeter
        U is the flow velocity in lpm
        theta is the angle of the bend in degree
        rhoP is the particle density in kg/m3
    Output:
        Transmission efficiency due to inertial deposition'''
    U=U/(60*np.pi/4*d**2)*10 #Conversion from LPM to m/s
    d=d*0.01 #convert into meters

    Stk=StokesInlet(T,P,Dp,d,U,rhoP)
    theta=theta*2*np.pi/360 #convert to rad
    Trans_Idep=(1+(Stk/0.171)**(0.452*Stk/0.171+2.242))**(-2*theta/np.pi)
    return Trans_Idep

def InertialDeposition_constr(T,P,Dp,di,do,U,theta=90,rhoP=1000.):
    '''This function calculates losses by inertial deposition due to flow constriction.
    It is based on Chen and Pui (1995) and is the same of the particle loss calculator.
    NOT FULLY TESTED
    Input:
        T in Kelvin
        P in hPa
        Dp particle diameter in nanometer
        di is the inner larger tube diameter in centimeter
        do is the inner smaller tube diameter in centimeter
        U is the flow velocity in lpm
        theta is the contraction angle in degree
        rhoP is the particle density in kg/m3
    Output:
        Transmission efficiency due to inertial deposition'''
    U=U/(60*np.pi/4*di**2)*10 #Conversion from LPM to m/s
    di=di*0.01 #convert into meters
    do=do*0.01 #convert into meters
    
    Stk=StokesInlet(T,P,Dp,do,U,rhoP)
    Trans_Idep=1-1/(1+(2*Stk*(1-(do/di)**2)/(3.14*np.exp(-0.0185*theta)))**(-1.24))
    return Trans_Idep

def Unpack_trajNEW(fileN): 
    """ Function to unpack backtrajectory file given the fullpath of the file.
    it returns the dataframe with the trajectory and a datatime object
    corresponding to the release time of the trajectories
    The output dataframe is already formatted with a multindex that include the
    trajectory number and the time associated to the air parcel travelling"""
    Trjtime=pd.to_datetime(fileN[55:59]+'-'+fileN[59:61]+'-'+fileN[61:63]+' '+fileN[64:66]+':00')
    
    Trj = pd.read_csv(fileN, delim_whitespace=True, skiprows=(0,3),skip_blank_lines=True,engine='python')
    
    #Removing nan
    Trj.replace(-999.99,np.nan,inplace=True)
    Trj.replace(-1000,np.nan,inplace=True)
    #Trj.dropna(inplace=True)
    
    #Creating a multiindex to account for the different trajectories
    thrddim=len(np.where(Trj['time']==0)[0]) #number of trajectories
    
    #here I simply create a sequential index (each trajectory has the same integer number)
    indexcol=np.zeros(len(Trj.index))
    for j in range(thrddim):
        indexcol[(81*j):((j+1)*81)]=j
    
    newidx=[indexcol,Trj['time']] #create new double index that will be the new multindex df
    Trj02=Trj.iloc[:,1:] #Remove time column because it will be used as index
    Trj02.index=newidx #create this new multindex
    
    return(Trj02,Trjtime)
    
def Unpack_traj(fileN):
    """ THIS IS THE OLD VERSION THAT RETURNS A BADLY FORMATTED MULTINDEX
    I KEPT IT ONLY FOR RETROCOMPATIBILITY
    Function to unpack backtrajectory file given the fullpath of the file.
    it returns the dataframe with the trajectory and a datatime object
    corresponding to the release time of the trajectories"""
    Trjtime=pd.to_datetime(fileN[32:36]+'-'+fileN[36:38]+'-'+fileN[38:40]+' '+fileN[41:43]+':00')
    
    Trj = pd.read_csv(fileN, delim_whitespace=True, skiprows=(0,3),skip_blank_lines=True,engine='python')
    
    #Removing nan
    Trj.replace(-999.99,np.nan,inplace=True)
    Trj.replace(-1000,np.nan,inplace=True)
    #Trj.dropna(inplace=True)
    
    #Creating a multiindex to account for the different trajectories
    thrddim=len(np.where(Trj['time']==0)[0]) #number of trajectories
    
    #here I simply create a sequential index (each trajectory has the same integer number)
    indexcol=np.zeros(len(Trj.index))
    for j in range(thrddim):
        indexcol[(81*j):((j+1)*81)]=j
    
    Trj['trajectory']=indexcol #I add a new index column
    Trj.set_index(Trj['trajectory'],inplace=True) #I create a multindex
    return(Trj,Trjtime)

#%% Fitting PSD functions
# =============================================================================
# Standard mode fitting functions
# =============================================================================
def lognormal(x, p0):
        return ((p0[0]/(np.log10(p0[1])*np.sqrt(2*np.pi)))*np.exp(-((np.log10(x/p0[2]))**2)/(2*np.log10(p0[1])**2)))
def bilognormal(x, p0):
    '''p0 is initguess in the form: N1,Sig1,Mu1,...'''
    return ((p0[0]/(np.log10(p0[1])*np.sqrt(2*np.pi)))*np.exp(-((np.log10(x/p0[2]))**2)/(2*np.log10(p0[1])**2))+
                (p0[3]/(np.log10(p0[4])*np.sqrt(2*np.pi)))*np.exp(-((np.log10(x/p0[5]))**2)/(2*np.log10(p0[4])**2)))
def trilognormal(x, p0):
    '''p0 is initguess in the form: N1,Sig1,Mu1,...'''
    return ((p0[0]/(np.log10(p0[1])*np.sqrt(2*np.pi)))*np.exp(-((np.log10(x/p0[2]))**2)/(2*np.log10(p0[1])**2))+
                (p0[3]/(np.log10(p0[4])*np.sqrt(2*np.pi)))*np.exp(-((np.log10(x/p0[5]))**2)/(2*np.log10(p0[4])**2))+
                    (p0[6]/(np.log10(p0[7])*np.sqrt(2*np.pi)))*np.exp(-((np.log10(x/p0[8]))**2)/(2*np.log10(p0[7])**2)))

def lognormal_fit(PSD,diameters,init_guess,fit_bounds):
    def lognormal_f(x, N1,sig1,mu1):
        '''I need a sperate fitting curve because I cannot use a list for the parameters'''
        return ((N1/(np.log10(sig1)*np.sqrt(2*np.pi)))*np.exp(-((np.log10(x/mu1))**2)/(2*np.log10(sig1)**2)))
    #fit_bounds=([0,0,2,0,0,3],[np.inf,1.25,5,np.inf,1.25,10])
    pu, pcovu = curve_fit(lognormal_f, diameters, PSD,p0=init_guess,bounds=fit_bounds,max_nfev=1e4)
    return(pu, pcovu)

def bilognormal_fit(PSD,diameters,init_guess,fit_bounds):
    def bilognormal_f(x, N1,sig1,mu1,N2,sig2,mu2):
        '''I need a sperate fitting curve because I cannot use a list for the parameters'''
        return ((N1/(np.log10(sig1)*np.sqrt(2*np.pi)))*np.exp(-((np.log10(x/mu1))**2)/(2*np.log10(sig1)**2))+
                    (N2/(np.log10(sig2)*np.sqrt(2*np.pi)))*np.exp(-((np.log10(x/mu2))**2)/(2*np.log10(sig2)**2)))
    #fit_bounds=([0,0,2,0,0,3],[np.inf,1.25,5,np.inf,1.25,10])
    pu, pcovu = curve_fit(bilognormal_f, diameters, PSD,p0=init_guess,bounds=fit_bounds,max_nfev=1e4)
    return(pu, pcovu)

def trilognormal_fit(PSD,diameters,init_guess,fit_bounds):
    def trilognormal_f(x, N1,sig1,mu1,N2,sig2,mu2,N3,sig3,mu3):
        '''I need a sperate fitting curve because I cannot use a list for the parameters'''
        return ((N1/(np.log10(sig1)*np.sqrt(2*np.pi)))*np.exp(-((np.log10(x/mu1))**2)/(2*np.log10(sig1)**2))+
                    (N2/(np.log10(sig2)*np.sqrt(2*np.pi)))*np.exp(-((np.log10(x/mu2))**2)/(2*np.log10(sig2)**2))+
                        (N3/(np.log10(sig3)*np.sqrt(2*np.pi)))*np.exp(-((np.log10(x/mu3))**2)/(2*np.log10(sig3)**2)))
    #fit_bounds=([0,0,2,0,0,3],[np.inf,1.25,5,np.inf,1.25,10])
    pu, pcovu = curve_fit(trilognormal_f, diameters, PSD,p0=init_guess,bounds=fit_bounds,max_nfev=1e4)
    return(pu, pcovu)

def fourlognormal_fit(PSD,diameters,init_guess,fit_bounds):
    def fourlognormal_f(x, N1,sig1,mu1,N2,sig2,mu2,N3,sig3,mu3,N4,sig4,mu4):
        '''I need a sperate fitting curve because I cannot use a list for the parameters'''
        return ((N1/(np.log10(sig1)*np.sqrt(2*np.pi)))*np.exp(-((np.log10(x/mu1))**2)/(2*np.log10(sig1)**2))+
                    (N2/(np.log10(sig2)*np.sqrt(2*np.pi)))*np.exp(-((np.log10(x/mu2))**2)/(2*np.log10(sig2)**2))+
                        (N3/(np.log10(sig3)*np.sqrt(2*np.pi)))*np.exp(-((np.log10(x/mu3))**2)/(2*np.log10(sig3)**2))+
                            (N4/(np.log10(sig4)*np.sqrt(2*np.pi)))*np.exp(-((np.log10(x/mu4))**2)/(2*np.log10(sig4)**2)))
    pu, pcovu = curve_fit(fourlognormal_f, diameters, PSD,p0=init_guess,bounds=fit_bounds,max_nfev=1e4)
    return(pu, pcovu)

# =============================================================================
# Specific NAIS fit functions
# =============================================================================
def NAIS_PSD_logfit(NAIS,start_t,end_t,numavg,Dp_start,Dp_end,init_guess,fit_bounds,modenum=3):
    '''
    This functions averages a defined number of consecutive NAIS measurements and fit a
    multimodal lognormal function to the averaged PSD

    Parameters
    ----------
    NAIS : Dataframe containing NAIS data
    start_t : string, averaging start time
    end_t : string, averaging end time
    numavg : int, number of consecutive measurements to average
    Dp_start : float, starting diameter for the fit
    Dp_end : float, final diameter for the fit
    init_guess : Array containing the fit initial guess 
    fit_bounds : Array containing the parameter bounds for the fit
    modenum : int, number of modes to fit, max is four.

    Returns
    -------
    NAIS_gr: Averaged NAIS PSD
    df_fit_results: dataframe containing the fit results
    std_err: mu and std relative error sum for each mode, useful to evaluate
            bad fit that should be discarded
    std_err_mu: standard error on mu fit
    '''
    if modenum > 4:
        print('More than 4 mode fitting not yet implemented')
        return()
    DpNAIS = np.array(NAIS.columns,dtype=float)
    NAIS_temp = NAIS[start_t:end_t]
    NAIS_gr = NAIS_temp.resample(str(numavg*210)+'s',base=NAIS_temp.index[0].second).mean().dropna()

    # Cut extreme sizes
    NAIS_gr = NAIS_gr.iloc[:,(DpNAIS>Dp_start)&(DpNAIS<Dp_end)]
    DpNAIS_gr=np.array(NAIS_gr.columns,dtype=float)
    
    fit_param_array=np.zeros((len(NAIS_gr.index),modenum*3))
    fit_param_err=np.zeros((len(NAIS_gr.index),modenum*3))
    
    for j in range(len(NAIS_gr.index)):
        try:
            if modenum ==1:
                pu, pcovu=bilognormal_fit(NAIS_gr.iloc[j],DpNAIS_gr,init_guess,fit_bounds)
            elif modenum == 2:
                pu, pcovu=bilognormal_fit(NAIS_gr.iloc[j],DpNAIS_gr,init_guess,fit_bounds)
            elif modenum == 3:
                pu, pcovu=trilognormal_fit(NAIS_gr.iloc[j],DpNAIS_gr,init_guess,fit_bounds)
            elif modenum == 4:
                pu, pcovu=fourlognormal_fit(NAIS_gr.iloc[j],DpNAIS_gr,init_guess,fit_bounds)
            fit_param_array[j]=pu
            fit_param_err[j] = np.sqrt(np.diag(pcovu))
        except RuntimeError:
                print('Error - curve_fit failed at time'+ str(NAIS_gr.iloc[j].name))
                fit_param_array[j]=np.nan
                fit_param_err[j]=np.nan
     
    colnames = ['N1','sig1','mu1','N2','sig2','mu2','N3','sig3','mu3','N4','sig4','mu4']
    df_fit_results=pd.DataFrame(fit_param_array,index=NAIS_gr.index,
                             columns=colnames[:3*modenum]).copy()

    # Calculate sum relative error
    sumerr = fit_param_err[:,1::3]/fit_param_array[:,1::3]+\
                fit_param_err[:,2::3]/fit_param_array[:,2::3]
    
    colnames = ['Mode1','Mode2','Mode3','Mode4']
    std_err = pd.DataFrame(sumerr,columns=colnames[:modenum],
                           index=df_fit_results.index)
    std_err_mu = pd.DataFrame(fit_param_err[:,2::3],columns=colnames[:modenum],
                           index=df_fit_results.index)
    return(NAIS_gr,df_fit_results,std_err,std_err_mu)

def GRlinfit(xarr,yarr,ConfInt = True):
    '''
    Growth rate calculation as a OLS fit of the mode diameter as a function of 
    time (in minute).

    Parameters
    ----------
    xarr : Array containing the minutes since the start of each measurement
    yarr : Array containing the mode diameter in nm
    ConfInt : if True it returns the 95% confidence interval as error estimate
                if False the output is the standard error of the fit
    Returns
    -------
    GR : Growth rate in nm/hour
    Err : according to ConfInt

    '''
    X=xarr
    X = sm.add_constant(X)
    mod = sm.OLS(yarr,X)
    Results = mod.fit()
    GR = Results.params['x1']*60
    if ConfInt:
        Err = Results.conf_int(alpha=0.05, cols=None)*60
        Err = Err.loc['x1']
    else:
        Err = Results.bse['x1']*60
    return(GR, Err)

def GRweilinfit(xarr,yarr,ConfInt = True):
    '''
    Growth rate calculation as a WLS fit of the mode diameter as a function of 
    time (in minute). Weights are calculate as the inverse of the squared error
    on each mode diameter.

    Parameters
    ----------
    xarr : Array containing the minutes since the start of each measurement
    yarr : Array containing the mode diameter in nm as first column and the 
            error associate to the mode on the second column.
    ConfInt : if True it returns the 95% confidence interval as error estimate
                if False the output is the standard error of the fit
    Returns
    -------
    GR : Growth rate in nm/hour
    Err : according to ConfInt

    '''
    X = xarr
    Y = yarr.iloc[:,0]
    weig = 1/yarr.iloc[:,1]**2
    X = sm.add_constant(X)
    mod = sm.WLS(Y,X,weights=weig)
    Results = mod.fit()
    GR = Results.params['x1']*60
    if ConfInt:
        Err = Results.conf_int(alpha=0.05, cols=None)*60
        Err = Err.loc['x1']
    else:
        Err = Results.bse['x1']*60
    return(GR, Err)

def bilognormal_plot(j,DpNAIS,NAIS,df_fit_results,std_err,threshold):
    '''
    Plot consecutive bilognormal fit results, to be described and improved
    j is the scan index
    '''
    plt.figure(figsize=(17,11))
    plt.plot(DpNAIS,NAIS.loc[j])
    plt.plot(DpNAIS,lognormal(DpNAIS,df_fit_results.loc[j][:3]),label='sigma %(sigma).2f , median=%(median).2f, N=%(N).1f'%
             {'sigma':df_fit_results.loc[j][1],'median':df_fit_results.loc[j][2],'N':df_fit_results.loc[j][0]})
    plt.plot(DpNAIS,lognormal(DpNAIS,df_fit_results.loc[j][3:]),label='sigma %(sigma).2f , median=%(median).2f, N=%(N).1f'%
             {'sigma':df_fit_results.loc[j][4],'median':df_fit_results.loc[j][5],'N':df_fit_results.loc[j][3]})
    
    plt.plot(DpNAIS,bilognormal(DpNAIS,df_fit_results.loc[j]))
    plt.xscale('log')
    plt.title(NAIS.loc[j].name)
    plt.legend()
    plt.text(5,0.5,str(std_err.loc[j]))
    if std_err.loc[j]>threshold:
        plt.text(5,100,'Discarded')
        
def trilognormal_plot(j,DpNAIS,NAIS,df_fit_results,std_err,threshold):
    '''
    Plot consecutive trilognormal fit results, to be described and improved
    '''
    plt.figure(figsize=(17,11))
    plt.plot(DpNAIS,NAIS.loc[j])
    plt.plot(DpNAIS,lognormal(DpNAIS,df_fit_results.loc[j][:3]),label='sigma %(sigma).2f , median=%(median).2f, N=%(N).1f'%
             {'sigma':df_fit_results.loc[j][1],'median':df_fit_results.loc[j][2],'N':df_fit_results.loc[j][0]})
    plt.plot(DpNAIS,lognormal(DpNAIS,df_fit_results.loc[j][3:6]),label='sigma %(sigma).2f , median=%(median).2f, N=%(N).1f'%
             {'sigma':df_fit_results.loc[j][4],'median':df_fit_results.loc[j][5],'N':df_fit_results.loc[j][3]})
    plt.plot(DpNAIS,lognormal(DpNAIS,df_fit_results.loc[j][6:]),label='sigma %(sigma).2f , median=%(median).2f, N=%(N).1f'%
             {'sigma':df_fit_results.loc[j][7],'median':df_fit_results.loc[j][8],'N':df_fit_results.loc[j][6]})
    plt.plot(DpNAIS,trilognormal(DpNAIS,df_fit_results.loc[j]))
    plt.xscale('log')
    plt.title(NAIS.loc[j].name)
    plt.legend()
    plt.text(5,0.5,str(std_err.loc[j]))
    if std_err.loc[j]>threshold:
        plt.text(5,100,'Discarded')
        
#%% Growth rate calculation
# =============================================================================
# Various functions needed to estimate GR
# =============================================================================
def Modified_Knudsen_num(dp,dv, Dp, Dv, Cp, Cv):
    """This function calculates the modified Knudsen number for growth
    rate calculation at given T and P. Follow Lehtinen and Kulmala (2003).
    Input:
        dp particle diameter in nm
        dv molecule diameter in nm
        Dp particle diffusion coefficient
        Dv molecule diffusion coefficient
        Cp particle thermal speed in cm/s
        Cv gas thermal speed in cm/s

    Output:
        Kn"""
    #Mean free path calculation (with conversion to nm)
    lambd=3*(Dp+Dv)/np.sqrt(Cp**2+Cv**2)*1e7
    
    Kn=2*lambd/(dp+dv) #Knudsen number
    return Kn

def FS_Corr_Fact(Kn, alpha):
    ''' Fuchs-Sutugin transition regime correction factor
    for mass flux calculation.
    Input:
        Kn modified Knudsen number
        alpha accomodation coefficient
    Output:
        betam'''
    
    betam=(1+Kn)/(1+(4/(3*alpha)+0.337)*Kn+4/(3*alpha)*Kn**2)
    return betam



def Thermal_speed_gas(T, m):
    ''' Calculate thermal speed of gas molecule.
    input:
        T temperature in kelvin
        m Molecular weight in g/mol 
    output:
        Cv in cm/s
    '''
    m_conv = m/1000 #conversion in kg
    R = 8.314 #gas constant
    Cv = np.sqrt(8*R*T/(np.pi*m_conv))*100
    
    return Cv
             
def Thermal_speed_aerosol(T, m):
    ''' Calculate thermal speed of aerosol.
    input:
        T Temperature in kelvin
        m Aerosol mass in ng
    output:
        Cp in cm/s
    '''
    m_conv = m/1e12 #conversion in kg
    k = 1.381e-23 # Boltzmann constant
    Cp = np.sqrt(8*k*T/(np.pi*m_conv))*100
    
    return Cp

def GR_estimate(Conc, dp, dv, mp, mv, alpha, rho, T, P):
    ''' Return growth rate based on a certain gas-phase
    concentration valid also for sub-10 nm particles.
    Input:
        Conc gas phase concentration in molecules per cm3
        dp particle diameter in nm
        dv molecule diameter in nm
        mp aerosol mass in ng
        mv molecular mass in g/mol
        alpha accomodation coefficient
        rho vapor condensed phase density in g/cm3
        T temperature in Kelvin
        P pressure in hPa
    Output:
        Gr in nm/h '''
    k = 1.381e-23 # Boltzmann constant
    mv_abs = mv/6.022*1e-14 # mass of gas molecule in ng
    # Calculate thermal speed
    Cv = Thermal_speed_gas(T, mv)
    Cp = Thermal_speed_aerosol(T, mp)
    
    # Calculate diffusion coefficient
    # I'm assuming SA diffusion for every gas
    Dv = SA_DiffCoeff(T,P)
    Cc = Cunningham(dp,P,T)
    Dp = Particle_DiffCoeff(T,Cc,dp)
    
    # Calculate mod Kn, beta and gamma
    Knmod = Modified_Knudsen_num(dp,dv, Dp, Dv, Cp, Cv)
    beta = FS_Corr_Fact(Knmod, alpha)
    gamma = 4/3 * Knmod * beta
    
    GR = gamma/(2*rho) * (1+dv/dp)**2 * np.sqrt(8*k*T/np.pi) \
        * np.sqrt(1/mp+1/mv_abs) *mv_abs*Conc * 3.6*1e9
    
    return GR

def GR_estimate_with_Coag(Conc, Kcoag, dp, mv, rho):
    ''' Return growth rate based on a certain collision
    rate and gas-phase concentration.
    Input:
        Conc gas phase concentration in molecules per cm3
        Kcoag collision in cm3/s
        dp particle diameter in nm
        mv molecular mass in g/mol
        rho vapor condensed phase density in g/cm3
    Output:
        Gr in nm/h '''
    mv_abs = mv/6.022*1e-23 # mass of gas molecule in g
    GR=Kcoag*mv_abs*Conc/(rho*np.pi/2*dp**2)*3.6*10**(24)
    return GR
    
    
def Coagulation_coeff_SA(dp, dv, mp, mv, alpha, T, P):
    ''' Calculate generalized SA coagulation coefficient.
    Input:
        dp particle diameter in nm
        dv molecule diameter in nm
        mp aerosol mass in ng
        mv molecular mass in g/mol
        alpha accomodation coefficient
        rho vapor condensed phase density in g/cm3
        T temperature in Kelvin
        P pressure in hPa
    Output:
        Coagulation coeff in cm3/s '''
    # k = 1.381e-23 # Boltzmann constant
    # mv_abs = mv/6.022*1e-14 # mass of gas molecule in ng
    # Calculate thermal speed
    Cv = Thermal_speed_gas(T, mv)
    Cp = Thermal_speed_aerosol(T, mp)
    
    # Calculate diffusion coefficient
    # I'm assuming SA diffusion for every gas
    Dv = SA_DiffCoeff(T,P)
    Cc = Cunningham(dp,P,T)
    Dp = Particle_DiffCoeff(T,Cc,dp)
    
    # Calculate mod Kn, beta and gamma
    Knmod = Modified_Knudsen_num(dp,dv, Dp, Dv, Cp, Cv)
    beta = FS_Corr_Fact(Knmod, alpha)
    gamma = 4/3 * Knmod * beta
    
    CoagCf=gamma*np.pi/4*(dp+dv)**2*np.sqrt((Cv**2+Cp**2))*1e-14
    return CoagCf

def Coagulation_coeff_kinetic(dp1, dp2, mp1, mp2, alpha, T, P):
    ''' Calculate coagulation coefficient in the kinetic
    limit. This is mostly just as a reference function,
    in general it is better to use the general coagulation function.
    
    Input:
        dp particle diameter in nm
        mp aerosol mass in ng
        alpha accomodation coefficient
        T temperature in Kelvin
        P pressure in hPa
    Output:
        Coagulation coeff in cm3/s '''
    # Calculate thermal speed
    Cp1 = Thermal_speed_aerosol(T, mp1)
    Cp2 = Thermal_speed_aerosol(T, mp2)
    
    # Calculate diffusion coefficient
    Cc = Cunningham(dp1,P,T)
    D1 = Particle_DiffCoeff(T,Cc,dp1)
    Cc = Cunningham(dp2,P,T)
    D2 = Particle_DiffCoeff(T,Cc,dp2)
    
    # Calculate mod Kn, beta and gamma
    Knmod = Modified_Knudsen_num(dp1,dp2, D1, D2, Cp1, Cp2)
    beta = FS_Corr_Fact(Knmod, alpha)
    gamma = 4/3 * Knmod * beta
    
    CoagCf=gamma*np.pi/4*(dp1+dp2)**2*np.sqrt((Cp1**2+Cp2**2))*1e-14
    return CoagCf

def FS_Corr_Fact_coag(dp1, dp2, D1, D2, Cp1, Cp2):
    ''' Fuchs-Sutugin transition regime correction factor
    for coagulation calculation.
    Input:
        
    Output:
        betam'''
    dp1 = dp1*1e-7
    dp2 = dp2*1e-7
    print(dp1)
    l1 = 8*D1/(np.pi*Cp1)
    l2 = 8*D2/(np.pi*Cp2)
    
    g1 = np.sqrt(2)/(3*dp1*l1)*((dp1+l1)**3-(dp1**2+l1**2)**(3/2))-dp1
    g2 = np.sqrt(2)/(3*dp2*l2)*((dp2+l2)**3-(dp2**2+l2**2)**(3/2))-dp2
    
    betam=((dp1+dp2)/(dp1+dp2+2*(g1**2+g2**2)**0.5)+(8*(D1+D2))/((Cp1**2+Cp2**2)**0.5*(dp1+dp2)))**(-1)
    return betam

def Coagulation_coeff(dp1, dp2, mp1, mp2, alpha, T, P):
    ''' Calculate generalized coagulation coefficient.
    Input:
        dp particle diameter in nm
        mp aerosol mass in ng
        alpha accomodation coefficient
        T temperature in Kelvin
        P pressure in hPa
    Output:
        Coagulation coeff in cm3/s '''
    # Calculate thermal speed
    Cp1 = Thermal_speed_aerosol(T, mp1)
    Cp2 = Thermal_speed_aerosol(T, mp2)
    
    # Calculate diffusion coefficient
    Cc = Cunningham(dp1,P,T)
    D1 = Particle_DiffCoeff(T,Cc,dp1)
    Cc = Cunningham(dp2,P,T)
    D2 = Particle_DiffCoeff(T,Cc,dp2)
    
    # Calculate mod Kn, beta and gamma
    Knmod = Modified_Knudsen_num(dp1,dp2, D1, D2, Cp1, Cp2)
    beta = FS_Corr_Fact_coag(dp1, dp2, D1, D2, Cp1, Cp2)
    
    CoagCf=2*np.pi*(dp1+dp2)*(D1+D2)*beta*1e-7
    return CoagCf
    
#%% Deposition velocity
def friction_velocity(Wspeed,height):
    k=0.4
    Z0=0.001
    u_fric=k*Wspeed/(np.log(height/Z0))
    return(u_fric)

def Aerod_Res(Fvel,height):
    k=0.4
    Z0=0.001
    A_Res=1/(k*Fvel)*np.log(height/Z0)
    return(A_Res)

def Schmidt_num(T,P,RH):
    # Dry and wet air density
    rho_d,rho_w=Air_density(T,P,RH)
    Sc=Air_viscosity(T)/(rho_w*SA_DiffCoeff(T,P)*1e-4) #1e4 factor is conversion from cm2 to m2
    return(Sc)

def Qlam_Res(Sc,FricVel):
    Q_Res=5*Sc**(2/3)/FricVel
    return Q_Res

def Deposition_Vel(Wspeed,Temperature,Pressure,RH,height):
    # Temperature must be in kelvin
    
    FricVel=friction_velocity(Wspeed,height)
    ARes=Aerod_Res(FricVel,height)
    ScN=Schmidt_num(Temperature,Pressure,RH)
    QRes=Qlam_Res(ScN,FricVel)
    
    Dvel=1/(ARes+QRes)
    return Dvel

#%% Kohler theory
def kohler(Ddry, Ddrop):
    Mw=18.01528
    sw=72.75 #has to be in milli newton per meter
    T=293
    R=8.3144
    rhoW=0.997 #has to be in gram per cubic centimeter
    Aw=4*Mw*sw/(R*T*rhoW)
    
    vs=3 #number of discrete ions in the formula
    rhos=1.77
    Ms=132.139 #ammonium sulfate molar mass
    
    Bs=rhos*vs*Mw*Ddry**3/(Ms*rhoW)
    
    RHeq=Aw/Ddrop-Bs/Ddrop**3
    return RHeq

def Kelvin(Ddrop):
    Mw=18.01528
    sw=72.75 #has to be in milli newton per meter
    T=293
    R=8.3144
    rhoW=0.997 #has to be in gram per cubic centimeter
    Aw=4*Mw*sw/(R*T*rhoW)
    
    RHeq=Aw/Ddrop
    return RHeq

def K_eqn(Ddrop,Ddry,SS):
    # calculate Aw
    Mw=18.01528
    sw=72.75 #has to be in milli newton per meter
    T=293
    R=8.3144
    rhoW=0.997 #has to be in gram per cubic centimeter
    Aw=4*Mw*sw/(R*T*rhoW)
    
    Kfunc=(Ddry**3-Ddrop**3)*(1-np.exp(Aw/Ddrop)+SS)/(Ddry**3*(1+SS))
    return Kfunc

def SS_eqn(Ddrop,Ddry,k):
    # calculate Aw
    Mw=18.01528
    sw=72.75 #has to be in milli newton per meter
    T=293
    R=8.3144
    rhoW=0.997 #has to be in gram per cubic centimeter
    Aw=4*Mw*sw/(R*T*rhoW)
    
    SSfunc=(Ddrop**3-Ddry**3)/(Ddrop**3-(1-k)*Ddry**3)*np.exp(Aw/Ddrop)-1
    return SSfunc
#%% Instrument load/process function
def resample_FSSP(Ndrop,Dpdrop):
    ''' Resample FSSP data on a regular diameter grid.
    The grid is very dense but uses linear interpolation so it
    doesn't introduce artefacts.
    input:
        -Ndrop: is the number of droplet per bin
        -Dpdrop: droplet diameter
    output:
        -Regularly log-spaced Dp grid
        -dN/dD'''
    #Define interpolating function
    f = interp1d(Dpdrop,Ndrop, kind='linear',bounds_error=False)
    #define new diameter grid
    Dp_new=np.logspace(-0.126,1.67,400)
    bin_width=Dp_new[2:]-Dp_new[:-2]
    #evaluate interpolating function (I exclude the 2 diam extreme)
    Ndrop_interp=f(Dp_new[1:-1])/bin_width
    return(Dp_new[1:-1],Ndrop_interp)

def ACES_DMPSread(datadir):
    DMPS_tot=pd.read_csv(datadir,delimiter='\t',index_col=0,header=None)
    
    #convert time (to be checked):
    days=pd.to_timedelta(DMPS_tot.index[1:],unit='d')
    index=datetime.datetime(2017,12,31)+days
    index=index-pd.Timedelta('4min')-pd.Timedelta('30s')
    
    DMPS=DMPS_tot.iloc[1:,2:]
    DMPS.index=index
    Dp=DMPS_tot.iloc[0,2:]#Diameter
    
    #I don't know the difference between these 2 columns, I need to ask Paul
    TotConc1=DMPS_tot.iloc[1:,0]
    TotConc1.index=index
    TotConc2=DMPS_tot.iloc[1:,1]
    TotConc2.index=index
    
    return(TotConc1,TotConc2,DMPS,Dp)

def Load_PICARRO(datadir):
    ''' Load PICARRO data, it just need the filepath'''
    Picarro_data=pd.read_csv(datadir,sep=r"\s+",engine='python',usecols=[4,10,12,14,15])
    idx=pd.to_datetime(Picarro_data['EPOCH_TIME'],unit='s')
    Picarro_data.index=idx

    return(Picarro_data[['CO2_dry','CO','CH4_dry','H2O']])

def Load_O32b(datadir):
    '''
    Load Ozone data from 2B monitor, it includes
    some cleaning from tipically wrongly formatted
    lines '''
    
    Ozone=pd.read_csv(datadir,header=None,delimiter=',',index_col=0,parse_dates=True,dayfirst=True,usecols=[0,1,2,3])
    Ozone.columns=['Ozone [ppb]','Temp Instr.','Press Instr.']
    Ozone.index.name='Datetime'

    # Remove non numeric data
    Ozone=Ozone.apply(lambda x: pd.to_numeric(x, errors='coerce')).dropna()
    # Identify other badly formatted lines
    str_index=np.array(Ozone.index.astype('|S20'))
    sizes = [len(i) for i in str_index]
    idx_to_drop=np.where(np.array(sizes)<19)[0]
    
    Ozone_clean=Ozone.drop(Ozone.index[idx_to_drop])
    
    Ozone_clean.index=pd.DatetimeIndex(Ozone_clean.index)

    return(Ozone_clean)

def read_APS(filename):
    APS = pd.read_csv(filename,delimiter=',',skiprows=6)
    Dp = np.array(APS.columns[5:56], dtype=float)
    Time = pd.to_datetime(APS['Date']+' '+APS['Start Time'], 
                          infer_datetime_format=True, errors='coerce')
    APS.index = Time
    APS_sel = APS.iloc[:,5:56].dropna()
    APS_sel.columns = Dp
    return APS_sel

#%% Miscellaneous
def Group_getindex(data,threshold=10):
    '''This function identify groups of data in a timeseries.
    It was written for the CCN timeseries but eventually it can
    be adapted to other dataset.
    Input:
        -Timeseries, should be a single column DataFrame
        -Threshold in minutes between groups
    Output:
        -Dataframe containing start and end time of each group'''

    ddf=data.dropna().reset_index()
    threshold_t = pd.Timedelta(threshold, 'm')
    starting = ddf['index'].loc[ddf['index'].diff() > threshold_t]
    ending = ddf['index'].loc[starting.index-1]

    #adding first row of ddf to starting, and last row of ddf to ending
    starting = pd.Series(ddf['index'].iloc[0]).append(starting)
    ending = ending.append(pd.Series(ddf['index'].iloc[-1]))

    #make a dataframe, each row contains starting and ending times of a group
    groups = pd.DataFrame({'start':starting.reset_index(drop=True), 'end':ending.reset_index(drop=True)})
    return(groups)


#%% Plotting functions
    
def PSD_plot(SMPS,Dp,Maxcolorscale=5000,fsize=(12,8)):
    # Matrix roation for plot
    matrix=np.array(SMPS)
    matrix[np.where(matrix<=0)]=0.01 #replace the zeros for graphical purposes (cannot handle log scale)
    size=np.shape(matrix)
    Tmatrix=np.zeros((size[1],size[0]))

    for j in range(size[1]):
        Tmatrix[j,:]=matrix[:,j]
    
    fig=plt.figure(figsize=fsize)
    ax=plt.subplot(111)
    pcm=ax.pcolormesh(SMPS.index,Dp,Tmatrix,norm=LogNorm(vmin=1,vmax=Maxcolorscale),cmap='viridis')
    
    ax.set_yscale('log')
    
    #colorbar
    fig.subplots_adjust(right=0.88)
    cbar_ax = fig.add_axes([0.9, 0.15, 0.05, 0.7])
    cbar=fig.colorbar(pcm,cax=cbar_ax, extend='both', orientation='vertical')
    
    cbar.set_label('dN/dlogDp [#/cm3]')
    plt.subplots_adjust(hspace=0.15)
    fig.text(0.065, 0.5, 'Size [nm]', ha='center', va='center', rotation='vertical',fontsize=20)
    ticks=ax.get_xticks()
    
    ax.set_xticks(ticks[::2])
    ax.xaxis.set_major_formatter(date.DateFormatter('%d-%m %H:%M'))
    ax.tick_params(labelsize=18)
    #yticks=ax.get_yticks(minor=True)
    #ax.set_yticks(yticks[1::2],minor=True)
    
    #ax.yaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter('%i'))
    #ax.yaxis.set_minor_formatter(matplotlib.ticker.ScalarFormatter(useOffset=False))
    ax.tick_params(axis='y',labelsize=18)
    ax.tick_params(which='minor',labelsize=16)
    ax.set_title('Particle Size Distribution',fontsize=18)
    
    plt.show()