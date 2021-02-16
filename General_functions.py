# -*- coding: utf-8 -*-
"""
Created on Tue Jan 22 17:14:17 2019

@author: baccarini_a
"""
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.gridspec as gridspec
import seaborn as sns
import numpy as np

def newfigure(num=None,x=-1800,y=50,w=1200,h=1100):
       hfig = plt.figure(num)
       plt.get_current_fig_manager().window.setGeometry(x,y,w,h)
       return hfig
   

def ConvertIgorTime(Igortime):
    #this function convert igor timestamp into normal dates
    Pytime=pd.to_datetime(Igortime, unit='s',origin=pd.Timestamp('1904-01-01'))
    return Pytime

def ConvertMatlabTime(matlabtime):
    #this function convert matlab timestamp into normal dates
    #719529 is the number of days from matlab epoch to Unix epoch
    Pytime=pd.to_datetime(matlabtime-719529,unit='d')#.round('s')
    return Pytime


class SeabornFig2Grid():

    def __init__(self, seaborngrid, fig,  subplot_spec):
        self.fig = fig
        self.sg = seaborngrid
        self.subplot = subplot_spec
        if isinstance(self.sg, sns.axisgrid.FacetGrid) or \
            isinstance(self.sg, sns.axisgrid.PairGrid):
            self._movegrid()
        elif isinstance(self.sg, sns.axisgrid.JointGrid):
            self._movejointgrid()
        self._finalize()

    def _movegrid(self):
        """ Move PairGrid or Facetgrid """
        self._resize()
        n = self.sg.axes.shape[0]
        m = self.sg.axes.shape[1]
        self.subgrid = gridspec.GridSpecFromSubplotSpec(n,m, subplot_spec=self.subplot)
        for i in range(n):
            for j in range(m):
                self._moveaxes(self.sg.axes[i,j], self.subgrid[i,j])

    def _movejointgrid(self):
        """ Move Jointgrid """
        h= self.sg.ax_joint.get_position().height
        h2= self.sg.ax_marg_x.get_position().height
        r = int(np.round(h/h2))
        self._resize()
        self.subgrid = gridspec.GridSpecFromSubplotSpec(r+1,r+1, subplot_spec=self.subplot)

        self._moveaxes(self.sg.ax_joint, self.subgrid[1:, :-1])
        self._moveaxes(self.sg.ax_marg_x, self.subgrid[0, :-1])
        self._moveaxes(self.sg.ax_marg_y, self.subgrid[1:, -1])

    def _moveaxes(self, ax, gs):
        #https://stackoverflow.com/a/46906599/4124317
        ax.remove()
        ax.figure=self.fig
        self.fig.axes.append(ax)
        self.fig.add_axes(ax)
        ax._subplotspec = gs
        ax.set_position(gs.get_position(self.fig))
        ax.set_subplotspec(gs)

    def _finalize(self):
        plt.close(self.sg.fig)
        self.fig.canvas.mpl_connect("resize_event", self._resize)
        self.fig.canvas.draw()

    def _resize(self, evt=None):
        self.sg.fig.set_size_inches(self.fig.get_size_inches())

#%% Signal processing
def Hampel_filter(vals_orig, k=7, t0=3):
    '''
    vals: pandas series of values from which to remove outliers
    k: size of window (including the sample; 7 is equal to 3 on either side of value)
    '''
    #Dropnan otherwise median is set to nan
    vals = vals_orig.dropna().copy()

    #Hampel Filter
    L = 1.4826
    
    rolling_median = vals.rolling(window=k, center=True).median()
    MAD = lambda x: np.median(np.abs(x - np.median(x)))
    rolling_MAD = vals.rolling(window=k, center=True).apply(MAD)
    threshold = t0 * L * rolling_MAD
    difference = np.abs(vals - rolling_median)

    if np.any(threshold==0):
        print('Threshold is zero, check data')

    outlier_idx = difference > threshold
    vals[outlier_idx] = np.nan

    #Refill the series with the originally removed nans
    vals=vals.reindex(vals_orig.index,method='asfreq')
    return(vals)

#%% Numpy based functions
def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return array[idx]

def find_nearest_idx(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx

#%%Fitting
# =============================================================================
# Linear regression without intercept
# =============================================================================
def LR_nointerc(x,y):
    '''Calculate the slope, I should add R2
    input should be arrays'''
    slope=np.sum(x*y)/np.sum(x**2)
    return(slope)

# =============================================================================
# Linear regression keeping into account errors, from M. Loreti 2006
# Functions are not carefully checked
# =============================================================================
def LinFit(x,y,sigma):
    N = len(x)
    Delta = N*np.sum(x**2)-np.sum(x)**2
    a = 1/Delta*(np.sum(x**2)*np.sum(y) - np.sum(x)*np.sum(x*y))
    b = 1/Delta*(N*np.sum(x*y) - np.sum(x)*np.sum(y))
    sig_a = sigma*np.sqrt(np.sum(x**2)/Delta)
    sig_b = sigma*np.sqrt(N/Delta)
    return(a,b,sig_a,sig_b)

def LinFitWeight(x,y,sigma):
    Delta = np.sum(1/sigma**2)*np.sum(x**2/sigma**2)-np.sum(x/sigma**2)**2
    a = 1/Delta*(np.sum(x**2/sigma**2)*np.sum(y/sigma**2) -\
                 np.sum(x/sigma**2)*np.sum(x*y/sigma**2))
    b = 1/Delta*(np.sum(1/sigma**2)*np.sum(x*y/sigma**2) -\
                 np.sum(x/sigma**2)*np.sum(y/sigma**2))
    sig_a = np.sum(x**2/sigma**2)/Delta
    sig_b = np.sum(1/sigma**2)/Delta
    return(a,b,sig_a,sig_b)

#%% Cartopy functions
def add_map_background(axis, debug = False, name = "natural-earth-1", resolution = "medium0512px", extent = None):
    import json
    import os
    '''
    Function adapted from https://github.com/Guymer/PyGuymer3 to modify the default background picture in cartopy
    it requires the correct system variable containing all the PNG files and .Json file
    
    '''
    os.environ['CARTOPY_USER_BACKGROUNDS'] = 'C:\\Anaconda2\\envs\\spyder-beta\\Lib\\site-packages\\cartopy\\data\\raster\\new_bkg\\'

    # Initialize trigger ...
    default = True

    # Check if the environment variable has been defined ...
    if "CARTOPY_USER_BACKGROUNDS" in os.environ:
        # Determine JSON path and check it exists ...
        jpath = os.path.join(os.environ["CARTOPY_USER_BACKGROUNDS"], "images.json")
        if os.path.exists(jpath):
            # Load JSON and check keys exist ...
            info = json.load(open(jpath, "rt"))
            if name in info:
                if resolution in info[name]:
                    # Determine image path and check it exists ...
                    ipath = os.path.join(os.environ["CARTOPY_USER_BACKGROUNDS"], info[name][resolution])
                    if os.path.exists(ipath):
                        default = False

    # Draw background image ...
    if default:
        if debug:
            print("INFO: Drawing default background.")
        axis.stock_img()
    else:
        if debug:
            print("INFO: Drawing user-requested background.")
        axis.background_img(name = name, resolution = resolution, extent = extent)
        