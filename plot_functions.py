# -*- coding: utf-8 -*-
"""
Created on Thu Aug 19 10:09:10 2021

@author: jo28dohe
"""


import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import cm

### TODO: typical figure sizes 
##- twocolumn paper: 3.375,3.375*3/4  (inches) or 8.57 cm , 8.57*3/4 cm

class paperfigure:
    '''
    returns an object containing figures with typical paper sizes. 
    figure_type can be:
        - 'twocolumn'
        - 'twocolumn_double'
        - 'onecolumn'
    '''
    
    def __init__(self,width_in_cols=1,aspect_ratio=4/3,
                 override_figsize=None):
        cm_to_in=0.3937
        self.width=width_in_cols*8.6*cm_to_in   ## has to be in inch for matplotlib
        self.aspect_ratio=aspect_ratio
        mpl.rcParams['figure.dpi']=600
        mpl.rcParams['axes.grid']=True
        mpl.rcParams['axes.labelsize']='medium'
        mpl.rcParams['xtick.labelsize']='small'
        mpl.rcParams['ytick.labelsize']='small'
        
        if width_in_cols<0.6:
            
            mpl.rcParams['font.size']=8
            plt.locator_params(nbins=4)
            mpl.rcParams['figure.subplot.left']=0.28
            mpl.rcParams['figure.subplot.right']=0.92
            mpl.rcParams['figure.subplot.bottom']=0.24
            mpl.rcParams['figure.subplot.top']=0.93
        else:
            plt.locator_params(nbins=6)
            
            mpl.rcParams['font.size']=9
            mpl.rcParams['figure.subplot.left']=0.15
            mpl.rcParams['figure.subplot.right']=0.97
            mpl.rcParams['figure.subplot.bottom']=0.15
            mpl.rcParams['figure.subplot.top']=0.95
            mpl.rcParams['figure.subplot.hspace']=0.02
        
        self.fig, self.ax =plt.subplots(figsize=(self.width,self.width/self.aspect_ratio))
        
class colorplot(paperfigure):
    '''
    creates a colorplot using plt.scatter()
    x_data shall be a list or df of lists/arrays/dataframes
    '''
    
    def __init__(self,x_data,y_data,c_data,
                 xlabel=None,ylabel=None,cmap=cm.nipy_spectral,
                 vmin=None,vmax=None,
                 **kwargs):
        super().__init__(**kwargs)
        
        ### make list if only one line of x_data is given:
        if np.shape(x_data)==():
            x_data=[x_data]
            y_data=[y_data]
            c_data=[c_data]
            
        if vmin==None:
            min_list=[min(x) for x in c_data]
            self.vmin=min(min_list)
        else:
            self.vmin=vmin
        if vmax==None:
            max_list=[max(x) for x in c_data]
            self.vmax=max(max_list)
        else:
            self.vmax=vmax
            
            
        ### make the colorplot using scatter 
        for x,y,c in zip(x_data,y_data,c_data):
        
            self.ax.scatter(x,y,c=c,cmap=cmap,
                            vmin=self.vmin,vmax=self.vmax)
        
        ###  label axes. If no name specified try to use name of xdata
        if xlabel is not None:
            self.ax.set_xlabel(xlabel)
        else:
            try:
                self.ax.set_xlabel(x_data.name) 
            except:
                pass
                
        if ylabel is not None:
            self.ax.set_ylabel(ylabel)
        else:
            try:
                self.ax.set_ylabel(y_data.name)
            except:
                pass
        
    
    
    