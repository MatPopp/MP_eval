# -*- coding: utf-8 -*-
"""
Created on Mon Aug 23 16:13:19 2021

@author: jo28dohe
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import cm

from LAP_eval import filehandling as fh
from LAP_eval import evaluation_functions as eval_func
from LAP_eval import physical_functions as phys_func
from LAP_eval import plot_functions as plot_func
from LAP_eval import refractive_index as ri

def read_data(datapath='data'):
    """this function shall contaion all the data reading and put all data into 
    a pandas DataFrame"""
    df=fh.data_from_directory(datapath, read_regex='_30s' ,
                              var_strings=['V_Piezo','V_SMU'],
                              var_regex_dict={'center_wavelength':'\D*(\d+\.?\d*)nm_30s'},
                              read_function=fh.spectrum_to_pd)
    return(df)
    
def evaluate_data(df):
    """this function shall do numerical evaluations and add the evaluation
    results to DataFrame. Putting all results into df will help
    - in keeping the code clean, everything findeable
    - saving evaluation results before plotting"""
    
    ## load calibration function, correct for list in nm 
    calib_fun_nm = fh.get_calibration_function(path = r'calibration_confocal2\Newton_OIympus_20x_vis150l_907_C_Dichro532nm_no_LP',
                                 regex = '.*calibration_combined.*')
    def calib_fun(lamb):
        return(calib_fun_nm(1e9*lamb))
    ## get all possible V_SMU values
    V_SMU_vals=df['V_SMU'].unique()
    center_wavelengths=df['center_wavelength'].unique()
    
    ## use V_Piezo and V_SMU as index
    df=df.set_index(['V_Piezo','V_SMU','center_wavelength']).sort_index()
    
    ## extract array-like counts and wavelengths from spectra
    df['wavelength'] = [np.array(spectrum.wavelength) for spectrum in df['data']]
    df['counts'] = [np.array(spectrum.counts) for spectrum in df['data']]
    ## remove cosmics
    df['counts'] = df['counts'].map(eval_func.remove_cosmics)
    
    
    ## background correction (the background counts are the counts at V_SMU=0)    
    def bg_corr(index):
        V_Piezo,V_SMU,center_wavelength=index
        return(df['counts'][V_Piezo,V_SMU,center_wavelength]-df['counts'][V_Piezo,0,center_wavelength])
        
    df['counts_corr']=df.index.map(bg_corr)
    
    ## stitch spectra
    
    df['counts_stitched']=None
    df['wavelength_stitched']=None
    
    for V_Piezo,V_SMU,center_wavelength in df.index.values:
        if V_SMU > 0 and center_wavelength==center_wavelengths[0]:
            lamb_list=list(df.wavelength.loc[V_Piezo,V_SMU,:].values)
            counts_list=list(df.counts_corr.loc[V_Piezo,V_SMU,:].values)
            
            df['wavelength_stitched'].loc[V_Piezo,V_SMU,center_wavelength],df['counts_stitched'].loc[V_Piezo,V_SMU,center_wavelength],coefficients=eval_func.stitchSpectra(lamb_list,counts_list)
   
    df['counts_stitched'] = df['counts_stitched'].fillna(np.nan)
    df['wavelength_stitched'] = df['wavelength_stitched'].fillna(np.nan)
    
    
    ## use instrument calibration function
    calib_list=calib_fun(df['wavelength_stitched'][df.index.unique(level=0)[0],df.index.unique(level=1)[1],df.index.unique(level=2)[0]])
    df['counts_calib'] = df['counts_stitched'].map(lambda counts: counts*calib_list)
    
    
    ## fit black-body+Fabry-Perot_Model
    
    def fit_func(index):
        if np.shape(df['counts_calib'][index])==() or df['wavelength_stitched'][index] is np.nan :
            print('doing nothing for V_Piezo,V_SMU,center_wavelength=',index,'because shape of counts is ()')
            return([np.nan,np.nan,np.nan])
        else:
            print('doing fit for V_Piezo,V_SMU,center_wavelength=',index)
            counts = df['counts_calib'][index]
            wavelength=df['wavelength_stitched'][index]
            counts=counts[wavelength<1000e-9]
            wavelength=wavelength[wavelength<1000e-9]
            
            
            print(wavelength)
            n_Au_instance = ri.n_from_string('Au')
            n_Au = n_Au_instance.get_n(wavelength)
            n_SiC_instance = ri.n_from_string('SiC')
            n_SiC = n_SiC_instance.get_n(wavelength)
            p_names=['d','A','T']
            p_min_max_steps_dict = {'d':[500e-9,2000e-9,20],'A':[0,40e-9,20],'T':[1200,2200,20]}
            
            fitted_args = eval_func.brute_leastsquare_fit(phys_func.thermal_radiation_mirror,
                                                wavelength, counts,
                                                p_names=p_names,
                                                p_min_max_steps_dict=p_min_max_steps_dict,
                                                visualize=True,
                                                const_params=[n_Au,n_SiC])
            return(fitted_args)
    
    df['fit_results']=df.index.map(fit_func)
    
    df[['d_fit','A_fit','T_fit']] = pd.DataFrame(df.fit_results.tolist(), index= df.index)
    
    ## save dataframe 
    save_directory='evaluated_data'
    fh.ensure_dir_exists(save_directory)
    df.to_pickle(save_directory+'/evaluated_stitch_evaluation_pandas_dataframe.pkl')
     
    return(df)
    

def visualize_data(df):
    """here, data shall be visualized and plotted"""
    pass
    


if __name__=='__main__':
    
    df=read_data(datapath='stitch_test_data')
    try:
        df=pd.read_pickle('evaluated_data/evaluated_stitch_evaluation_pandas_dataframe.pkl')
    except:
        df=evaluate_data(df)
    #df=evaluate_data(df)
    
    ## plot V_Piezo_dependant
    V_SMU=10
    plot_piezo_dep=plot_func.multiline_plot(df.wavelength_stitched.loc[:,V_SMU,700]*1e9,
                             df.counts_calib.loc[:,V_SMU,700],
                             df.loc[:,V_SMU,700].reset_index()['V_Piezo'],
                             xlabel=r'$\lambda$ in nm',
                             ylabel='counts in a.u.',
                             cmap=cm.jet,
                             width_in_cols=2/3)
    
    ## plot V_SMU_dependant
    V_Piezo=30
    plot_SMU_dep=plot_func.multiline_plot(df.wavelength_stitched.loc[:,V_SMU,700]*1e9,
                             df.counts_calib.loc[V_Piezo,:,700],
                             df.loc[V_Piezo,:,700].reset_index()['V_SMU'],
                             xlabel=r'$\lambda$ in nm',
                             ylabel='counts in a.u.',
                             rel_vmax=1.5,
                             vmin=6,
                             cmap=cm.hot,
                             width_in_cols=2/3)
    
    ## plot temeratures and Areas
    T_plot=plot_func.paperfigure(width_in_cols=1)
    #T_plot.ax.scatter(df.reset_index()['V_SMU'],df.reset_index()['A_fit'])
    T_plot.ax.scatter(df.reset_index()['V_SMU'],df.reset_index()['T_fit'],c='black')
    T_plot.ax.set_xlabel(r'$V_{SMU}$ in V')
    T_plot.ax.set_ylabel(r'$T_{fit}$ in K')
    
    A_ax=T_plot.ax.twinx()
    A_ax.scatter(df.reset_index()['V_SMU'],df.reset_index()['A_fit'],c='blue')
    A_ax.set_ylabel(r'$A_{fit}$')
    
   
    
    
    
    
    
         
         