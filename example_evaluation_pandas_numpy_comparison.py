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
    df=fh.data_from_directory(datapath, read_regex='_30s' ,var_strings=['V_Piezo','V_SMU'],
                              read_function=fh.spectrum_to_pd)  
    return(df)
    
def evaluate_data(df):
    """this function shall do numerical evaluations and add the evaluation
    results to DataFrame. Putting all results into df will help
    - in keeping the code clean, everything findeable
    - saving evaluation results before plotting"""
    
    
    ## load calibration function, correct for list in nm 
    calib_fun_nm = fh.get_calibration_function(path = r'calibration_confocal2\Newton_OIympus_20x_vis150l_907_C_Dichro_532_LP_572',
                                 regex = '.*calibration_combined.*')
    def calib_fun(lamb):
        return(calib_fun_nm(1e9*lamb))
    ## get all possible V_SMU values
    V_SMU_vals=df['V_SMU'].unique()
    
    ## use V_Piezo and V_SMU as index
    df=df.set_index(['V_Piezo','V_SMU']).sort_index()
    
    ## extract array-like counts and wavelengths from spectra
    df['wavelength'] = [np.array(spectrum.wavelength) for spectrum in df['data']]
    df['counts'] = [np.array(spectrum.counts) for spectrum in df['data']]
    ## remove cosmics
    df['counts'] = df['counts'].map(eval_func.remove_cosmics)
    
    ## use instrument calibration function
    calib_list=calib_fun(df['data'][0,0]['wavelength'])
    df['calibrated_counts'] = df['counts'].map(lambda counts: counts*calib_list)
     
    ## background correction (the background counts are the counts at V_SMU=0)
    dfu = df.unstack()
    for V in V_SMU_vals:
        dfu['corrected_counts',V] = dfu['calibrated_counts',V]-dfu['calibrated_counts',0]
    df=dfu.stack('V_SMU')
    
    
    ## fit black-body+Fabry-Perot_Model
    
    def fit_func(index):
        print('doing fit for V_Piezo,V_SMU=',index)
        counts = df['corrected_counts'][index]
        wavelength=df.data[0,0].wavelength
        n_Au_instance = ri.n_from_string('Au')
        n_Au = n_Au_instance.get_n(wavelength)
        n_SiC_instance = ri.n_from_string('SiC')
        n_SiC = n_SiC_instance.get_n(wavelength)
        p_names=['d','A','T']
        p_min_max_steps_dict = {'d':[2000e-9,3500e-9,20],'A':[0,40e-9,5],'T':[1500,2500,5]}
        
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
    df.to_pickle(save_directory+'/evaluated_pandas_dataframe.pkl')


def visualize_data(df):
    """here, data shall be visualized and plotted"""
    pass
    


if __name__=='__main__':
    
    df=read_data()
    try:
        df=pd.read_pickle('evaluated_data/evaluated_pandas_dataframe.pkl')
    except:
        df=evaluate_data(df)
    df['wavelength'] = [np.array(spectrum.wavelength) for spectrum in df['data']]
    visualize_data(df)
    
    
    
    ## plot example spectra for different applied SMU Voltage
    V_SMU_vals=df.index.unique(level=1)
    fig_Vdep,ax_Vdep=plt.subplots()
    V_Piezo_plot=15
    for V_SMU in V_SMU_vals:
        c=cm.hot((V_SMU-16)/(14))
        ax_Vdep.plot(df.wavelength[V_Piezo_plot,V_SMU]*1e9,df.corrected_counts[V_Piezo_plot,V_SMU],
                     c=c)
        ax_Vdep.grid(True)
    
    ax_Vdep.set_xlabel('wavelength in nm')
    ax_Vdep.set_ylabel('counts')
    
    fig_Vdep.savefig('V_dep_spectra.png')
        
    ## plot example spectra for different applied Piezo Voltage
    V_Piezo_vals=df.index.unique(level=0)
    fig_ddep,ax_ddep=plt.subplots()
    V_SMU_plot=24
    V_Piezo_start=1
    V_Piezo_stop=20
    
    for V_Piezo in V_Piezo_vals:
        if V_Piezo >= V_Piezo_start and V_Piezo <= V_Piezo_stop and V_Piezo%2==0:
            c=cm.jet(V_Piezo/V_Piezo_stop)
            ax_ddep.plot(df.wavelength[V_Piezo,V_SMU_plot]*1e9,df.corrected_counts[V_Piezo,V_SMU],
                         c=c)
            ax_ddep.grid(True)
    ax_ddep.set_xlabel('wavelength in nm')
    ax_ddep.set_ylabel('counts')
    
    fig_ddep.savefig('d_dep_spectra.png')
        
    ## plot calibrated intensity vs d 
    fig_intensity,ax_intensity=plt.subplots()
     
    V_SMU_plot=24
    intensity_plot=plot_func.colorplot(df.wavelength[:,V_SMU_plot]*1e9,
                        0*df.wavelength[:,V_SMU_plot]+df.d_fit[:,V_SMU_plot]*1e9,
                        df.corrected_counts[:,V_SMU_plot],
                        xlabel=r'$\lambda$ in nm',
                        ylabel=r'$d$ in nm')
    intensity_plot.fig.savefig('intensity.png')
    
         
         
     
     
    