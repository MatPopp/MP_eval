import numpy as np
import filehandling as fh
import evaluation_functions as eval_func
import keyboard 
import pandas as pd
import matplotlib.pyplot as plt

def read_data(datapath='data'):
    """this function shall contaion all the data reading and put all data into 
    a pandas DataFrame"""
    df=fh.data_from_directory('data', read_regex='_30s' ,var_strings=['V_Piezo','V_SMU'],
                              read_function=fh.spectrum_to_pd)  
    return(df)
    
def evaluate_data(df):
    """this function shall do numerical evaluations and add the evaluation
    results to DataFrame. Putting all results into df will help
    - in keeping the code clean, everything findeable
    - saving evaluation results before plotting"""
    
    fh.get_calibration_function(path = r'calibration_confocal2\Newton_OIympus_20x_vis150l_907_C_Dichro_532_LP_572',
                                 regex = '.*calibration_combined.*')
    ## extract array-like counts from spectra
    df['counts'] = [np.array(spectrum.counts) for spectrum in df['data']]
    ## remove cosmics
    df['counts'] = [eval_func.remove_cosmics(counts) for counts in df['counts']]
    
    ## correct background
    if_p
    V_SMU_vals=df['V_SMU'].unique()
    df_multi=df.set_index(['V_Piezo','V_SMU']).sort_index()
    
    unstacked = df_multi.unstack()
    for V in V_SMU_vals:
        unstacked['corrected_counts',V] = unstacked['counts',V]-unstacked['counts',0]
    restacked=unstacked.stack('V_SMU')
    
    
    df_multi['']
    
    
    return(df,restacked)

    
def visualize_data(df):
    """here, data shall be visualized and plotted"""
    return(df,test)


if __name__=='__main__':
    df=raw_data_df=read_data()
    
    fh.get_calibration_function(path = r'calibration_confocal2\Newton_OIympus_20x_vis150l_907_C_Dichro_532_LP_572',
                                 regex = '.*calibration_combined.*')
    ## extract array-like counts from spectra
    df['counts'] = [np.array(spectrum.counts) for spectrum in df['data']]
    ## remove cosmics
    df['counts'] = df['counts'].map(eval_func.remove_cosmics)
    
    
    
    ## correct background pandas-like
    if False:
        V_SMU_vals=df['V_SMU'].unique()
        df=df.set_index(['V_Piezo','V_SMU']).sort_index()
        
        dfu = df.unstack()
        for V in V_SMU_vals:
           dfu['corrected_counts',V] = dfu['counts',V]-dfu['counts',0]
        df=dfu.stack('V_SMU')
        
    ## correct background pandas-like
    if True:
        V_SMU_vals=df['V_SMU'].unique()
        df=df.set_index(['V_Piezo','V_SMU']).sort_index()
        backgrounds = df['counts'][:,0]
        V_SMU_list=df.index.unique(level=1)
        corr_spec_list=[]
        for V_SMU in V_SMU_list:
            print(V_SMU)
            raw_specs = df['counts'][:,V_SMU]
            corr_spec_series=raw_specs-backgrounds
            corr_spec_df=pd.DataFrame(corr_spec_series)
            corr_spec_df = corr_spec_df.rename(columns={'counts':'corrected_counts'})
            corr_spec_df['V_SMU']=V_SMU
            corr_spec_df.reset_index(level=0,inplace = True)
            corr_spec_df=corr_spec_df.set_index(['V_Piezo','V_SMU']).sort_index()
            corr_spec_list.append(corr_spec_df)
        corr_spec_df = pd.concat(corr_spec_list)
        concat = pd.concat([df,corr_spec_df],axis=1)
            
            
        corrected_counts=pd.concat([df.loc[(slice(None),V_SMU),'counts']-backgrounds for V_SMU in df.index.unique(level=1)])
        df['corrected_counts']=corrected_counts
        
    ## correct background numpy-like
    if False:
        df=df.set_index(['V_Piezo','V_SMU']).sort_index()
        backgrounds = df['counts'][df['filepath'].str.contains('background').values]
        spectra_list=[]
        V_SMU_list=[]
        for i in range(len(df.index.unique(level=1))):
            spectra_list.append()
            V_SMU_list.append()
        spectra = [df['counts'][:,V_SMU].values for V_SMU in df.index.unique(level=1)]
        spectra = [spectrum - backgrounds for spectrum in spectra]
        
        
    
   # visualize_data(evaluated_data_df)