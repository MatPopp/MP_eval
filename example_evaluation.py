import numpy as np
import filehandling as fh
import keyboard 

def read_data(datapath='data'):
    """this function shall contaion all the data reading and put all data into 
    a pandas DataFrame"""
    df=fh.data_from_directory('data',var_strings=['V_Piezo','V_SMU'], read_function=fh.spectrum_to_pd)  
    return(df)
    
def evaluate_data(df):
    """this function shall do numerical evaluations and add the evaluation
    results to DataFrame. Putting all results into df will help
    - in keeping the code clean, everything findeable
    - saving evaluation results before plotting"""
    
    
    
    return(df)
    
def visualize_data(df):
    """here, data shall be visualized and plotted"""
    return(df)


if __name__=='__main__':
    raw_data_df=read_data()
    evaluated_data_df = evaluate_data(raw_data_df)
    visualize_data(evaluated_data_df)