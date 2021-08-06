import os
import pandas as pd
import numpy as np


def ensure_dir_exists(dir_name):
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)
        
def dat_to_pd(filepath):
    return(pd.read_csv(filepath,header=0,delimiter='\t'))
    
def spectrum_to_pd(filepath):
    spec_df=pd.read_csv(filepath,names=['wavelength','counts'],skiprows=4,delimiter='\t')
    spec_df['wavelength']=spec_df['wavelength']*1e-9
    return(spec_df)
        
class dat_to_object():

    def __init__(self,filepath,comment_discriminator='#'):
        self.filepath=filepath
        try:
            self.load_with_delimiter('\t')
        except:
            try:
                self.load_with_delimiter(',')
            except:
                try:
                    self.load_with_delimiter(' ')
                except ValueError:
                    print("could not read datafile with filepath: "+filepath)
    def load_with_delimiter(self,delimiter):
        file = open(self.filepath,'r')
        filedata = file.readlines()
        self.headers = filedata[0].split(delimiter)
        ## remeove \n at end of headers (artefact of saving textfiles with csv.writer
        #print(self.headers)
        for i in range(len(self.headers)):
        
            if self.headers[i].endswith('\n'):
                self.headers[i] = self.headers[i][:-1]

        ## lade Daten in dictionary
        self.data = {}
        for i in range(len(self.headers)):
            self.data[self.headers[i]]=[]

        for line in filedata[1:]:
            lspl=line.split(delimiter)
        
            for i in range(len(self.headers)):
                self.data[self.headers[i]].append(float(lspl[i]))
if __name__ == "__main__":
    ## do test cases of all functions
    ensure_dir_exists('dummy_directory')
    data_object=dat_to_object('test_data/LAP_Measurment_output.dat')
    data_object=dat_to_object('test_data/space_separated.dat')
    dataframe=dat_to_pd('test_data/LAP_Measurment_output.dat')
    spec_data=spectrum_to_pd('test_data/spectrum.txt')
    print(spec_data)
    input('test finished, press Enter to quit')
    
    