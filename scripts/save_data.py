import os
import sys
import gc
from poloidal_flow import ABESConfig, ABESDataReader
import flap
    
if __name__ == '__main__':
    
    exp_id = sys.argv[1]
    
    savepath = os.path.abspath(sys.argv[2])
        
    print('Reading ABES data')
    print('-----------------')
    print(f'Exp. ID {exp_id}')
    print(f'Path to save: {savepath}')
    
    config = ABESConfig(exp_id, spatial_cal = False, bandpass_type = 'Butterworth', bandpass_range = [2e3, 10e3])  
    defl0, defl1 = ABESDataReader(config).read_data()
    
    print('Saving')
    
    flap.save(defl0, os.path.join(savepath, f'{exp_id}_bandpass_defl0.pkl'))
    flap.save(defl1, os.path.join(savepath, f'{exp_id}_bandpass_defl1.pkl'))