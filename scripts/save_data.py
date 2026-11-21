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
    
    #config = ABESConfig(exp_id, spatial_cal = False, bandpass_type = 'Butterworth', bandpass_range = [2e3, 10e3])
    config = ABESConfig(exp_id, spatial_cal = False, bandpass_type = None)  
    defl0, defl1 = ABESDataReader(config).read_data()
    
    defl0_d = defl0.detrend('Time', options = {'Trend removal': 'Mean'})
    defl1_d = defl1.detrend('Time', options = {'Trend removal': 'Mean'})
    
    print('Saving')
    
    flap.save(defl0_d, os.path.join(savepath, f'{exp_id}_detrend_defl0.pkl'))
    flap.save(defl1_d, os.path.join(savepath, f'{exp_id}_detrend_defl1.pkl'))