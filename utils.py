import numpy as np 


def load_stimulus_filtered_array(stim_arr_fname, stim_durn, dt):
    upsmp_fac = round(stim_durn/dt) 
    print(stim_durn, dt, upsmp_fac)
    stim_filt_arr = np.load(stim_arr_fname)
    print(stim_filt_arr.shape)

    stim_filt_arr_upsmp = np.repeat(stim_filt_arr.T, upsmp_fac, axis=1)

    return stim_filt_arr_upsmp
