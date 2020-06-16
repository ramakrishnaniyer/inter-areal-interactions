import numpy as np
import os,sys
import yaml
import pandas as pd
import time
import random
import argparse

#import allensdk
#from allensdk.brain_observatory.ecephys.ecephys_project_cache import EcephysProjectCache
from allensdk.brain_observatory.ecephys.ecephys_session import EcephysSession

from fit_methods import fit_methods
from fr_transform_methods import fr_transforms

from utils import load_stimulus_filtered_array
#from pop_cpl_util_funcs import load_stimulus_filtered_array, fit_lasso

class PC_sandbox:
    def __init__(self, yaml_fname = 'sg_a2a_full.yaml', expt_id = 715093703, cstr = 'VISp', unit_id = 950930407, stim_scale = 1):
        
        config = yaml.load(open(yaml_fname, 'r'), Loader=yaml.FullLoader)
        self.expt_id = str(expt_id) #str(config['expt_id'])
        self.data_dir = config['data_dir']
        self.nwb_path = os.path.join(self.data_dir,'ecephys_session_'+ self.expt_id + '.nwb')
        self.yaml_fname = yaml_fname
        
        self.cstr = cstr #config['cstr']
        self.unit_id = unit_id #config['unit_id']

        self.frac = config['frac']
        self.stim_name = config['stim_name']
        #self.dt = config['dt']
        
        self.stim_scale = stim_scale#config['stim_scale']
        self.cstr_scale = config['cstr_scale']
        
        self.fit_type = config['fit_type']
        self.glm_method = config['glm_method']
        
        self.bin_start = config['bin_start']
        self.bin_end = config['bin_end']
        self.bin_dt = config['bin_dt']
        self.bin_edges = np.arange(self.bin_start, self.bin_end + 0.001, self.bin_dt)
        #print(self.bin_start, self.bin_end, self.bin_edges, len(self.bin_edges))
        self.prs_savedir =  os.path.join(config['prs_savedir'], self.expt_id, self.cstr, \
            self.stim_name, 't1_'+str(self.bin_start) + '_t2_'+str(self.bin_end) + '_dt_'+str(self.bin_dt))

        if not os.path.exists(self.prs_savedir):
            os.makedirs(self.prs_savedir)
        
        separator = '_'
        prs_save_str = prs_save_str = separator.join([str(self.unit_id), 'fr_scale',str(self.cstr_scale), 'stim_scale', str(self.stim_scale)]) + '.pkl'
        self.prs_savename = os.path.join(self.prs_savedir,prs_save_str)
        
        if not os.path.isfile(self.prs_savename):
            ##### MAIN STEPS
            self.load_session_and_get_fr_df()
            
            self.make_output_vector()
            print(self.tot_fr_df.shape)
            fr_tr_methods = fr_transforms(self.tot_fr_df, 
                                          transform_method="identity",
                                          cstr_scale=self.cstr_scale,
                                          fit_type=self.fit_type)
            self.X_cstr = fr_tr_methods.make_cstr_input_matrix(unit_id=self.unit_id,
                                                               y_shape=self.y.shape)
            #self.make_cstr_input_matrix()
            self.make_stim_input_matrix()
            
            self.input_X = np.concatenate((self.X_cstr, self.X_stim.T), axis=1)
            print('Shape of input is :', self.input_X.shape)

            self.get_prs_df_col_names()
            self.get_glm_fit_prs_new()
            #################
        else:
            print('Params file for this unit already exists')

    def get_prs_df_col_names(self):
        self.col_names = ['sim_prs_yml','unit_id','alpha_val','mse','nnz_coef','cstr_scale','stim_scale','train_corr','test_corr']
        if self.cstr_scale != 0: 
            self.col_names = self.col_names + list(self.tot_fr_df.T.columns.values)
        if self.stim_scale != 0:
            self.col_names = self.col_names + list(range(self.X_stim.shape[0]))   

    def load_session_and_get_fr_df(self):
        session = EcephysSession.from_nwb_path(self.nwb_path, api_kwargs={
        "amplitude_cutoff_maximum": 0.1,
        "presence_ratio_minimum": 0.9,
        "isi_violations_maximum": 0.5
            })

        self.units_df = session.units
        self.stim_table = session.get_stimulus_table(self.stim_name)
        stim_pres_ids = self.stim_table.index.values
        tmp_binned_spt = session.presentationwise_spike_counts\
            (bin_edges = self.bin_edges, stimulus_presentation_ids=stim_pres_ids, unit_ids = session.units.index.values)

        num_pres,num_bins,num_cells = tmp_binned_spt.shape
        print(num_pres,num_bins,num_cells)

        tot_arr_fr_all = np.reshape(tmp_binned_spt.values, (num_pres*num_bins,num_cells))
        self.tot_fr_df = pd.DataFrame(tot_arr_fr_all.T, index = session.units.index.values)

        del tmp_binned_spt, session
        
    def make_output_vector(self):
        self.y = self.tot_fr_df.T[self.unit_id].values.astype(float)
    '''
    def make_cstr_input_matrix(self):
        self.X_cstr = np.array([]).reshape(self.tot_fr_df.shape[1],0)
        if self.cstr_scale != 0:
            df_X = self.tot_fr_df.T.copy()
            if self.fit_type == 'all_to_all':
                df_X[self.unit_id] = np.zeros(self.y.shape)
                self.X_cstr = self.cstr_scale*np.array(df_X)

            del df_X
        print('Cstr shape is :', self.X_cstr.shape)
    '''

    def make_stim_input_matrix(self):
        stim_arr_fname =  '/allen/programs/braintv/workgroups/cortexmodels/rami/Research/VC_NP_sklearn/gabor_filtered_'+self.stim_name+'_stim_corr.npy'
        self.X_stim = np.array([]).reshape(0,self.tot_fr_df.shape[1])
        if self.stim_scale != 0:
            stim_durn = self.bin_end - self.bin_start
            self.X_stim = self.stim_scale*load_stimulus_filtered_array(stim_arr_fname, stim_durn, self.bin_dt)
        print('Stim shape is :', self.X_stim.shape)

    def get_glm_fit_prs_new(self):
        
        print('Starting to analyze unit ',  self.unit_id)

        print('Maximum input is: ', np.amax(self.input_X))
        prs_df_unit = pd.DataFrame(index=range(1),columns = self.col_names)

        fitlasso = fit_methods(self.input_X, self.y, frac=self.frac, cv=10)

        alpha_val, train_corr, true_test_corr, params, nnz_coef, mse = fitlasso.fit()
        print('Finished analyzing unit ',  self.unit_id)
        print(self.unit_id, train_corr, true_test_corr, nnz_coef, mse)
        print('Optimal alpha_val: ',alpha_val)
        
        prs_df_unit.iloc[0]['sim_prs_yml'] = self.yaml_fname
        prs_df_unit.iloc[0]['unit_id'] = self.unit_id
        prs_df_unit.iloc[0]['alpha_val'] = alpha_val
        prs_df_unit.iloc[0]['mse'] = mse
        prs_df_unit.iloc[0]['nnz_coef'] = nnz_coef
        prs_df_unit.iloc[0]['cstr_scale'] = self.cstr_scale
        prs_df_unit.iloc[0]['stim_scale'] = self.stim_scale
        prs_df_unit.iloc[0]['train_corr'] = train_corr
        prs_df_unit.iloc[0]['test_corr'] = true_test_corr
        prs_df_unit.iloc[0,9:] = params

        print('Number of non-zero stim coeffs: ', np.count_nonzero(prs_df_unit.iloc[0,-120:].values))
        print(np.amax(prs_df_unit.iloc[0,-120:].values))
        print(np.amin(prs_df_unit.iloc[0,-120:].values))

        prs_df_unit.to_pickle(self.prs_savename)
   
if __name__ == '__main__':

    random.seed(9)

    # Parse arguments:
    parser = argparse.ArgumentParser(description='Number of clusters')

    parser.add_argument('--expt_id', type=str, required=True)
    parser.add_argument('--cstr', type = str, required = True)
    parser.add_argument('--unit_id', type = str, required = True)
    parser.add_argument('--prs_yaml_fname', type = str, required = True)
    parser.add_argument('--stim_scale', type = float, required = True)

    #Parse all arguments provided
    args = parser.parse_args()
    eid = args.expt_id
    uid = int(args.unit_id)
    cstr = args.cstr
    yaml_fname = args.prs_yaml_fname
    stim_scale = args.stim_scale

    sim_stt = time.time()
    pc = PC_sandbox(yaml_fname = yaml_fname, expt_id = eid, cstr = cstr, unit_id = uid, stim_scale = stim_scale)
    sim_endt = time.time()

    print('Time taken to fit: ', sim_endt-sim_stt)