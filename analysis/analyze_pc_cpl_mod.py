'''
Main steps involved:
1. For each group of prs .pkl files, first collect files with max test correlation : DONE
2. Make box plots of test correlation for units in each cstr
3. Add layer info to units where possible
4. Make coupling heatmaps from source areas to target area
5. Split couplings into positive and negative couplings
5. Count number of non-zero couplings from each area. Include layers where appropriate
6. symmetry measure of coupling matrix
7. statistics of cpl distributions

'''

import os, sys
import numpy as np
import pandas as pd
import pickle
import random

import itertools
from itertools import product
import json

from shutil import copyfile


def make_units_with_lyr_for_cstr(experiment_id, units_filename, layers_dir, cstr):
    '''
    Function to combine layer information with units dataframe
    '''
    
    cstr_lyr = pd.DataFrame()
    units_df = pd.read_pickle(units_filename)
    visp_units = units_df[units_df.ecephys_structure_acronym == cstr].index.values
    
    #layers_dir = '/Users/Ram/Dropbox/VC_NP_sklearn_copy/layer_est_from_Jung/'
    lyr_fname = os.path.join(layers_dir,experiment_id+'_layer.json')
    try:
        with open(lyr_fname) as datafile:
            data = json.load(datafile)
            
        lyr_df = pd.DataFrame(data.items(), index = data.keys())
        cstr_lyr = lyr_df.loc[list(units_df[units_df.ecephys_structure_acronym ==cstr].peak_channel_id.values.astype(str))]
        cstr_lyr.columns = ['peak_channel_id','layer']
        cstr_lyr['unit_id'] = visp_units
    except Exception as e:
        print(e)
        
    return units_df, cstr_lyr


def copy_params_files_to_max_test_corr_dir(params_df_dir, max_fname_list):
    '''
    Function to collect PC params files with max test correlation for all units in given cstr
    '''
    src_dir = params_df_dir
    dst_dir = os.path.join(params_df_dir,'max_test_corr')
    if not os.path.exists(dst_dir):
        os.makedirs(dst_dir)

    for max_fname in max_fname_list:
        src_fname = os.path.join(src_dir,max_fname)
        dst_fname = os.path.join(dst_dir, max_fname)
        if not os.path.isfile(dst_fname):
            #print('File already copied to max test corr dir')
        #else:
            copyfile(src_fname, dst_fname)
    
    return dst_dir

def get_scale_fnames_with_max_test_corr_wvis(prs_df_dir, expt_id, cstr, unit_id):
    '''
    Function to return PC params filenames with maximum test correlation 
    for different stimulus scales for given unit_id

    '''    
    separator = '_'
    
    cstr_scale = 1
    stim_scale_list = [0.0, 0.01, 0.05, 0.1, 0.5, 1.0, 5.0, 10.0, 50.0]
    tmp = pd.DataFrame()
    for stim_scale in stim_scale_list:
        
        fname = separator.join([str(unit_id), 'fr_scale',str(cstr_scale), \
                                                  'stim_scale', str(stim_scale)]) + '.pkl'
        
        if os.path.isfile(os.path.join(prs_df_dir,fname)):
            tmp_pd = pd.read_pickle(os.path.join(prs_df_dir,fname))
            tmp = pd.concat((tmp,tmp_pd),sort=False)
    
    max_ind = np.argmax(tmp.test_corr.values)
    max_scale = tmp.iloc[max_ind]['stim_scale']
    max_fname = separator.join([str(tmp.iloc[max_ind]['unit_id']), 'fr_scale',str(cstr_scale), \
                                                  'stim_scale', str(max_scale)]) + '.pkl' 
    
    return max_fname

def make_prs_df_from_given_dir(prs_df_dir, expt_id, cstr):
    '''
    Function to make dataframe from all params files in given directory
    '''
    tmp = pd.DataFrame()
    for fname in os.listdir(prs_df_dir):
        tmp_pd = pd.read_pickle(os.path.join(prs_df_dir,fname))
        tmp = pd.concat((tmp,tmp_pd), sort = False)
    
    tmp.index = tmp['unit_id'].values
    return tmp

def make_nnz_cpl_df(units_df, visp_units, tmp1):
    '''
    Functions to make dataframe with numbers of non-zero couplings
    '''
    
    cstr_list = sorted(units_df.ecephys_structure_acronym.unique())
    nnz_cpl_from_cstr_counts = pd.DataFrame(index = visp_units, columns = cstr_list + ['total_nnz','stim','layer'])
    regr_cols = [ii for ii in tmp1.columns if (isinstance(ii,int) or isinstance(ii,np.int64))]

    for ii, uid in enumerate(tmp1.unit_id.values):
        #nnz_cpl_from_cstr_counts.loc[uid]['total_nnz'] = tmp1.iloc[np.where(tmp1.unit_id == uid)[0], 11:].fillna(0).astype(bool).sum(axis=1).values[0]
        nnz_cpl_from_cstr_counts.loc[uid]['total_nnz'] = tmp1.iloc[np.where(tmp1.unit_id == uid)[0]][regr_cols].fillna(0).astype(bool).sum(axis=1).values[0]
        #tmp_nnz_vals_df = tmp1.iloc[np.where(tmp1.unit_id == uid)[0],8:].fillna(0).astype(bool)
        tmp_nnz_vals_df = tmp1.iloc[np.where(tmp1.unit_id == uid)[0]][regr_cols].fillna(0).astype(bool)
        cols = tmp_nnz_vals_df.columns
        for cstr_area in cstr_list:
            nnz_cols_all = cols[np.where(tmp_nnz_vals_df.iloc[0,:])]
            nnz_cols_cstr = nnz_cols_all[nnz_cols_all>120]
            nnz_counts = len(np.where(units_df.loc[nnz_cols_cstr].ecephys_structure_acronym==cstr_area)[0])
            nnz_cpl_from_cstr_counts.loc[uid][cstr_area] = nnz_counts
            nnz_cpl_from_cstr_counts.loc[uid]['layer'] = tmp1.iloc[np.where(tmp1.unit_id == uid)[0]]['layer'].values[0]

    nnz_cpl_from_cstr_counts['stim'] = nnz_cpl_from_cstr_counts['total_nnz'] - nnz_cpl_from_cstr_counts.iloc[:,:-3].sum(axis=1)
    nnz_cpl_from_cstr_counts = nnz_cpl_from_cstr_counts.dropna()
    #nnz_cpl_from_cstr_counts['layer'] = tmp1[nnz_cpl_from_cstr_counts.index.isin(tmp1.unit_id.values)]['layer'].values

    return nnz_cpl_from_cstr_counts

def get_num_units_in_cstr(units_df):
    '''
    Functions to make heatmap of numbers of non-zero couplings 
    '''
    
    cstr_list = sorted(units_df.ecephys_structure_acronym.unique())
    num_units_in_cstr = pd.DataFrame(index = range(1), columns = cstr_list)
    for cstr_name in cstr_list:
        num_units = np.shape(units_df[units_df.ecephys_structure_acronym == cstr_name])[0]
        num_units_in_cstr[cstr_name] = num_units
    
    cstr_to_drop = [cc for cc in num_units_in_cstr.columns.values if num_units_in_cstr[cc].values < 2.]

    return num_units_in_cstr, cstr_to_drop
