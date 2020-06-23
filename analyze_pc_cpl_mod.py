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
import matplotlib.pyplot as plt
import matplotlib.gridspec as gs
from mpl_toolkits.axes_grid1 import make_axes_locatable
import pylab

import itertools
from itertools import product
import json

from shutil import copyfile

SMALL_SIZE = 20
MEDIUM_SIZE = 30
BIGGER_SIZE = 40

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title


'''
Function to split coupling matrix by src and tgt layers and +ve/-ve couplings and obtain mean 'strength' and 'nsyn' 
'''

def get_mean_nnz_cpl_by_lyr(src_tmp, tgt_tmp):
    mean_lyr_pos = np.zeros((8,8))
    nnz_lyr_pos = np.zeros((4,4))
    nnz_lyr_neg = np.zeros((4,4))

    for jj,src_lyr in enumerate([2,4,5,6]):
        src_units = src_tmp[src_tmp.layer==src_lyr].unit_id.values
        tmp_df_main = tgt_tmp[list(src_units)]
        for ii,tgt_lyr in enumerate([2,4,5,6]):
            tmp_df = tmp_df_main.iloc[np.where(tgt_tmp.layer==tgt_lyr)[0],:]
            nsyn_tot = len(~np.isnan(tmp_df.values.astype(float).flatten()))#tmp_df.shape[0]*tmp_df.shape[1]

            mean_lyr_pos[ii,jj] = np.nanmean(tmp_df[tmp_df>0].values.astype(float).flatten())
            mean_lyr_pos[ii+4,jj+4] = np.nanmean(tmp_df[tmp_df<0].values.astype(float).flatten())

            nnz_pos = np.count_nonzero(~np.isnan(tmp_df[tmp_df>0].values.astype(float).flatten()))
            nnz_neg = np.count_nonzero(~np.isnan(tmp_df[tmp_df<0].values.astype(float).flatten()))
            nnz_lyr_pos[ii,jj] = nnz_pos/nsyn_tot
            nnz_lyr_neg[ii,jj] = nnz_neg/nsyn_tot
            
    return mean_lyr_pos, nnz_lyr_pos, nnz_lyr_neg

'''
Functions to plot heatmap of connection probability
'''
def make_plot_on_gs(arr, fig, gs0, cmap_val = 'magma', max_val = None):
    ax0 = fig.add_subplot(gs0)
    if max_val is not None:
        im0 = ax0.imshow(arr, cmap = cmap_val, vmin = 0, vmax = max_val)
    else:
        im0 = ax0.imshow(arr, cmap = cmap_val)
        
    divider = make_axes_locatable(ax0)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    fig.colorbar(im0, cax=cax, orientation='vertical')
    
    ax0.set_xticks([0,1,2,3])
    ax0.set_yticks([0,1,2,3])
    
    ax0.set_xticklabels(['2/3','4','5','6'])
    ax0.set_yticklabels(['2/3','4','5','6'])

def plot_hm_conn_prob(mean_lyr_pos, nnz_lyr_pos, nnz_lyr_neg, fig_savename = None):

    mv_wts_pos = np.nanmax(np.abs(mean_lyr_pos[:4,:4]))
    mv_wts_neg = np.nanmax(np.abs(mean_lyr_pos[4:,4:]))

    mv_nsyn_pos = np.nanmax(np.abs(nnz_lyr_pos))
    mv_nsyn_neg = np.nanmax(np.abs(nnz_lyr_neg))
    
    conn_pos = np.multiply(mean_lyr_pos[:4,:4],nnz_lyr_pos)#mean_lyr_pos[:4,:4]*nnz_lyr_pos
    conn_neg = np.multiply(np.abs(mean_lyr_pos[4:,4:]),nnz_lyr_neg)#np.abs(mean_lyr_pos[4:,4:])*nnz_lyr_neg 
    
    mv_w = np.amax([mv_wts_pos, np.abs(mv_wts_neg)])
    mv_n = np.amax([mv_nsyn_pos, np.abs(mv_nsyn_neg)])
    mv_prob = np.amax([np.nanmax(conn_pos), np.nanmax(conn_neg)])

    fig = plt.figure(figsize = (20,10))
    gs = fig.add_gridspec(2, 3, figure=fig, width_ratios=[1., 1., 1.])

    arr_pos_list = [mean_lyr_pos[:4,:4], nnz_lyr_pos, conn_pos]
    #mval_pos_list = [mv_wts_pos, mv_nsyn_pos,None]

    arr_neg_list = [np.abs(mean_lyr_pos[4:,4:]), nnz_lyr_neg, conn_neg]
    #mval_neg_list = [mv_wts_neg, mv_nsyn_neg, None]
    
    mval_list = [mv_w, mv_n, mv_prob]

    for ii, arr1, mv in zip(range(3), arr_pos_list, mval_list):
        make_plot_on_gs(arr1, fig, gs[0,ii], cmap_val = 'magma', max_val = mv)

    for ii, arr2, mv in zip(range(3), arr_neg_list, mval_list):
        make_plot_on_gs(arr2, fig, gs[1,ii], cmap_val = 'magma', max_val = mv)
        
    if fig_savename is not None:
        plt.savefig(fig_savename)
    
#######################################################################################################################    
    
if __name__ == '__main__':
    
    # Provide initial variables
    units_df_basedir = '/Users/Ram/Dropbox/VC_NP_sklearn_copy/units_dfs'
    layers_dir = '/Users/Ram/Dropbox/VC_NP_sklearn_copy/layer_est_from_Jung'
    basedir = '/Users/Ram/Dropbox/VC_NP_sklearn_mod/save_prs_five_epochs'
    stim_name = 'static_gratings'
   
    #bin_start = 0.2 #0.05
    #bin_end = 0.25 #0.15
    bin_dt = 0.01
    
    bin_start_list = [(round(ii,2)) for ii in np.arange(0.000,0.201,0.050)]
    bin_end_list = [(round(ii,2)) for ii in np.arange(0.050,0.251,0.050)]
    
    ### ====== Everything above can be rolled in to already existing .yaml config files and loaded

    expt_id = '715093703' #['719161530']#['715093703'] #,'719161530'
    cstr = 'VISl' #['LGd','LP','VISam','VISpm','VISp','VISl','VISrl']
    
    ### ==== The above two variables can be looped over using a for loop
    
    for (bin_start, bin_end) in zip(bin_start_list, bin_end_list):
        
        print(expt_id, cstr, bin_start, bin_end)
        
        ## START MAIN SCRIPT ##

        #Get units df for specified cstr from session and combine with layer information
        units_fname = os.path.join(units_df_basedir,expt_id + '_units_df.pkl')
        units_df, cstr_lyr_df = make_units_with_lyr_for_cstr(expt_id, units_fname, layers_dir, cstr)

        #Get unit IDs from units_df
        visp_units = units_df[units_df.ecephys_structure_acronym == cstr].index.values

        #Specify directory containing parameters for specified cstr and time intervals
        prs_df_dir =  os.path.join(basedir, expt_id, cstr, \
           stim_name, 't1_'+str(bin_start) + '_t2_'+str(bin_end) + '_dt_'+str(bin_dt))

        #Make figures directory inside directory containing parameters
        figs_savedir = os.path.join(basedir, expt_id, cstr, stim_name, 'Figures')
        #if not os.path.exists(figs_savedir):
        #    os.makedirs(figs_savedir)

        #Collect all filenames with maximum test correlation among stim scales
        max_fname_list = []
        for unit_id in visp_units:
            try:
                max_fname = get_scale_fnames_with_max_test_corr_wvis(prs_df_dir, expt_id, cstr, unit_id)
            except:
                print('prs .pkl does not exist for: ',unit_id)
                pass
                
            max_fname_list.append(max_fname)

        max_test_corr_dir = copy_params_files_to_max_test_corr_dir(prs_df_dir, max_fname_list)

        #Make dataframe with parameters from maximum test correlations' files
        prs_df_with_lyr = make_prs_df_from_given_dir(max_test_corr_dir, expt_id, cstr)

        #Merge PC params dataframe with units/layers dataframe
        tmp = pd.merge(cstr_lyr_df, prs_df_with_lyr, on=['unit_id'])
        tmp1 = tmp.sort_values('layer')

        #Make test correlation boxplot
        fig_title = expt_id + '_' + cstr + '_wvis'
        
        fig_dir = os.path.join(figs_savedir, 'test_corr')
        if not os.path.exists(fig_dir):
            os.makedirs(fig_dir)
        fig_summary = os.path.join(fig_dir, 't1_'+str(bin_start) + '_t2_'+str(bin_end) + '_dt_'+str(bin_dt) +  '_wvis_summary.png')
        
        #if not os.path.isfile(fig_summary):
        make_test_corr_box_plot(tmp1, fig_title, fig_savename = fig_summary)

        #Make stimulus coupling heatmap
        fig_dir = os.path.join(figs_savedir, 'stim_cpl')
        if not os.path.exists(fig_dir):
            os.makedirs(fig_dir)     
        fig_stim_nnz = os.path.join(fig_dir, 't1_'+str(bin_start) + '_t2_'+str(bin_end) + '_dt_'+str(bin_dt) + '_stim_cpl.png')
        if not os.path.isfile(fig_stim_nnz):
            make_stim_nnz_cpl_df_and_plot(tmp1,fig_savename = fig_stim_nnz)

#         #Make heatmap of NUMBER of non-zero cstr couplings
#         fig_cstr_nnz = os.path.join(figs_savedir, 't1_'+str(bin_start) + '_t2_'+str(bin_end) + '_dt_'+str(bin_dt) + '_wvis_nnz_cpl.png')
        
#         if not os.path.isfile(fig_cstr_nnz):
#             nnz_cpl_from_cstr_counts = make_nnz_cpl_df(units_df, visp_units, tmp1)
#             normalize_and_plot_nnz_cpl_df(nnz_cpl_from_cstr_counts,units_df,visp_units,tmp1,\
#                                        cols_to_keep = None, fig_savename = fig_cstr_nnz)

        #Make heatmaps of intra and inter-areal coupling matrices
        src_cstr = 'VISp'
        src_units_df, src_cstr_lyr_df_unsrt = make_units_with_lyr_for_cstr(expt_id, units_fname, layers_dir, src_cstr)
        src_cstr_lyr_df = src_cstr_lyr_df_unsrt.sort_values('layer')
        
        fig_dir = os.path.join(figs_savedir, 'cstr_cpl', src_cstr + '_to_' + cstr)
        if not os.path.exists(fig_dir):
            os.makedirs(fig_dir)
        fig_hm_pc = os.path.join(fig_dir, 't1_'+str(bin_start) + '_t2_'+str(bin_end) + '_dt_'+str(bin_dt) + '_' + src_cstr + '_to_' + cstr + '.png')
        if not os.path.isfile(fig_hm_pc):     
            cpl_mat = plot_cpl_matrix_hm(cstr, tmp1, src_cstr, src_cstr_lyr_df, fig_savename = fig_hm_pc)
        
        #Plot eigenvalues of coupling matrices
        fig_dir = os.path.join(figs_savedir, 'cstr_eig_val', src_cstr + '_to_' + cstr)
        if not os.path.exists(fig_dir):
            os.makedirs(fig_dir)
        fig_eval_pc = os.path.join(fig_dir, 't1_'+str(bin_start) + '_t2_'+str(bin_end) + '_dt_'+str(bin_dt) + '_' + src_cstr + '_to_' + cstr + '_eig_val.png')
        if not os.path.isfile(fig_eval_pc):
            w = get_and_plot_eval_cpl_mat(src_cstr_lyr_df, tmp1, fig_savename = fig_eval_pc)
            
        #Obtain and plot heatmaps of connection probability between laminae
        mean_lyr_pos, nnz_lyr_pos, nnz_lyr_neg = get_mean_nnz_cpl_by_lyr(src_cstr_lyr_df, tmp1)
        fig_dir = os.path.join(figs_savedir, 'conn_prob', src_cstr + '_to_' + cstr)
        if not os.path.exists(fig_dir):
            os.makedirs(fig_dir)
        fig_conn_prob_pc = os.path.join(fig_dir, 't1_'+str(bin_start) + '_t2_'+str(bin_end) + '_dt_'+str(bin_dt) + '_' + src_cstr + '_to_' + cstr + '_conn_prob.png')
        if not os.path.isfile(fig_conn_prob_pc):
            plot_hm_conn_prob(mean_lyr_pos, nnz_lyr_pos, nnz_lyr_neg, fig_savename = fig_conn_prob_pc)


