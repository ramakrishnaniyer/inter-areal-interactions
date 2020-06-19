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
Function to combine layer information with units dataframe
'''
def make_units_with_lyr_for_cstr(experiment_id, units_filename, layers_dir, cstr):
    
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


'''
Function to collect PC params files with max test correlation for all units in given cstr
'''

def copy_params_files_to_max_test_corr_dir(params_df_dir, max_fname_list):
    
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

'''
Function to return PC params filenames with maximum test correlation for different stimulus scales for given unit_id
'''
def get_scale_fnames_with_max_test_corr_wvis(prs_df_dir, expt_id, cstr, unit_id):
    
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

'''
Function to make dataframe from all params files in given directory
'''
def make_prs_df_from_given_dir(prs_df_dir, expt_id, cstr):
    tmp = pd.DataFrame()
    for fname in os.listdir(prs_df_dir):
        tmp_pd = pd.read_pickle(os.path.join(prs_df_dir,fname))
        tmp = pd.concat((tmp,tmp_pd), sort = False)
    
    tmp.index = tmp['unit_id'].values
    return tmp

'''
Function to make boxplot of test correlation values
'''

def make_test_corr_box_plot(tmp1_df, fig_title, fig_savename = None):
    
    max_test_corr_vals_2 = tmp1_df.test_corr.fillna(0).values

    fig = pylab.figure(figsize = (12,8))

    boxprops = dict(linestyle='-', linewidth=3, color='b')
    medianprops = dict(linestyle='-', linewidth=3, color='b')
    whiskerprops = dict(linestyle='-',linewidth=3, color='b')

    df=pd.DataFrame(np.array(max_test_corr_vals_2))
    box_plot=df.boxplot(boxprops=boxprops, whiskerprops = whiskerprops, medianprops = medianprops)

    x=1.9+0.2*np.random.rand(len(max_test_corr_vals_2))
    pylab.scatter(x,max_test_corr_vals_2,s=30)
    pylab.yticks()#(fontsize=30)
    print (box_plot)
    
    pylab.ylim([-0.1, 0.8])
    
    pylab.title(fig_title)#, fontsize = 30)#(expt_id+ '_' + cstr, fontsize = 30)
    if fig_savename is not None:
        pylab.savefig(fig_savename)
        
    pylab.show(block = False)


'''
Function to plot heatmap of couplings with visual stimulus
'''
def make_stim_nnz_cpl_df_and_plot(tmp1_df, fig_savename = None):
    
    yv = np.where(np.diff(tmp1_df.layer.values))[0]+0.5

    stim_cpl_df = tmp1_df[range(120)]
    stim_cpl_df_nnz = stim_cpl_df.loc[:, (stim_cpl_df != 0).any(axis=0)]
    xv = np.where(np.mod(1.+stim_cpl_df_nnz.columns.values.astype(float),20)==0.)[0]+0.5
    
    stim_cpl_matrix = stim_cpl_df_nnz.values.astype(float)
    max_cpl = np.amax(np.abs(stim_cpl_matrix))
    fig,ax = plt.subplots(1,1, figsize = (15,10))
    plt.imshow(stim_cpl_matrix, aspect='auto', cmap = 'bwr', vmin = -max_cpl, vmax = max_cpl)
    ax.set_yticks(range(stim_cpl_matrix.shape[0]))
    ax.set_yticklabels(tmp1_df.layer.values,fontsize=10,rotation=0)
    ax.set_xticks(range(stim_cpl_matrix.shape[1]))
    ax.set_xticklabels(stim_cpl_df_nnz.columns.values,fontsize=10,rotation=45)
    ax.vlines(xv, 0, 1, transform=ax.get_xaxis_transform(), color='g')
    ax.hlines(yv, 0, 1, transform=ax.get_yaxis_transform(), color='k')
    
    plt.colorbar()
    plt.tight_layout()
    if fig_savename is not None:
        plt.savefig(fig_savename)
    plt.show(block = False)

    
'''
Functions to make dataframe with numbers of non-zero couplings
'''
def make_nnz_cpl_df(units_df, visp_units, tmp1):
    
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

'''
Functions to make heatmap of numbers of non-zero couplings 
'''
def get_num_units_in_cstr(units_df):
    
    cstr_list = sorted(units_df.ecephys_structure_acronym.unique())
    num_units_in_cstr = pd.DataFrame(index = range(1), columns = cstr_list)
    for cstr_name in cstr_list:
        num_units = np.shape(units_df[units_df.ecephys_structure_acronym == cstr_name])[0]
        num_units_in_cstr[cstr_name] = num_units
    
    cstr_to_drop = [cc for cc in num_units_in_cstr.columns.values if num_units_in_cstr[cc].values < 2.]

    return num_units_in_cstr, cstr_to_drop

def normalize_and_plot_nnz_cpl_df(nnz_cpl_from_cstr_counts, units_df, visp_units, \
                                  tmp1_df, cols_to_keep = None, fig_savename=None):
    
    yv = np.where(np.diff(tmp1_df.layer.values))[0]+0.5
    
    cstr_list = sorted(units_df.ecephys_structure_acronym.unique())
    num_units_in_cstr_df, cstr_to_drop = get_num_units_in_cstr(units_df)
    norm_nnz_df = nnz_cpl_from_cstr_counts.iloc[:,:-3]/np.squeeze(num_units_in_cstr_df.values)
    norm_nnz_df = norm_nnz_df.drop(cstr_to_drop,axis=1)
    cstr_list.remove(cstr_to_drop[0])
    
    if cols_to_keep is not None:
        norm_nnz_df = norm_nnz_df[cols_to_keep]
        cstr_list = cols_to_keep
    
    norm_nnz = norm_nnz_df.values.astype(float)

    fig,ax = plt.subplots(1,1,figsize = (15,10))
    #im = plt.imshow(norm_nnz, aspect='auto',vmin=0,vmax = 1.0*np.amax(norm_nnz))
    im = plt.imshow(norm_nnz, aspect='auto',vmin= 0.0, vmax = 1.0, cmap = 'magma')
    ax.set_xticks(range(len(cstr_list)))
    ax.set_xticklabels(cstr_list,fontsize=30,rotation=45)
    ax.set_yticks(range(norm_nnz.shape[0]))
    ax.set_yticklabels(tmp1_df.layer.values,fontsize=10,rotation=0)
    plt.hlines(yv, -1.,len(cstr_list)-1, transform=ax.get_yaxis_transform(), color='g')

    cbar = fig.colorbar(im, shrink=0.99, aspect=20, fraction=.12,pad=.02)
    cbar.ax.tick_params(labelsize=30)

    fig_str = expt_id+ '_' + cstr + '_nnz_cpl'
    plt.title(fig_str, fontsize = 30)
    plt.tight_layout()
    if fig_savename is not None:
        plt.savefig(fig_savename)
        
    plt.show(block = False)

    
'''
Functions to make and plot coupling heatmap for given source and target cstr
'''
def plot_cpl_matrix_hm(tgt_cstr, tgt_df, src_cstr, src_cstr_lyr_df, fig_savename=None):
    
    xv = np.where(np.diff(src_cstr_lyr_df.layer.values))[0]+0.5
    yv = np.where(np.diff(tgt_df.layer.values))[0]+0.5
    
    src_units = src_cstr_lyr_df.unit_id.values
    if len(src_units)>0:
        cpl_matrix = tgt_df[src_units].values.astype(float)
        max_cpl = np.amax(np.abs(cpl_matrix))
        fig,ax = plt.subplots(1,1, figsize = (15,10))
        plt.imshow(cpl_matrix, aspect='auto', cmap = 'bwr', vmin = -max_cpl, vmax = max_cpl)
        
        ax.set_yticks(range(cpl_matrix.shape[0]))
        ax.set_yticklabels(tgt_df.layer.values,fontsize=10,rotation=0)
        
        ax.set_xticks(range(cpl_matrix.shape[1]))
        ax.set_xticklabels(src_cstr_lyr_df.layer.values,fontsize=10,rotation=0)
        
        ax.vlines(xv, 0, 1, transform=ax.get_xaxis_transform(), color='k')
        ax.hlines(yv, 0, 1, transform=ax.get_yaxis_transform(), color='k')
        plt.colorbar()
        
        plt.xlabel('From ' + src_cstr, fontsize = 30)
        plt.ylabel('To ' + tgt_cstr, fontsize = 30, rotation = 90)
        
        plt.tight_layout()
        if fig_savename is not None:
            plt.savefig(fig_savename)
            #plt.close()
        plt.show(block = False)
        
        return cpl_matrix

'''
Function to compute eigenvalues/singular values of coupling matrix and plot
'''
def get_and_plot_eval_cpl_mat(src_cstr_lyr_df, tgt_df, fig_savename = None):
    
    rec_mat = tgt_df[src_cstr_lyr_df.unit_id.values].fillna(0).values
    print(rec_mat.shape)
    if rec_mat.shape[0] == rec_mat.shape[1]:
        w,v = np.linalg.eig(rec_mat)
    else:
        w1,v1 = np.linalg.eig(np.matmul(rec_mat, rec_mat.T))
        w = np.sqrt(w1)

    plt.figure(figsize = (5,5))
    plt.plot(w.real,w.imag,'b.')
    plt.xlim([-0.1, 0.2])
    plt.ylim([-0.005, 0.005])
    plt.vlines(0,-0.005, 0.005)
    plt.xlabel('Re (w)')
    plt.ylabel('Im (w)')
    plt.tight_layout()
    if fig_savename is not None:
        plt.savefig(fig_savename)
        #plt.close()
    
    plt.show(block = False)
    
    return w  

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


