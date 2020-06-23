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


def make_test_corr_box_plot(tmp1_df, fig_title, fig_savename = None):
    '''
    Function to make boxplot of test correlation values
    '''    
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


def make_stim_nnz_cpl_df_and_plot(tmp1_df, fig_savename = None):
    '''
    Function to plot heatmap of couplings with visual stimulus
    '''
    
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

def compute_cpl_matrix(tgt_df, src_cstr_lyr_df):
    '''
    Function to compute cpl_matrix
    '''
    src_units = src_cstr_lyr_df.unit_id.values
    cpl_matrix = []
    if len(src_units)>0:
        cpl_matrix = tgt_df[src_units].values.astype(float)
    return cpl_matrix

def plot_cpl_matrix_hm(tgt_cstr, tgt_df, src_cstr, src_cstr_lyr_df, fig_savename=None):
    '''
    Functions to make and plot coupling heatmap for given source and target cstr
    '''
    
    xv = np.where(np.diff(src_cstr_lyr_df.layer.values))[0]+0.5
    yv = np.where(np.diff(tgt_df.layer.values))[0]+0.5
    
    cpl_matrix = compute_cpl_matrix(tgt_df,src_cstr_lyr_df)
    src_units = src_cstr_lyr_df.unit_id.values

    if len(src_units)>0:
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


def compute_w(src_cstr_lyr_df, tgt_df):
    '''
    Function to compute eigenvalues/singular values of coupling matrix
    '''
    rec_mat = tgt_df[src_cstr_lyr_df.unit_id.values].fillna(0).values

    if rec_mat.shape[0] == rec_mat.shape[1]:
        w,v = np.linalg.eig(rec_mat)
    else:
        w1,v1 = np.linalg.eig(np.matmul(rec_mat, rec_mat.T))
        w = np.sqrt(w1)
    return w

def get_and_plot_eval_cpl_mat(src_cstr_lyr_df, tgt_df, fig_savename = None):
    '''
    Function to compute eigenvalues/singular values of coupling matrix and plot
    '''
    w = compute_w(src_cstr_lyr_df, tgt_df)
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