import sys
sys.path.append('/allen/aibs/mat/RamIyer/pbstools/')
#sys.path.append('/allen/programs/braintv/workgroups/cortexmodels/rami/Research/pbstools/')
from pbstools import PythonJob
import os
import numpy as np
import pandas as pd
import subprocess
import time
 
dryrun = False # if True, prints what script would be to screen but DOESN'T RUN
conda_env = 'py36'

job_settings = {'queue': 'braintv',
                'mem': '10g',
                'priority': 'high',
                'walltime': '1:00:00',
                'ppn': 1#20
                }
#=======================================================================================================================

basedir = '/allen/programs/braintv/workgroups/cortexmodels/rami/Research/VC_NP_sklearn_mod'
units_savedir = os.path.join('/allen/programs/braintv/workgroups/cortexmodels/rami/Research/VC_NP_sklearn', 'units_dfs')

expt_id_list = ['715093703'] #['719161530','732592105'] #'847657808']#['719161530']#['715093703'] #,'719161530'
cstr_list = ['VISp']#,'VISl','LP','LGd','VISal','VISam','VISpm','VISrl','TH'] #'VISp'
prs_yaml_fname = os.path.join(basedir,'sg_a2a_full.yaml')
stim_scale_list = [1.0, 0.01, 0.05, 0.1, 0.5, 5.0, 10., 50., 0.0]


#dt = 0.005#0.025
#stim_list = ['static_gratings']#,'natural_movie_one','spontaneous','natural_scenes', 'drifting_gratings', 'natural_movie_three'] #',static_gratings']
#for stim_name in stim_list:

for expt_id in expt_id_list:
    
    out_err_dir = os.path.join(basedir,'logfiles', expt_id + '_logfiles_sga2a_t150_t250_c1_stim_allscales_May29')
    if not os.path.exists(out_err_dir):
        os.makedirs(out_err_dir, exist_ok=True)
    
    job_settings.update({
                        'outfile':os.path.join(out_err_dir,'$PBS_JOBID.out'),
                        'errfile':os.path.join(out_err_dir,'$PBS_JOBID.err'),
                        'email': 'rami@alleninstitute.org',
                        'email_options': 'a'
                        })
    jobname = 'sg_mod_aapc'

    units_savename = os.path.join(units_savedir,expt_id + '_units_df.pkl')
    units_df = pd.read_pickle(units_savename)

    jobcount = 0

    for cstr in cstr_list:
        visp_units = units_df[units_df.ecephys_structure_acronym ==cstr].index.values

        no_of_jobs = len(visp_units) * len(stim_scale_list)
        while (jobcount + no_of_jobs) >= 8000:
            time.sleep(30)
            out = subprocess.Popen(["qstat", "-u rami"], stdout=subprocess.PIPE)
            out2 = subprocess.Popen(["wc", "-l"], stdin=out.stdout, 
                                     stdout=subprocess.PIPE).communicate()[0].strip('\n')
            jobcount = int(float(out2))

        if len(visp_units) > 0:
            jobcount = jobcount + no_of_jobs
            for unit_id in visp_units:
                for stim_scale in stim_scale_list:
                    job_settings['jobname'] = '%s:%s:%s:%s:%s' % (jobname, expt_id, cstr, unit_id, stim_scale)
                        
                    PythonJob(os.path.join(basedir,'sandbox_pop_coupling_with_modules.py'),
                            python_args='--cstr=%s --expt_id=%s  --unit_id=%s --prs_yaml_fname=%s --stim_scale=%s' % (cstr, expt_id, unit_id, prs_yaml_fname, stim_scale),
                            conda_env=conda_env,
                            python_path=None,     
                            **job_settings).run(dryrun=dryrun)