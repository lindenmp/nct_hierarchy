#!/usr/bin/env python
# coding: utf-8

# # Submit jobs to cubic

# In[ ]:


import os
import pandas as pd
import numpy as np
import subprocess
import json

# 1) basic setup
py_exec = '/cbica/home/parkesl/miniconda3/envs/pfactor_gradients/bin/python'

parc_scale = 400
connectome_spec = 'schaefer_'+str(parc_scale)+'_streamlineCount'

py_script = '/cbica/home/parkesl/research_projects/pfactor_gradients/1_code/cluster/compute_control_energy.py'

indir = '/cbica/home/parkesl/research_projects/pfactor_gradients/2_pipeline/0_get_sample/out/'
outputdir = '/cbica/home/parkesl/research_projects/pfactor_gradients/2_pipeline/2_compute_energy/out/'+connectome_spec
if not os.path.exists(outputdir): os.makedirs(outputdir)


# 2) subject ids
# subjids = ['disc_mean_A_s4', 'disc_mean_A_s5', 'disc_mean_A_s6', 'disc_mean_A_s7', 'disc_mean_A_s8']
# subjids = ['disc_mean_A_s6',]

df_file = indir+connectome_spec+'_df.csv'
df = pd.read_csv(df_file)
df.set_index(['bblid', 'scanid'], inplace = True)
df = df.loc[df['disc_repl'] == 0,:]
print(df.shape)


# 3) control params
T_list = [1,]; print(T_list)
B_list = ['x0xfwb',]; print(B_list)
rho_list = [1,]; print(rho_list)
n_subsamples = 20

gradients_file = '/cbica/home/parkesl/research_projects/pfactor_gradients/2_pipeline/1_compute_gradient/out/'+connectome_spec+'_pnc_grads_template.txt'
n_clusters = int(parc_scale/20); print(n_clusters)

# control_list = ['minimum', 'minimum_nonh', 'minimum_taylor']; print(control_list)
control_list = ['minimum','minimum_taylor']; print(control_list)

split_clusters = False


# ### Submit

# In[ ]:


for control in control_list:
    if control == 'minimum_taylor': mem_amount = '3'
    else: mem_amount = '1'

#     for subjid in subjids:
#         subjid_short = subjid
#         A_file = indir+connectome_spec+'_'+subjid+'.npy'
    for s in np.arange(df.shape[0]):
        subjid = str(df.iloc[s].name[0])+'_'+str(df.iloc[s].name[1])
        subjid_short = 's'+str(df.iloc[s].name[0])
        A_file = indir+connectome_spec+'_'+subjid+'_A.npy'
        
        for T in T_list:
            for B_ver in B_list:
                if split_clusters == False:
                    subprocess_str = '{0} {1} -subjid {2} -A_file {3} -outputdir {4} -control {5} -T {6} -B_ver {7} -n_subsamples {8} -gradients_file {9} -n_clusters {10}'                     .format(py_exec, py_script, subjid, A_file, outputdir, control, T, B_ver, n_subsamples, gradients_file, n_clusters)

                    name = subjid_short+'_'+control+str(T)+B_ver
                    qsub_call = 'qsub -N {0} -l h_vmem={1}G,s_vmem={1}G -pe threaded 1 -j y -b y -o /cbica/home/parkesl/sge/ -e /cbica/home/parkesl/sge/ '.format(name, mem_amount)

                    os.system(qsub_call + subprocess_str)
                elif split_clusters == True:
                    for i in np.arange(n_clusters):
                        for j in np.arange(n_clusters):
                            if i != j:
                                subprocess_str = '{0} {1} -subjid {2} -A_file {3} -outputdir {4} -control {5} -T {6} -B_ver {7} -n_subsamples {8} -gradients_file {9} -n_clusters {10} -i {11} -j {12}'                                 .format(py_exec, py_script, subjid, A_file, outputdir, control, T, B_ver, n_subsamples, gradients_file, n_clusters, i, j)

                                name = subjid_short+'_'+control+str(T)+B_ver+'_'+str(i)+str(j)
                                qsub_call = 'qsub -N {0} -l h_vmem={1}G,s_vmem={1}G -pe threaded 1 -j y -b y -o /cbica/home/parkesl/sge/ -e /cbica/home/parkesl/sge/ '.format(name, mem_amount)

                                os.system(qsub_call + subprocess_str)


# ### Assemble outputs

# In[ ]:


for subjid in subjids:
    for control in control_list:
        for T in T_list:
            for B_ver in B_list:
                file_label = subjid+'_'+control+'_T-'+str(T)+'_B-'+B_ver+'-g'+str(n_clusters)
#                 file_label = subjid+'_'+control+'_T-'+str(T)+'_B-'+B_ver+'_rho-'+str(rho)+'-g'+str(n_clusters)

                if control == 'minimum_taylor':
                    E_tmp = np.load(os.path.join(outputdir,file_label+'_E_01.npy'))
                    E = np.zeros((n_clusters, n_clusters, n_subsamples, E_tmp.shape[3]))
                else:
                    E = np.zeros((n_clusters, n_clusters, n_subsamples))
                
                for i in np.arange(n_clusters):
                    for j in np.arange(n_clusters):
                        if i != j:
                            E[i,j,:] = np.load(os.path.join(outputdir,file_label+'_E_i'+str(i)+'j'+str(j)+'.npy'))
                
                np.save(os.path.join(outputdir,file_label+'_E_ij'), E)

