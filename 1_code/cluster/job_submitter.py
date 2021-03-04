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


# # 2) Compute energy

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


# # 3) Compute adjacency matrix statistics

# In[ ]:


py_script = '/cbica/home/parkesl/research_projects/pfactor_gradients/1_code/cluster/compute_adj_stats.py'


# ## individuals

# In[ ]:


outputdir = '/cbica/home/parkesl/research_projects/pfactor_gradients/2_pipeline/3_compute_adj_stats/out/'+connectome_spec
if not os.path.exists(outputdir): os.makedirs(outputdir)


# In[ ]:


for i in np.arange(df.shape[0]):
    subjid = str(df.iloc[i].name[0])+'_'+str(df.iloc[i].name[1])
    subjid_short = str(df.iloc[i].name[0])
    A_file = '/cbica/home/parkesl/research_projects/pfactor_gradients/2_pipeline/0_get_sample/out/'+connectome_spec+'_'+subjid+'_A.npy'

    subprocess_str = '{0} {1} -subjid {2} -A_file {3} -gradients_file {4} -outputdir {5}'.format(py_exec, py_script, subjid, A_file, gradients_file, outputdir)

    name = 's_'+subjid_short
    qsub_call = 'qsub -N {0} -l h_vmem=1G,s_vmem=1G -pe threaded 1 -j y -b y -o /cbica/home/parkesl/sge/ -e /cbica/home/parkesl/sge/ '.format(name)

    os.system(qsub_call + subprocess_str) 


# # 4) Compute adjacency matrix statistics surrogates

# In[ ]:


py_script = '/cbica/home/parkesl/research_projects/pfactor_gradients/1_code/cluster/compute_adj_stats_gradient.py'


# ## mean adj.

# In[ ]:


storedir = '/cbica/home/parkesl/research_projects/pfactor_gradients/2_pipeline/4_compute_adj_stats_surrogates/store/'+connectome_spec
if not os.path.exists(storedir): os.makedirs(storedir)

centroids_file = '/cbica/home/parkesl/research_projects/pfactor_gradients/figs_support/labels/schaefer'+str(parc_scale)+'/Schaefer2018_'+str(parc_scale)+'Parcels_17Networks_order_FSLMNI152_1mm.Centroid_RAS.csv'
    
subjids = ['disc_mean_A_s6',]
surr_list = ['spatial_wsp_grad_cmni', 'spatial_wssp_grad_cmni', 'spatial_wwp_grad_cmni',
             'spatial_wsp_mni_cgrad', 'spatial_wssp_mni_cgrad','spatial_wwp_mni_cgrad',
             'spatial_wsp_hybrid', 'spatial_wssp_hybrid','spatial_wwp_hybrid']
# surr_list = ['standard', 'spatial_wwp', 'spatial_wsp', 'spatial_wssp', 'spatial_wwp_grad', 'spatial_wsp_grad', 'spatial_wssp_grad']
# surr_list = ['spatial_wwp_grad_cmni', 'spatial_wwp_mni_cgrad']
# surr_list = ['spatial_wsp_grad_cmni', 'spatial_wsp_mni_cgrad', 'spatial_wssp_grad_cmni', 'spatial_wssp_mni_cgrad']


# In[ ]:


for surr_type in surr_list:
    for subjid in subjids:
        A_file = indir+connectome_spec+'_'+subjid+'.npy'

        subprocess_str = '{0} {1} -subjid {2} -A_file {3} -gradients_file {4} -n_clusters {5} -outputdir {6} -surr_type {7} -centroids_file {8}'         .format(py_exec, py_script, subjid, A_file, gradients_file, n_clusters, storedir, surr_type, centroids_file)

        name = surr_type
        qsub_call = 'qsub -N {0} -l h_vmem=1G,s_vmem=1G -pe threaded 1 -j y -b y -o /cbica/home/parkesl/sge/ -e /cbica/home/parkesl/sge/ -t 1:10000 '.format(name)

        os.system(qsub_call + subprocess_str)


# ### Assemble outputs

# In[ ]:


outputdir = '/cbica/home/parkesl/research_projects/pfactor_gradients/2_pipeline/4_compute_adj_stats_surrogates/out/'+connectome_spec
if not os.path.exists(outputdir): os.makedirs(outputdir)

num_surrogates = 10000

for surr_type in surr_list:
    for subjid in subjids:
        tm_var_surr = np.zeros((n_clusters, n_clusters, num_surrogates))
        smv_var_surr = np.zeros((n_clusters, n_clusters, num_surrogates))
        joint_var_surr = np.zeros((n_clusters, n_clusters, num_surrogates))
        
        for surr_seed in np.arange(num_surrogates):
            file_label = subjid+'_'+surr_type+str(surr_seed)+'_grad'+str(n_clusters)
            adj_stats = np.load(os.path.join(storedir, file_label+'_adj_stats.npy'), allow_pickle = True)
            adj_stats = adj_stats.item()

            tm_var_surr[:,:,surr_seed] = adj_stats['tm_var']
            smv_var_surr[:,:,surr_seed] = adj_stats['smv_var']
            joint_var_surr[:,:,surr_seed] = adj_stats['joint_var']
        
        file_label = subjid+'_'+surr_type+'_grad'+str(n_clusters)
        np.save(os.path.join(outputdir, file_label+'_tm_var_surr'), tm_var_surr)
        np.save(os.path.join(outputdir, file_label+'_smv_var_surr'), smv_var_surr)
        np.save(os.path.join(outputdir, file_label+'_joint_var_surr'), joint_var_surr)


# # 5) Compute minimum energy taylor optimized for replication dataset

# In[ ]:


df_file = indir+connectome_spec+'_df.csv'
df = pd.read_csv(df_file)
df.set_index(['bblid', 'scanid'], inplace = True)
df = df.loc[df['disc_repl'] == 1,:]
print(df.shape)

n_taylor = 7


# In[ ]:


py_script = '/cbica/home/parkesl/research_projects/pfactor_gradients/1_code/cluster/compute_control_energy_taylor_optimized.py'


# In[ ]:


# drop_taylor_file = '/cbica/home/parkesl/research_projects/pfactor_gradients/3_output/results/out/Overall_Psychopathology_r_max_corr_taylor.npy'
# outputdir = '/cbica/home/parkesl/research_projects/pfactor_gradients/2_pipeline/5_compute_minimum_energy_taylor_optimized/out/'+connectome_spec+'/Overall_Psychopathology'
drop_taylor_file = '/cbica/home/parkesl/research_projects/pfactor_gradients/3_output/results/out/Overall_Psychopathology_mprage_antsCT_vol_TBV_dti64MeanRelRMS_r_max_corr_taylor.npy'
outputdir = '/cbica/home/parkesl/research_projects/pfactor_gradients/2_pipeline/5_compute_minimum_energy_taylor_optimized/out/'+connectome_spec+'/Overall_Psychopathology_mprage_antsCT_vol_TBV_dti64MeanRelRMS'
if not os.path.exists(outputdir): os.makedirs(outputdir)


# In[ ]:


for s in np.arange(df.shape[0]):
    subjid = str(df.iloc[s].name[0])+'_'+str(df.iloc[s].name[1])
    subjid_short = 's'+str(df.iloc[s].name[0])
    A_file = indir+connectome_spec+'_'+subjid+'_A.npy'

    for T in T_list:
        for B_ver in B_list:
            subprocess_str = '{0} {1} -subjid {2} -A_file {3} -outputdir {4} -T {5} -B_ver {6} -n_subsamples {7} -n_taylor {8} -drop_taylor_file {9} -gradients_file {10}'             .format(py_exec, py_script, subjid, A_file, outputdir, T, B_ver, n_subsamples, n_taylor, drop_taylor_file, gradients_file)

            name = subjid_short+'_'+str(T)+B_ver
            qsub_call = 'qsub -N {0} -l h_vmem=3G,s_vmem=3G -pe threaded 1 -j y -b y -o /cbica/home/parkesl/sge/ -e /cbica/home/parkesl/sge/ '.format(name)

            os.system(qsub_call + subprocess_str)

