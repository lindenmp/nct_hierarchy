#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Essentials
import os, sys, glob
import pandas as pd
import numpy as np
import nibabel as nib
import scipy.io as sio
from tqdm import tqdm

# Stats
import scipy as sp
from scipy import stats
import statsmodels.api as sm
import pingouin as pg

# Plotting
import seaborn as sns
import matplotlib.pyplot as plt
plt.rcParams['svg.fonttype'] = 'none'


# In[2]:


from brainspace.datasets import load_group_fc, load_parcellation, load_conte69
from brainspace.plotting import plot_hemispheres
from brainspace.gradient import GradientMaps
from brainspace.utils.parcellation import map_to_labels
from brainspace.gradient.utils import dominant_set


# In[3]:


sys.path.append('/Users/lindenmp/Google-Drive-Penn/work/research_projects/pfactor_gradients/1_code/')
from func import set_proj_env


# In[4]:


parc_str = 'schaefer'
parc_scale = 200
edge_weight = 'streamlineCount'
set_proj_env()


# In[5]:


# output file prefix
outfile_prefix = parc_str+'_'+str(parc_scale)+'_'
outfile_prefix


# In[6]:


# Parcellation specifications
if parc_str == 'schaefer':
    # Names of parcels
    parcel_names = np.genfromtxt(os.path.join(os.environ['PROJDIR'], 'figs_support/labels/schaefer' + str(parc_scale) + 'NodeNames.txt'), dtype='str')
    num_parcels = parcel_names.shape[0]
    
    scdir = os.path.join(os.environ['DERIVSDIR'], 'processedData/diffusion/deterministic_20171118'); os.environ['SCDIR'] = scdir
    sc_name_tmp = 'bblid/*xscanid/tractography/connectivity/bblid_*xscanid_SchaeferPNC_' + str(parc_scale) + '_dti_' + edge_weight + '_connectivity.mat'

    rstsdir = os.path.join(os.environ['DERIVSDIR'], 'processedData/restbold/restbold_201607151621')
    if parc_scale == 200:
        rsts_name_tmp = 'bblid/*xscanid/net/Schaefer' + str(parc_scale) + 'PNC/bblid_*xscanid_Schaefer' + str(parc_scale) + 'PNC_ts.1D'
    elif parc_scale == 400:
        rsts_name_tmp = 'bblid/*xscanid/net/SchaeferPNC/bblid_*xscanid_SchaeferPNC_ts.1D'


# ### Setup directory variables

# In[7]:


print(os.environ['PIPELINEDIR'])
if not os.path.exists(os.environ['PIPELINEDIR']): os.makedirs(os.environ['PIPELINEDIR'])


# In[8]:


outputdir = os.path.join(os.environ['PIPELINEDIR'], '2_compute_gradient', 'out')
print(outputdir)
if not os.path.exists(outputdir): os.makedirs(outputdir)


# In[9]:


figdir = os.path.join(os.environ['OUTPUTDIR'], 'figs')
print(figdir)
if not os.path.exists(figdir): os.makedirs(figdir)


# ## Load data

# In[10]:


# Load data
df = pd.read_csv(os.path.join(os.environ['PIPELINEDIR'], '0_get_sample', 'out', 'df_gradients.csv'))
df.set_index(['bblid', 'scanid'], inplace = True)
# retain discovery sample only
df = df.loc[df['disc_repl'] == 0,:]
print(df.shape)


# In[11]:


df['ageAtScan1_Years'].mean()


# In[12]:


df['ageAtScan1_Years'].std()


# In[13]:


num_subs = df.shape[0]; print(num_subs)
num_time = 120

num_connections = num_parcels * (num_parcels - 1) / 2; print(num_connections)


# ## Load in time series, compute FC

# In[14]:


# subject filter
subj_filt = np.zeros((df.shape[0],)).astype(bool)


# In[15]:


# fc stored as 3d matrix, subjects of 3rd dim
fc = np.zeros((num_parcels, num_parcels, num_subs))

for (i, (index, row)) in enumerate(df.iterrows()):
    file_name = rsts_name_tmp.replace("bblid", str(index[0]))
    file_name = file_name.replace("scanid", str(index[1]))
    full_path = glob.glob(os.path.join(rstsdir, file_name))
    if i == 0: print(full_path)
        
    if len(full_path) > 0:
        roi_ts = np.loadtxt(full_path[0])
        fc[:,:,i] = np.corrcoef(roi_ts, rowvar = False)
        # fisher r to z
        fc[:,:,i] = np.arctanh(fc[:,:,i])
        np.fill_diagonal(fc[:,:,i], 1)

        if np.any(np.isnan(fc[:,:,i])):
            subj_filt[i] = True

    elif len(full_path) == 0:
        print(file_name + ': NOT FOUND')
        subj_filt[i] = True
        roi_ts[:,:,i] = np.full((num_time, num_parcels), np.nan)
        fc[:,:,i] = np.full((num_parcels, num_parcels), np.nan)


# In[16]:


np.sum(subj_filt)


# In[17]:


if any(subj_filt):
    df = df.loc[~subj_filt]
    roi_ts = roi_ts[:,:,~subj_filt]
    fc = fc[:,:,~subj_filt]


# ### Generate participant gradients

# In[18]:


# Generate template
pnc_conn_mat = np.nanmean(fc, axis = 2)
pnc_conn_mat[np.eye(num_parcels, dtype = bool)] = 0
# pnc_conn_mat = dominant_set(pnc_conn_mat, 0.10, as_sparse = False)

gm_template = GradientMaps(n_components = 10, approach='dm', kernel='normalized_angle', random_state = 0)
gm_template.fit(pnc_conn_mat)

if parc_scale == 200 or parc_scale == 125:
    gradients = gm_template.gradients_ * -1
elif parc_scale == 400:
    gradients = gm_template.gradients_

np.savetxt(os.path.join(outputdir,outfile_prefix+'pnc_grads_template.txt'),gradients)


# # Plots

# In[19]:


if not os.path.exists(figdir): os.makedirs(figdir)
os.chdir(figdir)
sns.set(style='white', context = 'paper', font_scale = 1)


# In[20]:


f, ax = plt.subplots(1, figsize=(5, 5))
sns.heatmap(pnc_conn_mat, cmap = 'coolwarm', center = 0, square = True)
f.savefig(outfile_prefix+'mean_fc.png', dpi = 300, bbox_inches = 'tight')


# In[21]:


f, ax = plt.subplots(1, figsize=(5, 4))
ax.scatter(range(gm_template.lambdas_.size), gm_template.lambdas_)
ax.set_xlabel('Component Nb')
ax.set_ylabel('Eigenvalue')
f.savefig(outfile_prefix+'gradient_eigenvals.png', dpi = 300, bbox_inches = 'tight')


# In[22]:


from func import roi_to_vtx
from nilearn import datasets
from nilearn import plotting
atlas = datasets.fetch_atlas_schaefer_2018(n_rois=200, yeo_networks=17, resolution_mm=2)
parcellation = atlas['maps']
fsaverage = datasets.fetch_surf_fsaverage(mesh='fsaverage5')


# In[23]:


for g in np.arange(0,1):
    f, ax = plt.subplots(1, 4, figsize=(20, 5), subplot_kw={'projection': '3d'})
    plt.subplots_adjust(wspace=0, hspace=0)

    parc_file = os.path.join('/Users/lindenmp/Google-Drive-Penn/work/research_projects/pfactor_gradients/figs_support/Parcellations/FreeSurfer5.3/fsaverage5/label/lh.Schaefer2018_200Parcels_17Networks_order.annot')
    labels, ctab, surf_names = nib.freesurfer.read_annot(parc_file)
    vtx_data, plot_min, plot_max = roi_to_vtx(gradients[:,g], parcel_names, parc_file)
    vtx_data = vtx_data.astype(float)
    plotting.plot_surf_roi(fsaverage['infl_left'], roi_map=vtx_data,
                           hemi='left', view='lateral', vmin=plot_min, vmax=plot_max,
                           bg_map=fsaverage['sulc_left'], bg_on_data=False, axes=ax[0],
                           darkness=.5, cmap='viridis');

    plotting.plot_surf_roi(fsaverage['infl_left'], roi_map=vtx_data,
                           hemi='left', view='medial', vmin=plot_min, vmax=plot_max,
                           bg_map=fsaverage['sulc_left'], bg_on_data=False, axes=ax[1],
                           darkness=.5, cmap='viridis');

    parc_file = os.path.join('/Users/lindenmp/Google-Drive-Penn/work/research_projects/pfactor_gradients/figs_support/Parcellations/FreeSurfer5.3/fsaverage5/label/rh.Schaefer2018_200Parcels_17Networks_order.annot')
    labels, ctab, surf_names = nib.freesurfer.read_annot(parc_file)
    vtx_data, plot_min, plot_max = roi_to_vtx(gradients[:,g], parcel_names, parc_file)
    vtx_data = vtx_data.astype(float)
    plotting.plot_surf_roi(fsaverage['infl_right'], roi_map=vtx_data,
                           hemi='right', view='lateral', vmin=plot_min, vmax=plot_max,
                           bg_map=fsaverage['sulc_right'], bg_on_data=False, axes=ax[2],
                           darkness=.5, cmap='viridis');

    plotting.plot_surf_roi(fsaverage['infl_right'], roi_map=vtx_data,
                           hemi='right', view='medial', vmin=plot_min, vmax=plot_max,
                           bg_map=fsaverage['sulc_right'], bg_on_data=False, axes=ax[3],
                           darkness=.5, cmap='viridis');

    f.suptitle('Gradient '+str(g+1))
    f.savefig(outfile_prefix+'gradient_'+str(g)+'.png', dpi = 150, bbox_inches = 'tight', pad_inches = 0)

