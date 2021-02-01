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


sys.path.append('/Users/lindenmp/Google-Drive-Penn/work/research_projects/pathlength_tuning/1_code/')
from func import set_proj_env, my_get_cmap, rank_int


# In[3]:


parc_str = 'schaefer'
parc_scale = 200
edge_weight = 'streamlineCount'
set_proj_env()


# ### Setup directory variables

# In[4]:


print(os.environ['PIPELINEDIR'])
if not os.path.exists(os.environ['PIPELINEDIR']): os.makedirs(os.environ['PIPELINEDIR'])


# In[5]:


outputdir = os.path.join(os.environ['PIPELINEDIR'], '0_get_sample', 'out')
print(outputdir)
if not os.path.exists(outputdir): os.makedirs(outputdir)


# In[6]:


figdir = os.path.join(os.environ['OUTPUTDIR'], 'figs')
print(figdir)
if not os.path.exists(figdir): os.makedirs(figdir)


# # Load in demographic and symptom data

# In[7]:


# LTN and Health Status
health = pd.read_csv(os.path.join(os.environ['DATADIR'], 'external/pncDataFreeze20170905/n1601_dataFreeze/health/n1601_health_20170421.csv'))
# Protocol
prot = pd.read_csv(os.path.join(os.environ['DATADIR'], 'external/pncDataFreeze20170905/n1601_dataFreeze/neuroimaging/n1601_pnc_protocol_validation_params_status_20161220.csv'))
# T1 QA
t1_qa = pd.read_csv(os.path.join(os.environ['DATADIR'], 'external/pncDataFreeze20170905/n1601_dataFreeze/neuroimaging/t1struct/n1601_t1QaData_20170306.csv'))
# DTI QA
dti_qa = pd.read_csv(os.path.join(os.environ['DATADIR'], 'external/pncDataFreeze20170905/n1601_dataFreeze/neuroimaging/dti/n1601_dti_qa_20170301.csv'))
# Rest QA
rest_qa = pd.read_csv(os.path.join(os.environ['DATADIR'], 'external/pncDataFreeze20170905/n1601_dataFreeze/neuroimaging/rest/n1601_RestQAData_20170714.csv'))
# Demographics
demog = pd.read_csv(os.path.join(os.environ['DATADIR'], 'external/pncDataFreeze20170905/n1601_dataFreeze/demographics/n1601_demographics_go1_20161212.csv'))
# Brain volume
brain_vol = pd.read_csv(os.path.join(os.environ['DATADIR'], 'external/pncDataFreeze20170905/n1601_dataFreeze/neuroimaging/t1struct/n1601_ctVol20170412.csv'))
# Clinical diagnostic 
clinical = pd.read_csv(os.path.join(os.environ['DATADIR'], 'external/pncDataFreeze20170905/n1601_dataFreeze/clinical/n1601_goassess_psych_summary_vars_20131014.csv'))
clinical_ps = pd.read_csv(os.path.join(os.environ['DATADIR'], 'external/pncDataFreeze20170905/n1601_dataFreeze/clinical/n1601_diagnosis_dxpmr_20170509.csv'))
# GOASSESS Bifactor scores
goassess = pd.read_csv(os.path.join(os.environ['DATADIR'], 'external/GO1_clinical_factor_scores_psychosis_split_BIFACTOR.csv'))
# Cognition
cnb = pd.read_csv(os.path.join(os.environ['DATADIR'], 'external/pncDataFreeze20170905/n1601_dataFreeze/cnb/n1601_cnb_factor_scores_tymoore_20151006.csv'))

# merge
df = health
df = pd.merge(df, prot, on=['scanid', 'bblid']) # prot
df = pd.merge(df, t1_qa, on=['scanid', 'bblid']) # t1_qa
df = pd.merge(df, dti_qa, on=['scanid', 'bblid']) # dti_qa
df = pd.merge(df, rest_qa, on=['scanid', 'bblid']) # rest_qa
df = pd.merge(df, demog, on=['scanid', 'bblid']) # demog
df = pd.merge(df, brain_vol, on=['scanid', 'bblid']) # brain_vol
df = pd.merge(df, clinical, on=['scanid', 'bblid']) # clinical
df = pd.merge(df, clinical_ps, on=['scanid', 'bblid']) # clinical
df = pd.merge(df, goassess, on=['bblid']) # goassess
df = pd.merge(df, cnb, on=['scanid', 'bblid']) # goassess

print(df.shape[0])
df.set_index(['bblid', 'scanid'], inplace = True)


# # Filter subjects

# In[8]:


# 1) Primary sample filter
df = df[df['healthExcludev2'] == 0]
print('N after initial exclusion:', df.shape[0])

# 2) T1 exclusion
df = df[df['t1Exclude'] == 0]
print('N after T1 exclusion:', df.shape[0])

# 3) Diffusion exclusion
df = df[df['b0ProtocolValidationStatus'] == 1]
df = df[df['dti64ProtocolValidationStatus'] == 1]
df = df[df['dti64Exclude'] == 0]
print('N after Diffusion exclusion:', df.shape[0])

# 4) Medicated
df = df[df['psychoactiveMedPsychv2'] == 0]
print('N after medication exclusion:', df.shape[0])


# In[9]:


df['dti64QAManualScore'].unique()


# In[10]:


np.sum(df['averageManualRating'] == 2)


# In[11]:


np.sum(df['dti64QAManualScore'] == 2)


# In[12]:


# Convert age to years
df['ageAtScan1_Years'] = np.round(df.ageAtScan1/12, decimals=1)


# In[13]:


# find unique ages
age_unique = np.unique(df.ageAtScan1_Years)
print('There are', age_unique.shape[0], 'unique age points')


# # Define discovery and replication splits

# In[14]:


df['disc_repl'] = np.zeros(df.shape[0]).astype(int)

n=np.round(df.shape[0]*.5).astype(int)
print(n)

hold_out = df.sample(n=n, random_state=0, replace=False, axis=0).index
df.loc[hold_out,'disc_repl'] = 1
print('Train:', np.sum(df['disc_repl'] == 0), 'Test:', np.sum(df['disc_repl'] == 1))


# ## Symptom dimensions

# In[15]:


# phenos = ['Overall_Psychopathology','Psychosis_Positive','F3_Executive_Efficiency','Overall_Speed']
phenos = ['Overall_Psychopathology','Psychosis_Positive','Psychosis_NegativeDisorg','AnxiousMisery','Externalizing','Fear','F1_Exec_Comp_Res_Accuracy','F3_Executive_Efficiency','Overall_Speed']
print(phenos)


# In[16]:


for pheno in phenos:
    if df.loc[:,pheno].isna().any():
        print('NaN replacement: ', pheno)
        x = np.nanmedian(df.loc[:,pheno])
        df.loc[df.loc[:,pheno].isna(),pheno] = x


# In[17]:


# Normalize
rank_r = np.zeros(len(phenos),)

for i, pheno in enumerate(phenos):
    # normalize regional metric
#     x = sp.stats.yeojohnson(df.loc[:,pheno])[0]
    x = rank_int(df.loc[:,pheno])
    # check if rank order is preserved
    rank_r[i] = sp.stats.spearmanr(df.loc[:,pheno],x)[0]
    # store normalized version
    df.loc[:,pheno] = x

print(np.sum(rank_r < 1))


# In[18]:


df.loc[:,phenos].var()


# ## Export

# In[19]:


header = ['disc_repl', 'squeakycleanExclude','ageAtScan1', 'ageAtScan1_Years','sex','race2','handednessv2', 'averageManualRating', 'dti64QAManualScore', 'restProtocolValidationStatus', 'restExclude',
          'dti64MeanAbsRMS','dti64MeanRelRMS','dti64MaxAbsRMS','dti64MaxRelRMS','mprage_antsCT_vol_TBV', 'averageManualRating',  'goassessSmryMood', 'goassessSmryMan', 'goassessSmryDep',
          'goassessSmryEat', 'goassessSmryBul', 'goassessSmryAno', 'goassessSmryAnx', 'goassessSmryGad', 'goassessSmrySep', 'goassessSmryPhb', 'goassessSmrySoc', 'goassessSmryPan',
          'goassessSmryAgr', 'goassessSmryOcd', 'goassessSmryPtd', 'goassessSmryPsy', 'goassessSmryDel', 'goassessSmryHal', 'goassessSmryHalAv', 'goassessSmryHalAs', 'goassessSmryHalVh',
          'goassessSmryHalOh', 'goassessSmryHalTh', 'goassessSmryBeh', 'goassessSmryAdd', 'goassessSmryOdd', 'goassessSmryCon', 'goassessSmryPrimePos1', 'goassessSmryPrimeTot',
          'goassessSmryPrimePos2', 'goassessSmryPsychOverallRtg',
          'goassessDxpmr4'] + phenos
df.to_csv(os.path.join(outputdir, 'df.csv'), columns = header)


# # Plots

# In[20]:


if not os.path.exists(figdir): os.makedirs(figdir)
os.chdir(figdir)
sns.set(style='white', context = 'paper', font_scale = 1)
cmap = my_get_cmap('pair')

labels = ['Discovery', 'Replication']
# phenos_label_short = ['Ov. Psych.', 'Psy. (pos.)', 'Psy. (neg.)', 'Anx.-mis.', 'Ext.', 'Fear']
# phenos_label = ['Overall psychopathology','Psychosis (positive)','Psychosis (negative)','Anxious-misery','Externalizing','Fear']


# ## Age

# In[21]:


f, axes = plt.subplots(1,2)
f.set_figwidth(6.5)
f.set_figheight(2.5)
colormap = sns.color_palette("pastel", 2)

sns.distplot(df.loc[df['disc_repl'] == 0,'ageAtScan1_Years'], bins=20, hist=True, kde=False, rug=False, label = labels[0],
             hist_kws={"histtype": "step", "linewidth": 2, "alpha": 1}, color=list(cmap[0]), ax = axes[0]);
sns.distplot(df.loc[df['disc_repl'] == 1,'ageAtScan1_Years'], bins=20, hist=True, kde=False, rug=False, label = labels[1],
             hist_kws={"histtype": "step", "linewidth": 2, "alpha": 1}, color=list(cmap[1]), ax = axes[0]);
axes[0].set_xlabel('Age (years)');
axes[0].set_ylabel('Number of participants');
axes[0].set_xticks(np.arange(np.min(np.round(age_unique,0)), np.max(np.round(age_unique,0)), 2))

# set width of bar
barWidth = 0.25

# Sex
y_train = [np.sum(df.loc[df['disc_repl'] == 0,'sex'] == 1), np.sum(df.loc[df['disc_repl'] == 0,'sex'] == 2)]
y_test = [np.sum(df.loc[df['disc_repl'] == 1,'sex'] == 1), np.sum(df.loc[df['disc_repl'] == 1,'sex'] == 2)]
r1 = np.arange(len(y_train))+barWidth/2
r2 = [x + barWidth for x in r1]
axes[1].bar(r1, y_train, width = barWidth, color = cmap[0], label = labels[0])
axes[1].bar(r2, y_test, width = barWidth, color = cmap[1], label = labels[1])
axes[1].set_xlabel('Sex')
# axes[1].set_ylabel('Number of participants')
axes[1].set_xticks([r + barWidth for r in range(len(y_train))])
axes[1].set_xticklabels(['Male', 'Female'])

f.savefig('age_distributions.png', dpi = 150, bbox_inches = 'tight', pad_inches = 0)


# ## Symptom dimensions

# In[22]:


df_rc = pd.melt(df, id_vars = 'disc_repl', value_vars = phenos)

f, ax = plt.subplots()
f.set_figwidth(2.5)
f.set_figheight(4)
ax = sns.violinplot(y='variable', x='value', hue='disc_repl', data=df_rc, palette = cmap, split=True, scale='width', inner = 'quartile', orient = 'h')
ax.get_legend().remove()
ax.set_yticklabels(phenos)
ax.set_ylabel('Psychopathology phenotypes')
ax.set_xlabel('Phenotype score')
f.savefig('symptoms_distributions.png', dpi = 150, bbox_inches = 'tight', pad_inches = 0)


# ### Export sample for FC gradients

# In[23]:


# 4) rs-fMRI exclusion
df = df[df['restProtocolValidationStatus'] == 1]
df = df[df['restExclude'] == 0]
print('N after rs-fMRI exclusion:', df.shape[0])


# In[24]:


df.to_csv(os.path.join(outputdir, 'df_gradients.csv'), columns = header)

