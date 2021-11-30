import os
import numpy as np
import pandas as pd
from sklearn.metrics import pairwise_distances

# %%
workdir = '/Users/lindenmp/research_data/field-echos'

# %% load tau data from Gao et al. eLife
df_human = pd.read_csv(os.path.join(workdir, 'data', 'df_human.csv'), index_col=0)
electrode_coords = df_human.loc[:, ['x', 'y', 'z']]

# %% schaefer 400
centroids = pd.read_csv('/Users/lindenmp/research_data/parcellations/MNI/Centroid_coordinates/Schaefer2018_400Parcels_17Networks_order_FSLMNI152_1mm.Centroid_RAS.csv')
n_parcels = centroids.shape[0]
centroids.drop('ROI Index', axis=1, inplace=True)
centroids.set_index('Label Name', inplace=True)
centroids.drop('NONE', axis=0, inplace=True)

D = pairwise_distances(electrode_coords, centroids, metric='euclidean')
nearest_region = np.argmin(D, axis=1)
print(len(np.unique(nearest_region)))

mean_tau = pd.DataFrame(index=centroids.index, columns=['tau', 'log_tau'])

for i in np.arange(n_parcels):
    if np.any(nearest_region == i):
        mean_tau.iloc[i, 0] = df_human.loc[nearest_region == i, 'tau'].mean()
        mean_tau.iloc[i, 1] = df_human.loc[nearest_region == i, 'log_tau'].mean()

# save
mean_tau.to_csv(os.path.join(workdir, 'tau_Schaefer2018_400Parcels_17Networks.csv'))

# %% schaefer 200
centroids = pd.read_csv('/Users/lindenmp/research_data/parcellations/MNI/Centroid_coordinates/Schaefer2018_200Parcels_17Networks_order_FSLMNI152_1mm.Centroid_RAS.csv')
n_parcels = centroids.shape[0]
centroids.drop('ROI Index', axis=1, inplace=True)
centroids.set_index('Label Name', inplace=True)
centroids.drop('NONE', axis=0, inplace=True)

D = pairwise_distances(electrode_coords, centroids, metric='euclidean')
nearest_region = np.argmin(D, axis=1)
print(len(np.unique(nearest_region)))

mean_tau = pd.DataFrame(index=centroids.index, columns=['tau', 'log_tau'])

for i in np.arange(n_parcels):
    if np.any(nearest_region == i):
        mean_tau.iloc[i, 0] = df_human.loc[nearest_region == i, 'tau'].mean()
        mean_tau.iloc[i, 1] = df_human.loc[nearest_region == i, 'log_tau'].mean()

# save
mean_tau.to_csv(os.path.join(workdir, 'tau_Schaefer2018_200Parcels_17Networks.csv'))
