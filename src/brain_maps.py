import os
import numpy as np
import pandas as pd
from src.utils import get_parcelwise_average_surface
import wget
import nibabel as nib
from src.utils import load_schaefer_parc
from sklearn.metrics import pairwise_distances

# note, this code only works with Schaefer. Glasser not implemented

class BrainMapLoader:
    def __init__(self, computer='macbook', n_parcels=200):
        # analysis parameters
        self.computer = computer
        self.n_parcels = n_parcels

        # directories
        if self.computer == 'macbook':
            self.userdir = '/Users/lindenmp'
        elif self.computer == 'cbica':
            self.userdir = '/cbica/home/parkesl'

        self.research_data = os.path.join(self.userdir, 'research_data')
        self.bbw_dir = os.path.join(self.research_data, 'BigBrainWarp') # original BigBrainData downloaded ~June 2021
        # self.bbw_dir = os.path.join(self.research_data, 'BBW_BigData') # new BigBrainData downloaded August 2022
        self.fukutomi_dir = os.path.join(self.research_data, 'fukutomi_2018ni_brain_maps')
        self.outdir = os.path.join(self.research_data, 'brain_maps')

        if os.path.exists(self.outdir) == False:
            os.makedirs(self.outdir)


    def _get_parc_data(self, annot='fsaverage'):
        self.centroids, self.lh_annot_file, self.rh_annot_file, self.hcp_file = load_schaefer_parc(
            n_parcels=self.n_parcels,
            order=17,
            annot=annot,
            outdir='~/research_data/schaefer_parc')
        self.centroids.set_index('ROI Name', inplace=True)


    def load_cyto(self):
        self._get_parc_data(annot='fsaverage')

        lh_gifti_file = os.path.join(self.bbw_dir, 'spaces', 'fsaverage', 'Hist_G2_lh_fsaverage.shape.gii') # original BigBrainData downloaded ~June 2021
        rh_gifti_file = os.path.join(self.bbw_dir, 'spaces', 'fsaverage', 'Hist_G2_rh_fsaverage.shape.gii') # original BigBrainData downloaded ~June 2021
        # lh_gifti_file = os.path.join(self.bbw_dir, 'spaces', 'tpl-fsaverage', 'tpl-fsaverage_hemi-L_den-164k_desc-Hist_G2.shape.gii') # new BigBrainData downloaded August 2022
        # rh_gifti_file = os.path.join(self.bbw_dir, 'spaces', 'tpl-fsaverage', 'tpl-fsaverage_hemi-R_den-164k_desc-Hist_G2.shape.gii') # new BigBrainData downloaded August 2022
        # note, results from Figure 2A left, and Figure 2B left are consistent between BigBrain versions
        # All figures in paper use original download from June 2021.

        # get average values over parcels
        data_lh = get_parcelwise_average_surface(lh_gifti_file, self.lh_annot_file)
        data_rh = get_parcelwise_average_surface(rh_gifti_file, self.rh_annot_file)

        # drop first entry (corresponds to 0)
        data_lh = data_lh[1:]
        data_rh = data_rh[1:]

        self.cyto = np.hstack((data_lh, data_rh)).astype(float)


    def load_tau(self, return_log=False):
        self._get_parc_data()

        # download data
        remote_path = 'https://github.com/rdgao/field-echos/raw/master/data'
        file = 'df_human.csv'
        if os.path.exists(os.path.join(self.outdir, file)) == False:
            wget.download(os.path.join(remote_path, file), self.outdir)

        df_human = pd.read_csv(os.path.join(self.outdir, file), index_col=0)
        electrode_coords = df_human.loc[:, ['x', 'y', 'z']]

        D = pairwise_distances(electrode_coords, self.centroids, metric='euclidean')
        nearest_region = np.argmin(D, axis=1)

        mean_tau = pd.DataFrame(index=self.centroids.index, columns=['tau', 'log_tau'])

        for i in np.arange(self.n_parcels):
            if np.any(nearest_region == i):
                mean_tau.iloc[i, 0] = df_human.loc[nearest_region == i, 'tau'].mean()
                mean_tau.iloc[i, 1] = df_human.loc[nearest_region == i, 'log_tau'].mean()

        mean_tau['tau'] = mean_tau['tau'].astype(float)
        mean_tau['log_tau'] = mean_tau['log_tau'].astype(float)

        if return_log:
            self.tau = mean_tau['log_tau'].values
        else:
            self.tau = mean_tau['tau'].values

