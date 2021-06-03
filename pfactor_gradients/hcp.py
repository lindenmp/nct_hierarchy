import os
import numpy as np
from pfactor_gradients.utils import get_parcelwise_average_gii

class BrainMapLoader:
    def __init__(self):
        self.description = 'simple class for loading precomputed HCP brain maps taken from Fukutomi et al. 2018 NeuroImage'
        self.brainmap_dir = '/Volumes/T7/research_data/fukutomi_2018ni_brain_maps'


    def load_ct(self, lh_annot_file, rh_annot_file, order='lhrh'):
        indir = os.path.join(self.brainmap_dir, 'ct')
        order = order

        lh_gifti_file = os.path.join(indir, 'ave_corrThickness_MSMAll.fsaverage5.L.func.gii')
        rh_gifti_file = os.path.join(indir, 'ave_corrThickness_MSMAll.fsaverage5.R.func.gii')

        # get average values over parcels
        data_lh = get_parcelwise_average_gii(lh_gifti_file, lh_annot_file)
        data_rh = get_parcelwise_average_gii(rh_gifti_file, rh_annot_file)

        if 'schaefer' in lh_annot_file.lower() and 'schaefer' in rh_annot_file.lower():
            data_lh = data_lh[1:]
            data_rh = data_rh[1:]

        if order == 'lhrh':
            self.ct = np.hstack((data_lh, data_rh))
        elif order == 'rhlh':
            self.ct = np.hstack((data_rh, data_lh))


    def load_myelin(self, lh_annot_file, rh_annot_file, order='lhrh'):
        indir = os.path.join(self.brainmap_dir, 'myelin')
        order = order

        lh_gifti_file = os.path.join(indir, 'ave_MyelinMap_BC_MSMAll.fsaverage5.L.func.gii')
        rh_gifti_file = os.path.join(indir, 'ave_MyelinMap_BC_MSMAll.fsaverage5.R.func.gii')

        # get average values over parcels
        data_lh = get_parcelwise_average_gii(lh_gifti_file, lh_annot_file)
        data_rh = get_parcelwise_average_gii(rh_gifti_file, rh_annot_file)

        if 'schaefer' in lh_annot_file.lower() and 'schaefer' in rh_annot_file.lower():
            data_lh = data_lh[1:]
            data_rh = data_rh[1:]

        if order == 'lhrh':
            self.myelin = np.hstack((data_lh, data_rh))
        elif order == 'rhlh':
            self.myelin = np.hstack((data_rh, data_lh))


    def load_ndi(self, lh_annot_file, rh_annot_file, order='lhrh'):
        indir = os.path.join(self.brainmap_dir, 'ndi')
        order = order

        lh_gifti_file = os.path.join(indir, 'ave_ficvf_MSMAll.fsaverage5.L.func.gii')
        rh_gifti_file = os.path.join(indir, 'ave_ficvf_MSMAll.fsaverage5.R.func.gii')

        # get average values over parcels
        data_lh = get_parcelwise_average_gii(lh_gifti_file, lh_annot_file)
        data_rh = get_parcelwise_average_gii(rh_gifti_file, rh_annot_file)

        if 'schaefer' in lh_annot_file.lower() and 'schaefer' in rh_annot_file.lower():
            data_lh = data_lh[1:]
            data_rh = data_rh[1:]

        if order == 'lhrh':
            self.ndi = np.hstack((data_lh, data_rh))
        elif order == 'rhlh':
            self.ndi = np.hstack((data_rh, data_lh))


    def load_odi(self, lh_annot_file, rh_annot_file, order='lhrh'):
        indir = os.path.join(self.brainmap_dir, 'odi')
        order = order

        lh_gifti_file = os.path.join(indir, 'ave_odifromkappa_MSMAll.fsaverage5.L.func.gii')
        rh_gifti_file = os.path.join(indir, 'ave_odifromkappa_MSMAll.fsaverage5.R.func.gii')

        # get average values over parcels
        data_lh = get_parcelwise_average_gii(lh_gifti_file, lh_annot_file)
        data_rh = get_parcelwise_average_gii(rh_gifti_file, rh_annot_file)

        if 'schaefer' in lh_annot_file.lower() and 'schaefer' in rh_annot_file.lower():
            data_lh = data_lh[1:]
            data_rh = data_rh[1:]

        if order == 'lhrh':
            self.odi = np.hstack((data_lh, data_rh))
        elif order == 'rhlh':
            self.odi = np.hstack((data_rh, data_lh))