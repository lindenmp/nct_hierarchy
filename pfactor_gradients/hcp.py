import os
import numpy as np
from pfactor_gradients.utils import get_parcelwise_average_surface

class BrainMapLoader:
    def __init__(self, computer='macbook'):
        # analysis parameters
        self.computer = computer

        # directories
        if self.computer == 'macbook':
            self.userdir = '/Users/lindenmp'
            self.projdir = os.path.join(self.userdir, 'Google-Drive-Penn', 'work', 'research_projects', 'pfactor_gradients')
            self.rootdir = '/Volumes'
            self.research_data = os.path.join(self.rootdir, 'T7', 'research_data')
        elif self.computer == 'cbica':
            self.userdir = '/cbica/home/parkesl'
            self.projdir = os.path.join(self.userdir, 'research_projects', 'pfactor_gradients')
            self.research_data = os.path.join(self.userdir, 'research_data')

        self.brainmap_dir = os.path.join(self.research_data, 'fukutomi_2018ni_brain_maps')


    def load_ct(self, lh_annot_file, rh_annot_file, order='lhrh'):
        indir = os.path.join(self.brainmap_dir, 'ct')
        order = order

        lh_gifti_file = os.path.join(indir, 'ave_corrThickness_MSMAll.fsaverage5.L.func.gii')
        rh_gifti_file = os.path.join(indir, 'ave_corrThickness_MSMAll.fsaverage5.R.func.gii')

        # get average values over parcels
        data_lh = get_parcelwise_average_surface(lh_gifti_file, lh_annot_file)
        data_rh = get_parcelwise_average_surface(rh_gifti_file, rh_annot_file)

        # drop first entry (corresponds to 0)
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
        data_lh = get_parcelwise_average_surface(lh_gifti_file, lh_annot_file)
        data_rh = get_parcelwise_average_surface(rh_gifti_file, rh_annot_file)

        # drop first entry (corresponds to 0)
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
        data_lh = get_parcelwise_average_surface(lh_gifti_file, lh_annot_file)
        data_rh = get_parcelwise_average_surface(rh_gifti_file, rh_annot_file)

        # drop first entry (corresponds to 0)
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
        data_lh = get_parcelwise_average_surface(lh_gifti_file, lh_annot_file)
        data_rh = get_parcelwise_average_surface(rh_gifti_file, rh_annot_file)

        # drop first entry (corresponds to 0)
        data_lh = data_lh[1:]
        data_rh = data_rh[1:]

        if order == 'lhrh':
            self.odi = np.hstack((data_lh, data_rh))
        elif order == 'rhlh':
            self.odi = np.hstack((data_rh, data_lh))
