import os
import numpy as np
from pfactor_gradients.utils import get_parcelwise_average_gii

class MiscLoader:
    def __init__(self):
        self.myelin_dir = '/Volumes/T7/research_data/Brain_Organization/Myelin'

    def load_brain_map(self, lh_annot_file, rh_annot_file, metric='myelin', order='lhrh'):
        self.metric = metric
        self.order = order

        if metric == 'myelin':
            lh_gifti_file = os.path.join(self.myelin_dir, 'MyelinMap.lh.fsaverage5.func.gii')
            rh_gifti_file = os.path.join(self.myelin_dir, 'MyelinMap.rh.fsaverage5.func.gii')

        # get average values over parcels
        data_lh = get_parcelwise_average_gii(lh_gifti_file, lh_annot_file)
        data_rh = get_parcelwise_average_gii(rh_gifti_file, rh_annot_file)

        if 'schaefer' in lh_annot_file.lower() and 'schaefer' in rh_annot_file.lower():
            data_lh = data_lh[1:]
            data_rh = data_rh[1:]

        if self.order == 'lhrh':
            self.myelin = np.hstack((data_lh, data_rh))
        elif self.order == 'rhlh':
            self.myelin = np.hstack((data_rh, data_lh))
