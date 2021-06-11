import os
import numpy as np
import nibabel as nib
from tqdm import tqdm

def get_parcelwise_average_surface(data_file, annot_file):
    # load parcellation
    labels, ctab, surf_names = nib.freesurfer.read_annot(annot_file)
    unique_labels = np.unique(labels)

    # load gifti file
    data = nib.load(data_file)
    file_name, file_extension = os.path.splitext(data_file)
    if file_extension == '.gii':
        data = data.darrays[0].data
    elif file_extension == '.mgh':
        data = data.get_fdata().squeeze()

    # mean over labels
    data_mean = []
    for i in unique_labels:
        data_mean.append(np.mean(data[labels == i]))

    return np.asarray(data_mean)

# %% BigBrainWarp maps
workdir='/Volumes/T7/research_data/BigBrainWarp/spaces/fsaverage'

# schaefer 400
lh_annot_file = '/Volumes/T7/research_data/parcellations/FreeSurfer5.3/fsaverage/label/lh.Schaefer2018_400Parcels_17Networks_order.annot'
rh_annot_file = '/Volumes/T7/research_data/parcellations/FreeSurfer5.3/fsaverage/label/rh.Schaefer2018_400Parcels_17Networks_order.annot'

data_lh = get_parcelwise_average_surface(os.path.join(workdir, 'Hist_G1_lh_fsaverage.shape.gii'), lh_annot_file)
data_rh = get_parcelwise_average_surface(os.path.join(workdir, 'Hist_G1_rh_fsaverage.shape.gii'), rh_annot_file)
data = np.hstack((data_lh[1:], data_rh[1:]))
np.savetxt(os.path.join(workdir, 'Hist_G1_Schaefer2018_400Parcels_17Networks.txt'), data)

data_lh = get_parcelwise_average_surface(os.path.join(workdir, 'Hist_G2_lh_fsaverage.shape.gii'), lh_annot_file)
data_rh = get_parcelwise_average_surface(os.path.join(workdir, 'Hist_G2_rh_fsaverage.shape.gii'), rh_annot_file)
data = np.hstack((data_lh[1:], data_rh[1:]))
np.savetxt(os.path.join(workdir, 'Hist_G2_Schaefer2018_400Parcels_17Networks.txt'), data)

# schaefer 200
lh_annot_file = '/Volumes/T7/research_data/parcellations/FreeSurfer5.3/fsaverage/label/lh.Schaefer2018_200Parcels_17Networks_order.annot'
rh_annot_file = '/Volumes/T7/research_data/parcellations/FreeSurfer5.3/fsaverage/label/rh.Schaefer2018_200Parcels_17Networks_order.annot'

data_lh = get_parcelwise_average_surface(os.path.join(workdir, 'Hist_G1_lh_fsaverage.shape.gii'), lh_annot_file)
data_rh = get_parcelwise_average_surface(os.path.join(workdir, 'Hist_G1_rh_fsaverage.shape.gii'), rh_annot_file)
data = np.hstack((data_lh[1:], data_rh[1:]))
np.savetxt(os.path.join(workdir, 'Hist_G1_Schaefer2018_200Parcels_17Networks.txt'), data)

data_lh = get_parcelwise_average_surface(os.path.join(workdir, 'Hist_G2_lh_fsaverage.shape.gii'), lh_annot_file)
data_rh = get_parcelwise_average_surface(os.path.join(workdir, 'Hist_G2_rh_fsaverage.shape.gii'), rh_annot_file)
data = np.hstack((data_lh[1:], data_rh[1:]))
np.savetxt(os.path.join(workdir, 'Hist_G2_Schaefer2018_200Parcels_17Networks.txt'), data)

# glasser 360
lh_annot_file = '/Volumes/T7/research_data/parcellations/Glasser_et_al_2016_HCP_MMP1.0_qN_RVVG/HCP-MMP1.fsaverage.L.annot'
rh_annot_file = '/Volumes/T7/research_data/parcellations/Glasser_et_al_2016_HCP_MMP1.0_qN_RVVG/HCP-MMP1.fsaverage.R.annot'

data_lh = get_parcelwise_average_surface(os.path.join(workdir, 'Hist_G1_lh_fsaverage.shape.gii'), lh_annot_file)
data_rh = get_parcelwise_average_surface(os.path.join(workdir, 'Hist_G1_rh_fsaverage.shape.gii'), rh_annot_file)
data = np.hstack((data_rh[1:], data_lh[1:]))
np.savetxt(os.path.join(workdir, 'Hist_G1_HCP-MMP1.txt'), data)

data_lh = get_parcelwise_average_surface(os.path.join(workdir, 'Hist_G2_lh_fsaverage.shape.gii'), lh_annot_file)
data_rh = get_parcelwise_average_surface(os.path.join(workdir, 'Hist_G2_rh_fsaverage.shape.gii'), rh_annot_file)
data = np.hstack((data_rh[1:], data_lh[1:]))
np.savetxt(os.path.join(workdir, 'Hist_G2_HCP-MMP1.txt'), data)

# %% pnc freesurfer files
workdir='/Volumes/T7/research_data/pnc/processedData/structural/freesurfer53'
subjs = os.listdir(workdir)

# %% schaefer 400
print('schaefer 400')
for subj in tqdm(subjs):
    subjdir = os.path.join(workdir, subj)
    scans = os.listdir(subjdir)
    for scan in scans:
        lh_annot_file = '/Volumes/T7/research_data/parcellations/FreeSurfer5.3/fsaverage/label/lh.Schaefer2018_400Parcels_17Networks_order.annot'
        rh_annot_file = '/Volumes/T7/research_data/parcellations/FreeSurfer5.3/fsaverage/label/rh.Schaefer2018_400Parcels_17Networks_order.annot'
        # surface area
        try:
            data_lh = get_parcelwise_average_surface(os.path.join(subjdir, scan, 'surf', 'lh.area.pial.fsaverage.mgh'), lh_annot_file)
            data_rh = get_parcelwise_average_surface(os.path.join(subjdir, scan, 'surf', 'rh.area.pial.fsaverage.mgh'), rh_annot_file)
            data = np.hstack((data_lh[1:], data_rh[1:]))
            np.savetxt(os.path.join(subjdir, scan, 'surf', 'area_pial_Schaefer2018_400Parcels_17Networks.txt'), data)
        except FileNotFoundError:
            pass

        # thickness
        try:
            data_lh = get_parcelwise_average_surface(os.path.join(subjdir, scan, 'surf', 'lh.thickness.fsaverage.mgh'), lh_annot_file)
            data_rh = get_parcelwise_average_surface(os.path.join(subjdir, scan, 'surf', 'rh.thickness.fsaverage.mgh'), rh_annot_file)
            data = np.hstack((data_lh[1:], data_rh[1:]))
            np.savetxt(os.path.join(subjdir, scan, 'surf', 'thickness_Schaefer2018_400Parcels_17Networks.txt'), data)
        except FileNotFoundError:
            pass

# schaefer 200
print('schaefer 200')
for subj in tqdm(subjs):
    subjdir = os.path.join(workdir, subj)
    scans = os.listdir(subjdir)
    for scan in scans:
        lh_annot_file = '/Volumes/T7/research_data/parcellations/FreeSurfer5.3/fsaverage/label/lh.Schaefer2018_200Parcels_17Networks_order.annot'
        rh_annot_file = '/Volumes/T7/research_data/parcellations/FreeSurfer5.3/fsaverage/label/rh.Schaefer2018_200Parcels_17Networks_order.annot'
        # surface area
        try:
            data_lh = get_parcelwise_average_surface(os.path.join(subjdir, scan, 'surf', 'lh.area.pial.fsaverage.mgh'), lh_annot_file)
            data_rh = get_parcelwise_average_surface(os.path.join(subjdir, scan, 'surf', 'rh.area.pial.fsaverage.mgh'), rh_annot_file)
            data = np.hstack((data_lh[1:], data_rh[1:]))
            np.savetxt(os.path.join(subjdir, scan, 'surf', 'area_pial_Schaefer2018_200Parcels_17Networks.txt'), data)
        except FileNotFoundError:
            pass

        # thickness
        try:
            data_lh = get_parcelwise_average_surface(os.path.join(subjdir, scan, 'surf', 'lh.thickness.fsaverage.mgh'), lh_annot_file)
            data_rh = get_parcelwise_average_surface(os.path.join(subjdir, scan, 'surf', 'rh.thickness.fsaverage.mgh'), rh_annot_file)
            data = np.hstack((data_lh[1:], data_rh[1:]))
            np.savetxt(os.path.join(subjdir, scan, 'surf', 'thickness_Schaefer2018_200Parcels_17Networks.txt'), data)
        except FileNotFoundError:
            pass

# glasser 360
print('glasser 360')
for subj in tqdm(subjs):
    subjdir = os.path.join(workdir, subj)
    scans = os.listdir(subjdir)
    for scan in scans:
        lh_annot_file = '/Volumes/T7/research_data/parcellations/Glasser_et_al_2016_HCP_MMP1.0_qN_RVVG/HCP-MMP1.fsaverage.L.annot'
        rh_annot_file = '/Volumes/T7/research_data/parcellations/Glasser_et_al_2016_HCP_MMP1.0_qN_RVVG/HCP-MMP1.fsaverage.R.annot'
        # surface area
        try:
            data_lh = get_parcelwise_average_surface(os.path.join(subjdir, scan, 'surf', 'lh.area.pial.fsaverage.mgh'), lh_annot_file)
            data_rh = get_parcelwise_average_surface(os.path.join(subjdir, scan, 'surf', 'rh.area.pial.fsaverage.mgh'), rh_annot_file)
            data = np.hstack((data_rh[1:], data_lh[1:]))
            np.savetxt(os.path.join(subjdir, scan, 'surf', 'area_pial_HCP-MMP1.txt'), data)
        except FileNotFoundError:
            pass

        # thickness
        try:
            data_lh = get_parcelwise_average_surface(os.path.join(subjdir, scan, 'surf', 'lh.thickness.fsaverage.mgh'), lh_annot_file)
            data_rh = get_parcelwise_average_surface(os.path.join(subjdir, scan, 'surf', 'rh.thickness.fsaverage.mgh'), rh_annot_file)
            data = np.hstack((data_rh[1:], data_lh[1:]))
            np.savetxt(os.path.join(subjdir, scan, 'surf', 'thickness_HCP-MMP1.txt'), data)
        except FileNotFoundError:
            pass
