FREESURFER_HOME='/Applications/freesurfer/7.3.2'
source $FREESURFER_HOME/SetUpFreeSurfer.sh

workbenchdir='/Applications/workbench/bin_macosx64'
standard_mesh_atlases='/Users/lindenmp/Google-Drive-Penn/work/git/HCPpipelines/global/templates/standard_mesh_atlases'
indir='/Users/lindenmp/research_data/Glasser_et_al_2016_HCP_MMP1.0_kN_RVVG/HCP_PhaseTwo/Q1-Q6_RelatedParcellation210/MNINonLinear/fsaverage_LR32k'
outdir='/Users/lindenmp/research_data/Glasser_et_al_2016_HCP_MMP1.0_kN_RVVG/prepare_glasser'
mkdir ${outdir}

# convert cifti to gifti
${workbenchdir}/wb_command -cifti-separate ${indir}/Q1-Q6_RelatedParcellation210.L.CorticalAreas_dil_Colors.32k_fs_LR.dlabel.nii \
COLUMN -label CORTEX_LEFT ${outdir}/Q1-Q6_RelatedParcellation210.L.CorticalAreas_dil_Colors.32k_fs_LR.label.gii

${workbenchdir}/wb_command -cifti-separate ${indir}/Q1-Q6_RelatedParcellation210.R.CorticalAreas_dil_Colors.32k_fs_LR.dlabel.nii \
COLUMN -label CORTEX_RIGHT ${outdir}/Q1-Q6_RelatedParcellation210.R.CorticalAreas_dil_Colors.32k_fs_LR.label.gii

# convert to fsaverage(5)

# left hemi
# step 1: resample from original space to fsaverage 164k
${workbenchdir}/wb_command -label-resample ${outdir}/Q1-Q6_RelatedParcellation210.L.CorticalAreas_dil_Colors.32k_fs_LR.label.gii \
${standard_mesh_atlases}/L.sphere.32k_fs_LR.surf.gii \
${standard_mesh_atlases}/fs_L/fs_L-to-fs_LR_fsaverage.L_LR.spherical_std.164k_fs_L.surf.gii \
BARYCENTRIC \
${outdir}/HCP_MMP1.fsaverage.L.label.gii

# convert to .annot
mris_convert --annot ${outdir}/HCP_MMP1.fsaverage.L.label.gii \
${standard_mesh_atlases}/fs_L/fs_L-to-fs_LR_fsaverage.L_LR.spherical_std.164k_fs_L.surf.gii \
${outdir}/HCP-MMP1.fsaverage.L.annot

# step 2: downsample from fsaverage 164k to fsaverage5 10k
${workbenchdir}/wb_command -label-resample ${outdir}/HCP_MMP1.fsaverage.L.label.gii \
${standard_mesh_atlases}/fs_L/fsaverage.L.sphere.164k_fs_L.surf.gii \
${standard_mesh_atlases}/resample_fsaverage/fsaverage5_std_sphere.L.10k_fsavg_L.surf.gii \
BARYCENTRIC \
${outdir}/HCP_MMP1.fsaverage5.L.label.gii

# convert to .annot
mris_convert --annot ${outdir}/HCP_MMP1.fsaverage5.L.label.gii \
${standard_mesh_atlases}/resample_fsaverage/fsaverage5_std_sphere.L.10k_fsavg_L.surf.gii \
${outdir}/HCP-MMP1.fsaverage5.L.annot



# right hemi
# step 1: resample from original space to fsaverage 164k
${workbenchdir}/wb_command -label-resample ${outdir}/Q1-Q6_RelatedParcellation210.R.CorticalAreas_dil_Colors.32k_fs_LR.label.gii \
${standard_mesh_atlases}/R.sphere.32k_fs_LR.surf.gii \
${standard_mesh_atlases}/fs_R/fs_R-to-fs_LR_fsaverage.R_LR.spherical_std.164k_fs_R.surf.gii \
BARYCENTRIC \
${outdir}/HCP_MMP1.fsaverage.R.label.gii

# convert to .annot
mris_convert --annot ${outdir}/HCP_MMP1.fsaverage.R.label.gii \
${standard_mesh_atlases}/fs_R/fs_R-to-fs_LR_fsaverage.R_LR.spherical_std.164k_fs_R.surf.gii \
${outdir}/HCP-MMP1.fsaverage.R.annot

# step 2: downsample from fsaverage 164k to fsaverage5 10k
${workbenchdir}/wb_command -label-resample ${outdir}/HCP_MMP1.fsaverage.R.label.gii \
${standard_mesh_atlases}/fs_R/fsaverage.R.sphere.164k_fs_R.surf.gii \
${standard_mesh_atlases}/resample_fsaverage/fsaverage5_std_sphere.R.10k_fsavg_R.surf.gii \
BARYCENTRIC \
${outdir}/HCP_MMP1.fsaverage5.R.label.gii

# convert to .annot
mris_convert --annot ${outdir}/HCP_MMP1.fsaverage5.R.label.gii \
${standard_mesh_atlases}/resample_fsaverage/fsaverage5_std_sphere.R.10k_fsavg_R.surf.gii \
${outdir}/HCP-MMP1.fsaverage5.R.annot
