# This script takes the Schaefer parcellation in PNC template and multiplies it by each subject's GM hard segmentation
# mask (which is also in PNC template space)
# This produces a Schaefer parc in PNC template space for each subject where only voxels inside GM are retained.
# This new mask is used to extract region CT measures for each subjects

# input data directories
in_dir='/Volumes/T7/research_data/PNC/processedData/gm_masks_template_pnc'
cd ${in_dir}

for gm_mask in *_atropos3class_seg_GmMask_Template.nii.gz; do
    echo ${gm_mask}
    out_label=$(echo "$gm_mask" | cut -f 1 -d '.')

    # schaefer 400
    parc_file='/Volumes/T7/research_data/PNC/template/schaefer2018PNC2mm/Schaefer2018_400_17Networks_PNC_2mm.nii.gz'
    fslmaths ${parc_file} -mul ${gm_mask} ${out_label}_Schaefer400_17.nii.gz

    # schaefer 200
    parc_file='/Volumes/T7/research_data/PNC/template/schaefer2018PNC2mm/Schaefer2018_200_17Networks_PNC_2mm.nii.gz'
    fslmaths ${parc_file} -mul ${gm_mask} ${out_label}_Schaefer200_17.nii.gz
done
