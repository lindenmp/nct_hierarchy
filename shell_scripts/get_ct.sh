# input data directories
indir='/Volumes/T7/research_data/PNC/processedData/antsCt/voxelwiseMaps_antsCt'
outdir='/Volumes/T7/research_data/PNC/processedData/antsCt/parcelwise_antsCt/'
cd ${indir}

for scan in *_CorticalThicknessNormalizedToTemplate2mm.nii.gz; do
    echo ${scan}
    out_label=$(echo "$scan" | cut -f 1 -d '.')
    scanid=$(echo "$out_label" | cut -f 1 -d '_')

    # schaefer 400
    parc_file='/Volumes/T7/research_data/PNC/processedData/gm_masks_template_pnc/'${scanid}'_atropos3class_seg_GmMask_Template_Schaefer400_17.nii.gz'
    fslmeants -i ${scan} --label=${parc_file} -o ${outdir}${out_label}_schaefer400_17.txt

    # schaefer 200
    parc_file='/Volumes/T7/research_data/PNC/processedData/gm_masks_template_pnc/'${scanid}'_atropos3class_seg_GmMask_Template_Schaefer200_17.nii.gz'
    fslmeants -i ${scan} --label=${parc_file} -o ${outdir}${out_label}_schaefer200_17.txt

    # glasser 360
    parc_file='/Volumes/T7/research_data/PNC/template/glasser/pncTemplateGlasser_Labels_2mm.nii.gz'
    fslmeants -i ${scan} --label=${parc_file} -o ${outdir}${out_label}_glasser.txt
done
