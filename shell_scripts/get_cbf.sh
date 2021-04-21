indir=/Volumes/T7/research_data/PNC/processedData/asl/voxelwiseMaps_cbf/
outdir=/Volumes/T7/research_data/PNC/processedData/asl/parcelwise_cbf/

# schaefer 400
parc_file=/Volumes/T7/research_data/PNC/Schaefer2018_400_17Networks_PNC_2mm.nii.gz
cd ${indir}

for i in *.nii.gz; do
  name=$(echo "$i" | cut -f 1 -d '.')
  echo $name
  fslmeants -i ${indir}${i} --label=${parc_file} -o ${outdir}${name}_schaefer400_17.txt
done

# schaefer 200
parc_file=/Volumes/T7/research_data/PNC/Schaefer2018_200_17Networks_PNC_2mm.nii.gz
cd ${indir}

for i in *.nii.gz; do
  name=$(echo "$i" | cut -f 1 -d '.')
  echo $name
  fslmeants -i ${indir}${i} --label=${parc_file} -o ${outdir}${name}_schaefer200_17.txt
done