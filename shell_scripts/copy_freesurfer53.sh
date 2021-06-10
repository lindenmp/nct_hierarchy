server_dir=/data/joy/BBL/studies/pnc/processedData/structural/freesurfer53/
out_dir=/data/jag/bassett-lab/lindenmp/pnc/processedData/structural/freesurfer53/

# /surf/
cd ${server_dir}
for i in */; do
  cd ${server_dir}${i}
  for j in 2*/; do
    echo ${i}${j}
    mkdir -p ${out_dir}${i}${j}surf/
    cp ${server_dir}${i}${j}surf/lh.area.fsaverage.mgh ${out_dir}${i}${j}surf/
    cp ${server_dir}${i}${j}surf/lh.area.pial.fsaverage.mgh ${out_dir}${i}${j}surf/
    cp ${server_dir}${i}${j}surf/lh.curv.fsaverage.mgh ${out_dir}${i}${j}surf/
    cp ${server_dir}${i}${j}surf/lh.sulc.fsaverage.mgh ${out_dir}${i}${j}surf/
    cp ${server_dir}${i}${j}surf/lh.thickness.fsaverage.mgh ${out_dir}${i}${j}surf/
    cp ${server_dir}${i}${j}surf/lh.volume.fsaverage.mgh ${out_dir}${i}${j}surf/

    cp ${server_dir}${i}${j}surf/rh.area.fsaverage.mgh ${out_dir}${i}${j}surf/
    cp ${server_dir}${i}${j}surf/rh.area.pial.fsaverage.mgh ${out_dir}${i}${j}surf/
    cp ${server_dir}${i}${j}surf/rh.curv.fsaverage.mgh ${out_dir}${i}${j}surf/
    cp ${server_dir}${i}${j}surf/rh.sulc.fsaverage.mgh ${out_dir}${i}${j}surf/
    cp ${server_dir}${i}${j}surf/rh.thickness.fsaverage.mgh ${out_dir}${i}${j}surf/
    cp ${server_dir}${i}${j}surf/rh.volume.fsaverage.mgh ${out_dir}${i}${j}surf/
  done
done
