conda activate pfactor_gradients

# 1) average a matrix, null
# run first
qsub -N e_null -l h_vmem=3G,s_vmem=3G -pe threaded 2 -j y -b y -o /cbica/home/parkesl/sge/ -e /cbica/home/parkesl/sge/ \
-t 1 \
/cbica/home/parkesl/miniconda3/envs/pfactor_gradients/bin/python \
/cbica/home/parkesl/research_projects/pfactor_gradients/scripts/average_adj_control_energy_nulls.py

# run second when above is finished
qsub -N e_null -l h_vmem=3G,s_vmem=3G -pe threaded 2 -j y -b y -o /cbica/home/parkesl/sge/ -e /cbica/home/parkesl/sge/ \
-t 2:10000 -tc 500 \
/cbica/home/parkesl/miniconda3/envs/pfactor_gradients/bin/python \
/cbica/home/parkesl/research_projects/pfactor_gradients/scripts/average_adj_control_energy_nulls.py

# 2) pnc subjects, array job
qsub -N subjects -l h_vmem=3G,s_vmem=3G -pe threaded 1 -j y -b y -o /cbica/home/parkesl/sge/ -e /cbica/home/parkesl/sge/ \
-t 1:775 \
/cbica/home/parkesl/miniconda3/envs/pfactor_gradients/bin/python \
/cbica/home/parkesl/research_projects/pfactor_gradients/scripts/pnc_compute_control_energy.py

# 3) prediction
X_list=('wb' 'ct' 'cbf' 'reho' 'alff')
y_list=('Overall_Psychopathology' 'F3_Executive_Efficiency' 'F1_Exec_Comp_Res_Accuracy')
alg_list=('rr')
score_list=('rmse' 'corr')
runpca_list=('1%' '80%' 50)

for X_name in "${X_list[@]}"; do
  for y_name in "${y_list[@]}"; do
    for alg in "${alg_list[@]}"; do
      for score in "${score_list[@]}"; do
        for runpca in "${runpca_list[@]}"; do
          qsub -N prediction -l h_vmem=3G,s_vmem=3G -pe threaded 8 -j y -b y -o \
          /cbica/home/parkesl/sge/ -e /cbica/home/parkesl/sge/ \
          /cbica/home/parkesl/miniconda3/envs/pfactor_gradients/bin/python \
          /cbica/home/parkesl/research_projects/pfactor_gradients/scripts/pnc_run_prediction.py \
          -X_name $X_name -y_name $y_name -alg $alg -score $score -runpca $runpca
        done
      done
    done
  done
done
