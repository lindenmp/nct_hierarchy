ssh cbica

conda activate pfactor_gradients

# average a matrix
qsub -N avg_adj -l h_vmem=3G,s_vmem=3G -pe threaded 2 -j y -b y -o /cbica/home/parkesl/sge/ -e /cbica/home/parkesl/sge/ \
/cbica/home/parkesl/miniconda3/envs/pfactor_gradients/bin/python \
/cbica/home/parkesl/research_projects/pfactor_gradients/scripts/pnc_average_adj_compute_control_energy.py

# average a matrix, null
qsub -N avg_adj_null -l h_vmem=3G,s_vmem=3G -pe threaded 2 -j y -b y -o /cbica/home/parkesl/sge/ -e /cbica/home/parkesl/sge/ \
-t 1:1000 -tc 100 \
/cbica/home/parkesl/miniconda3/envs/pfactor_gradients/bin/python \
/cbica/home/parkesl/research_projects/pfactor_gradients/scripts/pnc_average_adj_compute_control_energy_null.py

# pnc subjects, array job
qsub -N subjects -l h_vmem=3G,s_vmem=3G -pe threaded 1 -j y -b y -o /cbica/home/parkesl/sge/ -e /cbica/home/parkesl/sge/ \
-t 1:775 \
/cbica/home/parkesl/miniconda3/envs/pfactor_gradients/bin/python \
/cbica/home/parkesl/research_projects/pfactor_gradients/scripts/pnc_compute_control_energy.py

# prediction
qsub -N prediction -l h_vmem=3G,s_vmem=3G -pe threaded 8 -j y -b y -o /cbica/home/parkesl/sge/ -e /cbica/home/parkesl/sge/ \
-t 1:20 \
/cbica/home/parkesl/miniconda3/envs/pfactor_gradients/bin/python \
/cbica/home/parkesl/research_projects/pfactor_gradients/scripts/pnc_run_prediction.py
