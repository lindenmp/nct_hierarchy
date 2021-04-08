ssh cbica

conda activate pfactor_gradients

# average a matrix
qsub -N pnc_average_adj -l h_vmem=3G,s_vmem=3G -pe threaded 8 -j y -b y -o /cbica/home/parkesl/sge/ -e /cbica/home/parkesl/sge/ \
-t 1:4 \
/cbica/home/parkesl/miniconda3/envs/pfactor_gradients/bin/python \
/cbica/home/parkesl/research_projects/pfactor_gradients/scripts/pnc_average_adj_compute_control_energy_tmp.py

# pnc subjects, array job
qsub -N subjects -l h_vmem=3G,s_vmem=3G -pe threaded 4 -j y -b y -o /cbica/home/parkesl/sge/ -e /cbica/home/parkesl/sge/ \
-t 1:100 \
/cbica/home/parkesl/miniconda3/envs/pfactor_gradients/bin/python \
/cbica/home/parkesl/research_projects/pfactor_gradients/scripts/pnc_compute_control_energy.py