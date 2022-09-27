# clean up on cbica
rm -rf core.*
rm -rf ./sge/*

conda activate pfactor_gradients

# 1) pnc subjects, array job
# 793 (schaefer200) | 766 (schaefer400) | 792 (glasser360)
qsub -N subjects -l h_vmem=8G,s_vmem=8G -pe threaded 2 -j y -b y -o /cbica/home/parkesl/sge/ -e /cbica/home/parkesl/sge/ \
-t 1:793 \
/cbica/home/parkesl/miniconda3/envs/pfactor_gradients/bin/python \
/cbica/home/parkesl/research_projects/nct_hierarchy/scripts/subjects_compute_control_energy.py

# 2) bootstrap mean adj matrix and compute energy
qsub -N e_bootstrap -l h_vmem=4G,s_vmem=4G -pe threaded 1 -j y -b y -o /cbica/home/parkesl/sge/ -e /cbica/home/parkesl/sge/ \
-t 1:1000 \
/cbica/home/parkesl/miniconda3/envs/pfactor_gradients/bin/python \
/cbica/home/parkesl/research_projects/nct_hierarchy/scripts/ave_adj_generate_energy_bootstraps.py
