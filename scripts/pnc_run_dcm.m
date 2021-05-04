clear all; clc
outdir = '/Users/lindenmp/Google-Drive-Penn/work/research_projects/pfactor_gradients/output_local/pnc/schaefer_400_streamlineCount/pipelines/spdcm/'
spmdir = '/Users/lindenmp/Google-Drive-Penn/work/matlab_tools/spm12';
rsts_file = [outdir 'rsts_states_ns-40.npy'];
rsts = readNPY(rsts_file);
size(rsts)
tr = 3;
te = 0.032;
cd('/Users/lindenmp/Google-Drive-Penn/work/research_projects/pfactor_gradients/matlab_functions')
spdcm_firstlevel(spmdir, rsts, tr, te, outdir)
