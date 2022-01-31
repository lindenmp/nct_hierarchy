clear all; clc
outdir = '/Users/lindenmp/research_projects/pfactor_gradients/output_local/pnc/schaefer_200_streamlineCount/pipelines/spdcm/'
spmdir = '/Users/lindenmp/Google-Drive-Penn/work/matlab_tools/spm12';
rsts_file = [outdir 'rsts_states_hist-g2_ns-20.npy'];
%rsts_file = [outdir 'rsts_states_func-g1_ns-20.npy'];
rsts = readNPY(rsts_file);
size(rsts)
tr = 3;
te = 0.032;

%cd('/Users/lindenmp/Google-Drive-Penn/work/research_projects/pfactor_gradients/matlab_functions')
%spdcm_firstlevel(spmdir, rsts, tr, te, outdir)
cd('/Users/lindenmp/Google-Drive-Penn/work/research_projects/pfactor_gradients/matlab_functions')
spdcm_firstlevel_loop(spmdir, rsts, tr, te, outdir)

%corr(A(eye(10) ~= 1), DCM.Ep.A(eye(10) ~= 1), 'Type', 'Pearson')