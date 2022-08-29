clear all; clc
outdir = '/Users/lindenmp/research_projects/nct_hierarchy/output_local/pnc/schaefer_200_streamlineCount/pipelines/spdcm/'
spmdir = '/Users/lindenmp/Google-Drive-Penn/work/matlab_tools/spm12';
fileprefix = 'hist-g2'
%fileprefix = 'func-g1'
rsts_file = [outdir 'rsts_states_', fileprefix, '_ns-20.npy'];
rsts = readNPY(rsts_file);
size(rsts)
tr = 3;
te = 0.032;

%cd('/Users/lindenmp/Google-Drive-Penn/work/research_projects/nct_hierarchy/matlab_functions')
%spdcm_firstlevel(spmdir, rsts, tr, te, outdir)
cd('/Users/lindenmp/Google-Drive-Penn/work/research_projects/nct_hierarchy/matlab_functions')
spdcm_firstlevel_loop(spmdir, rsts, tr, te, outdir, fileprefix)

%corr(A(eye(10) ~= 1), DCM.Ep.A(eye(10) ~= 1), 'Type', 'Pearson')