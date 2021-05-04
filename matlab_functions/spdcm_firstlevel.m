function [] = spdcm_firstlevel(spmdir, rsts, tr, te, outdir)
    % rng('default')

    % where spm12 is
    % spmdir = '/Users/lindenmp/Google-Drive-Penn/work/matlab_tools/spm12';
    addpath(spmdir)

    cd(outdir)

    % rsts_file = '/Users/lindenmp/Google-Drive-Penn/work/research_projects/pfactor_gradients/output_cluster/pnc/schaefer_400_streamlineCount/pipelines/spdcm/rsts_states.npy';
    % rsts = readNPY(rsts_file);
    dims = size(rsts);
    n_rois = dims(2);
    % n_rois = 5;
    n_trs = dims(1);
    % tr = 3;
    % te = 0.032;

    % initialize
    spm('defaults','fmri')
    spm_jobman('initcfg')
    spm_get_defaults('mask.thresh',-Inf)

    % setup dcm
    clear dcm
    dcm.n = n_rois; % number of regions.
    dcm.v = n_trs; % number of time points.

    % experimental inputs (there are none)
    dcm.U.u = zeros(n_rois,1);
    dcm.U.name = {'null'};

    dcm.Y.dt  = tr; % add tr
    dcm.Y.X0  = ones(n_rois,1); % confounds. none, add constant

    % add time series
    for i = 1:dcm.n
        dcm.Y.y(:,i) = rsts(:, i);
        dcm.Y.name{i} = i;
    end

    % add precision components
    dcm.Y.Q = spm_Ce(ones(1,dcm.n)*dcm.v);

    % dcm parameters and options
    dcm.delays = repmat(tr/2,dcm.n,1);
    dcm.TE = te; % add rsfmri te

    % set options
    dcm.options.nonlinear = 0;
    dcm.options.two_state = 0;
    dcm.options.stochastic = 0;
    dcm.options.centre = 1;
    dcm.options.induced = 1;
    dcm.options.maxnodes = n_rois;

    % add connections to model
     a = ones(n_rois, n_rois); % specify fully connected model
%    a = eye(n_rois);
%    a = a + diag(ones(n_rois-1,1),1);
%    a = a + diag(ones(n_rois-1,1),-1);
    dcm.a = a;
    dcm.b = zeros(dcm.n,dcm.n);
    dcm.c = zeros(dcm.n,1);
    dcm.d = zeros(dcm.n,dcm.n,0);
    % dcm.d = double.empty(dcm.n,dcm.n,0);

    % wrap dcm
    DCM{1} = dcm;
    DCM{1}

    % invert fully connected model
    fprintf(1, '\nInverting model... \n');
    DCM{1} = spm_dcm_fmri_csd(DCM{1});
    % DCM{1} = spm_dcm_fit(DCM{1});
    % DCM{1} = DCM{1}{1};
end
