function [] = spdcm_firstlevel(spmdir, rsts, tr, te, outdir)
    % where spm12 is
    addpath(spmdir)
    cd(outdir)

    dims = size(rsts);
    n_rois = dims(2);
    n_trs = dims(1);

    % initialize
    spm('defaults','fmri')
    spm_jobman('initcfg')
    spm_get_defaults('mask.thresh',-Inf)

    % setup dcm
    clear dcm
    dcm.n = n_rois; % number of regions.
    dcm.v = n_trs; % number of time points.

    % add time series
    dcm.Y.dt = tr; % add tr
    dcm.Y.y = rsts;
    for i = 1:dcm.n
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
%    dcm.d = zeros(dcm.n,dcm.n,0);
    dcm.d = double.empty(dcm.n,dcm.n,0);

    % wrap dcm
    DCM{1} = dcm;
    DCM{1}.name = ['dcm_ns-', num2str(n_rois)]
    DCM{1}

    % invert fully connected model
    fprintf(1, '\nInverting model... \n');
    DCM{1} = spm_dcm_fmri_csd(DCM{1});
end
