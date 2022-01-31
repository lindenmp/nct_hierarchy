function [] = spdcm_firstlevel_loop(spmdir, rsts, tr, te, outdir, fileprefix)
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

    % setup outputs
    A = zeros(n_rois, n_rois);

    % setup dcm
    for i = 1:n_rois
        for j = 1:n_rois
            if j > i
                clear dcm
                dcm.n = 2; % number of regions.
                dcm.v = n_trs; % number of time points.

                % add time series
                dcm.Y.dt = tr; % add tr
                dcm.Y.y(:,1) = rsts(:, i);
                dcm.Y.name{1} = i;
                dcm.Y.y(:,2) = rsts(:, j);
                dcm.Y.name{2} = j;

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
                dcm.options.maxnodes = 2;

                % add connections to model
                a = ones(2, 2); % specify fully connected model
                dcm.a = a;
                dcm.b = zeros(dcm.n,dcm.n);
                dcm.c = zeros(dcm.n,1);
                dcm.d = zeros(dcm.n,dcm.n,0);

                % wrap dcm
                DCM{1} = dcm;
                filename = ['dcm_', fileprefix, '_ns-', num2str(n_rois), '-i', num2str(i), 'j', num2str(j)]
                DCM{1}.name = filename
                DCM{1}

                % invert model
                fprintf(1, '\nInverting model... \n');
                DCM{1} = spm_dcm_fmri_csd(DCM{1});

                % store
                A(i, j) = DCM{1}.Ep.A(1, 2);
                A(j, i) = DCM{1}.Ep.A(2, 1);

                % delete interim dcm file
                delete([filename,'.mat'])
            end
        end
    end
    save(['dcm_', fileprefix, '_ns-', num2str(n_rois),'_A.mat'], 'A')
end
