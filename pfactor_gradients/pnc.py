import os, glob
import numpy as np
import scipy.io as sio
import pandas as pd
from nilearn import datasets

from pfactor_gradients.imaging_derivs import DataMatrix, compute_fc, compute_rlfp

class Environment():
    def __init__(self, computer='macbook', parc='schaefer', n_parcels=400, sc_edge_weight='streamlineCount'):
        # analysis parameters
        self.computer = computer
        self.parc = parc
        self.n_parcels = n_parcels
        self.sc_edge_weight = sc_edge_weight

        # directories
        if self.computer == 'macbook':
            self.userdir = '/Users/lindenmp'
            # self.projdir = os.path.join(self.userdir, 'Google-Drive-Penn', 'work', 'research_projects', 'pfactor_gradients')
            self.projdir = os.path.join(self.userdir, 'research_projects', 'pfactor_gradients')
            self.rootdir = '/Volumes'
            # self.research_data = os.path.join(self.rootdir, 'T7', 'research_data')
            self.research_data = os.path.join(self.userdir, 'research_data')

            self.outputdir = os.path.join(self.projdir, 'output_local', 'pnc', '{0}_{1}_{2}'.format(self.parc, self.n_parcels, self.sc_edge_weight))
        elif self.computer == 'cbica':
            self.userdir = '/cbica/home/parkesl'
            self.projdir = os.path.join(self.userdir, 'research_projects', 'pfactor_gradients')
            self.research_data = os.path.join(self.userdir, 'research_data')

            self.outputdir = os.path.join(self.projdir, 'output_cluster', 'pnc', '{0}_{1}_{2}'.format(self.parc, self.n_parcels, self.sc_edge_weight))
        # self.outputdir = os.path.join(self.projdir, 'output_cluster', 'pnc', '{0}_{1}_{2}'.format(self.parc, self.n_parcels, self.sc_edge_weight))

        self.pipelinedir = os.path.join(self.outputdir, 'pipelines')
        self.figdir = os.path.join(self.outputdir, 'figures')
        self.datadir = os.path.join(self.research_data, 'pnc')
        self.freezedir = os.path.join(self.datadir, 'pncDataFreeze20170905', 'n1601_dataFreeze')
        if self.parc == 'schaefer':
            self.scdir = os.path.join(self.datadir, 'processedData', 'diffusion', 'deterministic_20171118')
        elif self.parc == 'glasser':
            self.scdir = os.path.join(self.datadir, 'processedData', 'diffusion', 'deterministic_dec2016')
        # self.ctdir = os.path.join(self.datadir, 'processedData', 'antsCt', 'parcelwise_antsCt')
        self.ctdir = os.path.join(self.datadir, 'processedData', 'structural', 'freesurfer53')
        self.sadir = os.path.join(self.datadir, 'processedData', 'structural', 'freesurfer53')
        self.rstsdir = os.path.join(self.datadir, 'processedData', 'restbold', 'restbold_201607151621')

        # imaging parameters
        self.rsfmri_tr = 3
        self.n_trs = 120
        self.rsfmri_te = 0.032
        self.rsfmri_low = 0.01
        self.rsfmri_high = 0.08

    def make_output_dirs(self):
        if not os.path.exists(self.pipelinedir): os.makedirs(self.pipelinedir)
        if not os.path.exists(self.outputdir): os.makedirs(self.outputdir)
        if not os.path.exists(self.figdir): os.makedirs(self.figdir)

    def load_metadata(self, filters=[]):
        """
        Parameters
        ----------
        filters :
            dictionary of filters to apply to dataframe (see PNC documentation for details)

        Returns
        -------
        df : pd.DataFrame (n_subjects,n_variables)
            dataframe of subject meta data for the PNC
        """

        print("Data loader: loading metadata for PNC...")

        # LTN and Health Status
        health = pd.read_csv(os.path.join(self.freezedir, 'health', 'n1601_health_20170421.csv'))
        # Protocol
        prot = pd.read_csv(os.path.join(self.freezedir, 'neuroimaging', 'n1601_pnc_protocol_validation_params_status_20161220.csv'))
        # T1 QA
        t1_qa = pd.read_csv(os.path.join(self.freezedir, 'neuroimaging', 't1struct', 'n1601_t1QaData_20170306.csv'))
        # DTI QA
        dti_qa = pd.read_csv(os.path.join(self.freezedir, 'neuroimaging', 'dti', 'n1601_dti_qa_20170301.csv'))
        # Rest QA
        rest_qa = pd.read_csv(os.path.join(self.freezedir, 'neuroimaging', 'rest', 'n1601_RestQAData_20170714.csv'))
        # Demographics
        demog = pd.read_csv(os.path.join(self.freezedir, 'demographics', 'n1601_demographics_go1_20161212.csv'))
        # Brain volume
        brain_vol = pd.read_csv(os.path.join(self.freezedir, 'neuroimaging', 't1struct', 'n1601_ctVol20170412.csv'))
        # Clinical diagnostic
        clinical = pd.read_csv(os.path.join(self.freezedir, 'clinical', 'n1601_goassess_psych_summary_vars_20131014.csv'))
        clinical_ps = pd.read_csv(os.path.join(self.freezedir, 'clinical', 'n1601_diagnosis_dxpmr_20170509.csv'))
        # GOASSESS Bifactor scores
        goassess = pd.read_csv(os.path.join(self.datadir, 'GO1_clinical_factor_scores_psychosis_split_BIFACTOR.csv'))
        # Cognition
        cnb = pd.read_csv(os.path.join(self.freezedir, 'cnb', 'n1601_cnb_factor_scores_tymoore_20151006.csv'))

        # merge
        df = health
        df = pd.merge(df, prot, on=['scanid', 'bblid'])  # prot
        df = pd.merge(df, t1_qa, on=['scanid', 'bblid'])  # t1_qa
        df = pd.merge(df, dti_qa, on=['scanid', 'bblid'])  # dti_qa
        df = pd.merge(df, rest_qa, on=['scanid', 'bblid'])  # rest_qa
        df = pd.merge(df, demog, on=['scanid', 'bblid'])  # demog
        df = pd.merge(df, brain_vol, on=['scanid', 'bblid'])  # brain_vol
        df = pd.merge(df, clinical, on=['scanid', 'bblid'])  # clinical
        df = pd.merge(df, clinical_ps, on=['scanid', 'bblid'])  # clinical
        df = pd.merge(df, goassess, on=['bblid'])  # goassess
        df = pd.merge(df, cnb, on=['scanid', 'bblid'])  # goassess

        # df.set_index(['bblid', 'scanid'], inplace=True)
        df['subjid'] = df['bblid'].astype(str) + '_' + df['scanid'].astype(str)
        df.set_index(['subjid'], inplace=True)

        # filter dataframe
        print('\tInitial sample: {0} subjects with {1} columns'.format(df.shape[0], df.shape[1]))
        if len(filters) > 0:
            for filter in filters:
                n_excluded = df.shape[0] - df[df[filter] == filters[filter]].shape[0]
                df = df[df[filter] == filters[filter]]
                n_retained = df.shape[0]
                print('\tN after initial {0} exclusion: {1} ({2} excluded)'.format(filter, n_retained, n_excluded))

        print('\tFinal sample: {0} subjects with {1} columns'.format(df.shape[0], df.shape[1]))

        self.df = df

    def load_parc_data(self):
        if self.parc == 'schaefer':
            self.parcel_names = list(np.genfromtxt(os.path.join(self.research_data, 'parcellations', 'support_files',
                                                           'schaefer{0}NodeNames.txt'.format(self.n_parcels)),
                                                   dtype='str'))
            self.fsaverage = datasets.fetch_surf_fsaverage(mesh='fsaverage5')
            self.lh_annot_file = os.path.join(self.research_data, 'parcellations', 'FreeSurfer5.3',
                                              'fsaverage5', 'label',
                                              'lh.Schaefer2018_{0}Parcels_17Networks_order.annot'.format(self.n_parcels))
            self.rh_annot_file = os.path.join(self.research_data, 'parcellations', 'FreeSurfer5.3',
                                              'fsaverage5', 'label',
                                              'rh.Schaefer2018_{0}Parcels_17Networks_order.annot'.format(self.n_parcels))
            self.centroids = pd.read_csv(os.path.join(self.research_data, 'parcellations', 'MNI', 'Centroid_coordinates',
                                                      'Schaefer2018_{0}Parcels_17Networks_order_FSLMNI152_1mm.Centroid_RAS.csv'.format(self.n_parcels)))
            self.centroids.drop('ROI Index', axis=1, inplace=True)
            self.centroids.set_index('Label Name', inplace=True)
            self.centroids.drop('NONE', axis=0, inplace=True)
            self.spun_indices = np.genfromtxt(os.path.join(self.research_data, 'parcellations', 'spin_test',
                                                        'rotated_ind_schaefer{0}.csv'.format(self.n_parcels)),
                                              delimiter=',', dtype=int)
            self.spun_indices = self.spun_indices - 1

        elif self.parc == 'glasser':
            self.parcel_names = list(np.genfromtxt(os.path.join(self.research_data, 'parcellations', 'support_files',
                                                           'glasser{0}NodeNames.txt'.format(self.n_parcels)),
                                                   dtype='str'))
            self.fsaverage = datasets.fetch_surf_fsaverage(mesh='fsaverage5')
            self.lh_annot_file = os.path.join(self.research_data, 'parcellations',
                                              'Glasser_et_al_2016_HCP_MMP1.0_qN_RVVG', 'HCP-MMP1.fsaverage5.L.annot')
            self.rh_annot_file = os.path.join(self.research_data, 'parcellations',
                                              'Glasser_et_al_2016_HCP_MMP1.0_qN_RVVG', 'HCP-MMP1.fsaverage5.R.annot')

            self.centroids = pd.read_csv(os.path.join(self.research_data, 'parcellations', 'support_files',
                                                    'HCP-MMP1_UniqueRegionList.csv'))
            self.centroids = pd.concat((self.centroids.iloc[180:, :], self.centroids.iloc[0:180, :]), axis=0)
            self.centroids.reset_index(inplace=True, drop=True)
            self.centroids = self.centroids.loc[:, ['x-cog', 'y-cog', 'z-cog']]
            self.spun_indices = np.genfromtxt(os.path.join(self.research_data, 'parcellations', 'spin_test',
                                                        'rotated_ind_glasser{0}.csv'.format(self.n_parcels)),
                                              delimiter=',', dtype=int)
            self.spun_indices = self.spun_indices - 1

class Subject():
    def __init__(self, environment=Environment(), subjid='81287_2738'):
        self.environment = environment
        self.subjid = subjid
        self.bblid = subjid.split('_')[0]
        self.scanid = subjid.split('_')[1]

    def get_file_names(self):
        if self.environment.parc == 'schaefer':
            sc_filename = os.path.join('{0}'.format(self.bblid),
                                       '*x{0}'.format(self.scanid),
                                       'tractography', 'connectivity',
                                       '{0}_*x{1}_SchaeferPNC_{2}_dti_{3}_connectivity.mat' \
                                       .format(self.bblid, self.scanid, self.environment.n_parcels, self.environment.sc_edge_weight))
            sc_filename = glob.glob(os.path.join(self.environment.scdir, sc_filename))

            ct_filename = os.path.join('{0}'.format(self.bblid),
                                       '*x{0}'.format(self.scanid),
                                       'surf',
                                       'thickness_Schaefer2018_{0}Parcels_17Networks.txt' \
                                       .format(self.environment.n_parcels))
            # ct_filename = os.path.join('{0}_CorticalThicknessNormalizedToTemplate2mm_schaefer{1}_17.txt' \
            #                            .format(self.scanid, self.environment.n_parcels))
            ct_filename = glob.glob(os.path.join(self.environment.ctdir, ct_filename))

            sa_filename = os.path.join('{0}'.format(self.bblid),
                                       '*x{0}'.format(self.scanid),
                                       'surf',
                                       'area_pial_Schaefer2018_{0}Parcels_17Networks.txt' \
                                       .format(self.environment.n_parcels))
            sa_filename = glob.glob(os.path.join(self.environment.sadir, sa_filename))

            if self.environment.n_parcels == 200:
                rsts_filename = os.path.join('{0}'.format(self.bblid),
                                             '*x{0}'.format(self.scanid),
                                             'net',
                                             'Schaefer{0}PNC'.format(self.environment.n_parcels),
                                             '{0}_*x{1}_Schaefer{2}PNC_ts.1D' \
                                             .format(self.bblid, self.scanid, self.environment.n_parcels))
                rsts_filename = glob.glob(os.path.join(self.environment.rstsdir, rsts_filename))

                alff_filename = os.path.join('{0}'.format(self.bblid),
                                             '*x{0}'.format(self.scanid),
                                             'alff', 'roi', 'Schaefer{0}PNC'.format(self.environment.n_parcels),
                                             '{0}_*x{1}_Schaefer{2}PNC_val_alff.1D' \
                                             .format(self.bblid, self.scanid, self.environment.n_parcels))
                alff_filename = glob.glob(os.path.join(self.environment.rstsdir, alff_filename))
            elif self.environment.n_parcels == 400:
                rsts_filename = os.path.join('{0}'.format(self.bblid),
                                             '*x{0}'.format(self.scanid),
                                             'net',
                                             'SchaeferPNC',
                                             '{0}_*x{1}_SchaeferPNC_ts.1D' \
                                             .format(self.bblid, self.scanid))
                rsts_filename = glob.glob(os.path.join(self.environment.rstsdir, rsts_filename))

                alff_filename = os.path.join('{0}'.format(self.bblid),
                                             '*x{0}'.format(self.scanid),
                                             'alff', 'roi', 'SchaeferPNC',
                                             '{0}_*x{1}_SchaeferPNC_val_alff.1D' \
                                             .format(self.bblid, self.scanid))
                alff_filename = glob.glob(os.path.join(self.environment.rstsdir, alff_filename))
        elif self.environment.parc == 'glasser':
            sc_filename = os.path.join('{0}'.format(self.environment.sc_edge_weight), 'GlasserPNC',
                                       '{0}_{1}_GlasserPNC.mat' \
                                       .format(self.scanid, self.environment.sc_edge_weight))
            sc_filename = glob.glob(os.path.join(self.environment.scdir, sc_filename))

            ct_filename = os.path.join('{0}'.format(self.bblid),
                                       '*x{0}'.format(self.scanid),
                                       'surf',
                                       'thickness_HCP-MMP1.txt')
            # ct_filename = os.path.join('{0}_CorticalThicknessNormalizedToTemplate2mm_glasser.txt'.format(self.scanid))
            ct_filename = glob.glob(os.path.join(self.environment.ctdir, ct_filename))

            sa_filename = os.path.join('{0}'.format(self.bblid),
                                       '*x{0}'.format(self.scanid),
                                       'surf',
                                       'area_pial_HCP-MMP1.txt')
            sa_filename = glob.glob(os.path.join(self.environment.sadir, sa_filename))

            rsts_filename = os.path.join('{0}'.format(self.bblid),
                                         '*x{0}'.format(self.scanid),
                                         'net',
                                         'GlasserPNC',
                                         '{0}_*x{1}_GlasserPNC_ts.1D' \
                                         .format(self.bblid, self.scanid))
            rsts_filename = glob.glob(os.path.join(self.environment.rstsdir, rsts_filename))

            alff_filename = os.path.join('{0}'.format(self.bblid),
                                         '*x{0}'.format(self.scanid),
                                         'alff', 'roi', 'GlasserPNC',
                                         '{0}_*x{1}_GlasserPNC_val_alff.1D' \
                                         .format(self.bblid, self.scanid))
            alff_filename = glob.glob(os.path.join(self.environment.rstsdir, alff_filename))

        try: self.sc_filename = sc_filename[0]
        except: self.sc_filename = []

        try: self.ct_filename = ct_filename[0]
        except: self.ct_filename = []

        try: self.sa_filename = sa_filename[0]
        except: self.sa_filename = []

        try: self.rsts_filename = rsts_filename[0]
        except: self.rsts_filename = []

        try: self.alff_filename = alff_filename[0]
        except: self.alff_filename = []

    def load_sc(self):
        try:
            mat_contents = sio.loadmat(self.sc_filename)
            self.sc = DataMatrix(data=mat_contents['connectivity'])
        except:
            sc = np.zeros((self.environment.n_parcels, self.environment.n_parcels))
            sc[:] = np.nan
            self.sc = DataMatrix(data=sc)

        if self.sc.data.shape[0] != self.environment.n_parcels or self.sc.data.shape[1] != self.environment.n_parcels:
            sc = np.zeros((self.environment.n_parcels, self.environment.n_parcels))
            sc[:] = np.nan
            self.sc = DataMatrix(data=sc)

    def load_ct(self):
        if not self.ct_filename:
            self.ct = np.zeros((self.environment.n_parcels,))
            self.ct[:] = np.nan
        else:
            self.ct = np.loadtxt(self.ct_filename)

    def load_sa(self):
        if not self.sa_filename:
            self.sa = np.zeros((self.environment.n_parcels,))
            self.sa[:] = np.nan
        else:
            self.sa = np.loadtxt(self.sa_filename)

    def load_rsts(self):
        if not self.rsts_filename:
            self.rsts = np.zeros((self.environment.n_trs, self.environment.n_parcels))
            self.rsts[:] = np.nan
        else:
            self.rsts = np.loadtxt(self.rsts_filename)

    def load_rsfc(self):
        self.load_rsts()

        if np.all(np.isnan(self.rsts)):
            rsfc = np.zeros((self.environment.n_parcels, self.environment.n_parcels))
            rsfc[:] = np.nan
            self.rsfc = DataMatrix(data=rsfc)
        else:
            self.rsfc = DataMatrix(data=compute_fc(self.rsts))

    def load_rlfp(self):
        if not self.rsts_filename:
            self.rlfp = np.zeros((self.environment.n_parcels,))
            self.rlfp[:] = np.nan
        else:
            rsts = np.loadtxt(self.rsts_filename)
            n_parcels = rsts.shape[1]

            rlfp = np.zeros((n_parcels,))
            for i in np.arange(n_parcels):
                rlfp[i] = compute_rlfp(rsts[:, i], tr=self.environment.rsfmri_tr,
                                       low=self.environment.rsfmri_low, high=self.environment.rsfmri_high,
                                       num_bands=5, band_of_interest=1)

            self.rlfp = rlfp

    def load_alff(self):
        if not self.alff_filename:
            self.alff = np.zeros((self.environment.n_parcels,))
            self.alff[:] = np.nan
        else:
            self.alff = np.genfromtxt(self.alff_filename, skip_header=1)[2:]
