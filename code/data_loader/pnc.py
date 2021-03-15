import os, glob
import numpy as np
import scipy.io as sio
import pandas as pd

from code.utils.imaging_derivs import compute_fc, compute_rlfp


class Environment():
    def __init__(self):
        # directories
        self.projdir = '/Users/lindenmp/Google-Drive-Penn/work/research_projects/pfactor_gradients'
        self.datadir = os.path.join(self.projdir, '0_data')
        self.pipelinedir = os.path.join(self.projdir, '2_pipeline')
        self.outputdir = os.path.join(self.projdir, '3_output')

        self.derivsdir = '/Volumes/work_ssd/research_data/PNC/'
        self.scdir = os.path.join(self.derivsdir, 'processedData', 'diffusion', 'deterministic_20171118')
        self.ctdir = os.path.join(self.derivsdir, 'processedData', 'antsCorticalThickness')
        self.rstsdir = os.path.join(self.derivsdir, 'processedData', 'restbold', 'restbold_201607151621')

        # imaging parameters
        self.rsfmri_tr = 3


class Subject(Environment):
    def __init__(self, bblid=0, scanid=0):
        Environment.__init__(self)
        self.bblid = bblid
        self.scanid = scanid

    def get_file_names(self, parc='schaefer', parc_res=400, sc_edge_weight='streamlineCount'):
        self.parc = parc
        self.parc_res = parc_res
        self.sc_edge_weight = sc_edge_weight

        if parc == 'schaefer':
            sc_filename = os.path.join('{0}'.format(self.bblid),
                                       '*x{0}'.format(self.scanid),
                                       'tractography', 'connectivity',
                                       '{0}_*x{1}_SchaeferPNC_{2}_dti_{3}_connectivity.mat' \
                                       .format(self.bblid, self.scanid, self.parc_res, self.sc_edge_weight))
            sc_filename = glob.glob(os.path.join(self.scdir, sc_filename))[0]

            ct_filename = os.path.join('{0}'.format(self.bblid),
                                       '*x{0}'.format(self.scanid),
                                       'ct_schaefer{0}_17.txt'.format(self.parc_res))
            ct_filename = glob.glob(os.path.join(self.ctdir, ct_filename))[0]

            if parc_res == 200:
                rsts_filename = os.path.join('{0}'.format(self.bblid),
                                             '*x{0}'.format(self.scanid),
                                             'net', 'Schaefer{0}PNC'.format(self.parc_res),
                                             '{0}_*x{1}_Schaefer{2}PNC_ts.1D' \
                                             .format(self.bblid, self.scanid, self.parc_res))
                rsts_filename = glob.glob(os.path.join(self.rstsdir, rsts_filename))[0]
            elif parc_res == 400:
                rsts_filename = os.path.join('{0}'.format(self.bblid),
                                             '*x{0}'.format(self.scanid),
                                             'net', 'SchaeferPNC',
                                             '{0}_*x{1}_SchaeferPNC_ts.1D' \
                                             .format(self.bblid, self.scanid))
                rsts_filename = glob.glob(os.path.join(self.rstsdir, rsts_filename))[0]

        self.sc_filename = sc_filename
        self.ct_filename = ct_filename
        self.rsts_filename = rsts_filename

    def load_adj_matrix(self):
        mat_contents = sio.loadmat(self.sc_filename)

        self.adj_matrix = mat_contents['connectivity']

    def load_ct(self):
        self.ct = np.loadtxt(self.ct_filename)

    def load_rsfc(self):
        rsts = np.loadtxt(self.rsts_filename)

        self.rsfc = compute_fc(rsts)

    def load_rlfp(self):
        rsts = np.loadtxt(self.rsts_filename)
        n_parcels = rsts.shape[1]

        rlfp = np.zeros(n_parcels)
        for i in np.arange(n_parcels):
            rlfp[i] = compute_rlfp(rsts[:, i], tr=self.rsfmri_tr, num_bands=5, band_of_interest=1)

        self.rlfp = rlfp


def load_metadata(environment, filters = []):
    """
    Parameters
    ----------
    environment :
        an instance of Environment() from above

    Returns
    -------
    df : pd.DataFrame (n_subjects,n_variables)
        dataframe of subject meta data for the PNC
    """
    # LTN and Health Status
    health = pd.read_csv(os.path.join(environment.datadir, 'external', 'pncDataFreeze20170905', 'n1601_dataFreeze',
                                      'health', 'n1601_health_20170421.csv'))
    # Protocol
    prot = pd.read_csv(os.path.join(environment.datadir, 'external', 'pncDataFreeze20170905', 'n1601_dataFreeze',
                                    'neuroimaging', 'n1601_pnc_protocol_validation_params_status_20161220.csv'))
    # T1 QA
    t1_qa = pd.read_csv(os.path.join(environment.datadir, 'external', 'pncDataFreeze20170905', 'n1601_dataFreeze',
                                     'neuroimaging', 't1struct', 'n1601_t1QaData_20170306.csv'))
    # DTI QA
    dti_qa = pd.read_csv(os.path.join(environment.datadir, 'external', 'pncDataFreeze20170905', 'n1601_dataFreeze',
                                      'neuroimaging', 'dti', 'n1601_dti_qa_20170301.csv'))
    # Rest QA
    rest_qa = pd.read_csv(os.path.join(environment.datadir, 'external', 'pncDataFreeze20170905', 'n1601_dataFreeze',
                                       'neuroimaging', 'rest', 'n1601_RestQAData_20170714.csv'))
    # Demographics
    demog = pd.read_csv(os.path.join(environment.datadir, 'external', 'pncDataFreeze20170905', 'n1601_dataFreeze',
                                     'demographics', 'n1601_demographics_go1_20161212.csv'))
    # Brain volume
    brain_vol = pd.read_csv(os.path.join(environment.datadir, 'external', 'pncDataFreeze20170905', 'n1601_dataFreeze',
                                         'neuroimaging', 't1struct', 'n1601_ctVol20170412.csv'))
    # Clinical diagnostic
    clinical = pd.read_csv(os.path.join(environment.datadir, 'external', 'pncDataFreeze20170905', 'n1601_dataFreeze',
                                        'clinical', 'n1601_goassess_psych_summary_vars_20131014.csv'))
    clinical_ps = pd.read_csv(os.path.join(environment.datadir, 'external', 'pncDataFreeze20170905', 'n1601_dataFreeze',
                                           'clinical', 'n1601_diagnosis_dxpmr_20170509.csv'))
    # GOASSESS Bifactor scores
    goassess = pd.read_csv(os.path.join(environment.datadir, 'external',
                                        'GO1_clinical_factor_scores_psychosis_split_BIFACTOR.csv'))
    # Cognition
    cnb = pd.read_csv(os.path.join(environment.datadir, 'external', 'pncDataFreeze20170905', 'n1601_dataFreeze',
                                   'cnb', 'n1601_cnb_factor_scores_tymoore_20151006.csv'))

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

    df.set_index(['bblid', 'scanid'], inplace=True)

    # filter dataframe
    if len(filters) > 0:
        for filter in filters:
            df = df[df[filter] == filters[filter]]
            print('N after initial {0} exclusion: {1}'.format(filter, df.shape[0]))

    return df

