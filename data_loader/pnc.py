import os, glob


class Project():
    def __init__(self):
        self.projdir = '/Users/lindenmp/Google-Drive-Penn/work/research_projects/pfactor_gradients'
        self.datadir = os.path.join(self.projdir, '0_data')
        self.pipelinedir = os.path.join(self.projdir, '2_pipeline')
        self.outputdir = os.path.join(self.projdir, '3_output')

        self.derivsdir = '/Volumes/work_ssd/research_data/PNC/'
        self.scdir = os.path.join(self.derivsdir, 'processedData', 'diffusion', 'deterministic_20171118')
        self.ctdir = os.path.join(self.derivsdir, 'processedData', 'antsCorticalThickness')
        self.rstsdir = os.path.join(self.derivsdir, 'processedData', 'restbold', 'restbold_201607151621')


class Subject(Project):
    def __init__(self, bblid=0, scanid=0):
        Project.__init__(self)
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
            sc_filename = glob.glob(os.path.join(self.scdir, sc_filename))

            ct_filename = os.path.join('{0}'.format(self.bblid),
                                       '*x{0}'.format(self.scanid),
                                       'ct_schaefer{0}_17.txt'.format(self.parc_res))
            ct_filename = glob.glob(os.path.join(self.ctdir, ct_filename))

            if parc_res == 200:
                rsts_filename = os.path.join('{0}'.format(self.bblid),
                                             '*x{0}'.format(self.scanid),
                                             'net', 'Schaefer{0}PNC'.format(self.parc_res),
                                             '{0}_*x{1}_Schaefer{2}PNC_ts.1D' \
                                             .format(self.bblid, self.scanid, self.parc_res))
                rsts_filename = glob.glob(os.path.join(self.rstsdir, rsts_filename))
            elif parc_res == 400:
                rsts_filename = os.path.join('{0}'.format(self.bblid),
                                             '*x{0}'.format(self.scanid),
                                             'net', 'SchaeferPNC',
                                             '{0}_*x{1}_SchaeferPNC_ts.1D' \
                                             .format(self.bblid, self.scanid))
                rsts_filename = glob.glob(os.path.join(self.rstsdir, rsts_filename))

        self.sc_filename = sc_filename
        self.ct_filename = ct_filename
        self.rsts_filename = rsts_filename
