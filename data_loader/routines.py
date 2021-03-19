import numpy as np


class LoadSC():
    def __init__(self, environment, Subject):
        self.environment = environment
        self.Subject = Subject

    def run(self):
        print('Routine: loading SC data')
        n_subs = self.environment.df.shape[0]
        self.df = self.environment.df.copy()
        self.A = np.zeros((self.environment.n_parcels, self.environment.n_parcels, n_subs))

        # subject filter
        subj_filt = np.zeros((n_subs,)).astype(bool)

        for i in np.arange(n_subs):
            subject = self.Subject(subjid=self.df.index[i])
            subject.get_file_names()
            subject.load_sc()
            self.A[:, :, i] = subject.sc.data.copy()

            subject.sc.check_disconnected_nodes()
            if subject.sc.disconnected_nodes:
                subj_filt[i] = True

        # filter subjects with disconnected nodes from sc matrix
        if np.any(subj_filt):
            print('\t{0} subjects had disconnected nodes in sc matrices'.format(np.sum(subj_filt)))
            self.df = self.df.loc[~subj_filt]
            self.A = self.A[:, :, ~subj_filt]


class LoadFC():
    def __init__(self, environment, Subject):
        self.environment = environment
        self.Subject = Subject

    def run(self):
        print('Routine: loading resting state FC data')
        n_subs = self.environment.df.shape[0]
        self.df = self.environment.df.copy()
        self.fc = np.zeros((self.environment.n_parcels, self.environment.n_parcels, n_subs))

        # subject filter
        subj_filt = np.zeros((n_subs,)).astype(bool)

        for i in np.arange(n_subs):
            subject = self.Subject(subjid=self.df.index[i])
            subject.get_file_names()
            subject.load_rsfc()
            self.fc[:, :, i] = subject.rsfc.data.copy()

            subject.load_sc()
            subject.sc.check_disconnected_nodes()
            if subject.sc.disconnected_nodes:
                subj_filt[i] = True

        # filter subjects with disconnected nodes from sc matrix
        if np.any(subj_filt):
            print('\t{0} subjects had disconnected nodes in sc matrices'.format(np.sum(subj_filt)))
            self.df = self.df.loc[~subj_filt]
            self.fc = self.fc[:, :, ~subj_filt]


class LoadRLFP():
    def __init__(self, environment, Subject):
        self.environment = environment
        self.Subject = Subject

    def run(self):
        print('Routine: loading resting state RLFP data')
        n_subs = self.environment.df.shape[0]
        self.df = self.environment.df.copy()
        self.rlfp = np.zeros((n_subs, self.environment.n_parcels))

        # subject filter
        subj_filt = np.zeros((n_subs,)).astype(bool)

        for i in np.arange(n_subs):
            subject = self.Subject(subjid=self.df.index[i])
            subject.get_file_names()
            subject.load_rlfp()
            self.rlfp[i, :] = subject.rlfp.copy()

            subject.load_sc()
            subject.sc.check_disconnected_nodes()
            if subject.sc.disconnected_nodes:
                subj_filt[i] = True

        # filter subjects with disconnected nodes from sc matrix
        if np.any(subj_filt):
            print('\t{0} subjects had disconnected nodes in sc matrices'.format(np.sum(subj_filt)))
            self.df = self.df.loc[~subj_filt]
            self.rlfp = self.rlfp[~subj_filt, :]
