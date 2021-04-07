import numpy as np
import time

class LoadSC():
    def __init__(self, environment, Subject):
        self.environment = environment
        self.Subject = Subject

    def run(self):
        print('Routine: loading SC data')
        start_time = time.time()
        n_subs = self.environment.df.shape[0]
        self.df = self.environment.df.copy()
        self.A = np.zeros((self.environment.n_parcels, self.environment.n_parcels, n_subs))

        # subject filter
        subj_filt = np.zeros((n_subs,)).astype(bool)

        for i in np.arange(n_subs):
            subject = self.Subject(environment=self.environment, subjid=self.df.index[i])
            subject.get_file_names()
            subject.load_sc()
            self.A[:, :, i] = subject.sc.data.copy()

            subject.sc.check_disconnected_nodes()
            if subject.sc.disconnected_nodes:
                subj_filt[i] = True
        print("\t --- finished in {:.0f} seconds ---".format((time.time() - start_time)))

        # filter subjects with disconnected nodes from sc matrix
        if np.any(subj_filt):
            print('\t{0} subjects had disconnected nodes in sc matrices'.format(np.sum(subj_filt)))
            self.df = self.df.loc[~subj_filt]
            self.A = self.A[:, :, ~subj_filt]


class LoadAverageSC():
    def __init__(self, load_sc, spars_thresh=0.06):
        self.load_sc = load_sc
        self.spars_thresh = spars_thresh

    def _check_inputs(self):
        try:
            A = self.load_sc.A
        except AttributeError:
            self.load_sc.run()

    def _print_settings(self):
        print('\tsettings:')
        print('\t\tsparsity: {0}'.format(self.spars_thresh))

    def run(self):
        print('Routine: loading average SC matrix')
        self._check_inputs()
        self._print_settings()

        A = self.load_sc.A
        n_subs = self.load_sc.df.shape[0]
        print('\tnumber of subjects in average adj matrix: {0}'.format(n_subs))

        # Get streamline count and network density
        A_d = np.zeros((n_subs,))
        for i in range(n_subs):
            A_d[i] = np.count_nonzero(np.triu(A[:, :, i])) / (
                        (A[:, :, i].shape[0] ** 2 - A[:, :, i].shape[0]) / 2)

        # Get group average adj. matrix
        A = np.mean(A, 2)
        thresh = np.percentile(A, 100 - (self.spars_thresh * 100))
        A[A < thresh] = 0

        print('\tactual matrix sparsity = {:.2f}'.format(
            np.count_nonzero(np.triu(A)) / ((A.shape[0] ** 2 - A.shape[0]) / 2)))

        self.A = A

class LoadFC():
    def __init__(self, environment, Subject):
        self.environment = environment
        self.Subject = Subject

    def run(self):
        print('Routine: loading resting state FC data')
        start_time = time.time()
        n_subs = self.environment.df.shape[0]
        self.df = self.environment.df.copy()
        self.fc = np.zeros((self.environment.n_parcels, self.environment.n_parcels, n_subs))

        # subject filter
        subj_filt = np.zeros((n_subs,)).astype(bool)

        for i in np.arange(n_subs):
            subject = self.Subject(environment=self.environment, subjid=self.df.index[i])
            subject.get_file_names()
            subject.load_rsfc()
            self.fc[:, :, i] = subject.rsfc.data.copy()

            subject.load_sc()
            subject.sc.check_disconnected_nodes()
            if subject.sc.disconnected_nodes:
                subj_filt[i] = True
        print("\t --- finished in {:.0f} seconds ---".format((time.time() - start_time)))

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
        start_time = time.time()
        n_subs = self.environment.df.shape[0]
        self.df = self.environment.df.copy()
        self.rlfp = np.zeros((n_subs, self.environment.n_parcels))

        # subject filter
        subj_filt = np.zeros((n_subs,)).astype(bool)

        for i in np.arange(n_subs):
            subject = self.Subject(environment=self.environment, subjid=self.df.index[i])
            subject.get_file_names()
            subject.load_rlfp()
            self.rlfp[i, :] = subject.rlfp.copy()

            subject.load_sc()
            subject.sc.check_disconnected_nodes()
            if subject.sc.disconnected_nodes:
                subj_filt[i] = True
        print("\t --- finished in {:.0f} seconds ---".format((time.time() - start_time)))

        # filter subjects with disconnected nodes from sc matrix
        if np.any(subj_filt):
            print('\t{0} subjects had disconnected nodes in sc matrices'.format(np.sum(subj_filt)))
            self.df = self.df.loc[~subj_filt]
            self.rlfp = self.rlfp[~subj_filt, :]


class LoadCT():
    def __init__(self, environment, Subject):
        self.environment = environment
        self.Subject = Subject

    def run(self):
        print('Routine: loading cortical thickness data')
        start_time = time.time()
        n_subs = self.environment.df.shape[0]
        self.df = self.environment.df.copy()
        self.ct = np.zeros((n_subs, self.environment.n_parcels))

        # subject filter
        subj_filt = np.zeros((n_subs,)).astype(bool)

        for i in np.arange(n_subs):
            subject = self.Subject(environment=self.environment, subjid=self.df.index[i])
            subject.get_file_names()
            subject.load_ct()
            self.ct[i, :] = subject.ct.copy()

            subject.load_sc()
            subject.sc.check_disconnected_nodes()
            if subject.sc.disconnected_nodes:
                subj_filt[i] = True
        print("\t --- finished in {:.0f} seconds ---".format((time.time() - start_time)))

        # filter subjects with disconnected nodes from sc matrix
        if np.any(subj_filt):
            print('\t{0} subjects had disconnected nodes in sc matrices'.format(np.sum(subj_filt)))
            self.df = self.df.loc[~subj_filt]
            self.ct = self.ct[~subj_filt, :]