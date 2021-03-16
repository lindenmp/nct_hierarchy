import numpy as np


def load_sc(environment):
    print('Loading SC data')

    # pull out dataframe, just for convenience
    df = environment.df
    Subject = environment.Subject

    n_subs = df.shape[0]
    n_parcels = environment.n_parcels
    A = np.zeros((n_parcels, n_parcels, n_subs))

    # subject filter
    subj_filt = np.zeros((n_subs,)).astype(bool)

    for i in np.arange(n_subs):
        subject = Subject(df.index[i][0], df.index[i][1])
        subject.get_file_names()
        subject.load_sc()
        A[:, :, i] = subject.sc.data.copy()

        subject.sc.check_disconnected_nodes()
        if subject.sc.disconnected_nodes:
            subj_filt[i] = True

    # filter subjects with disconnected nodes from sc matrix
    if np.any(subj_filt):
        print('{0} subjects had disconnected nodes in sc matrices'.format(np.sum(subj_filt)))
        df = df.loc[~subj_filt]
        A = A[:, :, ~subj_filt]

    environment.df = df
    environment.A = A

    return environment


def load_fc(environment):
    print('Loading resting state FC data')

    # pull out dataframe, just for convenience
    df = environment.df
    Subject = environment.Subject

    n_subs = df.shape[0]
    n_parcels = environment.n_parcels
    fc = np.zeros((n_parcels, n_parcels, n_subs))

    # subject filter
    subj_filt = np.zeros((n_subs,)).astype(bool)

    for i in np.arange(n_subs):
        subject = Subject(df.index[i][0], df.index[i][1])
        subject.get_file_names()
        subject.load_rsfc()
        fc[:, :, i] = subject.rsfc.data.copy()

        subject.load_sc()
        subject.sc.check_disconnected_nodes()
        if subject.sc.disconnected_nodes:
            subj_filt[i] = True

    # filter subjects with disconnected nodes from sc matrix
    if np.any(subj_filt):
        print('{0} subjects had disconnected nodes in sc matrices'.format(np.sum(subj_filt)))
        df = df.loc[~subj_filt]
        fc = fc[:, :, ~subj_filt]

    environment.df = df
    environment.fc = fc

    return environment
