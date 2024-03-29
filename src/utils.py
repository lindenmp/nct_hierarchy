import os
import numpy as np
import pandas as pd
import scipy as sp
from statsmodels.stats import multitest
import nibabel as nib
import wget

def get_pdist_clusters(coords, labels, method='mean'):
    unique, counts = np.unique(labels, return_counts=True)
    n_clusters = len(unique)

    dist = np.zeros((n_clusters, n_clusters))

    for i in np.arange(n_clusters):
        for j in np.arange(n_clusters):
            x0 = labels == i
            xf = labels == j

            x0_coords = coords[x0, :]
            xf_coords = coords[xf, :]

            tmp = []
            for r1 in np.arange(x0_coords.shape[0]):
                for r2 in np.arange(xf_coords.shape[0]):
                    d = (x0_coords[r1, :] - xf_coords[r2, :]) ** 2
                    d = np.sum(d)
                    d = np.sqrt(d)
                    tmp.append(d)

            if method == 'mean':
                dist[i, j] = np.mean(tmp)
            elif method == 'min':
                dist[i, j] = np.min(tmp)
            elif method == 'median':
                dist[i, j] = np.median(tmp)

    return dist


def get_disc_repl(df, frac=0.5):
    df_out = pd.Series(data=np.zeros(df.shape[0], dtype=bool), index=df.index)

    n = np.round(df_out.shape[0] * frac).astype(int)
    hold_out = df_out.sample(n=n, random_state=0, replace=False, axis=0).index
    df_out[hold_out] = True
    print('Train:', np.sum(df_out == 0), 'Test:', np.sum(df_out == 1))

    return df_out


def mean_over_clusters(x, cluster_labels):
    unique = np.unique(cluster_labels, return_counts=False)
    n_clusters = len(unique)

    x_out = np.zeros((n_clusters, n_clusters))
    for i in np.arange(n_clusters):
        for j in np.arange(n_clusters):
            x_out[i, j] = np.nanmean(np.nanmean(x[cluster_labels == i, :], axis=0)[cluster_labels == j])

    return x_out


def get_exact_p(x, y):
    p_val = 2 * np.min([np.mean(x - y >= 0),
                        np.mean(x - y <= 0)])

    return p_val


def get_null_p(x, null, version='standard', abs=False):
    if abs:
        x = np.abs(x)
        null = np.abs(null)

    if version == 'standard':
        p_val = np.sum(null >= x) / len(null)
    elif version == 'reverse':
        p_val = np.sum(x >= null) / len(null)
    elif version == 'smallest':
        p_val = np.min([np.sum(null >= x) / len(null),
                        np.sum(x >= null) / len(null)])

    return p_val


def get_fdr_p(p_vals, alpha=0.05):
    if p_vals.ndim == 2:
        do_reshape = True
        dims = p_vals.shape
        p_vals = p_vals.flatten()
    else:
        do_reshape = False

    out = multitest.multipletests(p_vals, alpha=alpha, method='fdr_bh')
    p_fdr = out[1]

    if do_reshape:
        p_fdr = p_fdr.reshape(dims)

    return p_fdr


def rank_to_normal(data, c, n):
    # Standard quantile function
    data = (data - c) / (n - 2 * c + 1)
    return sp.stats.norm.ppf(data)


def rank_int(data, c=3.0 / 8):
    if data.ndim > 1:
        do_reshape = True
        dims = data.shape
        data = data.flatten()
    else:
        do_reshape = False

    # Set seed
    np.random.seed(0)

    # Get rank, ties are averaged
    data = sp.stats.rankdata(data, method="average")

    # Convert rank to normal distribution
    transformed = rank_to_normal(data=data, c=c, n=len(data))

    if do_reshape:
        transformed = transformed.reshape(dims)

    return transformed


def pearsonr_permutation(x, y, n_perms=1e4):
    # for reproducibility
    np.random.seed(0)
    observed_r = sp.stats.pearsonr(x, y)[0]

    null_r = np.zeros(n_perms)
    for i in np.arange(n_perms):
        x_shuf = np.random.permutation(x)
        null_r[i] = sp.stats.pearsonr(x_shuf, y)[0]

    p_value = np.sum(np.abs(null_r) >= np.abs(observed_r)) / n_perms

    return observed_r, null_r, p_value


def fit_hyperplane(data, type='linear'):
    # regular grid covering the domain of the data
    mn = np.min(data, axis=0)
    mx = np.max(data, axis=0)
    X, Y = np.meshgrid(np.linspace(mn[0], mx[0], 20), np.linspace(mn[1], mx[1], 20))
    XX = X.flatten()
    YY = Y.flatten()

    # best-fit quadratic curve
    if type == 'linear':
        # best-fit linear plane
        a = np.c_[data[:, 0], data[:, 1], np.ones(data.shape[0])]
        c, resids, _, _ = sp.linalg.lstsq(a, data[:, 2])  # coefficients

        # evaluate it on grid
        Z = c[0] * X + c[1] * Y + c[2]
    elif type == 'quad':
        a = np.c_[np.ones(data.shape[0]), data[:, :2], np.prod(data[:, :2], axis=1), data[:, :2] ** 2]
        c, resids, _, _ = sp.linalg.lstsq(a, data[:, 2])

        # evaluate it on a grid
        Z = np.dot(np.c_[np.ones(XX.shape), XX, YY, XX * YY, XX ** 2, YY ** 2], c).reshape(X.shape)
    # elif type == 'cubic':
    #     a = np.c_[np.ones(data.shape[0]), data[:, :2], np.prod(data[:, :2], axis=1), data[:, :2] ** 2, data[:, :2] ** 3]
    #     c, resids, _, _ = sp.linalg.lstsq(a, data[:, 2])
    #
    #     # evaluate it on a grid
    #     Z = np.dot(np.c_[np.ones(XX.shape), XX, YY, XX * YY, XX ** 2, YY ** 2, XX ** 3, YY ** 3], c).reshape(X.shape)

    # compute coefficient of determination
    r2 = 1 - resids / np.sum(np.square(data[:, 2] - data[:, 2].mean()))
    mse = resids / data[:, 2].shape[0]
    rmse = np.sqrt(mse)

    return X, Y, Z, c, r2, mse, rmse


def get_xyz_slope(X, Y, Z):
    # this is more a sanity check function than anything else.. can just use values in c from fit_hyperplane
    p1 = np.array([X[0, 0], Y[0, 0], Z[0, 0]]) # corner at lowest x,y position
    p2 = np.array([X[0, -1], Y[0, 0], Z[0, -1]]) # corner at highest x, lowest y
    p3 = np.array([X[0, -1], Y[-1, 0], Z[-1, -1]]) # corner at highest x,y position

    slope_x = (p2[-1] - p1[-1]) / sp.spatial.distance.pdist(np.vstack((p1, p2)), 'euclidean')
    slope_y = (p3[-1] - p2[-1]) / sp.spatial.distance.pdist(np.vstack((p2, p3)), 'euclidean')

    return list([slope_x[0], slope_y[0]])


def helper_null_mean(e, e_null, indices):
    n_perms = e_null.shape[2]
    # compute energy asymmetry
    ed = e.transpose() - e

    # containers
    ed_null = np.zeros(e_null.shape)
    asymm_null = np.zeros(n_perms)

    for i in np.arange(e_null.shape[2]):
        # compute null asymmetry matrix
        ed_null[:, :, i] = e_null[:, :, i].transpose() - e_null[:, :, i]
        # get mean of null asymmetry
        asymm_null[i] = np.mean(ed_null[:, :, i][indices])

    # get observed
    observed = np.mean(ed[indices])
    # get p val
    p_val = get_null_p(observed, asymm_null, abs=True)

    return asymm_null, observed, p_val


def helper_null_hyperplane(e, e_null, indices):
    n_perms = e_null.shape[2]
    # compute energy asymmetry
    ed = e.transpose() - e

    # containers
    asymm_nulls = np.zeros((n_perms, 3))

    for i in np.arange(e_null.shape[2]):
        # compute null asymmetry matrix
        ed_null = e_null[:, :, i].transpose() - e_null[:, :, i]

        data = np.concatenate((indices[0].reshape(-1, 1),
                               indices[1].reshape(-1, 1),
                               ed_null[indices].reshape(-1, 1)), axis=1)
        data = (data - data.mean(axis=0)) / data.std(axis=0)
        _, _, _, c, r2, _, _ = fit_hyperplane(data)
        asymm_nulls[i, 0] = r2
        asymm_nulls[i, 1] = c[0]
        asymm_nulls[i, 2] = c[1]

    # get observed
    data = np.concatenate((indices[0].reshape(-1, 1),
                           indices[1].reshape(-1, 1),
                           ed[indices].reshape(-1, 1)), axis=1)
    data = (data - data.mean(axis=0)) / data.std(axis=0)
    _, _, _, c, r2, _, _ = fit_hyperplane(data)
    observed = list([r2, c[0], c[1]])

    # get p val
    p_vals = []
    p_vals.append(get_null_p(observed[0], asymm_nulls[:, 0]))
    p_vals.append(get_null_p(observed[1], asymm_nulls[:, 1], abs=True))
    p_vals.append(get_null_p(observed[2], asymm_nulls[:, 2], abs=True))

    return asymm_nulls, observed, p_vals


def get_p_val_string(p_val):
    if p_val == 0.0:
        p_str = "-log10($\mathit{:}$)>25".format('{p}')
    elif p_val < 0.05:
        p_str = '$\mathit{:}$ = {:0.0e}'.format('{p}', p_val)
    # elif p_val < 0.001:
    #     p_str = '$\mathit{:}$ < 0.001'.format('{p}')
    # elif p_val >= 0.001 and p_val < 0.05:
    #     p_str = '$\mathit{:}$ < 0.05'.format('{p}')
    else:
        p_str = "$\mathit{:}$ = {:.3f}".format('{p}', p_val)

    return p_str


def get_parcelwise_average_surface(data_file, annot_file):
    # load parcellation
    labels, ctab, surf_names = nib.freesurfer.read_annot(annot_file)
    unique_labels = np.unique(labels)

    file_name, file_extension = os.path.splitext(data_file)

    # load gifti file
    if file_extension == '.gii':
        data = nib.load(data_file)
        data = data.darrays[0].data
    elif file_extension == '.mgh':
        data = nib.load(data_file)
        data = data.get_fdata().squeeze()
    elif file_extension == '.txt':
        data = np.loadtxt(data_file)

    # mean over labels
    data_mean = []
    for i in unique_labels:
        data_mean.append(np.mean(data[labels == i]))

    return np.asarray(data_mean)


def get_offset_diag(n, version='lower', return_indices=False):

    a = np.zeros((n, n))
    b = np.ones(n-1)

    if version == 'upper':
        np.fill_diagonal(a[:, 1:], b)
    elif version == 'lower':
        np.fill_diagonal(a[1:], b)

    if return_indices == True:
        a = np.where(a)
    else:
        a = a.astype(np.bool)

    return a


def get_states_from_brain_map(brain_map, n_bins):
    n_parcels = len(brain_map)
    bin_size = int(n_parcels / n_bins)

    states = np.array([])
    for i in np.arange(n_bins):
        states = np.append(states, np.ones(bin_size) * i)

    if len(states) < n_parcels:
        states = np.append(states, np.ones(bin_size) * (n_bins - 1))

    if len(states) > n_parcels:
        states = states[:n_parcels]

    states = states.astype(int)
    sort_idx = np.argsort(brain_map)
    unsorted_idx = np.argsort(sort_idx)
    states = states[unsorted_idx]

    return states


def threshold_consistency(A, thr=0.60):
    """
    Thresholds edges from a group averaged adjacency matrix by retaining edges that are non-zero in some proportion of
    subjects (defined by thr)

    Args:
        A: n x n x m structural adjacency matrix with subjects along m
        thr: proportion of subjects that an edge needs to exist in order to be retained

    Returns:
        Am: thresholded mean adjacency matrix

    """
    n_parcels = A.shape[0]
    n_subs = A.shape[2]

    # get group averaged A matrix
    Am = np.mean(A, axis=2)

    # find non-zero elements in A
    Ab = A > 0

    # get count of non-zero elements along subject dimension
    Ab = np.sum(Ab, axis=2)

    # compute fraction of non-zero elements over subjects
    Ab = Ab / n_subs

    # define mask using threshold
    mask = Ab < thr

    # set elements within mask in mean A matrix to zero
    Am[mask] = 0

    return Am


def get_bootstrap_indices(d_size=5, frac=1, n_samples=10000):
    bootstrap_indices = np.zeros((n_samples, int(d_size*frac)))

    # for reproducibility
    np.random.seed(0)

    for i in np.arange(n_samples):
        bootstrap_indices[i, :] = np.random.choice(np.arange(d_size), size=int(d_size*frac), replace=True)

    return bootstrap_indices.astype(int)


def mean_over_states(matrix, states):
    if matrix.ndim == 1:
        is_matrix = False
    else:
        is_matrix = True

    unique = np.unique(states, return_counts=False)
    n_states = len(unique)

    out = np.zeros((n_states, n_states))

    for i in np.arange(n_states):
        for j in np.arange(n_states):
            if is_matrix:
                out[i, j] = matrix[np.ix_(states == i, states == j)].mean()
            else:
                out[i, j] = matrix[states == i].mean() - matrix[states == j].mean()

    return out


def load_schaefer_parc(n_parcels=200, order=17, annot='fsaverage', outdir='~/research_data/schaefer_parc'):
    # output dir
    outdir = os.path.expanduser(outdir)
    if os.path.exists(outdir) == False:
        os.makedirs(outdir)

    # github link
    remote_path = 'https://github.com/ThomasYeoLab/CBIG/raw/master/stable_projects/brain_parcellation/Schaefer2018_LocalGlobal'

    # roi names and coords in MNI
    file = 'Schaefer2018_{0}Parcels_{1}Networks_order_FSLMNI152_1mm.Centroid_RAS.csv'.format(n_parcels, order)
    interim_dir = 'Parcellations/MNI/Centroid_coordinates'
    if os.path.exists(os.path.join(outdir, file)) == False:
        wget.download(os.path.join(remote_path, interim_dir, file), outdir)

    centroids = pd.read_csv(os.path.join(outdir, file), index_col=0)

    # annotation files for fsaverage
    if os.path.exists(os.path.join(outdir, annot)) == False:
        os.makedirs(os.path.join(outdir, annot))
    interim_dir = 'Parcellations/FreeSurfer5.3/{0}/label'.format(annot)
    files = ['lh.Schaefer2018_{0}Parcels_{1}Networks_order.annot'.format(n_parcels, order),
             'rh.Schaefer2018_{0}Parcels_{1}Networks_order.annot'.format(n_parcels, order)]
    for file in files:
        if os.path.exists(os.path.join(outdir, annot, file)) == False:
            wget.download(os.path.join(remote_path, interim_dir, file), os.path.join(outdir, annot))
    lh_annot_file = os.path.join(outdir, annot, files[0])
    rh_annot_file = os.path.join(outdir, annot, files[1])

    # cifti file for HCP space (32k fs_LR)
    # file = 'Schaefer2018_{0}Parcels_{1}Networks_order.dscalar.nii'.format(n_parcels, order)
    file = 'Schaefer2018_{0}Parcels_{1}Networks_order.dlabel.nii'.format(n_parcels, order)
    interim_dir = 'Parcellations/HCP/fslr32k/cifti'
    if os.path.exists(os.path.join(outdir, file)) == False:
        wget.download(os.path.join(remote_path, interim_dir, file), outdir)
    hcp_file = os.path.join(outdir, file)

    return centroids, lh_annot_file, rh_annot_file, hcp_file


def load_glasser_parc(glasser_dir='~/research_data/Glasser_et_al_2016_HCP_MMP1.0_kN_RVVG',
                      annot='fsaverage5', outdir='~/research_data/glasser_parc'):
    glasser_dir = os.path.expanduser(glasser_dir)

    # annot files
    lh_annot_file = os.path.join(glasser_dir, 'prepare_glasser', 'HCP-MMP1.{0}.L.annot'.format(annot))
    rh_annot_file = os.path.join(glasser_dir, 'prepare_glasser', 'HCP-MMP1.{0}.R.annot'.format(annot))

    hcp_file = os.path.join(glasser_dir, 'HCP_PhaseTwo', 'Q1-Q6_RelatedParcellation210', 'MNINonLinear', 'fsaverage_LR32k',
                            'Q1-Q6_RelatedParcellation210.CorticalAreas_dil_Colors.32k_fs_LR.dlabel.nii')

    # output dir
    outdir = os.path.expanduser(outdir)
    if os.path.exists(outdir) == False:
        os.makedirs(outdir)

    # roi names
    remote_path = 'https://github.com/PennLINC/xcpEngine/raw/master/atlas/glasser360'
    file = 'glasser360NodeNames.txt'
    if os.path.exists(os.path.join(outdir, file)) == False:
        wget.download(os.path.join(remote_path, file), outdir)

    parcel_names = list(np.genfromtxt(os.path.join(outdir, 'glasser360NodeNames.txt'), dtype='str'))

    # roi centroids
    remote_path = 'https://bitbucket.org/dpat/tools/raw/master/REF/ATLASES'
    file = 'HCP-MMP1_UniqueRegionList.csv'
    if os.path.exists(os.path.join(outdir, file)) == False:
        wget.download(os.path.join(remote_path, file), outdir)

    centroids = pd.read_csv(os.path.join(outdir, 'HCP-MMP1_UniqueRegionList.csv'))
    centroids = pd.concat((centroids.iloc[180:, :], centroids.iloc[0:180, :]), axis=0)
    centroids.reset_index(inplace=True, drop=True)
    centroids = centroids.loc[:, ['x-cog', 'y-cog', 'z-cog']]
    centroids['regionName'] = parcel_names

    return centroids, lh_annot_file, rh_annot_file, hcp_file
