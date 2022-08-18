import numpy as np
import scipy as sp
from scipy import stats

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from sklearn.linear_model import Ridge, Lasso, LinearRegression
from sklearn.kernel_ridge import KernelRidge
from sklearn.svm import SVR
from sklearn.decomposition import PCA
import copy
from tqdm import tqdm

def corr_true_pred(y_true, y_pred):
    if type(y_true) == np.ndarray:
        y_true = y_true.flatten()
    if type(y_pred) == np.ndarray:
        y_pred = y_pred.flatten()

    r, p = sp.stats.pearsonr(y_true, y_pred)
    return r


def root_mean_squared_error(y_true, y_pred):
    mse = np.mean((y_true - y_pred) ** 2, axis=0)
    rmse = np.sqrt(mse)
    return rmse


def shuffle_data(X, y, c, seed=0):
    np.random.seed(seed)
    idx = np.arange(y.shape[0])
    np.random.shuffle(idx)

    try:
        X_shuf = X[idx, :]
    except IndexError:
        X_shuf = X[idx]

    try:
        c_shuf = c[idx, :]
    except IndexError:
        c_shuf = c[idx]
    except TypeError:
        c_shuf = None

    y_shuf = y[idx]

    return X_shuf, y_shuf, c_shuf


def get_reg():
    regs = {'linear': LinearRegression(),
            'rr': Ridge(),
            'lr': Lasso(),
            'krr_lin': KernelRidge(kernel='linear'),
            'krr_rbf': KernelRidge(kernel='rbf'),
            'svr_lin': SVR(kernel='linear'),
            'svr_rbf': SVR(kernel='rbf')
            }

    return regs


def get_cv(y, n_splits=10):
    my_cv = []

    kf = KFold(n_splits=n_splits, shuffle=False)

    for train_idx, test_idx in kf.split(y):
        my_cv.append((train_idx, test_idx))

    return my_cv


def get_stratified_cv(X, y, c=None, n_splits=10):
    # sort data on outcome variable in ascending order
    idx = np.argsort(y)
    y_sort = y[idx]

    if X.ndim == 2:
        X_sort = X[idx, :]
    elif X.ndim == 1:
        X_sort = X[idx]

    if c is not None:
        if c.ndim == 2:
            c_sort = c[idx, :]
        elif c.ndim == 1:
            c_sort = c[idx]

    # create custom stratified kfold on outcome variable
    my_cv = []
    for k in range(n_splits):
        my_bool = np.zeros(y.shape[0]).astype(bool)
        my_bool[np.arange(k, y.shape[0], n_splits)] = True

        train_idx = np.where(my_bool == False)[0]
        test_idx = np.where(my_bool == True)[0]
        my_cv.append((train_idx, test_idx))

    if c is not None:
        return X_sort, y_sort, my_cv, c_sort
    else:
        return X_sort, y_sort, my_cv


def my_cross_val_score(X, y, c, my_cv, reg, scorer, runpca=False):
    accuracy = np.zeros(len(my_cv), )
    y_pred_out = np.zeros(y.shape)

    # find number of PCs
    if type(runpca) == str:
        pca = PCA(n_components=np.min(X.shape), svd_solver='full')
        pca.fit(StandardScaler().fit_transform(X))

        if runpca == '80%':
            cum_var = np.cumsum(pca.explained_variance_ratio_)
            n_components = np.where(cum_var >= 0.8)[0][0] + 1
        elif runpca == '1%':
            var_idx = pca.explained_variance_ratio_ >= .01
            n_components = np.sum(var_idx)

    elif type(runpca) == int:
        n_components = runpca

    for k in np.arange(len(my_cv)):
        tr = my_cv[k][0]
        te = my_cv[k][1]

        # Split into train test
        try:
            X_train = X[tr, :]
            X_test = X[te, :]
        except IndexError:
            X_train = X[tr]
            X_test = X[te]

        try:
            c_train = c[tr, :]
            c_test = c[te, :]
        except IndexError:
            c_train = c[tr]
            c_test = c[te]
        except TypeError:
            pass

        y_train = y[tr]
        y_test = y[te]

        # standardize predictors
        sc = StandardScaler()
        sc.fit(X_train)
        X_train = sc.transform(X_train)
        X_test = sc.transform(X_test)

        try:
            # standardize covariates
            sc = StandardScaler()
            sc.fit(c_train)
            c_train = sc.transform(c_train)
            c_test = sc.transform(c_test)

            # regress nuisance (X)
            nuis_reg = copy.deepcopy(reg)
            nuis_reg.fit(c_train, X_train)
            X_pred = nuis_reg.predict(c_train)
            X_train = X_train - X_pred
            X_pred = nuis_reg.predict(c_test)
            X_test = X_test - X_pred
        except:
            pass

        if type(runpca) == str or type(runpca) == int:
            pca = PCA(n_components=n_components, svd_solver='full')
            pca.fit(X_train)
            X_train = pca.transform(X_train)
            X_test = pca.transform(X_test)

        reg.fit(X_train, y_train)
        accuracy[k] = scorer(reg, X_test, y_test)
        y_pred_out[te] = reg.predict(X_test)

    return accuracy, y_pred_out


def run_reg(X, y, c, reg, scorer, n_splits=10, runpca=False, seed=0):
    X_shuf, y_shuf, c_shuf = shuffle_data(X=X, y=y, c=c, seed=seed)

    my_cv = get_cv(y_shuf, n_splits=n_splits)

    accuracy, y_pred_out = my_cross_val_score(X=X_shuf, y=y_shuf, c=c_shuf, my_cv=my_cv, reg=reg,
                                              scorer=scorer, runpca=runpca)

    return accuracy, y_pred_out


def run_reg_scv(X, y, c, reg, scorer, n_splits=10, runpca=False):
    if c is None:
        X_sort, y_sort, my_cv = get_stratified_cv(X=X, y=y, c=c, n_splits=n_splits)
        accuracy, y_pred_out = my_cross_val_score(X=X_sort, y=y_sort, c=c, my_cv=my_cv, reg=reg,
                                                  scorer=scorer, runpca=runpca)
    else:
        X_sort, y_sort, my_cv, c_sort = get_stratified_cv(X=X, y=y, c=c, n_splits=n_splits)
        accuracy, y_pred_out = my_cross_val_score(X=X_sort, y=y_sort, c=c_sort, my_cv=my_cv, reg=reg,
                                                  scorer=scorer, runpca=runpca)

    return accuracy, y_pred_out


def run_perm(X, y, c, reg, scorer, n_splits=10, runpca=False, n_perm=int(5e3)):
    my_cv = get_cv(y, n_splits=n_splits)

    permuted_acc = np.zeros(n_perm)

    for i in tqdm(np.arange(n_perm)):
        np.random.seed(i)
        idx = np.arange(y.shape[0])
        np.random.shuffle(idx)

        y_perm = y[idx].copy()

        temp_acc, y_pred_out_tmp = my_cross_val_score(X=X, y=y_perm, c=c, my_cv=my_cv, reg=reg,
                                                      scorer=scorer, runpca=runpca)
        permuted_acc[i] = temp_acc.mean()

    return permuted_acc


def get_reg_weights(X, y, reg, c=None):
    sc = StandardScaler()
    X = sc.fit_transform(X)
    if c is None:
        pass
    else:
        c = sc.fit_transform(c)

        nuis_reg = copy.deepcopy(reg)
        nuis_reg.fit(c, X)
        X_pred = nuis_reg.predict(c)
        X = X - X_pred

    reg.fit(X, y)

    coefs = reg.coef_

    return coefs
