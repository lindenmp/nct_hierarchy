import os
import numpy as np
import pandas as pd
import scipy as sp
from pfactor_gradients.routines import LoadFC
from pfactor_gradients.imaging_derivs import DataVector, DataMatrix
from pfactor_gradients.energy import matrix_normalization, expand_states, control_energy_helper, \
    get_gmat, grad_descent_b, minimum_energy_fast
from pfactor_gradients.prediction import corr_true_pred, root_mean_squared_error, run_reg, run_perm
from brainspace.gradient import GradientMaps
from sklearn.cluster import KMeans
from sklearn.linear_model import Ridge, Lasso, LinearRegression
from sklearn.kernel_ridge import KernelRidge
from sklearn.svm import SVR
from sklearn.metrics import make_scorer, r2_score, mean_squared_error, mean_absolute_error
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import abagen
from tqdm import tqdm

# %% plotting
import seaborn as sns
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import ListedColormap
from pfactor_gradients.plotting import set_plotting_params
set_plotting_params(format='png')
figsize = 1.5

# %% pipeline classes
class ComputeGradients():
    def __init__(self, environment, Subject):
        self.environment = environment
        self.Subject = Subject

    def _output_dir(self):
        return os.path.join(self.environment.pipelinedir, 'gradients')

    @staticmethod
    def _output_file():
        return 'gradients.txt'

    def _check_outputs(self):
        if os.path.exists(self._output_dir()) and os.path.isfile(os.path.join(self._output_dir(), self._output_file())):
            return True
        else:
            return False

    def run(self):
        print('Pipeline: getting functional cortical gradients')
        if self._check_outputs():
            print('\toutput already exists...skipping')
            self.gradients = np.loadtxt(os.path.join(self._output_dir(), self._output_file()))
        else:
            # Load FC data
            loadfc = LoadFC(environment=self.environment, Subject=self.Subject)
            loadfc.run()
            fc = loadfc.values

            # Average over subjects
            pnc_conn_mat = np.nanmean(fc, axis=2)
            pnc_conn_mat[np.eye(self.environment.n_parcels, dtype=bool)] = 0
            # pnc_conn_mat = dominant_set(pnc_conn_mat, 0.10, as_sparse = False)

            # Plot mean fc matrix
            f, ax = plt.subplots(1, figsize=(5, 5))
            sns.heatmap(pnc_conn_mat, cmap='coolwarm', center=0, square=True)
            ax.tick_params(pad=-2.5)
            f.savefig(os.path.join(self.environment.figdir, 'mean_fc.png'), dpi=300, bbox_inches='tight')

            # Generate gradients
            gm_template = GradientMaps(n_components=2, approach='dm', kernel=None, random_state=0)
            gm_template.fit(pnc_conn_mat)

            # Plot eigenvalues
            f, ax = plt.subplots(1, figsize=(5, 4))
            ax.scatter(range(gm_template.lambdas_.size), gm_template.lambdas_)
            ax.set_xlabel('Component Nb')
            ax.set_ylabel('Eigenvalue')
            ax.tick_params(pad=-2.5)
            f.savefig(os.path.join(self.environment.figdir, 'gradient_eigenvals.png'), dpi=300, bbox_inches='tight')

            if self.environment.n_parcels == 200:
                gm_template.gradients_ = gm_template.gradients_ * -1
                self.gradients = np.zeros(gm_template.gradients_.shape)
                self.gradients[:, 0], self.gradients[:, 1] = gm_template.gradients_[:, 1], gm_template.gradients_[:, 0]
            elif self.environment.n_parcels == 400:
                self.gradients = np.zeros(gm_template.gradients_.shape)
                self.gradients[:, 0], self.gradients[:, 1] = gm_template.gradients_[:, 1] * -1, gm_template.gradients_[:, 0]
            else:
                gm_template.gradients_ = gm_template.gradients_ * -1
                self.gradients = np.zeros(gm_template.gradients_.shape)
                self.gradients[:, 0], self.gradients[:, 1] = gm_template.gradients_[:, 1], gm_template.gradients_[:, 0]

            # save outputs
            if not os.path.exists(self._output_dir()): os.makedirs(self._output_dir())
            np.savetxt(os.path.join(self._output_dir(), self._output_file()), self.gradients)

            # Plot first two gradients
            for g in np.arange(0, 1):
                gradient = DataVector(data=self.gradients[:, g], name='gradient_{0}'.format(g))
                gradient.brain_surface_plot(self.environment)


class LoadGeneExpression():
    def __init__(self, environment):
        self.environment = environment

    def _output_dir(self):
        return os.path.join(self.environment.pipelinedir, 'abagen')

    @staticmethod
    def _output_file():
        return 'expression.csv'

    def _check_outputs(self):
        if os.path.exists(self._output_dir()) and os.path.isfile(os.path.join(self._output_dir(), self._output_file())):
            return True
        else:
            return False

    def run(self):
        print('Pipeline: getting AHBA gene expression')
        if self._check_outputs():
            print('\toutput already exists...skipping')
            self.expression = pd.read_csv(os.path.join(self._output_dir(), self._output_file()))
            self.expression.set_index('label', inplace=True)
        else:
            if self.environment.parc == 'schaefer':
                parc_file = os.path.join(self.environment.research_data, 'parcellations', 'MNI',
                                         'Schaefer2018_{0}Parcels_17Networks_order_FSLMNI152_1mm.nii.gz' \
                                         .format(self.environment.n_parcels))
            elif self.environment.parc == 'glasser':
                pass

            if self.environment.parc == 'schaefer':
                self.expression = abagen.get_expression_data(parc_file,
                                                             data_dir=os.path.join(self.environment.research_data,
                                                                                   'abagen'),
                                                             verbose=2)

                # save outputs
                if not os.path.exists(self._output_dir()): os.makedirs(self._output_dir())
                self.expression.to_csv(os.path.join(self._output_dir(), self._output_file()))


class ComputeMinimumControlEnergy():
    def __init__(self, environment, A, states, B, T=1, control='minimum_fast', file_prefix='',
                 force_rerun=False, save_outputs=True, verbose=True):
        self.environment = environment
        self.A = A
        self.states = states
        self.B = B

        self.T = T
        self.control = control

        self.file_prefix = file_prefix

        self.force_rerun = force_rerun
        self.save_outputs = save_outputs
        self.verbose = verbose

    def _check_inputs(self):
        try:
            A_norm = self.A_norm
        except AttributeError:
            self.A_norm = matrix_normalization(self.A, version='continuous')

    def _output_dir(self):
        return os.path.join(self.environment.pipelinedir, 'minimum_control_energy')

    def _print_settings(self):
        unique = np.unique(self.states, return_counts=False)
        print('\tsettings:')
        print('\t\tn_states: {0}'.format(len(unique)))
        print('\t\tcontrol: {0}'.format(self.control))
        print('\t\tT: {0}'.format(self.T))

        if type(self.B) == DataMatrix:
            print('\t\tB: {0}'.format(self.B.name))
        elif type(self.B) == str:
            print('\t\tB: {0}'.format(self.B))
        else:
            print('\t\tB: unknown')

    def _get_file_prefix(self):
        unique = np.unique(self.states, return_counts=False)
        file_prefix = self.file_prefix+'ns-{0}_ctrl-{1}_T-{2}'.format(len(unique), self.control, self.T)
        if type(self.B) == DataMatrix:
            file_prefix = file_prefix+'_B-{0}_'.format(self.B.name)
        elif type(self.B) == str:
            file_prefix = file_prefix+'_B-{0}_'.format(self.B)
        else:
            file_prefix = file_prefix+'_B-unknown_'

        return file_prefix

    def _get_gmat(self):
        self.B = 'optimized'
        file_prefix = self._get_file_prefix()

        if os.path.exists(self._output_dir()) and \
                os.path.isfile(os.path.join(self._output_dir(), file_prefix+'gmat.npy')):

            self.gmat = np.load(os.path.join(self._output_dir(), file_prefix+'gmat.npy'))
            print('\tgmat loaded.')
        else:
            print('\tcomputing gmat...')
            self.gmat = get_gmat(A=self.A_norm, T=self.T)

            # save outputs
            if self.save_outputs:
                if not os.path.exists(self._output_dir()): os.makedirs(self._output_dir())
                np.save(os.path.join(self._output_dir(), file_prefix+'gmat'), self.gmat)

    def run(self):
        file_prefix = self._get_file_prefix()

        if self.verbose:
            print('Pipeline: getting minimum control energy')
            self._print_settings()
            print('\t' + file_prefix)

        self._check_inputs()

        if type(self.B) == DataMatrix:
            B = self.B.data
        else:
            B = self.B

        if os.path.exists(self._output_dir()) and \
                os.path.isfile(os.path.join(self._output_dir(), file_prefix+'E.npy')) and \
                self.force_rerun == False:

            if self.verbose:
                print('\toutput already exists...skipping')

            self.E = np.load(os.path.join(self._output_dir(), file_prefix+'E.npy'))
            if self.control != 'minimum_fast':
                self.n_err = np.load(os.path.join(self._output_dir(), file_prefix+'n_err.npy'))
        else:
            self.E, self.n_err = control_energy_helper(A=self.A_norm, states=self.states, B=B,
                                                       T=self.T, control=self.control)

            # save outputs
            if self.save_outputs:
                if not os.path.exists(self._output_dir()): os.makedirs(self._output_dir())
                np.save(os.path.join(self._output_dir(), file_prefix+'E'), self.E)
                if self.control != 'minimum_fast':
                    np.save(os.path.join(self._output_dir(), file_prefix+'n_err'), self.n_err)


    def run_with_optimized_b(self, n=1):
        # self.B = 'optimized'
        self.B = 'optimized-n-{0}'.format(n)
        file_prefix = self._get_file_prefix()

        if self.verbose:
            print('Pipeline: getting minimum control energy using optimized B weights')
            self._print_settings()
            print('\t' + file_prefix)
            print('\tNOTE: original B settings will be ignored and weights will be initialized using identity!')

        self._check_inputs()

        if os.path.exists(self._output_dir()) and \
                os.path.isfile(os.path.join(self._output_dir(), file_prefix+'E.npy')) and \
                self.force_rerun == False:

            if self.verbose:
                print('\toutput already exists...skipping')

            self.E_opt = np.load(os.path.join(self._output_dir(), file_prefix+'E.npy'))
            self.B_opt = np.load(os.path.join(self._output_dir(), file_prefix+'weights.npy'))
        else:
            self._get_gmat()

            x0_mat, xf_mat = expand_states(self.states)

            B0 = np.ones((self.A_norm.shape[0], x0_mat.shape[1]))

            B_opt, E_opt = grad_descent_b(A=self.A_norm, B0=B0, x0_mat=x0_mat, xf_mat=xf_mat,
                                          gmat=self.gmat, n=n, ds=0.1, T=self.T)

            # unique = np.unique(self.states, return_counts=False)
            # n_states = len(unique)
            # E_opt = E_opt[:, -1].reshape(n_states, n_states)

            self.B_opt = B_opt
            self.E_opt = E_opt

            # save outputs
            if self.save_outputs:
                if not os.path.exists(self._output_dir()): os.makedirs(self._output_dir())
                np.save(os.path.join(self._output_dir(), file_prefix+'E'), self.E_opt)
                np.save(os.path.join(self._output_dir(), file_prefix+'weights'), self.B_opt)

class Regression():
    def __init__(self, environment, X, y, c, X_name='X', y_name='y', c_name='c',
                 alg='rr', score='rmse', n_splits=10, runpca=False, n_rand_splits=100,
                 force_rerun=False):
        self.environment = environment
        self.X = X
        self.y = y
        self.c = c

        self.X_name = X_name
        self.y_name = y_name
        self.c_name = c_name

        self.alg = alg
        self.score = score
        self.n_splits = n_splits
        self.runpca = runpca
        self.n_rand_splits = n_rand_splits

        self.force_rerun = force_rerun

    def _output_dir(self):
        return os.path.join(self.environment.pipelinedir, 'prediction')

    def _get_file_prefix(self):
        file_prefix = '{0}_y-{1}_c-{2}_alg-{3}_score-{4}_pca-{5}'.format(self.X_name, self.y_name, self.c_name,
                                                                     self.alg, self.score, self.runpca)

        return file_prefix

    def _output_files(self):
        file_prefix = self._get_file_prefix()

        primary_files = ['{0}_accuracy_mean.txt'.format(file_prefix), '{0}_accuracy_std.txt'.format(file_prefix)]
        secondary_file = '{0}_accuracy_perm.txt'.format(file_prefix)

        return primary_files, secondary_file

    def _check_primary_outputs(self):
        if os.path.exists(self._output_dir()):
            primary_files, secondary_file = self._output_files()
            file_exists = []
            for file in primary_files:
                file_exists.append(os.path.isfile(os.path.join(self._output_dir(), file)))

            if np.all(file_exists):
                return True
            else:
                return False
        else:
            return False

    def _check_secondary_outputs(self):
        primary_files, secondary_file = self._output_files()

        if os.path.exists(self._output_dir()) and os.path.isfile(os.path.join(self._output_dir(), secondary_file)):
            return True
        else:
            return False

    def _print_settings(self):
        print('\tsettings:')
        print('\t\tX: {0}'.format(self.X_name))
        print('\t\ty: {0}'.format(self.y_name))
        print('\t\tc: {0}'.format(self.c_name))
        print('\t\talg: {0}'.format(self.alg))
        print('\t\tscore: {0}'.format(self.score))
        print('\t\tn_splits: {0}'.format(self.n_splits))
        print('\t\trunpca: {0}'.format(self.runpca))
        print('\t\tn_rand_splits: {0}'.format(self.n_rand_splits))

    def _get_reg(self):
        regs = {'linear': LinearRegression(),
                'rr': Ridge(),
                'lr': Lasso(),
                'krr_lin': KernelRidge(kernel='linear'),
                'krr_rbf': KernelRidge(kernel='rbf'),
                'svr_lin': SVR(kernel='linear'),
                'svr_rbf': SVR(kernel='rbf')
                }
        self.reg = regs[self.alg]

    def _get_scorer(self):
        if self.score == 'r2':
            self.scorer = make_scorer(r2_score, greater_is_better=True)
        elif self.score == 'corr':
            self.scorer = make_scorer(corr_true_pred, greater_is_better=True)
        elif self.score == 'mse':
            self.scorer = make_scorer(mean_squared_error, greater_is_better=False)
        elif self.score == 'rmse':
            self.scorer = make_scorer(root_mean_squared_error, greater_is_better=False)
        elif self.score == 'mae':
            self.scorer = make_scorer(mean_absolute_error, greater_is_better=False)

    def run(self):
        print('Pipeline: prediction (out-of-sample regression)')
        self._print_settings()
        file_prefix = self._get_file_prefix()
        print('\t' + file_prefix)

        if self._check_primary_outputs() and self.force_rerun == False:
            print('\toutput already exists...skipping')
            primary_files, secondary_file = self._output_files()
            self.accuracy_mean = np.loadtxt(os.path.join(self._output_dir(), primary_files[0]))
            self.accuracy_std = np.loadtxt(os.path.join(self._output_dir(), primary_files[1]))
        else:
            self._get_reg()
            self._get_scorer()

            accuracy_mean = np.zeros(self.n_rand_splits)
            accuracy_std = np.zeros(self.n_rand_splits)

            for i in tqdm(np.arange(self.n_rand_splits)):
                accuracy, y_pred_out = run_reg(X=self.X, y=self.y, c=self.c, reg=self.reg,
                                               scorer=self.scorer, n_splits=self.n_splits, runpca=self.runpca,
                                               seed=i)
                accuracy_mean[i] = accuracy.mean()
                accuracy_std[i] = accuracy.std()

            self.accuracy_mean = accuracy_mean
            self.accuracy_std = accuracy_std

            # save outputs
            if not os.path.exists(self._output_dir()): os.makedirs(self._output_dir())
            primary_files, secondary_file = self._output_files()
            np.savetxt(os.path.join(self._output_dir(), primary_files[0]), self.accuracy_mean)
            np.savetxt(os.path.join(self._output_dir(), primary_files[1]), self.accuracy_std)

    def run_perm(self):
        print('Pipeline: prediction, permutation test')
        if self._check_secondary_outputs() and self.force_rerun == False:
            print('\toutput already exists...skipping')
            primary_files, secondary_file = self._output_files()
            self.accuracy_perm = np.loadtxt(os.path.join(self._output_dir(), secondary_file))
        else:
            self._get_reg()
            self._get_scorer()

            accuracy_perm = run_perm(X=self.X, y=self.y, c=self.c, reg=self.reg,
                                    scorer=self.scorer, n_splits=self.n_splits, runpca=self.runpca)

            self.accuracy_perm = accuracy_perm

            # save outputs
            if not os.path.exists(self._output_dir()): os.makedirs(self._output_dir())
            primary_files, secondary_file = self._output_files()
            np.savetxt(os.path.join(self._output_dir(), secondary_file), self.accuracy_perm)


class DCM():
    def __init__(self, environment, Subject, states,
                 force_rerun=False):
        self.environment = environment
        self.Subject = Subject
        self.states = states
        self.force_rerun = force_rerun

    def _output_dir(self):
        return os.path.join(self.environment.pipelinedir, 'spdcm')

    def _output_file(self):
        unique = np.unique(self.states)
        n_states = len(unique)

        return 'rsts_states_ns-{0}.npy'.format(n_states)

    def _check_outputs(self):
        if os.path.exists(self._output_dir()) and os.path.isfile(os.path.join(self._output_dir(), self._output_file())):
            return True
        else:
            return False

    def run_mean_ts(self):
        print('Pipeline: getting time series for dcm')
        if self._check_outputs() and self.force_rerun == False:
            print('\toutput already exists...skipping')
            self.rsts_states = np.load(os.path.join(self._output_dir(), self._output_file()))
        else:
            n_subs = self.environment.df.shape[0]
            unique = np.unique(self.states)
            n_states = len(unique)

            rsts = np.zeros((self.environment.n_trs, self.environment.n_parcels, n_subs))
            # rsts_states = np.zeros((self.environment.n_trs, n_states, n_subs))
            rsts_states = np.zeros((self.environment.n_trs, n_states))

            for i in tqdm(np.arange(n_subs)):
                subject = self.Subject(environment=self.environment, subjid=self.environment.df.index[i])
                subject.get_file_names()
                subject.load_rsts()
                rsts[:, :, i] = sp.stats.zscore(subject.rsts, axis=0)

                # # get state time series for each subject
                # for j in np.arange(n_states):
                #     rsts_tmp = subject.rsts[:, self.states == j]
                #     # rsts_tmp = np.mean(rsts_tmp, axis=1)
                #
                #     # sc = StandardScaler()
                #     # sc.fit(rsts_tmp)
                #     # rsts_tmp = sc.transform(rsts_tmp)
                #     pca = PCA(n_components=1, svd_solver='full')
                #     pca.fit(rsts_tmp)
                #     rsts_tmp = pca.transform(rsts_tmp)
                #
                #     rsts_states[:, j, i] = rsts_tmp[:, 0]

            # mean over subjects
            rsts = np.mean(rsts, axis=2)

            for j in np.arange(n_states):
                rsts_tmp = rsts[:, self.states == j]

                sc = StandardScaler()
                sc.fit(rsts_tmp)
                rsts_tmp = sc.transform(rsts_tmp)
                pca = PCA(n_components=1, svd_solver='full')
                pca.fit(rsts_tmp)
                rsts_tmp = pca.transform(rsts_tmp)

                rsts_states[:, j] = rsts_tmp[:, 0]

            self.rsts = rsts
            self.rsts_states = rsts_states

            # save outputs
            if not os.path.exists(self._output_dir()): os.makedirs(self._output_dir())
            np.save(os.path.join(self._output_dir(), self._output_file()), self.rsts_states)

    def run_concat_ts(self):
        print('Pipeline: getting time series for dcm')
        if self._check_outputs() and self.force_rerun == False:
            print('\toutput already exists...skipping')
            self.rsts_states = np.load(os.path.join(self._output_dir(), self._output_file()))
        else:
            n_subs = self.environment.df.shape[0]
            n_trs = self.environment.n_trs * n_subs
            unique = np.unique(self.states)
            n_states = len(unique)

            rsts_states = np.zeros((n_trs, n_states))

            for i in tqdm(np.arange(n_subs)):
                subject = self.Subject(environment=self.environment, subjid=self.environment.df.index[i])
                subject.get_file_names()
                subject.load_rsts()
                rsts = sp.stats.zscore(subject.rsts, axis=0)

                # mean over states
                rsts_mean = np.zeros((self.environment.n_trs, n_states))
                for j in np.arange(n_states):
                    rsts_mean[:, j] = np.mean(rsts[:, self.states == j], axis=1)

                # z score and store
                start_idx = i * self.environment.n_trs
                end_idx = start_idx + self.environment.n_trs
                rsts_states[start_idx:end_idx, :] = rsts_mean

            self.rsts_states = rsts_states

            # save outputs
            if not os.path.exists(self._output_dir()): os.makedirs(self._output_dir())
            np.save(os.path.join(self._output_dir(), self._output_file()), self.rsts_states)

    def run_concat_mean_ts(self):
        print('Pipeline: getting time series for dcm')
        if self._check_outputs() and self.force_rerun == False:
            print('\toutput already exists...skipping')
            self.rsts_states = np.load(os.path.join(self._output_dir(), self._output_file()))
        else:
            n_subs = self.environment.df.shape[0]
            if n_subs % 10 > 0:
                n_subs = n_subs - (n_subs % 10)

            n_trs = self.environment.n_trs * n_subs
            unique = np.unique(self.states)
            n_states = len(unique)

            rsts_states = np.zeros((n_trs, n_states))

            for i in tqdm(np.arange(n_subs)):
                subject = self.Subject(environment=self.environment, subjid=self.environment.df.index[i])
                subject.get_file_names()
                subject.load_rsts()
                rsts = sp.stats.zscore(subject.rsts, axis=0)

                # mean over states
                rsts_mean = np.zeros((self.environment.n_trs, n_states))
                for j in np.arange(n_states):
                    rsts_mean[:, j] = np.mean(rsts[:, self.states == j], axis=1)

                # z score and store
                start_idx = i * self.environment.n_trs
                end_idx = start_idx + self.environment.n_trs
                rsts_states[start_idx:end_idx, :] = rsts_mean

            # mean over subsets of subjects
            n_trs_e = self.environment.n_trs * 10
            n_cols = int(n_trs / n_trs_e)

            rsts_states = rsts_states.reshape(n_trs_e, n_cols, n_states, order='F')
            rsts_states = np.mean(rsts_states, axis=1)

            self.rsts_states = rsts_states

            # save outputs
            if not os.path.exists(self._output_dir()): os.makedirs(self._output_dir())
            np.save(os.path.join(self._output_dir(), self._output_file()), self.rsts_states)
