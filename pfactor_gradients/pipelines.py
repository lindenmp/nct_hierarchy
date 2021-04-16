import os
import numpy as np
import pandas as pd
from pfactor_gradients.routines import LoadFC
from pfactor_gradients.imaging_derivs import DataVector
from pfactor_gradients.energy import control_energy_helper
from pfactor_gradients.prediction import corr_true_pred, root_mean_squared_error, run_reg, run_perm
from brainspace.gradient import GradientMaps
from sklearn.cluster import KMeans
from sklearn.linear_model import Ridge, Lasso, LinearRegression
from sklearn.kernel_ridge import KernelRidge
from sklearn.svm import SVR
from sklearn.metrics import make_scorer, r2_score, mean_squared_error, mean_absolute_error
import abagen
from tqdm import tqdm

# %% Plotting
import matplotlib.pyplot as plt
import seaborn as sns

sns.set(style='white', context='talk', font_scale=1)
import matplotlib.font_manager as font_manager
try:
    fontpath = 'PublicSans-Thin.ttf'
    prop = font_manager.FontProperties(fname=fontpath)
    plt.rcParams['font.family'] = prop.get_name()
    plt.rcParams['svg.fonttype'] = 'none'
except:
    pass

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
            fc = loadfc.fc

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
            np.savetxt(os.path.join(self._output_dir(), 'gradients.txt'), self.gradients)

            # Plot first two gradients
            for g in np.arange(0, 2):
                gradient = DataVector(data=self.gradients[:, g], name='gradient_{0}'.format(g))
                gradient.brain_surface_plot(self.environment)

        # Cluster gradient
        self.n_clusters = int(self.environment.n_parcels * .05)
        self.kmeans = KMeans(n_clusters=self.n_clusters, random_state=0).fit(self.gradients)
        self.unique, self.counts = np.unique(self.kmeans.labels_, return_counts=True)

        # Plot clustered gradient
        # f, ax = plt.subplots(figsize=(5, 5))
        # ax.scatter(self.gradients[:, 1], self.gradients[:, 0], c=self.kmeans.labels_, cmap='Set3')
        # for i, txt in enumerate(np.arange(self.n_clusters)):
        #     ax.annotate(txt, (self.kmeans.cluster_centers_[i, 1], self.kmeans.cluster_centers_[i, 0]),
        #                 ha="center", va="center", size=15)
        # ax.set_xlabel('Gradient 2')
        # ax.set_ylabel('Gradient 1')
        # ax.tick_params(pad=-2.5)
        # f.savefig(os.path.join(self.environment.figdir, 'gradient_clusters.png'), dpi=150, bbox_inches='tight',
        #           pad_inches=0.1)

        # equally sized bins based on principal gradient
        bin_size = int(self.environment.n_parcels / (self.environment.n_parcels * .10))
        n_bins = int(self.environment.n_parcels / bin_size)

        grad_bins = np.array([])
        for i in np.arange(n_bins):
            grad_bins = np.append(grad_bins, np.ones(bin_size) * i)

        grad_bins = grad_bins.astype(int)
        sort_idx = np.argsort(self.gradients[:, 0])
        unsorted_idx = np.argsort(sort_idx)
        grad_bins = grad_bins[unsorted_idx]
        # DataVector(data=grad_bins+1, name='grad_bins').brain_surface_plot(self.environment, cmap='coolwarm')
        self.grad_bins = grad_bins

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
            self.expression = abagen.get_expression_data(os.path.join(self.environment.projdir, 'figs_support', 'labels',
                                                                      'schaefer{0}'.format(self.environment.n_parcels),
                                                                      'schaefer{0}MNI.nii.gz'.format(self.environment.n_parcels)),
                                                         data_dir=os.path.join(self.environment.external_ssd, 'abagen'), verbose=2)

            # save outputs
            if not os.path.exists(self._output_dir()): os.makedirs(self._output_dir())
            self.expression.to_csv(os.path.join(self._output_dir(), self._output_file()))


class ComputeMinimumControlEnergy():
    def __init__(self, environment, A, states, n_subsamples=0, control='minimum', T=1, B='wb', file_prefix='',
                 force_rerun=False, save_outputs=True, add_noise=False, verbose=True):
        self.environment = environment
        self.A = A
        self.states = states
        self.n_subsamples = n_subsamples

        self.control = control
        self.T = T
        self.B = B

        self.file_prefix = file_prefix

        self.force_rerun = force_rerun
        self.save_outputs = save_outputs
        self.add_noise = add_noise
        self.verbose = verbose

    def _output_dir(self):
        return os.path.join(self.environment.pipelinedir, 'minimum_control_energy')

    def _print_settings(self):
        unique = np.unique(self.states, return_counts=False)
        print('\tsettings:')
        print('\t\tn_states: {0}'.format(len(unique)))
        print('\t\tn_subsamples: {0}'.format(self.n_subsamples))
        print('\t\tcontrol: {0}'.format(self.control))
        print('\t\tT: {0}'.format(self.T))

        if type(self.B) == str:
            print('\t\tB: {0}'.format(self.B))
        elif type(self.B) == DataVector:
            print('\t\tB: {0}'.format(self.B.name))
        else:
            print('\t\tB: vector')

    def _get_file_prefix(self):
        unique = np.unique(self.states, return_counts=False)
        file_prefix = self.file_prefix+'ns-{0}-{1}_c-{2}_T-{3}'.format(len(unique), self.n_subsamples, self.control, self.T)
        if type(self.B) == str:
            file_prefix = file_prefix+'_B-{0}_'.format(self.B)
        elif type(self.B) == DataVector:
            file_prefix = file_prefix+'_B-{0}_'.format(self.B.name)
        else:
            file_prefix = file_prefix+'_B-vector_'

        return file_prefix

    def run(self):
        print('Pipeline: getting minimum control energy')
        if self.verbose:
            self._print_settings()
        file_prefix = self._get_file_prefix()
        if self.add_noise:
            file_prefix = 'noise_'+file_prefix

        print('\t' + file_prefix)

        if type(self.B) == DataVector:
            B = self.B.data
        else:
            B = self.B

        if os.path.exists(self._output_dir()) and \
                os.path.isfile(os.path.join(self._output_dir(), file_prefix+'E.npy')) and \
                self.force_rerun == False:
            print('\toutput already exists...skipping')
            self.E = np.load(os.path.join(self._output_dir(), file_prefix+'E.npy'))
            if self.control != 'minimum_fast':
                self.n_err = np.load(os.path.join(self._output_dir(), file_prefix+'n_err.npy'))
        else:
            self.E, self.n_err = control_energy_helper(self.A, self.states, n_subsamples=self.n_subsamples,
                                                       control=self.control, T=self.T, B=B, add_noise=self.add_noise)

            # save outputs
            if self.save_outputs:
                if not os.path.exists(self._output_dir()): os.makedirs(self._output_dir())
                np.save(os.path.join(self._output_dir(), file_prefix+'E'), self.E)
                if self.control != 'minimum_fast':
                    np.save(os.path.join(self._output_dir(), file_prefix+'n_err'), self.n_err)


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
        file_prefix = '{0}-{1}-{2}_alg-{3}_score-{4}_pca-{5}'.format(self.X_name, self.y_name, self.c_name,
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
