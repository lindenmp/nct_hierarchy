import os
import numpy as np
from sklearn.cluster import KMeans
from data_loader.routines import LoadFC
from utils.imaging_derivs import DataVector
from brainspace.gradient import GradientMaps
import nibabel as nib
import abagen
import pandas as pd

# %% Plotting
import matplotlib.pyplot as plt
import seaborn as sns
from utils.plotting import roi_to_vtx
import nibabel as nib
from nilearn import plotting
sns.set(style='white', context='talk', font_scale=1)
import matplotlib.font_manager as font_manager
fontpath = '/Users/lindenmp/Library/Fonts/PublicSans-Thin.ttf'
prop = font_manager.FontProperties(fname=fontpath)
plt.rcParams['font.family'] = prop.get_name()
plt.rcParams['svg.fonttype'] = 'none'

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
                gradient = DataVector(data=self.gradients[:, g])
                gradient.brain_surface_plot(self.environment, figname='gradient_{0}.png'.format(g))

        # Cluster gradient
        self.n_clusters = int(self.environment.n_parcels * .05)
        self.kmeans = KMeans(n_clusters=self.n_clusters, random_state=0).fit(self.gradients)
        self.unique, self.counts = np.unique(self.kmeans.labels_, return_counts=True)

        # Plot clustered gradient
        f, ax = plt.subplots(figsize=(5, 5))
        ax.scatter(self.gradients[:, 1], self.gradients[:, 0], c=self.kmeans.labels_, cmap='Set3')
        for i, txt in enumerate(np.arange(self.n_clusters)):
            ax.annotate(txt, (self.kmeans.cluster_centers_[i, 1], self.kmeans.cluster_centers_[i, 0]),
                        ha="center", va="center", size=15)
        ax.set_xlabel('Gradient 2')
        ax.set_ylabel('Gradient 1')
        ax.tick_params(pad=-2.5)
        f.savefig(os.path.join(self.environment.figdir, 'gradient_clusters.png'), dpi=150, bbox_inches='tight',
                  pad_inches=0.1)


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