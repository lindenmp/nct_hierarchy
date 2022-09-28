# %% import
import sys, os, platform

from bct.algorithms import retrieve_shortest_path, distance_wei_floyd
from src.communicability import matching_index, get_pt_cum

# %% import workspace
os.environ["MY_PYTHON_WORKSPACE"] = 'ave_adj'
os.environ["WHICH_BRAIN_MAP"] = 'hist-g2'
os.environ["INTRAHEMI"] = "False"
from setup_workspace import *

# %%
transform = 'inv'
D, hops, Pmat = distance_wei_floyd(A, transform=transform)
m = matching_index(A)

# %%
import seaborn as sns
import matplotlib.pyplot as plt
sns.set(style='white', context='paper', font_scale=1, font='Helvetica')
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42
plt.rcParams['savefig.format'] = 'svg'
plt.rcParams['font.size'] = 8
plt.rcParams['svg.fonttype'] = 'none'

color_1 = sns.color_palette("pastel")[0]
color_2 = sns.color_palette("pastel")[1]

hops_copy = hops.copy()

counter = 0
f, ax = plt.subplots(8, 4, figsize=(7, 9))
f.tight_layout()
for r in np.arange(8):
    for c in np.arange(4):
        p = np.argmax(hops_copy)
        i = np.unravel_index(p, hops_copy.shape)[0]
        j = np.unravel_index(p, hops_copy.shape)[1]
        print(p, i, j, hops_copy[i, j])

        path = retrieve_shortest_path(i, j, hops_copy, Pmat)
        K = len(path)

        pt_ij, pt_ji = get_pt_cum(m, path)

        ax[r, c].plot(pt_ij, color=color_1, marker='o', linewidth=.5, markersize=2, label="s —> t")
        ax[r, c].plot(pt_ji, color=color_2, marker='o', linewidth=.5, markersize=2, label="t —> s")
        if counter == 0:
            ax[r, c].legend()

        if c == 0:
            ax[r, c].set_ylabel('path transitivity')
        if r == 7:
            ax[r, c].set_xlabel('path segment')
        ax[r, c].set_title('path length: {0}'.format(K))
        ax[r, c].tick_params(pad=-2)

        hops_copy[i, j] = 0
        hops_copy[j, i] = 0
        counter += 1

f.savefig(os.path.join(environment.figdir, 'pt_ij_counter{0}'.format(counter)),
          dpi=600, bbox_inches='tight', pad_inches=0.01)
plt.close()
