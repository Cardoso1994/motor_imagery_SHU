#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
import mne
from sklearn.cluster import KMeans, SpectralClustering, DBSCAN
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

# global
SEED = 73
np.random.seed(seed=SEED)
RESAMPLE_FREQ = 256
DS_DIR = "../mi_ds/"
EDF_DIR = "edf/"

BAD_SUBS = [5, 6, 15, 22, 24, 25]

# for sub in range(1, 26):
#     if sub in BAD_SUBS:
#         continue
#     for ses in range(1, 6):
#         print(5 * '-' + f"Subject {sub} session {ses} " + 5 * '-')
#         _file = f"sub-{sub:03d}_ses-{ses:02d}_task_motorimagery_eeg.edf"
#         mne.io.read_raw_edf(f"{DS_DIR}{EDF_DIR}{_file}")

sub = 1
ses = 2
_file = f"sub-{sub:03d}_ses-{ses:02d}_task_motorimagery_eeg.edf"
raw = mne.io.read_raw_edf(f"{DS_DIR}{EDF_DIR}{_file}")
# raw.pick(['O2']).plot()
raw.compute_psd().plot()
plt.show()
