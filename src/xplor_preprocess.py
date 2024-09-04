"""
"""
import os
import sys

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

sub = int(sys.argv[1])
ses = int(sys.argv[2])
_file = f"sub-{sub:03d}_ses-{ses:02d}_task_motorimagery_eeg.edf"
raw = mne.io.read_raw_edf(f"{DS_DIR}{EDF_DIR}{_file}", preload=True)
raw.set_montage("standard_1020")


DIR = f"{DS_DIR}/events/"
file = f"sub-0{sub:02d}_ses-{ses:02d}_task_motorimagery_events.tsv"
events = np.loadtxt(f"{DIR}{file}", delimiter='\t', dtype='object',
                    skiprows=1)

events = events[:, [0, 1, -1]]
events[:, 0] = np.array([int(float(num.strip())) for num in events[:, 0]])
events[:, 1] = np.array([0 for _ in events[:, 1]])
events[:, -1] = np.array([int(float(num.strip())) for num in events[:, -1]])
events = events.astype(np.int32)
events[:, 0] -= 1


event_id = {"left": 1, "right": 2}
epochs = mne.Epochs(raw, tmin=0.0, tmax=3.99, events=events, event_id=event_id,
                    baseline=None, preload=True)
epochs_filt = epochs.copy()

epochs_filt.filter(l_freq=1, h_freq=40, pad='reflect_limited', verbose=True)

data_filtered = np.concatenate(epochs_filt.get_data(copy=True), axis=1)
data_filtered.shape, raw.get_data().shape

raw_filt = mne.io.RawArray(data_filtered, epochs_filt.info)

ica = mne.preprocessing.ICA(n_components=32, random_state=SEED)
ica.fit(raw_filt)

eog_indices, eog_scores = ica.find_bads_eog(raw, ch_name=['Fp1', 'Fp2'])

print(eog_indices)
# ica.plot_scores(eog_scores)

print(ica.get_explained_variance_ratio(raw))
print(eog_scores[0].shape)

CLUSTER = True
if CLUSTER:
    _shape = (eog_scores[0].shape[0], 2)
    _eog_scores = np.zeros(_shape)
    for i in range(2):
        _eog_scores[:, i] = np.abs(eog_scores[i])

    # get dist to origin
    _dist = np.linalg.norm(_eog_scores, axis=1)

    clstr = SpectralClustering(n_clusters=2, random_state=SEED,
                            assign_labels='discretize')
    clstr.fit(np.reshape(_dist, (-1, 1)))

    mask = clstr.labels_ == 0
    avg_0 = np.mean(_dist[mask])
    avg_1 = np.mean(_dist[~mask])

    exclude = 0 if avg_0 > avg_1 else 1
    ica.exclude = np.nonzero(clstr.labels_ == exclude)[0]


    # fig, ax = plt.subplots()
    # ax.plot(_eog_scores[clstr.labels_ == 0, 0], _eog_scores[clstr.labels_ == 0, 1], 'b*')
    # ax.plot(_eog_scores[clstr.labels_ == 1, 0], _eog_scores[clstr.labels_ == 1, 1], 'r*')
else:
    ica.exclude = eog_indices



# reconstruct raw data with excluded components
raw_corrected = raw.copy()
ica.apply(raw_corrected)


epochs = mne.Epochs(raw_corrected, tmin=0.0, tmax=3.99, events=events,
                    event_id=event_id, baseline=None, preload=True)


epochs = epochs.filter(l_freq=1, h_freq=40, pad='reflect_limited', verbose=True)

# find max amplitude in each epoch and channel
print()
print()
print()
print(12 * '-' + " INTERPOLATION " + 12 * '-')
epochs_backup = epochs.copy()
new_epochs = []
for epoch_idx, epoch in enumerate(epochs_backup):
    bad_channs = []
    max_amps = np.max(np.abs(epoch), axis=1)
    mask = max_amps > 100e-6

    # find bad channels
    if np.any(mask):
        print(f"Epoch {epoch_idx} has bad channels")
    idx = list(np.argwhere(mask).astype(np.int32).flatten())

    # append to list of bad channels
    for i in idx:
        _chann = epochs_backup.ch_names[i]
        print(f"channel {i} ({_chann})"
              + f" has max amplitude {max_amps[i] * 1e6:.2f}")
        bad_channs.append(epochs_backup.ch_names[i])
    
    # interpolate bad channels
    current_epoch = epochs_backup[epoch_idx]
    if len(bad_channs) > 0:
        current_epoch.info['bads'] = bad_channs
        print(f"epoch {epoch_idx} has {current_epoch.info['bads']} bad channels")
        current_epoch.plot(picks=bad_channs, show=False)
        current_epoch = current_epoch.interpolate_bads(reset_bads=True)
        current_epoch.plot(picks=bad_channs, show=False)
        # plt.show()

    new_epochs.append(current_epoch)
    del bad_channs

new_epochs = mne.concatenate_epochs(new_epochs, add_offset=False)
# print(new_epochs.get_data().shape)
# print(new_epochs.events)
# exit()

# new_epochs.average().plot(show=False)
# epochs.average().plot(show=False)

_cmap = 'viridis'
evoked = new_epochs.average()
volt_max = evoked.data.max() * 1e6
volt_min = evoked.data.min() * 1e6

topo_args = {'cmap': _cmap,
             'vlim': (volt_min, volt_max),
             'average': 0.1}

# new_epochs['right'].average().plot_joint(times='auto', show=False,
#                                          topomap_args=topo_args)
# new_epochs['left'].average().plot_joint(times='auto', show=False,
#                                         topomap_args=topo_args)

""" save preprocessed epochs """
SAVE = True
if SAVE:
    save_dir = os.path.join(f"../preprocess_first/sub_{sub:02d}")
    if not os.path.isdir(save_dir):
        os.mkdir(save_dir)

    save_dir_ses = os.path.join(save_dir, f"ses_{ses:02d}")
    if not os.path.isdir(save_dir_ses):
        os.mkdir(save_dir_ses)

    epochs_file = os.path.join(save_dir_ses, "epochs_preprocessed-epo.fif")

    print()
    print()
    print(10 * '-' + " Saving preprocessed epochs " + 10 * '-')
    new_epochs.save(epochs_file, overwrite=False)
