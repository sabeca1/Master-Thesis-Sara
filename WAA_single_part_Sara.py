# flow paper of robert luke

# import packages
import os
import os.path as op
import numpy as np
import matplotlib.pyplot as plt
import mne
import mne_nirs
import pandas as pd

from scipy.stats import sem
from scipy import stats
from itertools import compress
from pprint import pprint
from nilearn.plotting import plot_design_matrix

from mne.viz import plot_compare_evokeds
from mne.preprocessing.nirs import (optical_density,
                                    temporal_derivative_distribution_repair, scalp_coupling_index)

from mne_nirs.visualisation import plot_glm_surface_projection
from mne_nirs.experimental_design import make_first_level_design_matrix
from mne_nirs.statistics import run_glm
from mne_nirs.channels import (get_long_channels,
                               get_short_channels,
                               picks_pair_to_idx)

# define the path
datapath = 'subjects/sub-01/ses-02/nirs'

# load the data
raw_intensity = mne.io.read_raw_nirx(datapath).load_data()

# downsample the data to 3 hz
raw_intensity.resample(4, npad="auto")

raw_intensity.annotations.rename(
    {"1.0": "Control", "2.0": "Noise", "3.0": "Speech"}
)
raw_intensity.annotations.delete(raw_intensity.annotations.description == "4.0")
raw_intensity.annotations.delete(raw_intensity.annotations.description == "5.0")

raw_intensity.annotations.set_durations(5)

# convert to optical density
raw_od = optical_density(raw_intensity)

# do that the only channels that remain for the analysis are "Auditory_Left": ['S4_D2', 'S4_D3', 'S5_D2', 'S5_D3', 'S5_D4', 'S5_D5'],"Auditory_Right": ['S11_D12', 'S11_D11', 'S10_D12', 'S10_D10', 'S10_D11', 'S10_D9']
raw_od.info['bads'] = [
    'S1_D1 785', 'S1_D1 830',
    'S2_D1 785', 'S2_D1 830',
    'S2_D13 785', 'S2_D13 830',
    'S3_D1 785', 'S3_D1 830',
    'S3_D2 785', 'S3_D2 830',
    'S4_D14 785', 'S4_D14 830',
    'S5_D15 785', 'S5_D15 830',
    'S6_D6 785', 'S6_D6 830',
    'S6_D8 785', 'S6_D8 830',
    'S6_D16 785', 'S6_D16 830',
    'S7_D6 785', 'S7_D6 830',
    'S7_D7 785', 'S7_D7 830',
    'S7_D8 785', 'S7_D8 830',
    'S8_D7 785', 'S8_D7 830',
    'S8_D8 785', 'S8_D8 830',
    'S8_D17 785', 'S8_D17 830',
    'S9_D8 785', 'S9_D8 830',
    'S10_D18 785', 'S10_D18 830',
    'S11_D19 785', 'S11_D19 830',
    'S12_D1 785', 'S12_D1 830',
    'S12_D20 785', 'S12_D20 830'
]


raw_od= raw_od.copy().drop_channels(raw_od.info['bads'])

# calculate the scalp coupling index between 0.7 amd 1.45 hz
sci = scalp_coupling_index(raw_od, l_freq=0.7, h_freq=1.4)

# remove channels with bad sci
raw_od.info['bads'] = list(compress(raw_od.ch_names, sci < 0.15))

# remove the bad channels
raw_od.pick(picks=[ch for ch in raw_od.ch_names if ch not in raw_od.info['bads']])


print(f"Removed {len(raw_intensity.ch_names) - len(raw_od.ch_names)} channels")

# apply temppooral derivative distribution repair
raw_od = temporal_derivative_distribution_repair(raw_od)

# identify short and long channels
#short_chs = get_short_channels(raw_od)
raw_haemo = get_long_channels(raw_od, min_dist=0.02, max_dist=0.04)

# apply short channel regression
#raw_od_enhanced = mne_nirs.signal_enhancement.short_channel_regression(raw_od)

# beer lambert law with pf 0.1
raw_haemo = mne.preprocessing.nirs.beer_lambert_law(raw_haemo, ppf=0.1)

# exclude channels with Source detection outside 20-40 mm

# signal improvement using negative correlation of HbO and HbR
raw_haemo = mne_nirs.signal_enhancement.enhance_negative_correlation(raw_haemo)

# bandpass filter the data between 0.01 and 0.7 using transitions of 0.005 and 0.3 hz

raw_haemo = raw_haemo.filter(0.01, 0.7,  h_trans_bandwidth=0.3, l_trans_bandwidth=0.005)


events, event_dict = mne.events_from_annotations(raw_haemo)
#fig = mne.viz.plot_events(events, event_id=event_dict, sfreq=raw_haemo.info["sfreq"])
# Epochs from -3 to 14 seconds. apply linear detrend to each epoch. rejecting criteria of 100e-6.
reject_criteria = dict(hbo=100e-6)
tmin, tmax = -3, 14

epochs = mne.Epochs(
    raw_haemo,
    events,
    event_id=event_dict,
    tmin=tmin,
    tmax=tmax,
    reject=reject_criteria,
    reject_by_annotation=True,
    proj=True,
    baseline=(None, 0),
    preload=True,
    detrend=1,
    verbose=True,
)
epochs.plot_drop_log()

evoked_dict = {'Noise/HbO': epochs['Noise'].average(picks='hbo'),
                'Noise/HbR': epochs['Noise'].average(picks='hbr'),
                'Speech/HbO': epochs['Speech'].average(picks='hbo'),
                'Speech/HbR': epochs['Speech'].average(picks='hbr'),
                'Control/HbO': epochs['Control'].average(picks='hbo'),
                'Control/HbR': epochs['Control'].average(picks='hbr')} 

for condition in evoked_dict:
    evoked_dict[condition].rename_channels(lambda x: x[:-4])
    
color_dict = dict(HbO='#AA3377', HbR='b')
styles_dict = dict(Control=dict(linestyle='dotted'), Noise=dict(linestyle='dashed'))

mne.viz.plot_compare_evokeds(evoked_dict, combine="mean", ci=0.95, colors=color_dict, styles=styles_dict,show=True)
                                
                                




