import os
import numpy as np
import pandas as pd
import mne
import matplotlib.pyplot as plt
from itertools import compress
from mne.preprocessing.nirs import optical_density, temporal_derivative_distribution_repair
from mne_nirs.preprocessing import scalp_coupling_index_windowed
from mne_nirs.visualisation import plot_timechannel_quality_metric

plt.ioff()

# Initialize dataframe to store bad channel info
bad_channels_df = pd.DataFrame(columns=['Participant', 'Session', 'Num_Bads', 'Bad_Channels'])

# Loop through all participants and sessions
for i in range(1, 22):  # sub-01 to sub-20
    subj_id = f'sub-{i:02d}'
    print(f'Processing {subj_id}')
    
    for session in ['ses-01', 'ses-02']:
        datapath = f'subjects/{subj_id}/{session}/nirs'
        save_path = f'results/{subj_id}/{session}/'
        os.makedirs(save_path, exist_ok=True)
        
        # Load data
        raw_intensity = mne.io.read_raw_nirx(datapath).load_data()
        
        # Rename annotations
        raw_intensity.annotations.rename({"1.0": "Control", "2.0": "Noise", "3.0": "Speech", '4.0': 'XStop', '5.0': 'XStart'})
        
        # Get event timings
        Breaks, _ = mne.events_from_annotations(raw_intensity, {'XStop': 4, 'XStart': 5})
        AllEvents, _ = mne.events_from_annotations(raw_intensity)
        Breaks = Breaks[:, 0] / raw_intensity.info['sfreq']
        LastEvent = AllEvents[-1, 0] / raw_intensity.info['sfreq']

        # Ensure valid breaks structure
        if len(Breaks) % 2 == 0:
            raise ValueError("Breaks array should have an odd number of elements (first XStart, then XStop/XStart pairs).")

        # Get original duration
        original_duration = raw_intensity.times[-1] - raw_intensity.times[0]
        print(f"Original duration: {original_duration:.2f} seconds")

        # Create cropped dataset
        cropped_intensity = raw_intensity.copy().crop(Breaks[0], Breaks[1])
        for j in range(2, len(Breaks) - 1, 2):
            block = raw_intensity.copy().crop(Breaks[j], Breaks[j + 1])
            cropped_intensity.append(block)
        cropped_intensity.append(raw_intensity.copy().crop(Breaks[-1], LastEvent + 15.25))

        # Get cropped duration
        cropped_duration = cropped_intensity.times[-1] - cropped_intensity.times[0]
        print(f"Cropped duration: {cropped_duration:.2f} seconds")

        # Ensure cropping was successful
        if cropped_duration >= original_duration:
            print(f"WARNING: Cropping did not reduce duration for {subj_id} - {session}!")

        # Assign cropped data back to raw_intensity
        raw_intensity = cropped_intensity.copy()

        # Optical density and correction
        cropped_od = optical_density(raw_intensity)
        cropped_corrected_od = temporal_derivative_distribution_repair(cropped_od)
        
        # Scalp Coupling Index
        sci = mne.preprocessing.nirs.scalp_coupling_index(cropped_corrected_od)
        fig, ax = plt.subplots()
        ax.hist(sci)
        ax.set_title(f'Scalp Coupling Index - {subj_id} - {session}')
        
        
        # Remove break annotations
        raw_intensity.annotations.delete(np.where(
            (raw_intensity.annotations.description == 'XStart') | 
            (raw_intensity.annotations.description == 'XStop') | 
            (raw_intensity.annotations.description == 'BAD boundary') | 
            (raw_intensity.annotations.description == 'EDGE boundary')
        )[0])
        
        # Convert again to optical density and correct
        raw_od = optical_density(raw_intensity)
        corrected_od = temporal_derivative_distribution_repair(raw_od)
        raw_haemo = mne.preprocessing.nirs.beer_lambert_law(corrected_od, ppf=6.1)
        
        # Pre-filtering plot
        fig = raw_haemo.plot_psd(average=True, show=False)
        fig.suptitle(f'Before filtering - {subj_id} - {session}', weight='bold', size='x-large')
        fig.subplots_adjust(top=0.88)
        
        
        # Filtering
        raw_haemo = raw_haemo.filter(None, 0.4, filter_length=87, h_trans_bandwidth=0.2, fir_window='hamming', fir_design='firwin')
        raw_haemo = raw_haemo.filter(0.05, None, l_trans_bandwidth=0.02)
        
        # Post-filtering plot
        fig = raw_haemo.plot_psd(average=True, show=False)
        fig.suptitle(f'After filtering - {subj_id} - {session}', weight='bold', size='x-large')
        fig.subplots_adjust(top=0.88)
        plt.savefig(os.path.join(save_path, f'PSD_after_{subj_id}_{session}.png'))
        plt.close(fig)
        
        # Identify bad channels
        bad_channels = list(compress(raw_od.ch_names, sci <= 0.5))
        num_bads = len(bad_channels)
        print(f'{subj_id} - {session}: {num_bads} bad channels')
        
        # Save bad channel info
        bad_channels_df = bad_channels_df._append({'Participant': subj_id, 'Session': session, 'Num_Bads': num_bads, 'Bad_Channels': bad_channels}, ignore_index=True)
        
        # Scalp Coupling Index windowed plot
        _, scores, times = scalp_coupling_index_windowed(raw_od, time_window=10)
        plot_timechannel_quality_metric(raw_od, scores, times, threshold=0.5, title=f'Scalp Coupling Index Quality Evaluation - {subj_id} - {session}')
        plt.savefig(os.path.join(save_path, f'SCI_windowed_{subj_id}_{session}.png'))
        plt.close()

# Save DataFrame as CSV
bad_channels_df.to_csv('bad_channels_summary.csv', index=False)
