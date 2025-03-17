import os
import numpy as np
import pandas as pd
import mne
import matplotlib.pyplot as plt
from itertools import compress
from mne.preprocessing.nirs import optical_density, temporal_derivative_distribution_repair
from mne_nirs.preprocessing import scalp_coupling_index_windowed, peak_power
from mne_nirs.visualisation import plot_timechannel_quality_metric
import mne_nirs
from mne_nirs.channels import get_short_channels, picks_pair_to_idx
from mne_nirs.experimental_design import make_first_level_design_matrix, create_boxcar
from mne_nirs.statistics import run_glm
from nilearn.plotting import plot_design_matrix


def signal_quality_1_subject(datapath, save_path, subj_id, session):
    print(f"Processing subject {subj_id}, session {session}...")
    os.makedirs(save_path, exist_ok=True)
    
    # Load data
    print("Loading NIRS data...")
    raw_intensity_original = mne.io.read_raw_nirx(datapath).load_data()
    
    # Rename annotations
    print("Renaming annotations...")
    raw_intensity_original.annotations.rename({"1.0": "Control", "2.0": "Noise", "3.0": "Speech", '4.0': 'XStop', '5.0': 'XStart'})
    
    # Get event timings
    print("Extracting event timings...")
    Breaks, _ = mne.events_from_annotations(raw_intensity_original, {'XStop': 4, 'XStart': 5})
    AllEvents, _ = mne.events_from_annotations(raw_intensity_original)
    Breaks = Breaks[:, 0] / raw_intensity_original.info['sfreq']
    LastEvent = AllEvents[-1, 0] / raw_intensity_original.info['sfreq']
    
    if len(Breaks) % 2 == 0:
        raise ValueError("Breaks array should have an odd number of elements.")
    
    original_duration = raw_intensity_original.times[-1] - raw_intensity_original.times[0]
    print(f"Original duration: {original_duration:.2f} seconds")
    
    # Cropping dataset
    print("Cropping the dataset...")
    cropped_intensity = raw_intensity_original.copy().crop(Breaks[0], Breaks[1])
    for j in range(2, len(Breaks) - 1, 2):
        block = raw_intensity_original.copy().crop(Breaks[j], Breaks[j + 1])
        cropped_intensity.append(block)
    cropped_intensity.append(raw_intensity_original.copy().crop(Breaks[-1], LastEvent + 15.25))
    
    cropped_duration = cropped_intensity.times[-1] - cropped_intensity.times[0]
    print(f"Cropped duration: {cropped_duration:.2f} seconds")
    
    if cropped_duration >= original_duration:
        print(f"WARNING: Cropping did not reduce duration for {subj_id} - {session}!")
    
    raw_intensity_cropped = cropped_intensity.copy()

    

    
    # Remove break annotations
    print("Removing break annotations for the orginal raw...")
    raw_intensity_original.annotations.delete(np.where(
        (raw_intensity_original.annotations.description == 'XStart') | 
        (raw_intensity_original.annotations.description == 'XStop') | 
        (raw_intensity_original.annotations.description == 'BAD boundary') | 
        (raw_intensity_original.annotations.description == 'EDGE boundary')
    )[0])
    
    print("Removing break annotations for the cropped raw...")
    raw_intensity_cropped.annotations.delete(np.where(
        (raw_intensity_cropped.annotations.description == 'XStart') | 
        (raw_intensity_cropped.annotations.description == 'XStop') | 
        (raw_intensity_cropped.annotations.description == 'BAD boundary') | 
        (raw_intensity_cropped.annotations.description == 'EDGE boundary')
    )[0])
    
    # Optical density and correction
    print("Applying optical density and correction...")
    cropped_od = optical_density(raw_intensity_cropped)
    original_od= optical_density(raw_intensity_original)
    cropped_corrected_od = temporal_derivative_distribution_repair(cropped_od)
    original_corrected_od = temporal_derivative_distribution_repair(original_od)
    
    # Compute Scalp Coupling Index
    print("Computing Scalp Coupling Index (SCI)...")
    sci = mne.preprocessing.nirs.scalp_coupling_index(cropped_corrected_od)
    
    # Identify bad channels
    print("Identifying bad channels...")
    bad_channels = list(compress(cropped_corrected_od.ch_names, sci <= 0.7))
    num_bads = len(bad_channels)
    print(f"{subj_id} - {session}: {num_bads} bad channels detected: {bad_channels}")
    
    # Save bad channel info
    bad_channels_df = pd.DataFrame([{
        'Participant': subj_id,
        'Session': session,
        'Num_Bads': num_bads,
        'Bad_Channels': bad_channels
    }])
    
    cropped_corrected_od.info['bads'] = bad_channels
    original_corrected_od.info['bads'] = bad_channels
    
    # Remove bad channels
    print("Removing bad channels...")
    raw_cropped_cleaned = cropped_corrected_od.copy().drop_channels(bad_channels)
    raw_original_cleaned = original_corrected_od.copy().drop_channels(bad_channels)
    
    # Check if bad channels were successfully removed
    remaining_channels = set(raw_cropped_cleaned.ch_names)
    if any(bc in remaining_channels for bc in bad_channels):
        print("ERROR: Some bad channels were not removed correctly!")
    else:
        print("Bad channels successfully removed.")
    
    # Convert to hemoglobin
    print("Converting to hemoglobin concentration...")
    raw_haemo_cropped = mne.preprocessing.nirs.beer_lambert_law(raw_cropped_cleaned, ppf=6.1)
    raw_haemo_original = mne.preprocessing.nirs.beer_lambert_law(raw_original_cleaned, ppf=6.1)
    
    # Pre-filtering plot
    print("Plotting pre-filtering power spectral density...")
    raw_haemo_cropped.plot_psd(average=True, show=False, n_fft=4096)
    raw_haemo_cropped.plot_psd(average=True, show=False, fmin=0, fmax=0.4, n_fft=4096)
    
    # Filtering
    print("Applying filtering...")
    # raw_haemo = raw_haemo.filter(None, 0.35, filter_length=87, h_trans_bandwidth=0.3, fir_window='hamming', fir_design='firwin')
    # raw_haemo = raw_haemo.filter(0.025, None, l_trans_bandwidth=0.005)
    raw_haemo_cropped = raw_haemo_cropped.filter(0.025, 0.35, h_trans_bandwidth=0.3, l_trans_bandwidth=0.005)
    raw_haemo_original = raw_haemo_original.filter(0.025, 0.35, h_trans_bandwidth=0.3, l_trans_bandwidth=0.005)

    # Post-filtering plot
    print("Plotting post-filtering power spectral density...")
    raw_haemo_cropped.plot_psd(average=True, show=False, n_fft=4096)
    raw_haemo_cropped.plot_psd(average=True, show=False, fmin=0, fmax=0.4, n_fft=4096)
    
    # Quality Metrics
    print("Computing quality metrics...")
    _, scores, times = scalp_coupling_index_windowed(raw_cropped_cleaned, time_window=10)
    plot_timechannel_quality_metric(raw_cropped_cleaned, scores, times, threshold=0.7, title=f'SCI Quality - {subj_id} - {session}')
    
    raw_cropped_cleaned, scores, times = peak_power(raw_cropped_cleaned, time_window=10)
    plot_timechannel_quality_metric(raw_cropped_cleaned, scores, times, threshold=0.1, title="Peak Power Quality Evaluation")
    # close all figures
    plt.close('all')
    
    print(f"Processing complete for {subj_id} - {session}")
    return cropped_intensity, bad_channels_df, raw_haemo_cropped, raw_haemo_original, cropped_corrected_od, original_corrected_od 



def scatter_mean_optodes(avg_theta_df, save_path, subj_id, session, df_big, conditions=("Control", "Noise", "Speech")):
    """
    Computes and stores mean and std deviation of HbO and HbR values for each condition and optode.
    Creates a figure with multiple subplots, one per optode.
    """
    optodes = avg_theta_df['ch_name_clean'].unique()
    num_optodes = len(optodes)
    fig, axes = plt.subplots(num_optodes, 3, figsize=(15, 5 * num_optodes))
    
    for opt_idx, optode in enumerate(optodes):
        for cond_idx, condition in enumerate(conditions):
            ax = axes[opt_idx, cond_idx] if num_optodes > 1 else axes[cond_idx]
            df_condition = avg_theta_df[(avg_theta_df["Condition"] == condition) & (avg_theta_df["ch_name_clean"] == optode)]
            
            if df_condition.empty:
                ax.set_title(f"No data for {condition}")
                ax.axis("off")
                continue
            
            mean_values = df_condition.groupby("Chroma")["theta"].mean()
            std_values = df_condition.groupby("Chroma")["theta"].std()
            
            hbo_value = mean_values.get("hbo", np.nan) * 1e6
            hbr_value = mean_values.get("hbr", np.nan) * 1e6
            hbo_std = std_values.get("hbo", np.nan) * 1e6
            hbr_std = std_values.get("hbr", np.nan) * 1e6
            
            new_data = pd.DataFrame([[subj_id, session, optode, df_condition["ROI"].iloc[0], condition, hbo_value, hbo_std, hbr_value, hbr_std]], 
                                     columns=df_big.columns)
            df_big = pd.concat([df_big, new_data], ignore_index=True)
            
            x_positions = [0, 1]
            ax.errorbar(x_positions[0], hbo_value, yerr=hbo_std, fmt='o', color='red', label="HbO")
            ax.errorbar(x_positions[1], hbr_value, yerr=hbr_std, fmt='o', color='blue', label="HbR")
            
            ax.set_xticks(x_positions)
            ax.set_xticklabels(["HbO", "HbR"])
            ax.set_title(f"{optode} - {condition}")
            ax.set_ylim(-0.2, 0.2)
            ax.grid(True)
            ax.legend()
    
    fig.suptitle(f"GLM Theta Values - Optodes", fontsize=14, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    save_file = os.path.join(save_path, f"theta_values_optodes_{subj_id}.png")
    os.makedirs(os.path.dirname(save_file), exist_ok=True)
    plt.savefig(save_file)
    plt.close(fig)
    
    return df_big


def glm_single_subject_optodes(datapath_template, save_path_template, subj_id):
    sessions = ["ses-01", "ses-02"]
    roi_electrode_pairs = {
        "Frontal": ['S12_D1', 'S2_D1', 'S1_D1', 'S3_D1', 'S3_D2'],
        "Auditory_Left": ['S4_D2', 'S4_D3', 'S5_D2', 'S5_D3', 'S5_D4', 'S5_D5'],
        "Auditory_Right": ['S11_D12', 'S11_D11', 'S10_D12', 'S10_D10', 'S10_D11', 'S10_D9'],
        "Visual": ['S7_D6', 'S6_D6', 'S6_D8', 'S9_D8', 'S8_D8', 'S7_D7', 'S8_D7', 'S7_D8']
    }
    
    for session in sessions:
        datapath = datapath_template.format(subj_id=subj_id, session=session)
        save_path = save_path_template.format(subj_id=subj_id, session=session)
        os.makedirs(save_path, exist_ok=True)

        cropped_intensity, bad_channels_df, raw_haemo_cropped, raw_haemo_original, cropped_corrected_od, original_corrected_od = signal_quality_1_subject(datapath, save_path, subj_id, session)
        
        design_matrix = make_first_level_design_matrix(raw_haemo_cropped, drift_model='cosine', hrf_model='spm', stim_dur=5)
        glm_est = run_glm(raw_haemo_cropped, design_matrix)
        df_theta = glm_est.to_dataframe()
        df_theta['ch_name_clean'] = df_theta['ch_name'].str.split(' ').str[0]
        df_theta['ROI'] = df_theta['ch_name_clean'].map({ch: roi for roi, channels in roi_electrode_pairs.items() for ch in channels}).fillna('Unknown')
        
        df_big = pd.DataFrame(columns=["Subject", "Session", "Optode", "ROI", "Condition", "HbO_Mean", "HbO_Std", "HbR_Mean", "HbR_Std"])
        df_big = scatter_mean_optodes(df_theta, save_path, subj_id, session, df_big)
        
        csv_file = os.path.join(save_path, f"theta_values_optodes_{subj_id}_{session}.csv")
        df_big.to_csv(csv_file, index=False)
        print(f"Final DataFrame saved to {csv_file}")
    
    return df_big

for subj_id in range(1, 2):
    subj_id_str = f"{subj_id:02d}"
    glm_single_subject_optodes(
        datapath_template=f"subjects/sub-{subj_id_str}/{{session}}/nirs",
        save_path_template=f"results/sub-{subj_id_str}/{{session}}/",
        subj_id=subj_id_str
    )
