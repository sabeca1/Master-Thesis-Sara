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

def signal_quality_rois(datapath, save_path, subj_id, session):
    """Processes signal quality and updates ROIs after removing bad channels.

    Parameters
    ----------
    datapath : str
        Path to the subject's data folder.
    save_path : str
        Path to save processed results.
    subj_id : str
        Subject ID.
    session : str
        Session number.

    Returns
    -------
    raw_haemo_cropped : mne.io.Raw
        The cropped haemodynamic data after preprocessing.
    glm_ready_rois : dict
        Dictionary with updated ROIs after removing bad channels.
    """
    import mne  # Import MNE here to avoid unnecessary dependency issues

    # Ensure the datapath includes '/nirs'
    if not datapath.endswith("nirs"):
        datapath = os.path.join(datapath, "nirs")
    
    # Ensure save directory exists
    os.makedirs(save_path, exist_ok=True)

    # Process signal quality and get data
    cropped_intensity, bad_channels_df, raw_haemo_cropped, raw_haemo_original, cropped_corrected_od, original_corrected_od = signal_quality_1_subject(
        datapath, save_path, subj_id, session
    )

    # Extract bad channels
    if "Bad_Channels" in bad_channels_df.columns:
        bad_channels = bad_channels_df["Bad_Channels"].values
    else:
        bad_channels = []

    # Define ROIs (before removing bad channels)
    frontal = [[12, 1], [2, 1], [1, 1], [3, 1], [3, 2]]
    auditory_left = [[4, 2], [4, 3], [5, 2], [5, 3], [5, 4], [5, 5]]
    auditory_right = [[11, 12], [11, 11], [10, 12], [10, 10], [10, 11], [10, 9]]
    visual = [[7, 6], [6, 6], [6, 8], [9, 8], [7, 8], [8, 8], [7, 7], [8, 7]]

    # Function to filter out bad channels
    def filter_bad_channels(roi, bad_channels):
        return [pair for pair in roi if f"S{pair[0]}_D{pair[1]}" not in bad_channels]

    # Update ROIs to exclude bad channels
    updated_rois = {
        "Frontal": filter_bad_channels(frontal, bad_channels),
        "Auditory_Left": filter_bad_channels(auditory_left, bad_channels),
        "Auditory_Right": filter_bad_channels(auditory_right, bad_channels),
        "Visual": filter_bad_channels(visual, bad_channels),
    }

    # Convert ROI definitions to indices
    glm_ready_rois = {
        name: picks_pair_to_idx(raw_haemo_cropped, channels)
        for name, channels in updated_rois.items()
    }

    print(f"Updated ROIs for Subject {subj_id}, Session {session}:")
    for roi, channels in updated_rois.items():
        print(f"{roi}: {channels}")

    return raw_haemo_cropped, glm_ready_rois

def scatter_mean(glm_est, save_path,subj_id, session, conditions=(), exclude_no_interest=True, no_interest=None):
    """Scatter plot and box plot of the GLM results with mean HbO and HbR across all channels.

    Parameters
    ----------
    glm_est : object
        GLM estimation result.
    save_path : str
        Path to save the generated plot.
    conditions : list, optional
        List of condition names to plot. By default, it plots all regressors of interest.
    exclude_no_interest : bool, optional
        Exclude regressors of no interest from the figure.
    no_interest : list, optional
        List of regressors that are of no interest. If none are specified, defaults to ["drift", "constant", "short", "Short"].

    Returns
    -------
    None
        Displays and saves the figure.
    """

    if no_interest is None:
        no_interest = ["drift", "constant", "short", "Short"]

    # Convert GLM results to DataFrame
    df = glm_est.to_dataframe()

    x_column = "Condition"
    y_column = "theta"

    if "ContrastType" in df.columns:
        x_column = "ch_name"
        y_column = "effect"
        if len(conditions) == 0:
            conditions = ["t", "f"]
        df = df.query("ContrastType in @conditions")
    else:
        if len(conditions) == 0:
            conditions = glm_est.design.columns
        df = df.query("Condition in @conditions")

    if exclude_no_interest:
        for no_i in no_interest:
            df = df[~df[x_column].astype(str).str.startswith(no_i)]

    # Compute mean and std deviation for each condition and chromophore
    grouped_df = df.groupby(["Condition", "Chroma"])["theta"]
    mean_values = grouped_df.mean().reset_index()
    std_values = grouped_df.std().reset_index()

    # Print mean theta values for user inspection
    print("\n### Mean Theta Values (µM) per Condition ###")
    for _, row in mean_values.iterrows():
        print(f"{row['Condition']} ({row['Chroma']}): {row['theta'] * 1e6:.4f} µM")

    # Prepare data for plotting
    df_hbo = mean_values.query('Chroma == "hbo"')
    df_hbr = mean_values.query('Chroma == "hbr"')

    std_hbo = std_values.query('Chroma == "hbo"')["theta"]
    std_hbr = std_values.query('Chroma == "hbr"')["theta"]

    conditions_unique = mean_values["Condition"].unique()
    x_positions = np.arange(len(conditions_unique))

    # Create figure with scatter plot and box plot
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(10, 6))

    # Scatter plot for mean theta values
    ax.scatter(x_positions, df_hbo["theta"] * 1e6, c="r", label="Oxyhaemoglobin (HbO)")
    ax.scatter(x_positions, df_hbr["theta"] * 1e6, c="b", label="Deoxyhaemoglobin (HbR)")

    # Box plot for theta values
    ax.errorbar(x_positions, df_hbo["theta"] * 1e6, yerr=std_hbo * 1e6, fmt='o', color='red', alpha=0.5)
    ax.errorbar(x_positions, df_hbr["theta"] * 1e6, yerr=std_hbr * 1e6, fmt='o', color='blue', alpha=0.5)

    # Formatting
    ax.set_xticks(x_positions)
    ax.set_xticklabels(conditions_unique)
    ax.set_xlabel("Condition")
    ax.set_ylabel("Mean Theta Value (µM)")
    ax.set_ylim(-5, 5)  # Adjust Y-axis range
    ax.legend()
    ax.hlines([0.0], 0, len(conditions_unique) - 1, colors="gray", linestyles="dashed")
    ax.set_title(f"GLM Theta Values - Subject {subj_id}, Session {session}")  
    if len(conditions_unique) > 8:
        plt.xticks(rotation=45, ha="right")

    # Save figure
    os.makedirs(os.path.dirname(save_path), exist_ok=True)  # Ensure directory exists
    plt.savefig(save_path)
    print(f"\nPlot saved to: {save_path}")

    # Show plot
    plt.show()
 
 

def scatter_mean_rois(avg_theta_df, save_path, subj_id, session, roi, df_big, conditions=("Control", "Noise", "Speech")):
    """
    Computes and stores mean and std deviation of HbO and HbR values for each condition and ROI.
    Creates a figure with 3 subplots (1 row × 3 columns), one for each condition.

    Parameters
    ----------
    avg_theta_df : DataFrame
        DataFrame containing mean theta values and standard deviations for each condition and chroma.
    save_path : str
        Path to save the generated plots and CSV.
    subj_id : str
        Subject identifier for the plot title.
    session : str
        Session identifier for the plot title.
    roi : str
        Region of interest (ROI) for the plot title.
    df_big : DataFrame
        A DataFrame that accumulates data from multiple ROIs.
    conditions : tuple, optional
        Conditions to plot (default is ("Control", "Speech", "Noise")).

    Returns
    -------
    DataFrame
        Updated `df_big` containing all ROI data.
    """

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))  # 1 row, 3 columns

    for i, condition in enumerate(conditions):
        ax = axes[i]  # Select subplot for this condition
        
        # Filter for the current condition
        df_condition = avg_theta_df[avg_theta_df["Condition"] == condition]

        if df_condition.empty:
            print(f"No data found for {condition} in ROI {roi}. Skipping...")
            ax.set_title(f"No data for {condition}")
            ax.axis("off")  # Hide the subplot if no data
            continue

        # Compute mean and std deviation
        mean_values = df_condition.groupby("Chroma")["theta"].mean()
        std_values = df_condition.groupby("Chroma")["theta"].std()

        # Extract values for HbO and HbR
        hbo_value = mean_values.get("hbo", np.nan) * 1e6
        hbr_value = mean_values.get("hbr", np.nan) * 1e6
        hbo_std = std_values.get("hbo", np.nan) * 1e6
        hbr_std = std_values.get("hbr", np.nan) * 1e6

        # Append new data to df_big
        new_data = pd.DataFrame([[subj_id, session, roi, condition, hbo_value, hbo_std, hbr_value, hbr_std]], 
                                 columns=df_big.columns)
        df_big = pd.concat([df_big, new_data], ignore_index=True)

        # Scatter plot for mean values
        x_positions = [0, 1]  # 0 for HbO, 1 for HbR
        ax.errorbar(x_positions[0], hbo_value, yerr=hbo_std, fmt='o', color='red', label="HbO")
        ax.errorbar(x_positions[1], hbr_value, yerr=hbr_std, fmt='o', color='blue', label="HbR")

        ax.set_xticks(x_positions)
        ax.set_xticklabels(["HbO", "HbR"])
        ax.set_title(f"{condition}")
        ax.set_ylim(-0.2, 0.2)  # Adjust Y-axis range
        ax.grid(True)
        ax.set_ylabel("Mean Theta Value (µM)")
        ax.legend()

    # Adjust layout and save figure
    fig.suptitle(f"GLM Theta Values - ROI: {roi}", fontsize=14, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.96])  # Leave space for the main title

    save_file = os.path.join(save_path, f"theta_values_{roi}_{subj_id}.png")
    os.makedirs(os.path.dirname(save_file), exist_ok=True)  # Ensure directory exists
    plt.savefig(save_file)
    print(f"Saved: {save_file}")

    plt.close(fig)  # Close figure to free memory

    return df_big  # Return updated DataFrame


        

def glm_single_subject(datapath_template, save_path_template, subj_id):
    sessions = ["ses-01", "ses-02"]
    
    for session in sessions:
        datapath = datapath_template.format(subj_id=subj_id, session=session)
        save_path = save_path_template.format(subj_id=subj_id, session=session)
        
        # Process signal quality
        cropped_intensity, bad_channels_df, raw_haemo_cropped, raw_haemo_original, cropped_corrected_od, original_corrected_od = signal_quality_1_subject(datapath, save_path, subj_id, session)
        
        # Create design matrix
        design_matrix = make_first_level_design_matrix(raw_haemo_original,
                                                       drift_model='cosine',
                                                       hrf_model='spm',
                                                       stim_dur=5)
        
        # Run GLM
        glm_est = run_glm(raw_haemo_original, design_matrix)
        
        # Convert GLM results to DataFrame
        df_theta = glm_est.to_dataframe()
        
        # Compute averages for each condition and chromophore
        avg_theta = {
        "Control_HbO": df_theta[(df_theta["Condition"] == "Control") & (df_theta["Chroma"] == "hbo") ]["theta"].mean(),
        "Control_HbR": df_theta[(df_theta["Condition"] == "Control") & (df_theta["Chroma"] == "hbr") ]["theta"].mean(),
        "Noise_HbO": df_theta[(df_theta["Condition"] == "Noise") & (df_theta["Chroma"] == "hbo") ]["theta"].mean(),
        "Noise_HbR": df_theta[(df_theta["Condition"] == "Noise") & (df_theta["Chroma"] == "hbr")]["theta"].mean(),
        "Speech_HbO": df_theta[(df_theta["Condition"] == "Speech") & (df_theta["Chroma"] == "hbo")]["theta"].mean(),
        "Speech_HbR": df_theta[(df_theta["Condition"] == "Speech") & (df_theta["Chroma"] == "hbr")]["theta"].mean(),
        }

        print(f"Corrected Average Theta Values for {subj_id}, {session}:")
        for condition, value in avg_theta.items():
            print(f"{condition} = {value}")
        
        """# Create figure with two subplots
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
         # Scatter plot of averaged theta values
        conditions = ["Control", "Noise", "Speech"]
        x_positions = np.arange(len(conditions))  # X locations for each group
        
        hbo_values = [avg_theta["Control_HbO"], avg_theta["Noise_HbO"], avg_theta["Speech_HbO"]]
        hbr_values = [avg_theta["Control_HbR"], avg_theta["Noise_HbR"], avg_theta["Speech_HbR"]]
        
        ax = axes[0]
        ax.scatter(x_positions, hbo_values, color='red', label='HbO')
        ax.scatter(x_positions, hbr_values, color='blue', label='HbR')
        
        ax.set_xticks(x_positions)
        ax.set_xticklabels(conditions)
        ax.set_xlabel("Condition")
        ax.set_ylabel("Theta Value")
        ax.set_title(f"{subj_id}, {session}")
        ax.legend()
        
        # GLM scatter plot
        axes[1].set_title(f"GLM Scatter for {subj_id}, {session}") """
        
        scatter_mean(glm_est, 
             save_path=f"{save_path}/theta_scatter_plot.png", 
             subj_id=subj_id, 
             session=session, 
             conditions=["Control", "Noise", "Speech"])


    """
    Runs GLM analysis for different regions of interest (ROIs) and generates scatter plots.

    Parameters
    ----------
    datapath_template : str
        Path template for subject data files.
    save_path_template : str
        Path template for saving output.
    subj_id : str
        Subject identifier.

    Returns
    -------
    None
        Saves and displays the generated plots.
    """

    sessions = ["ses-01", "ses-02"]
    
    # Define electrode pairs for each region of interest (ROI)
    roi_electrodes = {
        "Visual": ["P3-P4", "O1-O2"],  # Example pairs for visual cortex
        "Auditory": ["T3-T4", "C3-C4"],  # Example pairs for auditory cortex
        "Frontal": ["F3-F4", "Fp1-Fp2"]  # Example pairs for frontal lobe
    }
    
    for session in sessions:
        datapath = datapath_template.format(subj_id=subj_id, session=session)
        save_path = save_path_template.format(subj_id=subj_id, session=session)

        # Ensure save directory exists
        os.makedirs(save_path, exist_ok=True)

        # Process signal quality
        cropped_intensity, bad_channels_df, raw_haemo_cropped, raw_haemo_original, cropped_corrected_od, original_corrected_od = signal_quality_1_subject(datapath, save_path, subj_id, session)

        # Create design matrix
        design_matrix = make_first_level_design_matrix(
            raw_haemo_original, drift_model='cosine', hrf_model='spm', stim_dur=5
        )

        # Initialize dictionary for GLM results per ROI
        glm_results = {}

        for roi, channels in roi_electrodes.items():
            # Select only relevant channels for this ROI
            roi_data = raw_haemo_original.copy().pick_channels(channels)

            # Run GLM for this ROI
            glm_results[roi] = run_glm(roi_data, design_matrix)

        # Call scatter_mean_rois function to generate ROI-wise plots
        scatter_mean_rois(
            glm_dict=glm_results,
            save_path=f"{save_path}/roi_theta_scatter_plots",
            subj_id=subj_id,
            session=session
        )

    """
    Runs GLM analysis for different regions of interest (ROIs) and generates scatter plots.

    Parameters
    ----------
    datapath_template : str
        Path template for subject data files.
    save_path_template : str
        Path template for saving output.
    subj_id : str
        Subject identifier.

    Returns
    -------
    None
        Saves and displays the generated plots.
    """

    sessions = ["ses-01", "ses-02"]

    for session in sessions:
        # Define file paths
        datapath = datapath_template.format(subj_id=subj_id, session=session)
        save_path = save_path_template.format(subj_id=subj_id, session=session)

        # Ensure save directory exists
        os.makedirs(save_path, exist_ok=True)

        # Process signal quality and get updated ROIs
        raw_haemo_cropped, glm_ready_rois = signal_quality_rois(datapath, save_path, subj_id, session)

        # Create design matrix
        design_matrix = make_first_level_design_matrix(
            raw_haemo_cropped, drift_model='cosine', hrf_model='spm', stim_dur=5
        )

        # Initialize dictionary for GLM results per ROI
        glm_results = {}

        for roi_name, roi_channels in glm_ready_rois.items():
            # Run GLM only for channels in this ROI
            glm_results[roi_name] = run_glm(raw_haemo_cropped, design_matrix)

        # Generate and save ROI-wise scatter plots
        scatter_mean_rois(
            glm_dict=glm_results,
            save_path=save_path,
            subj_id=subj_id,
            session=session
        )      
    
    """
    Runs GLM analysis for different regions of interest (ROIs) and generates scatter plots.

    Parameters
    ----------
    datapath_template : str
        Path template for subject data files.
    save_path_template : str
        Path template for saving output.
    subj_id : str
        Subject identifier.

    Returns
    -------
    None
        Saves and displays the generated plots.
    """

    sessions = ["ses-01", "ses-02"]

    # Define electrode pairs for each ROI
    roi_electrode_pairs = {
        "Frontal": [[12, 1], [2, 1], [1, 1], [3, 1], [3, 2]],
        "Auditory_Left": [[4, 2], [4, 3], [5, 2], [5, 3], [5, 4], [5, 5]],
        "Auditory_Right": [[11, 12], [11, 11], [10, 12], [10, 10], [10, 11], [10, 9]],
        "Visual": [[7, 6], [6, 6], [6, 8], [9, 8], [7, 8], [8, 8], [7, 7], [8, 7]]
    }

    for session in sessions:
        # Define file paths
        datapath = datapath_template.format(subj_id=subj_id, session=session)
        save_path = save_path_template.format(subj_id=subj_id, session=session)

        # Ensure save directory exists
        os.makedirs(save_path, exist_ok=True)

        # Process signal quality and get updated ROIs
        cropped_intensity, bad_channels_df, X, raw_haemo_original, cropped_corrected_od, original_corrected_od  = signal_quality_1_subject(datapath, save_path, subj_id, session)
        raw_haemo_cropped = raw_haemo_original.copy()
        # Create design matrix
        design_matrix = make_first_level_design_matrix(
            raw_haemo_cropped, drift_model='cosine', hrf_model='spm', stim_dur=5
        )

        # Run GLM on full dataset
        glm_est = run_glm(raw_haemo_cropped, design_matrix)

        # Convert GLM results to DataFrame
        df_theta = glm_est.to_dataframe()

        # Filter out bad channels
        valid_channels = [ch for ch in raw_haemo_cropped.ch_names if ch not in bad_channels_df["Bad_Channels"].values]

        # Create dictionary for ROI-based theta values
        roi_glm_results = {}

        for roi_name, electrode_pairs in roi_electrode_pairs.items():
            # Convert electrode pairs to channel indices
            roi_channels = picks_pair_to_idx(raw_haemo_cropped, electrode_pairs) #
            roi_channels = [ch for ch in roi_channels if ch in valid_channels]  # Keep only valid channels
            
            if roi_channels:
                # Extract GLM results for this ROI
                roi_results = glm_est.to_dataframe_region_of_interest(
                    groups={roi_name: roi_channels}, conditions=["Control", "Noise", "Speech"]
                )
                
                # Compute mean theta values per condition
                roi_mean_theta = roi_results.groupby(["ROI", "Condition", "Chroma"])["theta"].mean().reset_index()

                roi_glm_results[roi_name] = roi_mean_theta

        # Generate and save scatter plots for each ROI
        scatter_mean_rois(
            roi_glm_results,
            save_path=save_path,
            subj_id=subj_id,
            session=session
        )

def glm_single_subject_rois(datapath_template, save_path_template, subj_id):
    """
    Runs GLM analysis for different regions of interest (ROIs) and generates scatter plots.
    """
    sessions = ["ses-01", "ses-02"]

    """ roi_electrode_pairs = {
        "Frontal": [[12, 1], [2, 1], [1, 1], [3, 1], [3, 2]],
        "Auditory_Left": [[4, 2], [4, 3], [5, 2], [5, 3], [5, 4], [5, 5]],
        "Auditory_Right": [[11, 12], [11, 11], [10, 12], [10, 10], [10, 11], [10, 9]],
        "Visual": [[7, 6], [6, 6], [6, 8], [9, 8], [8, 8], [7, 7], [8, 7]]
    } """
    
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

        cropped_intensity, bad_channels_df, X, raw_haemo_original, cropped_corrected_od, original_corrected_od = signal_quality_1_subject(datapath, save_path, subj_id, session)
        raw_haemo_cropped = raw_haemo_original.copy()
        
        design_matrix = make_first_level_design_matrix(
            raw_haemo_cropped, drift_model='cosine', hrf_model='spm', stim_dur=5
        )

        glm_est = run_glm(raw_haemo_cropped, design_matrix)
        df_theta = glm_est.to_dataframe()
        ########
        # add a new column to the dataframe that contains the same values as the channel name until the first space
        df_theta['ch_name_clean'] = df_theta['ch_name'].str.split(' ').str[0]
        
        # add a new empty column to the dataframe that will contain the ROI name
        df_theta['ROI'] = ''
        
        
        for i in range(len(df_theta)):
            for roi, channels in roi_electrode_pairs.items():
                if df_theta['ch_name_clean'][i] in channels:
                    df_theta['ROI'][i] = roi
               
        # create a new dataframe that contains only the rows with a ROI name
        df_theta_roi = df_theta[df_theta['ROI'] != '']
        # create a csv with df_theta_roi
        #df_theta_roi.to_csv('df_theta_roi.csv')
        
        # creeate a dictionary with the ROI names as keys and the corresponding dataframes as values
        roi_glm_results = {}
        for roi in df_theta_roi['ROI'].unique():
            subset= df_theta_roi[df_theta_roi['ROI'] == roi]
            roi_glm_results[roi] = subset
            
        df_big = pd.DataFrame(columns=["Subject", "Session", "ROI", "Condition", "HbO_Mean", "HbO_Std", "HbR_Mean", "HbR_Std"])

        for roi, avg_theta_df in roi_glm_results.items():
            print(f"Processing ROI: {roi}")
            df_big = scatter_mean_rois(avg_theta_df, save_path, subj_id, session, roi, df_big)
        
        # Save the final CSV file (after all ROIs have been processed)
        csv_file = os.path.join(save_path, f"theta_values_{subj_id}_{session}.csv")
        df_big.to_csv(csv_file, index=False)
        print(f"Final DataFrame saved to {csv_file}")

       

    return roi_glm_results

# Example usage:
#glm_single_subject('subjects/{subj_id}/{session}/nirs', 'results/{subj_id}/{session}/', 'sub-19')
# glm_single_subject_rois(datapath_template="subjects/sub-{subj_id}/{session}/",save_path_template="results/sub-{subj_id}/{session}/",subj_id="19")
""" datapath_template = "C:/Users/sarab/Desktop/THESIS/Tutorial MNE/subjects/sub-{subj_id}/{session}/nirs"
save_path_template = "C:/Users/sarab/Desktop/THESIS/Tutorial MNE/results/sub-{subj_id}/{session}"
subj_id = "19" """


# glm_single_subject_rois(datapath_template="subjects/sub-{subj_id}/{session}/nirs",save_path_template="results/sub-{subj_id}/{session}/",subj_id="19")

""" for subj_id in range(1, 22):  # Loop from 1 to 21
    subj_id_str = f"{subj_id:02d}"  # Format as two-digit string (e.g., "01", "02")
    glm_single_subject_rois(
        datapath_template=f"subjects/sub-{subj_id_str}/{{session}}/nirs",
        save_path_template=f"results/sub-{subj_id_str}/{{session}}/",
        subj_id=subj_id_str
    ) """

# add which optodes are used per region per patient and region
# more detail info about specific optodes. --> how many times is each optode activate.
# optode based dataframe, do a statistical test for each optode.
# what activation means? what activation patterns I'm expecting?
# Lateralization differences (higher activation on left side)
# PCA or ICA to reduce the number of channels-->Yamada paper






