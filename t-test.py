import os
import numpy as np
import pandas as pd
import scipy.stats as stats

## PAIRED T-TEST

def perform_paired_t_tests(data):
    """
    Perform paired t-tests for HbO values across conditions (Noise vs Control, Speech vs Control),
    and add significance indicators if p-value < 0.05.
    """
    unique_optodes = data["Optode"].unique()
    results = []

    for optode in unique_optodes:
        df_optode = data[data["Optode"] == optode]
        
        noise = df_optode[df_optode["Condition"] == "Noise"]["HbR_Mean"].dropna()
        speech = df_optode[df_optode["Condition"] == "Speech"]["HbR_Mean"].dropna()
        control = df_optode[df_optode["Condition"] == "Control"]["HbR_Mean"].dropna()
        
        if len(noise) > 1 and len(control) > 1:
            t_stat_noise, p_value_noise = stats.ttest_rel(noise, control, nan_policy='omit')
            sig_noise = "Significant" if p_value_noise < 0.05 else "Not Significant"
        else:
            t_stat_noise, p_value_noise, sig_noise = np.nan, np.nan, "N/A"

        if len(speech) > 1 and len(control) > 1:
            t_stat_speech, p_value_speech = stats.ttest_rel(speech, control, nan_policy='omit')
            sig_speech = "Significant" if p_value_speech < 0.05 else "Not Significant"
        else:
            t_stat_speech, p_value_speech, sig_speech = np.nan, np.nan, "N/A"

        results.append([optode, t_stat_noise, p_value_noise, sig_noise, t_stat_speech, p_value_speech, sig_speech])

    results_df = pd.DataFrame(results, columns=[
        "Optode", 
        "T-Stat Noise vs Control", "P-Value Noise vs Control", "Significance Noise vs Control",
        "T-Stat Speech vs Control", "P-Value Speech vs Control", "Significance Speech vs Control"
    ])
    return results_df

def process_all_subjects(subj_ids, datapath_template, save_path_template):
    """
    Process multiple subjects, extract HbO data, and perform t-tests across conditions.
    """
    all_data = pd.DataFrame(columns=["Subject", "Session", "Optode", "ROI", "Condition", "HbR_Mean", "HbR_Std"])

    for subj_id in subj_ids:
        subj_id_str = f"{subj_id:02d}"
        for session in ["ses-01", "ses-02"]:
            datapath = datapath_template.format(subj_id=subj_id_str, session=session)
            save_path = save_path_template.format(subj_id=subj_id_str, session=session)
            os.makedirs(save_path, exist_ok=True)

           
            # Extract HbO values
            df_big = pd.read_csv(os.path.join(save_path, f"theta_values_optodes_{subj_id_str}_{session}.csv"))
            all_data = pd.concat([all_data, df_big], ignore_index=True)

    # Perform paired t-tests
    results_df = perform_paired_t_tests(all_data)

    # Save results
    results_file = "t_test_results_HbR_21.csv"
    results_df.to_csv(results_file, index=False)
    print(f"T-test results saved to {results_file}")

    return results_df



# Run for subjects 1-10
subj_ids = range(1, 22)
results = process_all_subjects(
    subj_ids,
    datapath_template="subjects/sub-{subj_id}/{session}/nirs",
    save_path_template="results/sub-{subj_id}/{session}/"
)
