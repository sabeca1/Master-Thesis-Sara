import pandas as pd

def analyze_significant_optodes(results_file):
    """
    Analyzes the t-test results CSV file to count and list significant optodes
    for Noise vs Control and Speech vs Control conditions.
    """
    # Load the results file
    results_df = pd.read_csv(results_file)

    # Filter significant optodes for Noise vs Control
    significant_noise = results_df[results_df["Significance Noise vs Control"] == "Significant"]["Optode"].tolist()
    num_significant_noise = len(significant_noise)

    # Filter significant optodes for Speech vs Control
    significant_speech = results_df[results_df["Significance Speech vs Control"] == "Significant"]["Optode"].tolist()
    num_significant_speech = len(significant_speech)

    # Print results
    print(f"\n Number of significant optodes (Noise vs Control): {num_significant_noise}")
    print(f"Significant optodes for Noise vs Control: {significant_noise}")

    print(f"\n Number of significant optodes (Speech vs Control): {num_significant_speech}")
    print(f"Significant optodes for Speech vs Control: {significant_speech}")

    return significant_noise, significant_speech

# Run the function with the t-test results file
#
num_part= 21
param= 'HbR'
if param == 'HbO':
    if num_part==10:
        results_file = ["t_test_results_10participants.csv", 't_test_results_10participants_ses-01.csv', 't_test_results_10participants_ses-02.csv']  # Update this if needed
    elif num_part==21:
        results_file = ["t_test_results_21participants.csv", 't_test_results_21participants_ses-01.csv', 't_test_results_21participants_ses-02.csv']  # Update this if needed
elif param == 'HbR':
    if num_part==10:
        results_file = ['t_test_results_HbR_10.csv', 't_test_results_HbR_10_ses-01.csv', 't_test_results_HbR_10_ses-02.csv']
    elif num_part==21:
        results_file = ['t_test_results_HbR_21.csv', 't_test_results_HbR_21_ses-01.csv', 't_test_results_HbR_21_ses-02.csv']
        
# Update this if needed
# access the first file
#results_file = results_file[0]
for i in range(3):
    if i==0:
        print(f"\nResults for {param} and {num_part} participants, both sessions:")
    elif i==1:
        print(f"\nResults for {param} and {num_part} participants for session 1:")
    else:
        print(f"\nResults for {param} and {num_part} participants for session 2:")
    
    significant_noise, significant_speech = analyze_significant_optodes(results_file[i])
