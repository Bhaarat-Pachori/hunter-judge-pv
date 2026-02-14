import pandas as pd
from statsmodels.stats.contingency_tables import mcnemar

# File paths
file_path_stage1 = '../hunter-judge-pv/hunter_results.csv'
file_path_stage2 = '../hunter-judge-pv/final_cadec_results_with_cot.csv'

# Load the datasets
df_stage1 = pd.read_csv(file_path_stage1)
df_stage2 = pd.read_csv(file_path_stage2)

# Merge the dataframes
merged_df = pd.merge(df_stage1, df_stage2, on='text', how='inner')

# Create boolean columns for correctness
merged_df['hunter_is_correct'] = merged_df['hunter_pred'] == merged_df['ground_truth']
merged_df['hybrid_is_correct'] = merged_df['final_pred'] == merged_df['label']

# Construct the contingency table
contingency_table = {
    'without_duplicates': {
        'B': merged_df[
            (merged_df['hunter_is_correct'] == True) & (merged_df['hybrid_is_correct'] == False)
        ].drop_duplicates(subset=['text']).shape[0],
        'C': merged_df[
            (merged_df['hunter_is_correct'] == False) & (merged_df['hybrid_is_correct'] == True)
        ].drop_duplicates(subset=['text']).shape[0]
    },
    'with_duplicates': {
        'B': merged_df[
            (merged_df['hunter_is_correct'] == True) & (merged_df['hybrid_is_correct'] == False)
        ].shape[0],
        'C': merged_df[
            (merged_df['hunter_is_correct'] == False) & (merged_df['hybrid_is_correct'] == True)
        ].shape[0]
    }
}

# Perform McNemar's test and print results for both with and without duplicates
for name, table in contingency_table.items():
    print(f"McNemar's Test Results ({name}):")
    print(f"Contingency Table:")
    print(f"Cell B (Hunter Right, Hybrid Wrong): {table['B']}")
    print(f"Cell C (Hunter Wrong, Hybrid Right): {table['C']}")

    # Perform McNemar's test
    # result = mcnemar([[table['B'], table['C']]], correction=True)
    result = mcnemar([[0, table['B']], [table['C'], 0]], correction=True)

    # Print the results
    print(f"McNememar's Test p-value: {result.pvalue:.3f}")
    print("------------------------------------")
