"""
This script aggregates Reddit user activity over a specified time period to compute ratio1 and ratio2 metrics describing users' geographic posting behavior.

Input:
- A single input folder containing:
    - p_postnum_YYYY-MM.csv files (total posts/comments per user, from c_03_post_num.py)
    - p_YYYY-MM.csv files (per-user, per-state monthly activity, from c_01_locate_users.py)
    - o_2005-06to2023-12_ratios.csv (fallback ratios for users with no data in the selected period)

Output:
- One CSV file saved in the output folder:
    - p_YYYY-MMtoYYYY-MM_ratios_postnum.csv (ratios merged with total post counts)

Columns in p_YYYY-MMtoYYYY-MM_ratios_postnum.csv:
- author: Username of the Reddit user
- all: Total number of posts and comments during the specified period
- ratio1: The ratio of the number of months a user was active in their top state to the total number of months they
          were active in any state.
- substate1: State with the highest number of months of activity
- ratio2: The ratio of the number of posts/comments a user made in their top state to the total number of posts
          they made in any state.
- substate2: State with the highest number of posts

Command line usage example:

python c_04_period_ratio_postnum.py --input_folder "C:/data/reddit_inputs" --output_folder "C:/data/reddit_outputs" --start_month 202001 --end_month 202002
"""

import pandas as pd
import glob
import os
from tqdm import tqdm
import argparse
import numpy as np

tqdm.pandas()

# Merge and sum all p_postnum_ files to get total posts/comments per user over the period
def merge_and_sum_all(input_folder, start_month, end_month):
    start_month_int, end_month_int = int(start_month), int(end_month)
    file_paths = glob.glob(os.path.join(input_folder, "p_postnum_*.csv"))
    filtered_files = [
        f for f in file_paths
        if start_month_int <= int(os.path.basename(f).split('_')[2].split('.')[0].replace('-', '')) <= end_month_int
    ]
    # Read and concatenate
    combined_df = pd.concat(
        [pd.read_csv(file, usecols=['author', 'all']) for file in filtered_files],
        ignore_index=True
    )
    # Group by author to sum total posts/comments
    result_df = combined_df.groupby('author', as_index=False)['all'].sum()
    return result_df

# Load and concatenate p_YYYY-MM.csv files to get per-user per-state records
def load_and_concatenate_csv_files(input_folder, start_month, end_month):
    start_month_int, end_month_int = int(start_month), int(end_month)
    file_paths = glob.glob(os.path.join(input_folder, "p_*.csv"))
    # Exclude p_postnum_ files
    filtered_files = [
        f for f in file_paths
        if "p_postnum_" not in os.path.basename(f)
        and start_month_int <= int(os.path.basename(f).split('_')[1].split('.')[0].replace('-', '')) <= end_month_int
    ]
    combined_df = pd.concat([pd.read_csv(file) for file in filtered_files], ignore_index=True)
    return combined_df

# This function calculates the ratio of activity concentration in the user's most active state.
# It first sorts the values descending to ensure the top state comes first.
# It finds the maximum value (either the highest count of active months or the highest number of posts).
# It collects all substates that have this maximum value (to handle ties).
# If there is only one state, the ratio is defined as infinity.
# Otherwise, the ratio is computed as (largest value) divided by (sum of all other values).
def calculate_ratio_and_max_substate(values, substates):
    df = pd.DataFrame({'value': values.values, 'state': substates.values}).copy()
    if df.empty:
        return np.nan, ""

    df = df.sort_values('value', ascending=False, kind='mergesort')
    top_val = df['value'].iloc[0]
    top_states = df.loc[df['value'] == top_val, 'state'].tolist()
    denom = df['value'].iloc[1:].sum() if len(df) > 1 else 0.0
    ratio = float('inf') if (len(df) == 1 or denom <= 0) else float(top_val) / float(denom)
    top_states = ", ".join(sorted(set(top_states)))
    return ratio, top_states

# This function takes the grouped data for a single user and computes:
# - ratio1: The ratio of the number of months active in the top state compared to all other states.
# - ratio2: The ratio of total posts in the top state compared to all other states.
# It returns these ratios along with the names of the top substates.
def calculate_ratios(row):
    ratio1, substate1 = calculate_ratio_and_max_substate(row['count'], row['substate'])
    ratio2, substate2 = calculate_ratio_and_max_substate(row['num_sum'], row['substate'])
    return pd.Series([ratio1, substate1, ratio2, substate2],
                     index=['ratio1', 'substate1', 'ratio2', 'substate2'])

# This function processes the full DataFrame of per-user, per-state records.
# It groups the data by (author, substate) to calculate:
# - count: how many months the user was active in each state.
# - num_sum: the total number of posts in each state.
# Then it groups by author and applies the ratio calculation to each user.
# If the input DataFrame is empty, it returns an empty DataFrame with the expected columns.
def process_data(df):
    if df.empty:
        return pd.DataFrame(columns=['author', 'ratio1', 'substate1', 'ratio2', 'substate2'])
    grouped = df.groupby(['author', 'substate']).agg(
        count=('substate', 'size'),
        num_sum=('num', 'sum')
    ).reset_index()
    return grouped.groupby('author').progress_apply(calculate_ratios).reset_index()


# Merge ratios with total posts and fallback ratios
def merge_and_fill_unique(result_df, combined_df_1, fallback_file_path):
    fallback_df = pd.read_csv(fallback_file_path, usecols=['author', 'ratio1', 'substate1', 'ratio2', 'substate2'])
    merged_df = pd.merge(
        pd.merge(combined_df_1, result_df, on='author', how='left'),
        fallback_df,
        on='author',
        how='left',
        suffixes=('', '_fallback')
    )
    # Fill missing values with fallback
    for col in ['ratio1', 'substate1', 'ratio2', 'substate2']:
        merged_df[col] = merged_df[col].combine_first(merged_df[f"{col}_fallback"])
    merged_df.drop(columns=[f"{col}_fallback" for col in ['ratio1', 'substate1', 'ratio2', 'substate2']], inplace=True)
    return merged_df

def main():
    parser = argparse.ArgumentParser(description="Compute ratio1 and ratio2 for Reddit user activity.")
    parser.add_argument("--input_folder", required=True, help="Folder containing p_postnum_*, p_*.csv, and o_2005-06to2023-12_ratios.csv")
    parser.add_argument("--output_folder", required=True, help="Output folder")
    parser.add_argument("--start_month", required=True, help="Start month in format YYYYMM")
    parser.add_argument("--end_month", required=True, help="End month in format YYYYMM")
    args = parser.parse_args()

    os.makedirs(args.output_folder, exist_ok=True)

    fallback_file = os.path.join(args.input_folder, "o_2005-06to2023-12_ratios.csv")
    if not os.path.exists(fallback_file):
        raise FileNotFoundError(f"Fallback file not found: {fallback_file}")

    # Load total posts/comments
    combined_df_1 = merge_and_sum_all(args.input_folder, args.start_month, args.end_month)
    # Load per-user per-state records
    combined_df_2 = load_and_concatenate_csv_files(args.input_folder, args.start_month, args.end_month)
    # Compute ratios
    result_df = process_data(combined_df_2)
    # Merge with totals and fallback
    merged_df = merge_and_fill_unique(result_df, combined_df_1, fallback_file)

    prefix = f"p_{args.start_month[:4]}-{args.start_month[4:6]}to{args.end_month[:4]}-{args.end_month[4:6]}"

    # where we get `2005-06to2023-12_ratios.csv`
    #  output_file_ratios = os.path.join(args.output_folder, f"o_{prefix}_ratios.csv")
    #  result_df.to_csv(output_file_ratios, index=False)
    #  print(f"Saved ratio summary to {output_file_ratios}")

    output_file_ratios_postnum = os.path.join(args.output_folder, f"{prefix}_ratios_postnum.csv")
    merged_df.to_csv(output_file_ratios_postnum, index=False)
    print(f"Saved ratios merged with total posts to {output_file_ratios_postnum}")



if __name__ == "__main__":
    main()

