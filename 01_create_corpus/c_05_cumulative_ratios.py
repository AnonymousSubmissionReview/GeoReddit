"""
This script generates cumulative distribution plots for two metrics (`ratio1` and `ratio2`)
that describe Reddit user posting behavior over a specified period.

Input:
- An input folder containing:
    - p_YYYY-MMtoYYYY-MM_ratios_postnum.csv (output from c_04)

Output:
- Two PNG files saved in the output folder:
    - p_YYYY-MMtoYYYY-MM_cumulative_ratios_User.png
    - p_YYYY-MMtoYYYY-MM_cumulative_ratios_Post.png

Each plot shows how the cumulative proportion of users or posts varies when applying different
thresholds to ratio1 or ratio2.

Command line example:
python c_05_cumulative_ratios.py --input_folder "C:/Users/u2288/Downloads/reddit_input" --output_folder "C:/Users/u2288/Desktop/reddit_output" --start_year 2005 --start_month 6 --end_year 2023 --end_month 12 --type User --cut_off_value 1.0

"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import argparse
from tqdm import tqdm

# Prepare sorted unique ratio values and their logarithms for plotting
def prepare_sorted_ratios(df, column):
    sorted_ratios_o = np.sort(df.loc[~np.isinf(df[column]), column].unique())
    sorted_ratios_log = np.log(sorted_ratios_o)
    return sorted_ratios_o, sorted_ratios_log

# Compute cumulative proportions (users or posts) above each ratio threshold
def compute_cumulative_counts(df, sorted_ratios_o, column, type_, total_users, total_posts):
    cumulative_counts = []
    for threshold in tqdm(sorted_ratios_o, desc=f"Processing cumulative {column} counts for {type_}"):
        if type_ == 'User':
            cumulative_counts.append(len(df[df[column] >= threshold]) / total_users)
        elif type_ == 'Post':
            cumulative_counts.append(df[df[column] >= threshold]['all'].sum() / total_posts)
    return cumulative_counts

# Plot a cumulative distribution curve and annotate thresholds and cut-offs
def plot_cumulative(sorted_ratios_log, sorted_ratios_o, cumulative_counts, column, type_, counts, df, cut_off_value=None, is_ratio2=False):
    plt.plot(sorted_ratios_log, cumulative_counts, marker='o', color='#5b8db8', alpha=0.5)

    plt.xlabel(r"log(Ratio Threshold)", fontsize=13)
    plt.ylabel(f"Cumulative {type_} Proportion", fontsize=13)
    plt.title(
        f"Cumulative {type_} Proportion Above log({column.capitalize()}) Threshold",
        fontsize=14, fontweight='bold'
    )
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.ylim([(int(counts * 10) / 10), 1])

    # Final threshold annotation
    plt.axvline(x=sorted_ratios_log[-1], color='black', linestyle='--', linewidth=1.5, label="Final Threshold")
    plt.text(
        sorted_ratios_log[-1], max(cumulative_counts) * 0.95,
        f'log({sorted_ratios_o[-1]:.0f}) = {sorted_ratios_log[-1]:.2f}',
        color='black', fontsize=12, ha='right'
    )
    plt.text(
        sorted_ratios_log[-1], cumulative_counts[-1],
        f'{cumulative_counts[-1]:.2f}',
        color='black', fontsize=12, ha='right', va='bottom'
    )
    plt.legend(loc='lower left')

    # Cut-off visualization for ratio2
    if is_ratio2 and cut_off_value is not None:
        cut_off_log = np.log(cut_off_value)
        if type_ == 'User':
            leave_proportion = len(df[df[column] > cut_off_value]) / len(df)
        elif type_ == 'Post':
            leave_proportion = df[df[column] > cut_off_value]['all'].sum() / df['all'].sum()
        plt.axvline(x=cut_off_log, color='red', linestyle='--', linewidth=1.5, label=f"Cut-off: > {cut_off_value:.2f}")
        plt.axhline(y=leave_proportion, color='red', linestyle='--', linewidth=1.5)
        plt.text(
            cut_off_log, leave_proportion,
            f'{leave_proportion:.2f}',
            color='red', fontsize=12, ha='right', va='bottom'
        )
        plt.legend(loc='lower left')

# Orchestrate loading data, computing cumulative curves, and saving plots
def ratio_threshold_cumulative(
    input_folder, output_folder,
    start_year, start_month, end_year, end_month,
    type_, cut_off_value
):
    file_name = f"p_{start_year}-{str(start_month).zfill(2)}to{end_year}-{str(end_month).zfill(2)}_ratios_postnum.csv"
    file_path = os.path.join(input_folder, file_name)
    df = pd.read_csv(file_path)

    total_posts = df['all'].sum()
    counts = df[np.isinf(df['ratio1'])]['all'].sum() / total_posts
    total_users = len(df)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 10))

    # Plot ratio1
    sorted_ratios_o, sorted_ratios_log = prepare_sorted_ratios(df, 'ratio1')
    cumulative_counts = compute_cumulative_counts(df, sorted_ratios_o, 'ratio1', type_, total_users, total_posts)
    plt.sca(ax1)
    plot_cumulative(sorted_ratios_log, sorted_ratios_o, cumulative_counts, 'ratio1', type_, counts, df)

    # Plot ratio2
    sorted_ratios_o2, sorted_ratios_log2 = prepare_sorted_ratios(df, 'ratio2')
    cumulative_counts2 = compute_cumulative_counts(df, sorted_ratios_o2, 'ratio2', type_, total_users, total_posts)
    plt.sca(ax2)
    plot_cumulative(
        sorted_ratios_log2, sorted_ratios_o2, cumulative_counts2,
        'ratio2', type_, counts, df,
        cut_off_value=cut_off_value, is_ratio2=True
    )

    plt.tight_layout()
    output_file_name = f"p_{start_year}-{str(start_month).zfill(2)}to{end_year}-{str(end_month).zfill(2)}_cumulative_ratios_{type_}.png"
    output_file_path = os.path.join(output_folder, output_file_name)
    plt.savefig(output_file_path, format='png', dpi=600, bbox_inches='tight')
    print(f"Plot saved to {output_file_path}")

    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate cumulative distribution plots for ratio1 and ratio2 in Reddit user activity."
    )
    parser.add_argument("--input_folder", required=True, help="Input folder containing the p_*_ratios_postnum.csv file")
    parser.add_argument("--output_folder", required=True, help="Output folder to save PNG plots")
    parser.add_argument("--start_year", type=int, required=True, help="Start year (e.g., 2005)")
    parser.add_argument("--start_month", type=int, required=True, help="Start month (e.g., 6)")
    parser.add_argument("--end_year", type=int, required=True, help="End year (e.g., 2023)")
    parser.add_argument("--end_month", type=int, required=True, help="End month (e.g., 12)")
    parser.add_argument("--type", choices=["User", "Post"], required=True, help="Type of cumulative plot: User or Post")
    parser.add_argument("--cut_off_value", type=float, default=1.0, help="Threshold value for ratio2 cut-off visualization")

    args = parser.parse_args()

    os.makedirs(args.output_folder, exist_ok=True)

    ratio_threshold_cumulative(
        input_folder=args.input_folder,
        output_folder=args.output_folder,
        start_year=args.start_year,
        start_month=args.start_month,
        end_year=args.end_year,
        end_month=args.end_month,
        type_=args.type,
        cut_off_value=args.cut_off_value
    )
