r"""
This script generates a .png on the overall count of unique authors posting about Pepsi and their mean NRCL_freq_anger per month with marking the analysed event month of April 2017. 
Furthermore, the script generates the a .csv file for the respective unique author count and men NRCL_freq_anger data. 
In addition, the script computes the total amount of authors for the relevant event month, and the relative change in authors commenting, and the relative change in anger-related words overall and per state.

Input:
- CSV files in the input folder, named like:
    - p_{topic}_sentiment_YYYY-MM.csv
        (e.g., p_Pepsi_sentiment_2017-04)
    etc.
  Each file contains:
    - author, subreddit, body, keyword, total_keyword_num, total_word_count, state, NRCL_count_*,  NRCL_prop_*, NRCL_freq_* 

Columns in p_{topic}_sentiment_YYYY-MM.csv:
- author: Reddit username
- subreddit: Subreddit where the post/comment was published
- body: Text content of the post or comment
- total_keyword_num: Total number of keyword matches in the text
- total_word_count: Total number of words in body (i.e., comment)
- state: U.S. state associated with the author
- NRCL_count_*: counts of words in a comment associated with a specific emotion or sentiment (i.e., positive or negative) based on NCRLex (National Research Council Emotion Lexicon)
- NRCL_prop_*: proportion of words in a comment associated with a specific emotion or sentiment (i.e., positive or negative) in comparison to the total count of emotional words ranging form 0 to 1 based on NCRLex (National Research Council Emotion Lexicon)
- NRCL_freq_*: proportion of words in a comment associated with a specific emotion or sentiment (i.e., positive or negative) in comparison to the total number of words ranging form 0 to 1 based on NCRLex (National Research Council Emotion Lexicon)

Output:
- .png files saved in the output folder:
    - Pepsi_combined_timelines.png
 - .csv files saved in the output folder:
    - Pepsi_monthly_unique_authors.csv
    - Pepsi_monthly_mean_NRCL_freq_anger_with_CI.csv
    

Example usage:
python e_v1_01_pepsi_author_count_and_anger.py `
--input_folder "C:/.../Pepsi_sentiment" `
--output_folder "C:/.../Pepsi_timetrend" `
--topic {topic}

"""


import os
import re
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from datetime import datetime
import argparse

plt.rcParams.update({
    'axes.titlesize': 14,
    'axes.labelsize': 16,
    'xtick.labelsize': 14,
    'ytick.labelsize': 14,
    'legend.fontsize': 16
}) 

def generate_timelines(input_folder, output_folder, topic="Pepsi"):
    # Match files for the specified topic from 2014 onward
    pattern = re.compile(rf"^p_{re.escape(topic)}_sentiment_(\d{{4}})-(\d{{2}})\.csv$")
    csv_files = []

    for f in os.listdir(input_folder):
        match = pattern.match(f)
        if match:
            year, month = int(match.group(1)), int(match.group(2))
            file_date = datetime(year, month, 1)
            if file_date >= datetime(2014, 1, 1):
                csv_files.append(os.path.join(input_folder, f))

    if not csv_files:
        print(f"No matching files found for topic '{topic}' in {input_folder}")
        return

    # Load and concatenate data
    df_list = []
    for file in sorted(csv_files):
        df_tmp = pd.read_csv(file)
        date_str = os.path.basename(file).split('_')[-1].replace('.csv', '')
        df_tmp['time'] = pd.to_datetime(date_str)
        df_list.append(df_tmp)

    df = pd.concat(df_list, ignore_index=True)

    # Basic cleaning
    df = df[df['state'].notna()]
    df = df[df['author'].notna()]
    df['state'] = df['state'].str.upper()

    # Ensure time is month-based
    df['month'] = df['time'].dt.to_period('M').dt.to_timestamp()

    # Create output folder
    os.makedirs(output_folder, exist_ok=True)

    ### ===== TIMELINE 1: Unique authors per month ===== ###
    author_counts = df.groupby('month')['author'].nunique().reset_index()
    author_counts.rename(columns={'author': 'unique_authors'}, inplace=True)
    author_counts.to_csv(os.path.join(output_folder, f"{topic}_monthly_unique_authors.csv"), index=False)

    ### ===== TIMELINE 2: Mean NRCL_freq_anger per author per month ===== ###
    if "NRCL_freq_anger" not in df.columns:
        print("Metric 'NRCL_freq_anger' not found in the data.")
        return

    anger_df = df.dropna(subset=["NRCL_freq_anger"]).copy()

    # Step 1: Get author-level average per month, preserving state
    author_monthly = (
        anger_df.groupby(['month', 'author', 'state'])['NRCL_freq_anger']
        .mean()
        .reset_index(name='author_avg')
    )

    # Step 2: Monthly mean of author-level averages + 95% CI
    summary_stats = (
        author_monthly.groupby('month')['author_avg']
        .agg(['mean', 'count', 'std'])
        .reset_index()
    )
    summary_stats['sem'] = summary_stats['std'] / summary_stats['count']**0.5
    ci = 1.96  # 95% confidence
    summary_stats['ci_lower'] = summary_stats['mean'] - ci * summary_stats['sem']
    summary_stats['ci_upper'] = summary_stats['mean'] + ci * summary_stats['sem']
    summary_stats.to_csv(os.path.join(output_folder, f"{topic}_monthly_mean_NRCL_freq_anger_with_CI.csv"), index=False)

    ### ===== PLOT BOTH TIMELINES TOGETHER ===== ###
    event_date = pd.Timestamp('2017-04-01')

    fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(14, 10), sharex=True)

    # Plot Timeline 1: Unique Authors
    sns.lineplot(data=author_counts, x='month', y='unique_authors', ax=axes[0])
    axes[0].axvline(event_date, color='gray', linestyle='--', label='Apr 2017 (Event)')
    #axes[0].set_title(f'{topic}: Monthly Unique Authors')
    axes[0].set_ylabel("Number of unique monthly authors")
    axes[0].set_ylim(1000, 10000)  
    axes[0].legend()
    axes[0].grid(True)

    # Plot Timeline 2: NRCL_freq_anger + CI
    sns.lineplot(data=summary_stats, x='month', y='mean', label='% anger-related words', ax=axes[1])
    axes[1].fill_between(summary_stats['month'], summary_stats['ci_lower'], summary_stats['ci_upper'],
                         alpha=0.3, label='95% CI')
    axes[1].axvline(event_date, color='gray', linestyle='--', label='Apr 2017 (Event)')
    #axes[1].set_title(f'{topic}: Monthly mean NRCL_freq_anger (Author-level Avg + 95% CI)')
    axes[1].set_xlabel("year")
    axes[1].set_ylabel("share of anger-related words")
    axes[1].set_ylim(0.0325, 0.0525)  # Set y-axis limits

    # Format y-axis ticks: 4 decimals, no leading 0
    axes[1].yaxis.set_major_formatter(
        mticker.FuncFormatter(lambda x, _: f"{x:.4f}".lstrip("0") if x < 1 else f"{x:.4f}")
    )

    axes[1].legend()
    axes[1].grid(True)

    for ax in axes:
        ax.tick_params(axis='x', rotation=0)

    plt.tight_layout()
    combined_plot_path = os.path.join(output_folder, f"{topic}_combined_timelines.png")
    plt.savefig(combined_plot_path)
    plt.close()
    print(f"Combined timeline saved: {combined_plot_path}")

    # === PRINT APRIL 2017 AND PRE-EVENT AVERAGE STATS ===
    april_date = pd.Timestamp("2017-04-01")
    pre_event_end = pd.Timestamp("2017-03-31")
    pre_event_start = pd.Timestamp("2014-01-01")

    # April 2017 mean
    april_mean = summary_stats.loc[summary_stats['month'] == april_date, 'mean']
    
    # Pre-event period mean: Jan 2014 - Mar 2017
    pre_event_mask = (summary_stats['month'] >= pre_event_start) & (summary_stats['month'] <= pre_event_end)
    pre_event_mean = summary_stats.loc[pre_event_mask, 'mean'].mean()

    if not april_mean.empty:
        april_val = april_mean.values[0]
        diff = april_val - pre_event_mean
        percent_increase = (diff / pre_event_mean) * 100

        print(f"\n--- Event Analysis ---")
        print(f"Mean NRCL_freq_anger in April 2017:      {april_val:.6f}")
        print(f"Mean NRCL_freq_anger Jan 2014–Mar 2017:  {pre_event_mean:.6f}")
        print(f"Absolute increase:                        {diff:.6f}")
        print(f"Percentual increase:                      {percent_increase:.2f}%")
    else:
        print("April 2017 data not found in summary_stats.")

    # === PERCENTUAL INCREASE BY STATE ===
    print("\n--- Percentual Increase by State (April 2017 vs Jan 2014–Mar 2017) ---")

    # Merge original df to get states back into author_monthly
    author_monthly_with_state = author_monthly.copy()

    # Compute mean anger per state per month
    state_monthly_avg = (
        author_monthly_with_state
        .groupby(['state', 'month'])['author_avg']
        .mean()
        .reset_index()
        .rename(columns={'author_avg': 'state_monthly_mean'})
    )

    results = []

    for state in sorted(state_monthly_avg['state'].dropna().unique()):
        state_data = state_monthly_avg[state_monthly_avg['state'] == state]

        # April 2017 value
        april_val_series = state_data[state_data['month'] == april_date]['state_monthly_mean']
        if april_val_series.empty:
            continue  # Skip states without April 2017 data
        april_val = april_val_series.values[0]

        # Pre-event mean (Jan 2014 - Mar 2017)
        pre_event_vals = state_data[
            (state_data['month'] >= pre_event_start) & (state_data['month'] <= pre_event_end)
        ]['state_monthly_mean']
        if pre_event_vals.empty:
            continue  # Skip states with no pre-event data
        pre_event_mean_val = pre_event_vals.mean()

        diff = april_val - pre_event_mean_val
        pct_increase = (diff / pre_event_mean_val) * 100

        results.append((state, april_val, pre_event_mean_val, diff, pct_increase))

    # Print sorted by percentual increase descending
    results_sorted = sorted(results, key=lambda x: x[4], reverse=True)

    print(f"{'State':<6} {'Apr 2017':>10} {'Pre-Event':>10} {'Abs Δ':>10} {'% Increase':>12}")
    for state, apr, pre, d, pct in results_sorted:
        print(f"{state:<6} {apr:>10.4f} {pre:>10.4f} {d:>10.4f} {pct:>11.2f}%")


    # === OVERALL AUTHOR & COMMENT COUNTS (Jan 2014 to Dec 2023) ===
    overall_start = pd.Timestamp("2014-01-01")
    overall_end = pd.Timestamp("2023-12-31")

    full_period_df = df[(df['month'] >= overall_start) & (df['month'] <= overall_end)]

    unique_authors_overall = full_period_df['author'].nunique()
    total_comments_overall = full_period_df.shape[0]

    print("\n--- Overall Summary: Jan 2014 to Dec 2023 ---")
    print(f"Total unique authors: {unique_authors_overall:,}")
    print(f"Total number of comments/posts: {total_comments_overall:,}")

    # Unique authors for March and April 2017
    march_date = pd.Timestamp("2017-03-01")
    april_date = pd.Timestamp("2017-04-01")

    march_authors_series = author_counts.loc[author_counts['month'] == march_date, 'unique_authors']
    april_authors_series = author_counts.loc[author_counts['month'] == april_date, 'unique_authors']

    if not march_authors_series.empty and not april_authors_series.empty:
        march_authors = march_authors_series.values[0]
        april_authors = april_authors_series.values[0]
        diff_authors = april_authors - march_authors
        pct_increase_authors = (diff_authors / march_authors) * 100 if march_authors > 0 else float('nan')

        print(f"\n--- Unique Authors Event Analysis ---")
        print(f"Unique authors in March 2017: {march_authors:,}")
        print(f"Unique authors in April 2017: {april_authors:,}")
        print(f"Absolute increase: {diff_authors:,}")
        print(f"Percentual increase: {pct_increase_authors:.2f}%")
    else:
        print("Data for March or April 2017 unique authors not found.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate timelines of unique authors and NRCL_freq_anger.")
    parser.add_argument("--input_folder", required=True, help="Folder with monthly sentiment CSVs.")
    parser.add_argument("--output_folder", required=True, help="Folder to save outputs.")
    parser.add_argument("--topic", default="Pepsi", help="Topic name in CSV filenames (e.g., 'Pepsi').")
    args = parser.parse_args()

    generate_timelines(args.input_folder, args.output_folder, args.topic)