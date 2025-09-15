"""
This script generates scatter plots to visualize the correlation between U.S. state resident population census or estimates
and Reddit user/post counts over a specified time period.

Input:
- A base input folder containing:
    - p_state_counts_cumulative.csv
    - o_census.csv, mapping U.S. state abbreviations to Resident Population Census (2010, 2020) or Estimates (2011-2019, 2021-2023)
      from https://web.archive.org/web/20210426210008/https://www.census.gov/data/tables/2020/dec/2020-apportionment-data.html and
      https://web.archive.org/web/20210426202643/https://www.census.gov/data/tables/2010/dec/2010-apportionment-data.html and
      https://www.census.gov/data/tables/time-series/demo/popest/2020s-state-total.html and
      https://www.census.gov/data/tables/time-series/demo/popest/2010s-state-total.html.

Output:

- A PNG scatter plot saved in the output folder:
    p_scatter_plot.png

The scatter plots include:
- Regression lines
- Pearson correlation coefficients (r and R^2)
- Annotated state names

- A PNG figure with 28 subplots saved in the output folder:
    p_all_years_scatter_plots.png

The 28-subplot figure includes:
- 4 rows × 7 columns:
    - Row 1: 2010–2016 User Count
    - Row 2: 2010–2016 Post Count
    - Row 3: 2017–2023 User Count
    - Row 4: 2017–2023 Post Count
- Each subplot:
    - Scatter plot with regression line
    - Pearson correlation (r and R^2)
    - Annotated state names

Example usage:
python v_01b_vali_pop_visual.py --input_folder /path/to/input --output_folder /path/to/output

"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
from scipy.stats import pearsonr
import argparse


def generate_scatter_plots(input_folder,output_folder):
    os.makedirs(output_folder, exist_ok=True)
    fig, axs = plt.subplots(4, 7, figsize=(28, 16))
    plt.rcParams.update({'font.family': 'Times New Roman', 'font.size': 10})

    years_top = list(range(2010, 2017))
    years_bottom = list(range(2017, 2024))
    all_years = years_top + years_bottom

    x_min = np.inf
    x_max = -np.inf
    y_user_min = np.inf
    y_user_max = -np.inf
    y_post_min = np.inf
    y_post_max = -np.inf

    merged_data_dict = {}
    census_file_path = os.path.join(input_folder, 'o_census.csv')
    pop_df_full = pd.read_csv(census_file_path, encoding='utf-8')
    reddit_file = f"p_state_counts_cumulative.csv"
    reddit_file_path = os.path.join(input_folder, reddit_file)
    df = pd.read_csv(reddit_file_path)
    for idx, year in enumerate(all_years):
        combined = df[['state', f'{year}_user', f'{year}_post']].copy()
        combined.rename(columns={f'{year}_user': 'User Count', f'{year}_post': 'Post Count'}, inplace=True)

        pop_df = pop_df_full[['state', str(year)]].rename(columns={str(year): 'Population'})
        pop_df['Population Log'] = np.log(pop_df['Population'])

        combined['User Count Log'] = np.log(combined['User Count'].replace(0, np.nan)).fillna(0)
        combined['Post Count Log'] = np.log(combined['Post Count'].replace(0, np.nan)).fillna(0)

        # 以 state 连接
        merged_data = combined.merge(pop_df, on='state', how='inner')
        merged_data_dict[year] = merged_data

        x_min = min(x_min, merged_data['Population Log'].min())
        x_max = max(x_max, merged_data['Population Log'].max())
        y_user_min = min(y_user_min, merged_data['User Count Log'].min())
        y_user_max = max(y_user_max, merged_data['User Count Log'].max())
        y_post_min = min(y_post_min, merged_data['Post Count Log'].min())
        y_post_max = max(y_post_max, merged_data['Post Count Log'].max())

    # Add margins
    x_margin = (x_max - x_min) * 0.05
    y_user_margin = (y_user_max - y_user_min) * 0.05
    y_post_margin = (y_post_max - y_post_min) * 0.05
    x_min -= x_margin
    x_max += x_margin
    y_user_min -= y_user_margin
    y_user_max += y_user_margin
    y_post_min -= y_post_margin
    y_post_max += y_post_margin

    for idx, year in enumerate(all_years):
        if idx < 7:
            row_user = 0
            row_post = 1
            col = idx
        else:
            row_user = 2
            row_post = 3
            col = idx - 7

        merged_data = merged_data_dict[year]

        for (row, label, color) in [
            (row_user, 'User Count', 'red'),
            (row_post, 'Post Count', 'blue')
        ]:
            ax = axs[row, col]
            log_col = f'{label} Log'
            pearson_corr, p_value = pearsonr(merged_data['Population Log'], merged_data[log_col])
            m, b = np.polyfit(merged_data['Population Log'], merged_data[log_col], 1)
            ax.scatter(merged_data['Population Log'], merged_data[log_col], color=color, alpha=0.6, s=10)
            ax.plot(merged_data['Population Log'], m * merged_data['Population Log'] + b, color=color, linestyle='--', linewidth=1)

            ax.set_xlim(x_min, x_max)
            if label == 'User Count':
                ax.set_ylim(y_user_min, y_user_max)
            else:
                ax.set_ylim(y_post_min, y_post_max)

            if label == 'User Count':
                ax.set_title(f'{year}', fontsize=15)
            else:
                ax.set_title("")

            if col == 0:
                ax.set_ylabel(f'Log {label}', fontsize=10)
            else:
                ax.set_ylabel("")

            if row == 1 or row == 3:
                ax.set_xlabel('Log Population', fontsize=10)
            else:
                ax.set_xlabel("")

            for _, row_data in merged_data.iterrows():
                ax.text(row_data['Population Log'], row_data[log_col], row_data['state'], fontsize=5)

            ax.text(
                0.05, 0.95,
                f'y = {m:.2f}x + {b:.2f}\n$R^2$ = {pearson_corr ** 2:.2f}\nr = {pearson_corr:.2f}\np = {p_value:.2e}',
                transform=ax.transAxes,
                fontsize=7,
                verticalalignment='top',
                bbox=dict(facecolor='white', alpha=0.6, edgecolor='black')
            )

  #  fig.suptitle(
  #      "Correlation between GeoReddit User/Post Count and U.S. State Resident Population (2010–2023)",
  #     fontsize=25,
   #     fontweight='bold'
   # )
    plt.tight_layout()
    big_fig_path = os.path.join(output_folder, "p_all_years_scatter_plots.png")
    plt.savefig(big_fig_path, dpi=600)
    print(f"All-year scatter plots saved to {big_fig_path}")

    last_year = all_years[-1]
    last = merged_data_dict[last_year]

    # Create 2 subplots
    fig, axs = plt.subplots(1, 2, figsize=(12, 6))
    plt.rcParams.update({'font.family': 'Times New Roman', 'font.size': 14})

    for i, (label, color) in enumerate([('User Count', 'red'), ('Post Count', 'blue')]):
        log_col = f'{label} Log'
        pearson_corr, p_value = pearsonr(last['Population Log'], last[log_col])
        m, b = np.polyfit(last['Population Log'], last[log_col], 1)

        axs[i].scatter(last['Population Log'], last[log_col], color=color, alpha=0.6)
        axs[i].plot(last['Population Log'], m * last['Population Log'] + b, color=color, linestyle='--',
                    linewidth=2)

        axs[i].set_title(f'2020 Census - {label}',fontsize=12)

        axs[i].set_xlabel('Log of Population', fontsize=14)
        axs[i].set_ylabel(f'Log of {label}', fontsize=14)
        axs[i].grid(True, linestyle='--', alpha=0.6)

        for _, row in last.iterrows():
            axs[i].text(row['Population Log'], row[log_col], row['state'], fontsize=8, ha='left')

        axs[i].text(
            0.05, 0.95,
            f'y = {m:.2f}x + {b:.2f}\n$R^2$ = {pearson_corr ** 2:.2f}\nr = {pearson_corr:.2f}\np = {p_value:.2e}',
            transform=axs[i].transAxes,
            fontsize=12,
            verticalalignment='top',
            bbox=dict(facecolor='white', alpha=0.7, edgecolor='black')
        )

    plt.tight_layout()
    output_file_name = f"p_scatter_plot.png"
    output_file_path = os.path.join(output_folder, output_file_name)
    plt.savefig(output_file_path, format='png', dpi=600, bbox_inches='tight')
    plt.close()
    print(f"Scatter plot saved to {output_file_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate scatter plots showing correlation between US state population and Reddit user/post counts."
    )
    parser.add_argument("--input_folder", required=True, help="Folder containing input files")
    parser.add_argument("--output_folder", required=True, help="Folder to save output files")

    args = parser.parse_args()

    generate_scatter_plots(
        input_folder=args.input_folder,
        output_folder=args.output_folder,
    )