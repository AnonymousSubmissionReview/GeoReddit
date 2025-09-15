"""
This script visualizes the distribution of log-transformed `ratio2` values for GeoReddit users or posts over a specified period.

Input:
- An input folder containing:
    - p_YYYY-MMtoYYYY-MM_ratios_postnum.csv

Output:
- One PNG image saved in the output folder:
    - p_YYYY-MMtoYYYY-MM_ratio2_distribution.png

The output image shows side-by-side log-scale histograms for:
    - Post counts
    - User counts

Each plot includes:
    - Log-transformed `ratio2` distribution
    - A vertical cut-off line indicating the specified threshold
    - Annotated statistics summarizing total counts, excluded infinite ratios,
      and proportions retained after applying the cut-off.

Command line example:
python c_06_log_ratio2_distribution.py --input_folder "C:/Users/u2288/Downloads/period" --output_folder "C:/Users/u2288/Downloads/period" --start_year 2005 --start_month 6 --end_year 2023 --end_month 12 --bins 50 --cut_off_value 1.0

"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import argparse


# Function to annotate the plot with statistics and draw the cut-off line
def annotate_and_cutoff(
    ax, df, col, entity, bin_data,
    cut_off_value, cut_off_log,
    total_count, infinity_count,
    start_year, start_month, end_year, end_month,
    is_user=False
):
    # Calculate the proportion of data retained after excluding infinite ratios
    remaining_probability = 1 - (infinity_count / total_count)

    # Compute how many entries remain above the cut-off threshold
    if is_user:
        cut_off_count = len(df[df[col] > cut_off_value])
        lost_proportion = (total_count - cut_off_count) / total_count
    else:
        cut_off_count = df[df[col] > cut_off_value]['all'].sum()
        lost_proportion = (total_count - cut_off_count) / total_count

    # Prepare annotation text with summary statistics
    annotation_text = (
        f"Total geolocated {entity.lower()}: {total_count:,}\n"
        f"Displayed probability: {remaining_probability:.2%}\n"
        f"Unique-location {entity.lower()} (ratio = ∞) excluded: {infinity_count / total_count:.2%}\n"
        f"Cut-off {entity.lower()}: {cut_off_count:,} ({lost_proportion:.2%} lost)"
    )

    # Add the annotation box to the plot
    ax.annotate(
        annotation_text,
        xy=(0.99, 0.85), xycoords='axes fraction', ha='right', va='top',
        fontsize=10,
        bbox=dict(facecolor='white', edgecolor='gray', boxstyle='round,pad=0.5', alpha=0.7)
    )

    # Draw the cut-off line
    ax.axvline(
        x=cut_off_log,
        color='red',
        linestyle='--',
        linewidth=1.5,
        label=f"Cut-off: > {cut_off_value:.2f}"
    )

    # Set plot titles and labels
    ax.set_title(
        f"{entity} Count Distribution ({start_year}.{start_month} - {end_year}.{end_month})",
        fontsize=16
    )
    ax.set_xlabel("Log(Ratio2)", fontsize=12)
    ax.set_ylabel(f"{entity} Count", fontsize=12)
    ax.legend(loc='upper right')


# Function to generate the ratio2 distribution plots for posts and users
def combined_ratio_distribution(
    input_folder, output_folder,
    start_year, start_month,
    end_year, end_month,
    bins, cut_off_value
):
    # Build the input file path
    file_name = f"p_{start_year}-{str(start_month).zfill(2)}to{end_year}-{str(end_month).zfill(2)}_ratios_postnum.csv"
    file_path = os.path.join(input_folder, file_name)

    if not os.path.isfile(file_path):
        print(f"No data file found: {file_path}")
        return

    # Load data
    df = pd.read_csv(file_path)

    col = 'ratio2'

    # Compute log-transformed ratio2
    df[f'log_{col}'] = np.log(df[col].replace([float('inf')], np.nan).dropna())
    cut_off_log = np.log(cut_off_value)

    # Define histogram bin edges and centers
    bin_edges = np.linspace(df[f'log_{col}'].min(), df[f'log_{col}'].max(), bins + 1)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

    # Aggregate post counts per bin
    post_bin_sums = [
        df.loc[
            (df[f'log_{col}'] >= bin_edges[j]) & (df[f'log_{col}'] < bin_edges[j + 1]),
            'all'
        ].sum()
        for j in range(len(bin_edges) - 1)
    ]

    # Count number of users per bin
    user_counts, _ = np.histogram(df[f'log_{col}'], bins=bin_edges)

    # Create side-by-side plots
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Apply common formatting to both axes
    for ax in axes:
        ax.grid(visible=True, which='both', color='gray', linestyle='--', linewidth=0.5, alpha=0.6)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    # Plot post distribution
    total_post_count = df['all'].sum()
    infinity_post_count = df.loc[np.isinf(df[col]), 'all'].sum()
    axes[0].bar(
        bin_centers, post_bin_sums,
        width=np.diff(bin_edges),
        color='#5b8db8',
        edgecolor='black',
        alpha=0.6
    )
    axes[0].plot(bin_centers, post_bin_sums, color='black', linestyle='-', linewidth=1.5)
    annotate_and_cutoff(
        axes[0], df, col, "Post", post_bin_sums,
        cut_off_value, cut_off_log,
        total_post_count, infinity_post_count,
        start_year, start_month, end_year, end_month,
        is_user=False
    )

    # Plot user distribution
    total_user_count = len(df)
    infinity_user_count = np.isinf(df[col]).sum()
    axes[1].bar(
        bin_centers, user_counts,
        width=np.diff(bin_edges),
        color='#5b8db8',
        edgecolor='black',
        alpha=0.6
    )
    axes[1].plot(bin_centers, user_counts, color='black', linestyle='-', linewidth=1.5)
    annotate_and_cutoff(
        axes[1], df, col, "User", user_counts,
        cut_off_value, cut_off_log,
        total_user_count, infinity_user_count,
        start_year, start_month, end_year, end_month,
        is_user=True
    )

    # Finalize and save figure
    plt.tight_layout()
    save_name = f"p_{start_year}-{str(start_month).zfill(2)}to{end_year}-{str(end_month).zfill(2)}_ratio2_distribution.png"
    save_path = os.path.join(output_folder, save_name)
    plt.savefig(save_path, format='png', dpi=600, bbox_inches='tight')
    print(f"Plot saved to {save_path}")
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate ratio2 distribution plots for posts and users over a specified period."
    )
    parser.add_argument("--input_folder", required=True, help="Path to folder containing the ratio CSV file.")
    parser.add_argument("--output_folder", required=True, help="Path to folder to save output plots.")
    parser.add_argument("--start_year", type=int, required=True, help="Start year.")
    parser.add_argument("--start_month", type=int, required=True, help="Start month.")
    parser.add_argument("--end_year", type=int, required=True, help="End year.")
    parser.add_argument("--end_month", type=int, required=True, help="End month.")
    parser.add_argument("--bins", type=int, default=50, help="Number of histogram bins.")
    parser.add_argument("--cut_off_value", type=float, default=1.0, help="Cut-off value for threshold line.")
    args = parser.parse_args()

    combined_ratio_distribution(
        input_folder=args.input_folder,
        output_folder=args.output_folder,
        start_year=args.start_year,
        start_month=args.start_month,
        end_year=args.end_year,
        end_month=args.end_month,
        bins=args.bins,
        cut_off_value=args.cut_off_value
    )
