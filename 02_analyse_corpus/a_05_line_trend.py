"""
This script generates a line plot visualizing topic trends across U.S. states over time
(either monthly, quarterly, or yearly aggregation), using matched Reddit post or user counts.
For normalized month-level situation, the code first clips extreme values above the 99th percentile to reduce
distortion from outliers. Then, it applies a 3-month moving average to smooth short-term fluctuations,
making the trend lines more stable and easier to interpret.

Input:
- An input folder containing:
    - p_{topic}_{data_type}_{freq}.csv
         absolute monthly matched counts per U.S. state)
    - or p_{topic}_norm_{data_type}_{freq}.csv
        (e.g., p_AI_norm_post_quarter.csv; already normalized and aggregated)

Output:
- A .png line chart saved in the output folder:
    - p_{topic}_{data_type}_{freq}_{norm/abs}_trend.png
      (showing state-wise trends over time, optionally normalized)

Example usage:

# Example 1: Plot absolute post counts of topic 'AI' per state (monthly)
python a_05_line_trend.py --input_folder "./input" --output_folder "./output" --topic AI --data_type post --freq month

# Example 2: Plot normalized author counts quarterly
python a_05_line_trend.py --input_folder "./input" --output_folder "./output" --topic AI --data_type author --freq quarter --normalized

# Example 3: Plot yearly keyword trends (absolute)
python a_05_line_trend.py --input_folder "./input" --output_folder "./output" --topic AI --data_type keyword --freq year
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import datetime


# Search for matching input file name based on topic, type, freq, and whether normalized
def find_input_file(input_folder, topic, data_type, freq, normalized):
    for fname in os.listdir(input_folder):
        if normalized and fname == f"p_{topic}_norm_{data_type}_{freq}.csv":
            return os.path.join(input_folder, fname), True
        elif not normalized and fname == f"p_{topic}_{data_type}_{freq}.csv":
            return os.path.join(input_folder, fname), False
    return None, None


# Plotting function: one line per state, time on x-axis
def plot_state_trend(input_folder, output_folder, topic, data_type, freq="month", normalized=False):
    input_path, is_normalized_file = find_input_file(input_folder, topic, data_type, freq, normalized)
    if not input_path:
        raise FileNotFoundError("Input file not found with expected naming.")

    # Load data and aggregate if necessary
    df = pd.read_csv(input_path)

    # If normalized, clip outliers to reduce distortion from spikes
    if normalized:
        q99 = df[df.columns[1:]].stack().quantile(0.99)
        df[df.columns[1:]] = df[df.columns[1:]].clip(upper=q99)

    # Create output folder if needed
    os.makedirs(output_folder, exist_ok=True)

    # Create plot and handle axis formatting
    plt.figure(figsize=(18, 9))

    if freq == "month":
        df["time"] = pd.to_datetime(df["time"])
        df = df.sort_values("time")

        # Apply moving average smoothing if normalized
        # if normalized:
        #     smoothing_window = 3  # You can change this to 5 or 7 depending on how much smoothing you want
        #     for col in df.columns[1:]:
        #         df[col] = df[col].rolling(window=smoothing_window, center=True, min_periods=1).mean()

        locator = mdates.MonthLocator(interval=6)  # month internal would change xlabel
        fmt = mdates.DateFormatter("%Y-%m")
        plt.gca().xaxis.set_major_locator(locator)
        plt.gca().xaxis.set_major_formatter(fmt)
        plt.xticks(rotation=45)

    else:
        df = df.sort_values("time")  # Keep string format like "2023Q2" or "2022"

    # Determine time range for title
    if freq == "month":
        start = df["time"].min().strftime("%Y-%m")
        end = df["time"].max().strftime("%Y-%m")
    else:
        start = df["time"].min()
        end = df["time"].max()

    # Draw line for each state using color + linestyle combination
    linestyles = ['-', '--', '-.', ':']
    color_cycle = plt.colormaps.get_cmap('tab20')(range(13))  # use 13 unique colors

    for idx, state in enumerate(df.columns[1:]):
        color = color_cycle[idx % len(color_cycle)]
        linestyle = linestyles[(idx // len(color_cycle)) % len(linestyles)]
        plt.plot(df["time"], df[state], label=state, linewidth=1, color=color, linestyle=linestyle)

    # Create plot title
    title_suffix = (
        "Users" if data_type == "author" else
        "Posts"
    )

    title = f'{topic.capitalize()} Topic Prevalence Based on Geolocated Reddit {title_suffix} ({start} to {end} {freq.capitalize()} Aggregation)'
    plt.title(title, fontsize=16)
    plt.xlabel("Time", fontsize=12)
    plt.ylabel("Normalized proportion" if normalized else "Count", fontsize=12)

    # Add legend inside the plot (upper-left corner), with a white background and black border
    plt.legend(
        loc="upper left",  # Anchor the legend's upper-left corner
        bbox_to_anchor=(0.01, 0.99),  # Slight offset from the top-left corner of the axes (x=1%, y=99%)
        fontsize=8,  # Font size for legend text
        ncol=8,  # Number of columns in the legend
        frameon=True,  # Show border/frame around the legend
        facecolor='white',  # Legend background color
        edgecolor='black'  # Legend border color
    )

    plt.tight_layout()

    # Save plot
    output_file = os.path.join(output_folder,
                               f"p_{topic}_{data_type}_{freq}_{'norm' if normalized else 'abs'}_trend.png")
    plt.savefig(output_file, dpi=300)
    print(f"Saved plot to {output_file}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Plot US state-level trend lines.")
    parser.add_argument("--input_folder", required=True, help="Path to folder with CSV input files")
    parser.add_argument("--output_folder", required=True, help="Path to save plot images")
    parser.add_argument("--topic", required=True, help="Topic name, e.g., AI")
    parser.add_argument("--data_type", required=True, help="Data type: post, author")
    parser.add_argument("--freq", choices=["month", "quarter", "year"], default="month", help="Time aggregation")
    parser.add_argument("--normalized", action="store_true", help="Use normalized input file")

    args = parser.parse_args()

    plot_state_trend(
        input_folder=args.input_folder,
        output_folder=args.output_folder,
        topic=args.topic,
        data_type=args.data_type,
        freq=args.freq,
        normalized=args.normalized
    )
