r"""
This script generates linegraphs of NRCL emotions and sentiment over time (monthly, quarterly, or yearly aggregation) as .png-files. 

Input:
- CSV files in the input folder, named like:
    - p_{topic}_sentiment_YYYY-MM.csv
        (e.g., p_Pepsi_sentiment_2017-04)
    etc.
  Each file contains:
    - author, subreddit, body, keyword, total_keyword_num, total_word_count, state, NRCL_count_*, NRCL_prop_*, NRCL_freq_*

Columns in p_{topic}_sentiment_YYYY-MM.csv:
- author: Reddit username
- subreddit: Subreddit where the post/comment was published
- body: Text content of the post or comment
- total_keyword_num: Total number of keyword matches in the text
- total_word_count: Total number of words in body (i.e., comment)
- state: U.S. state associated with the author
- NRCL_count_*: counts of words in a comment associated with a specific emotion or sentiment (i.e., positive or negative) based on NCRLex (National Research Council Emotion Lexicon)
- NRCL_prop_*: proportion of words in a comment associated with a specific emotion or sentiment (i.e., positive or negative) in comparison to the total count of emotional words ranging form 0 to 1 based on NCRLex (National Research Council Emotion Lexicon)
- NRCL_freq_*: proportion of words in a comment associated with a specific emotion or sentiment (i.e., positive or negative) in comparison to the total number of words

Output:
- .png saved in the output folder:
    - {topic}_sentiment_timelines_{freq}_{outcome_variable}.png
        (e.g., Pepsi_sentiment_timelines_month_NRCL_EMOTIONS)
        

--outcome_variable is one variable based on NRCLex
--aggregation_method is presented in each timeline either: 
        _per_comment (i.e., mean value is calculated across all posts within a state for the specified frequency)
        _per_author (i.e., mean value is calculated for author within a state for the specified frequency and in a subsequent step state estimate is calculated as mean across author means)

Example usage:
python a_10_sentiment_linegraph.py `
--input_folder "C:/.../{topic}_sentiment" `
--output_folder "C:/.../{topic}_timeline" `
--topic {topic}

"""

import pandas as pd
import argparse
import os
import re
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np


def compute_and_save_timelines(input_folder, output_folder, topic="Pepsi"):
    # Regex pattern to find relevant CSV files
    pattern = re.compile(rf"^p_{re.escape(topic)}_sentiment_\d{{4}}-\d{{2}}\.csv$")
    csv_files = [os.path.join(input_folder, f) for f in os.listdir(input_folder) if pattern.match(f)]

    if not csv_files:
        print(f"No files found for topic '{topic}' in {input_folder}")
        return

    df_list = []
    for file in csv_files:
        try:
            df_tmp = pd.read_csv(file)
            if df_tmp.empty:
                print(f"⚠️ Warning: file {file} is empty. Skipping.")
                continue
            # Extract date from filename
            date_str = os.path.basename(file).split('_')[-1].replace('.csv', '')
            df_tmp['time'] = pd.to_datetime(date_str)
            df_list.append(df_tmp)
        except Exception as e:
            print(f"⚠️ Warning: failed to read {file}: {e}. Skipping.")
            continue

    if not df_list:
        print("No data loaded after reading files.")
        return

    df = pd.concat(df_list, ignore_index=True)
    df = df[df['time'].dt.year >= 2015].copy()  # avoid SettingWithCopyWarning

    # Define emotion and sentiment features
    NRCL_EMOTIONS = ['anger', 'anticipation', 'disgust', 'fear', 'joy', 'sadness', 'surprise', 'trust']
    NRCL_SENTIMENTS = ['positive', 'negative']

    # List of features to include if present in dataframe
    features_to_plot = [
        *[f'NRCL_prop_{e}' for e in NRCL_EMOTIONS + NRCL_SENTIMENTS],
        *[f'NRCL_freq_{e}' for e in NRCL_EMOTIONS + NRCL_SENTIMENTS],
    ]
    features_to_plot = [f for f in features_to_plot if f in df.columns]

    if not features_to_plot:
        print("⚠️ None of the specified sentiment columns are present in the dataset.")
        return

    # Filter numeric columns only
    numeric_features = [f for f in features_to_plot if pd.api.types.is_numeric_dtype(df[f])]
    if not numeric_features:
        print("⚠️ None of the specified sentiment columns are numeric.")
        return

    # Group features by type for plotting
    NRCL_prop_emotions = [f for f in numeric_features if f.startswith('NRCL_prop_') and any(e in f for e in NRCL_EMOTIONS)]
    NRCL_prop_sentiments = [f for f in numeric_features if f.startswith('NRCL_prop_') and any(s in f for s in NRCL_SENTIMENTS)]
    NRCL_freq_emotions = [f for f in numeric_features if f.startswith('NRCL_freq_') and any(e in f for e in NRCL_EMOTIONS)]
    NRCL_freq_sentiments = [f for f in numeric_features if f.startswith('NRCL_freq_') and any(s in f for s in NRCL_SENTIMENTS)]

    feature_groups = {
        'NRCL_prop_emotions': NRCL_prop_emotions,
        'NRCL_prop_sentiments': NRCL_prop_sentiments,
        'NRCL_freq_emotions': NRCL_freq_emotions,
        'NRCL_freq_sentiments': NRCL_freq_sentiments,
    }

    # Make sure output folder exists
    os.makedirs(output_folder, exist_ok=True)

    for freq in ["month", "quarter", "year"]:
        # Create 'period' column for grouping
        if freq == "quarter":
            df["period"] = df["time"].dt.to_period("Q")
        elif freq == "year":
            df["period"] = df["time"].dt.to_period("Y")
        else:
            df["period"] = df["time"].dt.to_period("M")

        # Aggregate per author, then average per period
        df_per_author = df.groupby(["period", "author"])[numeric_features].mean().reset_index()
        df_per_author_agg = df_per_author.groupby("period")[numeric_features].mean().reset_index()
        df_per_comment_agg = df.groupby("period")[numeric_features].mean().reset_index()

        # Convert Period to Timestamp for plotting
        df_per_author_agg['time'] = df_per_author_agg['period'].dt.to_timestamp()
        df_per_comment_agg['time'] = df_per_comment_agg['period'].dt.to_timestamp()

        for group_name, features in feature_groups.items():
            if not features:
                continue

            # Decide if broken axis is needed
            use_broken_axis = group_name in ['NRCL_prop_sentiments', 'NRCL_freq_sentiments']

            if use_broken_axis:
                fig, (ax_upper, ax_lower) = plt.subplots(
                    2, 1, sharex=True, figsize=(18, 9),
                    gridspec_kw={'height_ratios': [2, 1]}
                )

                # Manual assignment of features to upper/lower plots
                lower_features = [f for f in features if 'negative' in f]
                upper_features = [f for f in features if 'positive' in f]
                
                # Determine y-limits for each subplot
                def get_feature_range(feature_list):
                    combined_vals = pd.concat(
                        [df_per_author_agg[feature_list], df_per_comment_agg[feature_list]],
                        axis=0
                    ).stack()
                    return combined_vals.min(), combined_vals.max()

                if lower_features:
                    ymin, ymax = get_feature_range(lower_features)
                    ax_lower.set_ylim(ymin, ymax)

                if upper_features:
                    ymin, ymax = get_feature_range(upper_features)
                    ax_upper.set_ylim(ymin, ymax)

                # Plotting
                n_features = len(features)
                colors = plt.cm.get_cmap('tab10', max(n_features, 1))

                for i, feature in enumerate(features):
                    color = colors(i)
                    if feature in lower_features:
                        ax_lower.plot(df_per_author_agg['time'], df_per_author_agg[feature], '-', color=color,
                                    label=f"{feature} (per_author)")
                        ax_lower.plot(df_per_comment_agg['time'], df_per_comment_agg[feature], ':', color=color,
                                    label=f"{feature} (per_comment)")
                    elif feature in upper_features:
                        ax_upper.plot(df_per_author_agg['time'], df_per_author_agg[feature], '-', color=color,
                                    label=f"{feature} (per_author)")
                        ax_upper.plot(df_per_comment_agg['time'], df_per_comment_agg[feature], ':', color=color,
                                    label=f"{feature} (per_comment)")

                # Hide spines for broken look
                ax_upper.spines['bottom'].set_visible(False)
                ax_lower.spines['top'].set_visible(False)

                # Draw diagonal lines
                d = .015
                kwargs = dict(transform=ax_upper.transAxes, color='k', clip_on=False)
                ax_upper.plot((-d, +d), (-d, +d), **kwargs)
                ax_upper.plot((1 - d, 1 + d), (-d, +d), **kwargs)

                kwargs.update(transform=ax_lower.transAxes)
                ax_lower.plot((-d, +d), (1 - d, 1 + d), **kwargs)
                ax_lower.plot((1 - d, 1 + d), (1 - d, 1 + d), **kwargs)

                ax_lower.xaxis.set_major_locator(mdates.YearLocator())
                ax_lower.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))

                # Combine legend entries from both axes
                handles_upper, labels_upper = ax_upper.get_legend_handles_labels()
                handles_lower, labels_lower = ax_lower.get_legend_handles_labels()

                # Merge and remove duplicates while preserving order
                seen = set()
                handles_combined = []
                labels_combined = []
                for h, l in zip(handles_upper + handles_lower, labels_upper + labels_lower):
                    if l not in seen:
                        handles_combined.append(h)
                        labels_combined.append(l)
                        seen.add(l)

                # Plot the merged legend
                ax_upper.legend(handles_combined, labels_combined, loc='upper left', bbox_to_anchor=(1.05, 1), fontsize=8)
                fig.suptitle(f"{topic} sentiment timelines ({freq}) - {group_name} (broken axis)", fontsize=14)
                fig.subplots_adjust(hspace=0.05)

                filename = os.path.join(output_folder, f"{topic}_sentiment_timelines_{freq}_{group_name}_broken_axis.png")
                plt.savefig(filename, dpi=300, bbox_inches='tight')
                plt.close()

            else:
                # Single axis plot without broken axis
                plt.figure(figsize=(18, 6))
                n_features = len(features)
                colors = plt.cm.get_cmap('tab10', max(n_features, 1))

                for i, feature in enumerate(features):
                    color = colors(i)
                    plt.plot(df_per_author_agg['time'], df_per_author_agg[feature], linestyle='-', color=color,
                             label=f"{feature} (per_author)")
                    plt.plot(df_per_comment_agg['time'], df_per_comment_agg[feature], linestyle=':', color=color,
                             label=f"{feature} (per_comment)")

                plt.title(f"{topic} sentiment timelines ({freq}) - {group_name}")
                plt.legend(loc='upper left', bbox_to_anchor=(1.05, 1), fontsize=8)
                plt.gca().xaxis.set_major_locator(mdates.YearLocator())
                plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
                plt.tight_layout(rect=[0, 0, 0.85, 1])

                filename = os.path.join(output_folder, f"{topic}_sentiment_timelines_{freq}_{group_name}.png")
                plt.savefig(filename, dpi=300)
                plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute and save sentiment timelines for a topic.")
    parser.add_argument("--input_folder", help="Input folder with sentiment CSV files")
    parser.add_argument("--output_folder", help="Folder to save timeline plots")
    parser.add_argument("--topic", default="Pepsi", help="Topic name (default: Pepsi)")
    args = parser.parse_args()

    compute_and_save_timelines(args.input_folder, args.output_folder, args.topic)