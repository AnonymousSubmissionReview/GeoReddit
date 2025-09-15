r"""
This script generates choropleth maps (and its corresponding data as .csv) to visualize the change in average sentiment (estimated by NRCLex anger) across US states between April 2017 and the reference period January 2014 - March 2017. Maps are created for NRCL_freq_anger or NRCL_prop_anger and the cutoffs by 30 or 50 observations.

Input:
- A base input folder containing:
    - files named p_{topic}_sentiment_{YYYY}-{MM}.csv"

Output:
- An .csv-, .html-, .png-, and .jpeg-file of difference in anger-related words by state:
    - Pepsi_{outcome_variable}_author_event_diff_summary.csv
    - Pepsi_{outcome_variable}_event_diff_map_author_n{threshold}.html
    - Pepsi_{outcome_variable}_event_diff_map_author_n{threshold}.png
    - Pepsi_{outcome_variable}_event_diff_map_author_n{threshold}.jpeg
      (all matched posts with detailed metadata)

Additional info:
- the python library kaleido (as part of plotly, which does not need to be explicitly called) needs google chrome, is commented out here together with creating .png and .jpeg files
- {outcome_variable} is NRCL_freq_anger or NRCL_prop_anger
- {threshold} is cutoff by 30 or 50 observations

## Pepsi

Example Usage:
python .../e_v1_02_pepsi_event_change_map.py `
--input_folder "C:/.../Pepsi_sentiment" `
--output_folder "C:/.../Pepsi_event_map" `
--topic Pepsi

"""

import os
import re
import pandas as pd
import plotly.express as px
from datetime import datetime
import argparse
#kaleido

def compute_event_diff_map(input_folder, output_folder_base, topic="Pepsi"):
    pattern = re.compile(rf"^p_{re.escape(topic)}_sentiment_(\d{{4}})-(\d{{2}})\.csv$")
    csv_files = []

    for f in os.listdir(input_folder):
        match = pattern.match(f)
        if match:
            year, month = int(match.group(1)), int(match.group(2))
            file_date = datetime(year, month, 1)
            if datetime(2014, 1, 1) <= file_date <= datetime(2017, 4, 1):
                csv_files.append(os.path.join(input_folder, f))

    if not csv_files:
        print(f"No matching files found for topic '{topic}' in {input_folder}")
        return

    df_list = []
    for file in sorted(csv_files):
        df_tmp = pd.read_csv(file)
        date_str = os.path.basename(file).split('_')[-1].replace('.csv', '')
        df_tmp['time'] = pd.to_datetime(date_str)
        df_list.append(df_tmp)

    df = pd.concat(df_list, ignore_index=True)
    df = df[df['state'].notna() & df['author'].notna()]
    df['state'] = df['state'].str.upper()

    ### Loop through outcome_variables and thresholds ###
    outcome_variables = ["NRCL_freq_anger", "NRCL_prop_anger"]
    thresholds = [30, 50]

    for outcome_variable in outcome_variables:
        if outcome_variable not in df.columns:
            print(f"outcome_variable {outcome_variable} not found in data.")
            continue

        data = df[['state', 'author', 'time', outcome_variable]].copy()
        data = data.dropna(subset=[outcome_variable])
        data['event'] = 0
        data.loc[data['time'] == pd.Timestamp('2017-04-01'), 'event'] = 1
        data = data[data['event'].isin([0, 1])]

        # Author-level aggregation
        author_avg = data.groupby(['state', 'event', 'author'])[outcome_variable].mean().reset_index()
        author_grouped = author_avg.groupby(['state', 'event'])[outcome_variable].agg(['mean', 'count']).unstack()
        author_grouped.columns = [
            f'event0_mean_author_{outcome_variable}',
            f'event1_mean_author_{outcome_variable}',
            f'n_event0_author_{outcome_variable}',
            f'n_event1_author_{outcome_variable}'
        ]
        author_grouped[f'diff_author_{outcome_variable}'] = (
            author_grouped[f'event1_mean_author_{outcome_variable}'] -
            author_grouped[f'event0_mean_author_{outcome_variable}']
        )
        author_grouped = author_grouped.reset_index()

        for threshold in thresholds:
            output_folder = os.path.join(output_folder_base, f"{outcome_variable}_n{threshold}")
            os.makedirs(output_folder, exist_ok=True)

            # Save CSV
            csv_output_path = os.path.join(output_folder, f"{topic}_{outcome_variable}_author_event_diff_summary.csv")
            author_grouped.to_csv(csv_output_path, index=False)
            print(f"Author-level summary CSV saved: {csv_output_path}")

            # Prepare map data
            author_map_df = author_grouped[[
                'state',
                f'event0_mean_author_{outcome_variable}',
                f'event1_mean_author_{outcome_variable}',
                f'n_event0_author_{outcome_variable}',
                f'n_event1_author_{outcome_variable}',
                f'diff_author_{outcome_variable}'
            ]].copy()
            author_map_df = author_map_df.rename(columns={f'diff_author_{outcome_variable}': 'diff'})

            # Filter threshold
            author_map_df.loc[author_map_df[f'n_event1_author_{outcome_variable}'] < threshold, 'diff'] = pd.NA

            # Generate Map
            fig = px.choropleth(
                author_map_df,
                locations='state',
                locationmode="USA-states",
                color='diff',
                color_continuous_scale='YlOrRd',
                scope="usa",
                labels={'diff': 'Change in share\n '},
                title=f"Change in anger-related words ({outcome_variable}), min N = {threshold}",
                hover_data={
                    'state': True,
                    'diff': ':.3f',
                    f'event0_mean_author_{outcome_variable}': ':.3f',
                    f'event1_mean_author_{outcome_variable}': ':.3f',
                    f'n_event0_author_{outcome_variable}': True,
                    f'n_event1_author_{outcome_variable}': True
                }
            )

            fig.update_layout(
                coloraxis_colorbar=dict(
                    title=dict(
                        text='Change in share of<br>anger-related words',
                        side='top',
                        font=dict(size=14)
                    ),
                    ypad=20,
                    tickformat=".3f",
                    x=0.95
                ),
                title_x=0.5,
                title_font_size=18,
                margin=dict(l=0, r=0, t=80, b=0)
            )

            base_filename = f"{topic}_{outcome_variable}_event_diff_map_author_n{threshold}"
            fig.write_html(os.path.join(output_folder, f"{base_filename}.html"))
            #fig.write_image(os.path.join(output_folder, f"{base_filename}.png"), width=1000, height=600, scale=2)
            #fig.write_image(os.path.join(output_folder, f"{base_filename}.jpeg"), width=1000, height=600, scale=2)

            print(f"Map files saved to: {output_folder}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate sentiment difference maps from monthly CSVs.")
    parser.add_argument("--input_folder", required=True, help="Folder with monthly sentiment CSVs.")
    parser.add_argument("--output_folder", required=True, help="Folder to save choropleth maps and CSV.")
    parser.add_argument("--topic", default="Pepsi", help="Topic name in CSV filenames (e.g., 'Pepsi').")
    args = parser.parse_args()

    compute_event_diff_map(args.input_folder, args.output_folder, args.topic)
