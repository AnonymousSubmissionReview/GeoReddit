r"""
This script generates interactive dynamic choropleth maps (and their underlying .csv data) to visualize average sentiment (estimated by NRCLex) across US states over time (monthly, quarterly, or yearly aggregation). 
This script provides different cutoff values >10, >30, >50, >100 observations per state x period combination. Also it gives an overview excel of how many observations there are for each state x period combination.

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
- HTML files saved in the output folder:
    - {topic}_{outcome_variable}_map_{freq}_{aggregation_method}.html
        (e.g., Pepsi_NRCL_freq_map_month_per_author)
        (animated choropleth map showing normalized sentiment prevalence)
 - csv files saved in the output folder:
    - {topic}_map_{freq}_{aggregation_method}.csv
 - Excel file saved in the output folder:
    - {topic}_observation_overview.xlsx       

--freq can be by month, quarter, or year
--outcome_variable is one variable based on NRCLex
--aggregation_method is either: 
        _per_comment (i.e., mean value is calculated across all posts within a state for the specified frequency)
        _per_author (i.e., mean value is calculated for author within a state for the specified frequency and in a subsequent step state estimate is calculated as mean across author means)

Example usage:
python a_09_sentiment_map_table.py `
--input_folder "C:/.../{topic}_sentiment" `
--output_folder "C:/.../{topic}_maps" `
--topic {topic}

"""

import plotly.express as px
import pandas as pd
import argparse
import os
import re
import numpy as np
from collections import defaultdict

def generate_sentiment_choropleths_and_counts(input_folder, output_base_folder, topic="Pepsi"):
    time_frequencies = ["month", "quarter", "year"]
    observation_thresholds = [10, 30, 50, 100]

    combined_csv_outputs = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))

    pattern = re.compile(rf"^p_{re.escape(topic)}_sentiment_\d{{4}}-\d{{2}}\.csv$")
    csv_files = [os.path.join(input_folder, f) for f in os.listdir(input_folder) if pattern.match(f)]
    if not csv_files:
        print(f"No files for topic '{topic}' in {input_folder}")
        return

    df_list = []
    for fpath in csv_files:
        try:
            tmp = pd.read_csv(fpath)
            if tmp.empty:
                continue
            date_str = os.path.basename(fpath).split('_')[-1].replace('.csv', '')
            tmp['time'] = pd.to_datetime(date_str)
            df_list.append(tmp)
        except Exception as e:
            print(f"⚠️ Warning reading {fpath}: {e}")

    if not df_list:
        print("No valid files loaded.")
        return

    df = pd.concat(df_list, ignore_index=True)
    df = df[df['time'].dt.year >= 2013]
    sentiment_columns = [c for c in df.columns if c.startswith(('NRCL_count', 'NRCL_prop', 'NRCL_freq'))]

    excel_path = os.path.join(output_base_folder, f"{topic}_observation_overview.xlsx")
    writer = pd.ExcelWriter(excel_path, engine='xlsxwriter')
    workbook = writer.book

    fmt_darkred = workbook.add_format({'bg_color': '#8B0000'})
    fmt_red = workbook.add_format({'bg_color': '#FF6347'})
    fmt_orange = workbook.add_format({'bg_color': '#FFA500'})
    fmt_yellow = workbook.add_format({'bg_color': '#FFFF99'})

    def colorize(ws, df_counts):
        ws.freeze_panes(1, 1)
        for r in range(1, len(df_counts)+1):
            for c in range(1, df_counts.shape[1]):
                val = df_counts.iat[r-1, c]
                if pd.isna(val):
                    continue
                if val < 10:
                    ws.write(r, c, val, fmt_darkred)
                elif val < 30:
                    ws.write(r, c, val, fmt_red)
                elif val < 50:
                    ws.write(r, c, val, fmt_orange)
                elif val < 100:
                    ws.write(r, c, val, fmt_yellow)

    for freq in time_frequencies:
        df0 = df.copy()
        if freq == "month":
            df0['period'] = df0['time'].dt.to_period("M")
            label = 'month'
        elif freq == "quarter":
            df0['period'] = df0['time'].dt.to_period("Q")
            label = 'quarter'
        else:
            df0['period'] = df0['time'].dt.to_period("Y")
            label = 'year'

        for mode in ['per_comment', 'per_author']:
            if mode == 'per_comment':
                df_c = df0.copy()
                df_c['count'] = 1
                grouped = df_c.groupby(['period', 'state']).count()['count']
            else:
                df_c = df0.groupby(['period', 'state', 'author']).size().reset_index(name='tmp')
                grouped = df_c.groupby(['period', 'state']).count()['tmp']

            df_count = grouped.unstack(fill_value=0).reset_index()
            sheet = f"{mode}_{label}"
            df_count.to_excel(writer, sheet_name=sheet, index=False)
            colorize(writer.sheets[sheet], df_count)

            coverage_df = pd.DataFrame()
            coverage_df['period'] = df_count['period'].astype(str)
            for t in observation_thresholds:
                coverage_df[f'>={t}'] = df_count.drop(columns='period').apply(lambda row: (row >= t).sum(), axis=1)

            cov_sheet = f"coverage_{mode}_{label}"
            coverage_df.to_excel(writer, sheet_name=cov_sheet, index=False)

        # Map + table generation
        for mode in ['per_comment', 'per_author']:
            for metric in sentiment_columns:
                if mode == 'per_author':
                    auth = df0.groupby(['period','state','author'])[metric].mean().reset_index()
                    state_sent = auth.groupby(['period','state']).agg(mean=(metric,'mean'), count=('author','count')).reset_index()
                else:
                    state_sent = df0.groupby(['period','state']).agg(mean=(metric,'mean'), count=(metric,'count')).reset_index()

                state_sent.rename(columns={'mean':'value','count':'count'}, inplace=True)
                state_sent['time'] = state_sent['period'].astype(str)

                for cutoff in observation_thresholds:
                    ss = state_sent.copy()
                    ss.loc[ss['count'] < cutoff, 'value'] = None

                    # Create and save the map (same as before)
                    fig = px.choropleth(
                        ss,
                        locations='state', locationmode='USA-states',
                        color='value', animation_frame='time',
                        color_continuous_scale='RdBu' if 'polarity' in metric or 'compound' in metric else 'PuBuGn',
                        range_color=(ss['value'].min(), ss['value'].max()),
                        scope='usa', labels={'value': f"{metric}"}
                    )

                    # Update each frame's trace with matching customdata and hovertemplate
                    for i, frame in enumerate(fig.frames):
                        frame_data = ss[ss['time'] == frame.name]
                        if frame_data.empty:
                            continue
                        frame_trace = frame.data[0]
                        frame_trace.customdata = np.array(frame_data[['count']])
                        frame_trace.hovertemplate = (
                            "<b>State:</b> %{location}<br>"
                            "<b>Value:</b> %{z}<br>"
                            "<b>N:</b> %{customdata[0]:.0f}<extra></extra>"
                        )

                    # Update the initial trace (first frame)
                    initial_time = fig.layout.sliders[0].active if fig.layout.sliders else ss['time'].iloc[0]
                    initial_data = ss[ss['time'] == initial_time]
                    if not initial_data.empty:
                        fig.data[0].customdata = np.array(initial_data[['count']])
                        fig.data[0].hovertemplate = (
                            "<b>State:</b> %{location}<br>"
                            "<b>Value:</b> %{z}<br>"
                            "<b>N:</b> %{customdata[0]:.0f}<extra></extra>"
                        )

                    fig.update_layout(title={
                        'text': f"{topic}: {metric} by State ({label}, {mode}, N≥{cutoff})",
                        'x': 0.5, 'xanchor': 'center'},
                        width=900, height=650,
                        margin={'r': 0, 't': 80, 'l': 0, 'b': 0}
                    )
                    if fig.layout.sliders:
                        fig.layout.sliders[0].y = 1.05
                        fig.layout.updatemenus[0].y = 1.2

                    subfolder = f"maps_by_{label}/maps_{cutoff}_obs"
                    full_folder = os.path.join(output_base_folder, subfolder)
                    os.makedirs(full_folder, exist_ok=True)

                    safe_metric = metric.replace(' ', '_').replace('/', '_')
                    fname_map = f"{topic}_{safe_metric}_{mode}_{label}_n{cutoff}.html"
                    fig.write_html(os.path.join(full_folder, fname_map))

                    # Prepare wide-format DataFrame for combined CSV
                    ss_wide = ss[['period', 'state', 'value']].copy()
                    metric_col_name = safe_metric
                    ss_wide = ss_wide.rename(columns={'value': metric_col_name})

                    # Initialize list for this cutoff if not exists
                    if cutoff not in combined_csv_outputs[mode][label]:
                        combined_csv_outputs[mode][label][cutoff] = []
                    combined_csv_outputs[mode][label][cutoff].append(ss_wide)
                    print(f"✅ {label}/{mode}/{metric}/n{cutoff} saved map+table.")

    writer.close()
    print(f"✅ Excel summary saved at: {excel_path}")

    # COMBINE .CSV FILES ACROSS METRICS BY CUTOFF VALUE (WIDE FORMAT)
    from functools import reduce

    for mode in combined_csv_outputs:
        for label in combined_csv_outputs[mode]:
            for cutoff in combined_csv_outputs[mode][label]:
                dfs = combined_csv_outputs[mode][label][cutoff]
                # Merge all metric dfs on ['period', 'state'] to get wide format
                combined_df = reduce(lambda left, right: pd.merge(left, right, on=['period', 'state'], how='outer'), dfs)
                
                output_file = os.path.join(output_base_folder, f"{topic}_combined_sentiment_{mode}_{label}_n{cutoff}.csv")
                combined_df.to_csv(output_file, index=False)
                print(f"📄 Saved combined CSV: {output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate maps and Excel count overview.")
    parser.add_argument('--input_folder', required=True)
    parser.add_argument('--output_folder', required=True)
    parser.add_argument('--topic', default='Pepsi')
    args = parser.parse_args()

    generate_sentiment_choropleths_and_counts(
        input_folder=args.input_folder,
        output_base_folder=args.output_folder,
        topic=args.topic
    )