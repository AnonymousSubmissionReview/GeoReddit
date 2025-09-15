"""
Aggregate counts and normalize against total Reddit author/post counts
by U.S. state and time window (monthly, quarterly, or yearly) for a given topic.

Inputs:
  --input_folder:
       • p_{topic}_author_month.csv
       • p_{topic}_post_month.csv
       • p_author_counts_per_{freq}.csv or p_post_counts_per_{freq}.csv
       • p_{topic}_YYYY-MM.csv
  --output_folder:
       • p_{topic}_author_{freq}.csv or p_{topic}_post_{freq}.csv  – absolute counts
       • p_{topic}_norm_{author|post}_{freq}.csv                  – normalized ratios

Usage examples:
  # Quarterly author aggregates for topic "AI"
  python a_03b_gif_map.py --input_folder ./data --output_folder ./out --topic AI --data_type author --freq quarter

"""
import os
import argparse
import pandas as pd

US_STATES = [
    'AL', 'AK', 'AZ', 'AR', 'CA', 'CO', 'CT', 'DC', 'DE', 'FL', 'GA', 'HI',
    'ID', 'IL', 'IN', 'IA', 'KS', 'KY', 'LA', 'ME', 'MD', 'MA', 'MI', 'MN',
    'MS', 'MO', 'MT', 'NE', 'NV', 'NH', 'NJ', 'NM', 'NY', 'NC', 'ND', 'OH',
    'OK', 'OR', 'PA', 'RI', 'SC', 'SD', 'TN', 'TX', 'UT', 'VT', 'VA', 'WA',
    'WV', 'WI', 'WY'
]

def generate_author_aggregates(input_folder, output_folder, topic, freq):
    monthly_files = [
        os.path.join(input_folder, f)
        for f in os.listdir(input_folder)
        if (f.startswith(f"p_{topic}_20")
        and f.endswith(".csv")
        and not f.endswith(f"_author_{freq}.csv"))
        ]

    if not monthly_files:
       return

    # 读取并合并数据
    dfs = []
    for file in sorted(monthly_files):
        time_str = os.path.basename(file).split('_')[-1].replace('.csv', '')
        df = pd.read_csv(file, usecols=['author', 'state'])
        df['time'] = time_str
        df['date'] = pd.to_datetime(df['time'])
        dfs.append(df)

    if not dfs:
        return

    combined = pd.concat(dfs)

    if freq == 'quarter':
        combined['period'] = combined['date'].dt.to_period('Q')
    else:  # year
        combined['period'] = combined['date'].dt.year

    result = (
        combined.groupby(['period', 'state'])
        ['author'].nunique()
        .unstack(fill_value=0)
        .reset_index()
    )

    for state in US_STATES:
        if state not in result.columns:
            result[state] = 0

    if freq == 'quarter':
        result['time'] = result['period'].apply(lambda x: f"{x.year}-Q{x.quarter}")  # 强制转为 YYYY-QX
    else:
        result['time'] = result['period'].astype(str)
    output_cols = ['time'] + US_STATES  # 严格按US_STATES顺序
    output_df = result[output_cols]
    output_df = output_df.sort_values('time')
    output_path = os.path.join(output_folder, f"p_{topic}_author_{freq}.csv")
    output_df.to_csv(output_path, index=False)
    print(f"Saved {freq} user counts to {output_path}")

def generate_post_aggregates(input_folder,output_folder, topic, freq):
    data_file = os.path.join(input_folder, f"p_{topic}_post_month.csv")
    if not os.path.exists(data_file):
        raise FileNotFoundError(f"Monthly post file not found: {data_file}")

    monthly_df = pd.read_csv(data_file)
    monthly_df['date'] = pd.to_datetime(monthly_df['time'])

    if freq == 'quarter':
        monthly_df['period'] = monthly_df['date'].dt.to_period('Q')
        time_format = lambda x: f"{x.year}-Q{x.quarter}"
    if freq == 'year':
        monthly_df['period'] = monthly_df['date'].dt.year
        time_format = str

    result = (
        monthly_df.groupby('period')
        [US_STATES].sum()
        .reset_index()
    )

    result['time'] = result['period'].apply(time_format)
    result.drop('period', axis=1, inplace=True)

    for state in US_STATES:
        if state not in result.columns:
            result[state] = 0

    output_cols = ['time'] + US_STATES
    result = result.sort_values('time')
    output_path = os.path.join(output_folder, f"p_{topic}_post_{freq}.csv")
    result[output_cols].to_csv(output_path, index=False)
    print(f"Saved {freq} post counts to {output_path}")


def generate_normalized_counts(input_folder,output_folder, topic, data_type, freq):
    if freq == 'month':
        topic_file = os.path.join(input_folder, f"p_{topic}_{data_type}_{freq}.csv")
    else:
        topic_file = os.path.join(output_folder, f"p_{topic}_{data_type}_{freq}.csv")
    baseline_file = os.path.join(input_folder, f"p_{data_type}_counts_per_{freq}.csv")

    output_file = os.path.join(output_folder, f"p_{topic}_norm_{data_type}_{freq}.csv")

    df_topic = pd.read_csv(topic_file).set_index('time')
    df_base = pd.read_csv(baseline_file).set_index('time')

    common_times = df_topic.index.intersection(df_base.index)
    df_topic = df_topic.loc[common_times]
    df_base = df_base.loc[common_times]

    df_norm = df_topic.div(df_base.replace(0, pd.NA)).fillna(0)
    df_norm = df_norm.sort_index()
    df_norm.reset_index().to_csv(output_file, index=False)
    print(f"Saved normalized {data_type} counts to {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate abs and normalized table with monthly, quarterly, or yearly aggregation."
    )
    parser.add_argument("--input_folder", required=True, help="Path to folder containing input CSV files.")
    parser.add_argument("--output_folder", required=True, help="Path to folder where output files will be saved.")
    parser.add_argument("--topic", required=True, help="Topic keyword (e.g., 'AI').")
    parser.add_argument("--data_type", choices=["author", "post"], default="author", help="Type of data to visualize.")
    parser.add_argument("--freq", choices=["month", "quarter", "year"], default="month", help="Time aggregation frequency.")

    args = parser.parse_args()

    os.makedirs(args.output_folder, exist_ok=True)

    if args.freq == 'month':
        generate_normalized_counts(args.input_folder, args.output_folder, args.topic, args.data_type, args.freq)
    else:
        if args.data_type == "author":
            generate_author_aggregates(args.input_folder,args.output_folder, args.topic, args.freq)
        else:
            generate_post_aggregates(args.input_folder,args.output_folder, args.topic, args.freq)

        generate_normalized_counts(args.input_folder,args.output_folder, args.topic, args.data_type, args.freq)

    print(f"Successfully processed {args.topic} {args.data_type} data at {args.freq} level")
