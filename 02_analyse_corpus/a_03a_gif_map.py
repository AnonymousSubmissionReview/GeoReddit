"""
This script aggregates Reddit post counts and unique author counts by US state and time window (month, quarter
and year), using geolocated author data with ratio2 > 1.

Inputs:
- An input folder containing:
    - Monthly post count CSV files named: p_postnum_YYYY-MM.csv
    - o_2005-06to2023-12_filtered_authors.csv

Outputs:
- CSV files saved in the output folder:
    - p_author_counts_per_month.csv: For each month, the number of unique authors per US state.
    - p_post_counts_per_month.csv: For each month, the total number of posts per US state.
    - p_author_counts_per_quarter.csv: For each quarter, the number of unique authors per US state.
    - p_post_counts_per_quarter.csv: For each quarter, the number of posts per US state.
    - p_author_counts_per_year.csv: For each year, the number of unique authors per US state.
    - p_post_counts_per_year.csv: For each year, the number of posts per US state.

Example usage:
Process all monthly files in the input folder
python a_03a_gif_map.py --input_folder "C:/data/reddit_input" --output_folder "C:/data/reddit_output"

"""
import argparse
import pandas as pd
import os
from datetime import datetime

US_STATES = [
    'AL', 'AK', 'AZ', 'AR', 'CA', 'CO', 'CT', 'DC', 'DE', 'FL', 'GA', 'HI',
    'ID', 'IL', 'IN', 'IA', 'KS', 'KY', 'LA', 'ME', 'MD', 'MA', 'MI', 'MN',
    'MS', 'MO', 'MT', 'NE', 'NV', 'NH', 'NJ', 'NM', 'NY', 'NC', 'ND', 'OH',
    'OK', 'OR', 'PA', 'RI', 'SC', 'SD', 'TN', 'TX', 'UT', 'VT', 'VA', 'WA',
    'WV', 'WI', 'WY'
]

def load_data_by_time_range(input_folder, filtered_df, start_month, end_month):
    all_data = []
    for file in os.listdir(input_folder):
        if not file.startswith('p_postnum_') or not file.endswith('.csv'):
            continue
        month_str = file.split('_')[2].split('.')[0]
        if start_month <= month_str <= end_month:
            month_data = pd.read_csv(os.path.join(input_folder, file))
            merged_data = pd.merge(month_data, filtered_df, on='author', how='inner')
            all_data.append(merged_data)
    if not all_data:
        return pd.DataFrame()
    return pd.concat(all_data)

def main(input_folder, output_folder):
    filtered_df = pd.read_csv(os.path.join(input_folder, 'o_2005-06to2023-12_filtered_authors.csv'))


    monthly_author = pd.DataFrame(columns=['time'] + US_STATES)
    monthly_post = pd.DataFrame(columns=['time'] + US_STATES)
    quarterly_author = pd.DataFrame(columns=['time'] + US_STATES)
    quarterly_post = pd.DataFrame(columns=['time'] + US_STATES)
    yearly_author = pd.DataFrame(columns=['time'] + US_STATES)
    yearly_post = pd.DataFrame(columns=['time'] + US_STATES)


    monthly_files = [f for f in os.listdir(input_folder) if f.startswith('p_postnum_') and f.endswith('.csv')]
    monthly_files.sort()

    for file in monthly_files:
        month_str = file.split('_')[2].split('.')[0]
        year, month = month_str.split('-')
        quarter = f"{year}-Q{(int(month) - 1) // 3 + 1}"


        month_data = pd.read_csv(os.path.join(input_folder, file))
        merged_data = pd.merge(month_data, filtered_df, on='author', how='inner')


        author_counts = merged_data.groupby('state')['author'].nunique().reindex(US_STATES).fillna(0).astype(int)
        post_counts = merged_data.groupby('state')['all'].sum().reindex(US_STATES).fillna(0).astype(int)


        monthly_author.loc[len(monthly_author)] = {'time': month_str, **author_counts.to_dict()}
        monthly_post.loc[len(monthly_post)] = {'time': month_str, **post_counts.to_dict()}


    quarters = set(monthly_author['time'].apply(lambda x: f"{x.split('-')[0]}-Q{(int(x.split('-')[1]) - 1) // 3 + 1}"))
    for q in sorted(quarters):
        year, q_num = q.split('-Q')
        start_month = f"{year}-{int(q_num) * 3 - 2:02d}"  # Q1 → 01, Q2 → 04, etc.
        end_month = f"{year}-{int(q_num) * 3:02d}"        # Q1 → 03, Q2 → 06, etc.


        q_data = load_data_by_time_range(input_folder, filtered_df, start_month, end_month)
        if q_data.empty:
            continue


        q_author = q_data.groupby('state')['author'].nunique().reindex(US_STATES).fillna(0).astype(int)
        q_post = q_data.groupby('state')['all'].sum().reindex(US_STATES).fillna(0).astype(int)

        quarterly_author.loc[len(quarterly_author)] = {'time': q, **q_author.to_dict()}
        quarterly_post.loc[len(quarterly_post)] = {'time': q, **q_post.to_dict()}


    years = set(month_str.split('-')[0] for month_str in monthly_author['time'])
    for year in sorted(years):

        y_data = load_data_by_time_range(input_folder, filtered_df, f"{year}-01", f"{year}-12")
        if y_data.empty:
            continue


        y_author = y_data.groupby('state')['author'].nunique().reindex(US_STATES).fillna(0).astype(int)
        y_post = y_data.groupby('state')['all'].sum().reindex(US_STATES).fillna(0).astype(int)

        yearly_author.loc[len(yearly_author)] = {'time': year, **y_author.to_dict()}
        yearly_post.loc[len(yearly_post)] = {'time': year, **y_post.to_dict()}

    monthly_author.to_csv(os.path.join(output_folder, 'p_author_counts_per_month.csv'), index=False)
    monthly_post.to_csv(os.path.join(output_folder, 'p_post_counts_per_month.csv'), index=False)
    quarterly_author.to_csv(os.path.join(output_folder, 'p_author_counts_per_quarter.csv'), index=False)
    quarterly_post.to_csv(os.path.join(output_folder, 'p_post_counts_per_quarter.csv'), index=False)
    yearly_author.to_csv(os.path.join(output_folder, 'p_author_counts_per_year.csv'), index=False)
    yearly_post.to_csv(os.path.join(output_folder, 'p_post_counts_per_year.csv'), index=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_folder", required=True, help="输入文件夹路径")
    parser.add_argument("--output_folder", required=True, help="输出文件夹路径")
    args = parser.parse_args()
    main(args.input_folder, args.output_folder)