"""
Aggregate, for each year from a start date through that year’s end, the unique Reddit user counts and total post counts
by U.S. state, based on geolocated authors, as input for 28 plots in v_01b.

Inputs:
  p_postnum_YYYY-MM.csv files
  o_2005-06to2023-12_filtered_authors.csv

Outputs:
  p_state_counts_cumulative.csv in the output_folder, with columns:
    state, {YYYY}_user, {YYYY}_post for each year from start_year to end_year.

Usage:
  python v_01a_vali_pop.py --input_folder /path/to/input --output_folder /path/to/output --start_year 2005
  --start_month 6 --end_year 2023   --end_month 12
"""
import argparse
import pandas as pd
import os
from datetime import datetime

US_STATES = [
    'AL', 'AK', 'AZ', 'AR', 'CA', 'CO', 'CT', 'DC', 'DE', 'FL', 'GA', 'HI', 'ID', 'IL', 'IN', 'IA', 'KS', 'KY',
    'LA', 'ME', 'MD', 'MA', 'MI', 'MN', 'MS', 'MO', 'MT', 'NE', 'NV', 'NH', 'NJ', 'NM', 'NY', 'NC', 'ND', 'OH',
    'OK', 'OR', 'PA', 'RI', 'SC', 'SD', 'TN', 'TX', 'UT', 'VT', 'VA', 'WA', 'WV', 'WI', 'WY'
]

#Generate list of YYYY-MM from (sy,sm) to (ey,em) inclusive.
def month_list(sy, sm, ey, em):

    start = datetime(sy, sm, 1)
    end = datetime(ey, em, 1)
    months = []
    cur = start
    while cur <= end:
        months.append(cur.strftime("%Y-%m"))
        y, m = cur.year + (cur.month // 12), (cur.month % 12) + 1
        cur = datetime(y, m, 1)
    return months


def main(input_folder, sy, sm, ey, em, output_folder):
    # 1. Load author→state mapping
    auth = pd.read_csv(
        os.path.join(input_folder, "o_2005-06to2023-12_filtered_authors.csv"),
        usecols=["author", "state"], dtype=str
    ).drop_duplicates("author")

    # 2. Prepare aggregation structure
    years = list(range(sy, ey + 1))
    agg = {"state": US_STATES}
    for y in years:
        agg[f"{y}_user"] = [0] * len(US_STATES)
        agg[f"{y}_post"] = [0] * len(US_STATES)

    # 3. Load ALL data
    all_months = month_list(sy, sm, ey, em)
    monthly_data = []

    for m in all_months:
        fn = os.path.join(input_folder, f"p_postnum_{m}.csv")
        if os.path.exists(fn):
            df = pd.read_csv(fn, usecols=["author", "all"], dtype={"author": str, "all": int})
            df = df[df["author"].isin(auth["author"])]
            df = df.merge(auth, on="author", how="left").dropna(subset=["state"])
            monthly_data.append(df)
        else:
            print(f"[WARN] Missing file, skipping: {fn}")

    if not monthly_data:
        raise ValueError("No valid monthly data found!")

    # 4. Cumulative aggregation for each cutoff year
    cumulative_df = pd.DataFrame()

    for y in years:
        # Filter data up to December of current year (or end_month for final year)
        if y < ey:
            cutoff_month = 12
        else:
            cutoff_month = em

        # Get all months from start_date through December of year y
        months_in_range = [m for m in all_months
                           if (int(m.split('-')[0]) < y) or
                           (int(m.split('-')[0]) == y and int(m.split('-')[1]) <= cutoff_month)]

        # Combine all relevant monthly data
        combined = pd.concat([df for df, m in zip(monthly_data, all_months) if m in months_in_range],
                             ignore_index=True)

        # Group by state
        gb = combined.groupby("state").agg(
            unique_authors=("author", "nunique"),
            total_posts=("all", "sum")
        ).reindex(US_STATES, fill_value=0)

        # Store results
        for i, state in enumerate(US_STATES):
            row = gb.loc[state]
            agg[f"{y}_user"][i] = int(row["unique_authors"])
            agg[f"{y}_post"][i] = int(row["total_posts"])

    # 5. Output
    os.makedirs(output_folder, exist_ok=True)
    out_df = pd.DataFrame(agg)
    out_path = os.path.join(output_folder, "p_state_counts_cumulative.csv")
    out_df.to_csv(out_path, index=False)
    print(f"Cumulative results saved to {out_path}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--input_folder", required=True,
                   help="Folder with p_postnum_YYYY-MM.csv and filtered authors CSV")
    p.add_argument("--output_folder", required=True,
                   help="Folder to save output CSV")
    p.add_argument("--start_year", type=int, required=True, help="2005 etc.")
    p.add_argument("--start_month", type=int, required=True, help="1–12")
    p.add_argument("--end_year", type=int, required=True)
    p.add_argument("--end_month", type=int, required=True)

    args = p.parse_args()
    main(args.input_folder,
         args.start_year, args.start_month,
         args.end_year, args.end_month,
         args.output_folder)
