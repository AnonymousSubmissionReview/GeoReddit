"""
Compare year-specific author→state assignments (ratio2 > 1) to
the global assignment, summarizing overlap, mismatches, and coverage by year.

Inputs (inside --input_folder):
  - o_2005-06to2023-12_filtered_authors.csv (global state assignment)
      Required columns: author,state
  - p_YYYY-01toYYYY-12_ratios_postnum.csv (multiple years 2006-2023)
      Required columns: author,ratio2,substate2

Output (inside --output_folder):
  - p_geo_robust.csv with columns:
      year, total_year, only_year, only_all, mismatch, match
  total_year  = the number of users in that year with ratio2 > 1
  only_year   = the number of users in year set but not in global set
  only_all    = the number of users in global set but not in year set
  mismatch    = among users in both sets, the number of users whose year state != global state
  match       = among users in both sets, the number of users whose year state == global state


python c_10_geo_robust.py \
  --input_folder /path/to/input_folder \
  --output_folder /path/to/output_folder
"""

import os
import re
import math
import argparse
import pandas as pd

#Read one 'ratios_postnum' file and keep authors with ratio2>1.
#Returns a DataFrame with columns: author, substate2.
def read_year_file(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = [c.strip().lower() for c in df.columns]

    required = {'author', 'ratio2', 'substate2'}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{os.path.basename(path)} missing required columns: {missing}")

    def keep_ratio(x):
        try:
            if pd.isna(x):
                return False
            x = float(x)
            return math.isinf(x) or (x > 1.0)
        except Exception:
            return False

    mask = df['ratio2'].apply(keep_ratio)
    sub = df.loc[mask, ['author', 'substate2']].dropna()
    sub['author'] = sub['author'].astype(str).str.strip()
    sub['substate2'] = sub['substate2'].astype(str).str.strip()
    return sub

#Scan input_folder for required files, compute yearly overlaps, and save p_geo_robust.csv.
def build_geo_robust(input_folder: str, output_folder: str) -> str:
    # Load global author→state map
    authors_all_path = os.path.join(input_folder, "o_2005-06to2023-12_filtered_authors.csv")
    if not os.path.exists(authors_all_path):
        raise FileNotFoundError("Missing o_2005-06to2023-12_filtered_authors.csv in input_folder")

    all_df = pd.read_csv(authors_all_path)
    all_df.columns = [c.strip().lower() for c in all_df.columns]

    # Accept alternate column names and normalize
    if not {'author', 'state'}.issubset(all_df.columns):
        all_df = all_df.rename(columns={'author_all': 'author', 'state_all': 'state'})
    if not {'author', 'state'}.issubset(all_df.columns):
        raise ValueError("Global table must have columns: author,state (or author_all,state_all)")

    all_df['author'] = all_df['author'].astype(str).str.strip()
    all_df['state'] = all_df['state'].astype(str).str.strip()

    author_all_set = set(all_df['author'])
    state_all_map = dict(zip(all_df['author'], all_df['state']))

    # Yearly file pattern
    pat = re.compile(r'^p_(\d{4})-01to\1-12_ratios_postnum\.csv$')

    rows = []
    for fname in sorted(os.listdir(input_folder)):
        m = pat.match(fname)
        if not m:
            continue

        year = int(m.group(1))
        fpath = os.path.join(input_folder, fname)
        ydf = read_year_file(fpath)

        author_year_set = set(ydf['author'])
        state_year_map = dict(zip(ydf['author'], ydf['substate2']))

        total_year = len(author_year_set)
        only_year = len(author_year_set - author_all_set)
        only_all = len(author_all_set - author_year_set)

        inter = author_year_set & author_all_set
        mismatch = sum(1 for a in inter if state_year_map.get(a) != state_all_map.get(a))
        match = len(inter) - mismatch

        rows.append({
            'year': year,
            'total_year': total_year,
            'only_year': only_year,
            'only_all': only_all,
            'mismatch': mismatch,
            'match': match
        })

    out_df = pd.DataFrame(rows).sort_values('year')
    os.makedirs(output_folder, exist_ok=True)
    out_path = os.path.join(output_folder, "p_geo_robust.csv")
    out_df.to_csv(out_path, index=False)
    return out_path


def main():
    parser = argparse.ArgumentParser(description="Build p_geo_robust.csv from yearly ratios_postnum files")
    parser.add_argument("--input_folder", required=True,
                        help="Folder containing yearly 'p_YYYY-01toYYYY-12_ratios_postnum.csv' and the global authors file")
    parser.add_argument("--output_folder", required=True,
                        help="Folder to write p_geo_robust.csv")
    args = parser.parse_args()

    out_path = build_geo_robust(args.input_folder, args.output_folder)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()