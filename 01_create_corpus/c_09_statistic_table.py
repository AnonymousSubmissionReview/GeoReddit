"""
It generates summary statistics about Reddit user comment activity over a specified period, to help you compare:
- All Reddit users (from the raw dataset)
- Geo-located users (users assigned to U.S. states)
- High-confidence geo-located users (those exceeding a specified ratio threshold, e.g., ratio2 > 1)

Input:
- An input folder containing:
    - Multiple CSV files named in the format:
        o_T-YYYY-MM.csv
        (total number of comments per user for each month in the original Reddit dataset)
        Each file should have columns:
            - author: Reddit username
            - postnum: Number of comments in that month
    - A single CSV file named:
        p_YYYY-MMtoYYYY-MM_ratios_postnum.csv
        (containing user ratios and total posts/comments over the period)

Output:
- A CSV file saved in the output folder:
    - p_statistics_YYYY-MMtoYYYY-MM.csv
      A summary table containing:
        - Statistics for all Reddit users (excluding '[deleted]')
        - Statistics for all geo-located users 
        - Statistics for geo-located users with the specified ratio filter
      The file also appends a note indicating how many comments came from '[deleted]' or missing usernames.

Example usage:

# Example 1: Process all months from June 2005 to December 2006
python c_09_statistic_table.py --input_folder "C:/data/reddit_input" --output_folder "C:/data/reddit_output" --start_year 2005 --start_month 6 --end_year 2006 --end_month 12 --ratio_column ratio2 --ratio_threshold 1.0

# Example 2: Process only June 2005
python c_09_statistic_table.py --input_folder "C:/data/reddit_input" --output_folder "C:/data/reddit_output" --start_year 2005 --end_year 2005 --start_month 6 --end_month 6 --ratio_column ratio2 --ratio_threshold 1.0

# Example 3: Process March to August 2006
python c_09_statistic_table.py --input_folder "C:/data/reddit_input" --output_folder "C:/data/reddit_output" --start_year 2006 --end_year 2006 --start_month 3 --end_month 8 --ratio_column ratio2 --ratio_threshold 1.0

# Example 4: Process the entire year 2005
python c_09_statistic_table.py --input_folder "C:/data/reddit_input" --output_folder "C:/data/reddit_output" --start_year 2005 --end_year 2005 --start_month 1 --end_month 12 --ratio_column ratio2 --ratio_threshold 1.0

"""
import pandas as pd
import glob
import os
from collections import defaultdict
import argparse

def main(
    input_folder,
    output_folder,
    start_year,
    start_month,
    end_year,
    end_month,
    ratio_column,
    ratio_threshold
):
    # Collect date keys for filtering filenames
    date_keys = []
    for year in range(start_year, end_year + 1):
        m_start = start_month if year == start_year else 1
        m_end = end_month if year == end_year else 12
        for month in range(m_start, m_end + 1):
            date_keys.append(f"{year}-{month:02}")

    # Find input files
    all_files = glob.glob(os.path.join(input_folder, "o_T-*.csv"))
    selected_files = [f for f in all_files if any(key in os.path.basename(f) for key in date_keys)]
    selected_files.sort()

    print(f"Found {len(selected_files)} files to process.\n")

    # Count posts per author
    author_counts = defaultdict(float)
    for idx, file in enumerate(selected_files, start=1):
        print(f"[{idx}/{len(selected_files)}] Processing: {os.path.basename(file)}")
        df = pd.read_csv(file, usecols=["author", "postnum"])
        grouped = df.groupby("author")["postnum"].sum()
        for author, postnum in grouped.items():
            author_counts[author] += postnum

    print("\nFinished processing o_T-*.csv files.")

    author_summary = pd.DataFrame(list(author_counts.items()), columns=["author", "postnum"])
    non_deleted = author_summary[author_summary["author"] != "[deleted]"]

    mean_postnum = non_deleted["postnum"].mean()
    median_postnum = non_deleted["postnum"].median()
    sum_postnum = non_deleted["postnum"].sum()
    non_deleted_count = len(non_deleted)
    deleted_postnum = author_counts.get("[deleted]", 0)

    print("\nStatistics (excluding [deleted]):")
    print(f"Non-deleted user count: {non_deleted_count}")
    print(f"Total comments: {sum_postnum}")
    print(f"Mean comments: {mean_postnum}")
    print(f"Median comments: {median_postnum}")
    print(f"Deleted comments: {deleted_postnum}")

    # Load ratios file
    period_str = f"{start_year}-{start_month:02}to{end_year}-{end_month:02}"
    ratios_filename = os.path.join(input_folder, f"p_{period_str}_ratios_postnum.csv")
    geo_df = pd.read_csv(ratios_filename)

    geo_user_count = len(geo_df)
    geo_total = geo_df["all"].sum()
    geo_mean = geo_df["all"].mean()
    geo_median = geo_df["all"].median()

    certain_geo_df = geo_df[geo_df[ratio_column] > ratio_threshold]
    certain_geo_user_count = len(certain_geo_df)
    certain_geo_total = certain_geo_df["all"].sum()
    certain_geo_mean = certain_geo_df["all"].mean()
    certain_geo_median = certain_geo_df["all"].median()

    # Prepare summary table
    summary_rows = [
        {
            "Category": "All Reddit Users",
            "User Count": non_deleted_count,
            "Total Comment Count": f"{int(sum_postnum)} (+{int(deleted_postnum)} deleted)",
            "Mean Comments per User": round(mean_postnum, 3),
            "Median Comments per User": int(median_postnum)
        },
        {
            "Category": "Geo-located Users",
            "User Count": geo_user_count,
            "Total Comment Count": int(geo_total),
            "Mean Comments per User": round(geo_mean, 3),
            "Median Comments per User": int(geo_median)
        },
        {
            "Category": f"Certain Geo-located Users ({ratio_column} > {ratio_threshold})",
            "User Count": certain_geo_user_count,
            "Total Comment Count": int(certain_geo_total),
            "Mean Comments per User": round(certain_geo_mean, 3),
            "Median Comments per User": int(certain_geo_median)
        }
    ]

    summary_df = pd.DataFrame(summary_rows)

    note_text = (
        f"Note: {int(deleted_postnum)} is the total number of comments "
        "from users whose usernames were lost or marked as '[deleted]'."
    )

    # Export summary
    os.makedirs(output_folder, exist_ok=True)
    summary_output_path = os.path.join(output_folder, f"p_statistics_{period_str}.csv")
    with open(summary_output_path, "w", encoding="utf-8") as f:
        summary_df.to_csv(f, index=False)
        f.write("\n")
        f.write(note_text + "\n")

    print("\nSummary table exported to:")
    print(summary_output_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate Reddit Comment Statistics Summary")
    parser.add_argument("--input_folder", required=True, help="Input folder path containing o_T-*.csv and p_*_ratios_postnum.csv")
    parser.add_argument("--output_folder", required=True, help="Output folder path")
    parser.add_argument("--start_year", type=int, required=True)
    parser.add_argument("--start_month", type=int, required=True)
    parser.add_argument("--end_year", type=int, required=True)
    parser.add_argument("--end_month", type=int, required=True)
    parser.add_argument("--ratio_column", default="ratio2", help="Ratio column to filter (ratio1 or ratio2)")
    parser.add_argument("--ratio_threshold", type=float, default=1.0, help="Threshold for the ratio column")

    args = parser.parse_args()

    main(
        input_folder=args.input_folder,
        output_folder=args.output_folder,
        start_year=args.start_year,
        start_month=args.start_month,
        end_year=args.end_year,
        end_month=args.end_month,
        ratio_column=args.ratio_column,
        ratio_threshold=args.ratio_threshold
    )

