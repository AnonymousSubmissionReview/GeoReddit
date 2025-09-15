"""
This script processes monthly Reddit submission and comment `.zst` files
(already filtered for geolocated users) to count the total number of posts and comments made by each user per month.

Input:
- A base input folder containing:
    - submissions/    (p_RS_YYYY-MM.zst)
    - comments/       (p_RC_YYYY-MM.zst)

Output:
- For each month: a CSV file saved in the output folder:
    - p_postnum_YYYY-MM.csv

Columns in p_postnum_YYYY-MM.csv:
- author: Reddit username
- all:    Total number of posts and comments made by the user in that month

Example usage:

# Example 1: Process data from January 2021 to December 2022
python c_03_post_num.py --input_folder "C:/data/reddit_input" --output_folder "C:/data/reddit_output/post_counts" --start_year 2021 --end_year 2022 --start_month 1 --end_month 12 --processes 4

# Example 2: Process March to June 2020
python c_03_post_num.py --input_folder "C:/data/reddit_input" --output_folder "C:/data/reddit_output" --start_year 2020 --end_year 2020 --start_month 3 --end_month 6

# Example 3: Process the entire year 2019
python c_03_post_num.py --input_folder "C:/data/reddit_input" --output_folder "C:/data/reddit_output" --start_year 2019 --end_year 2019 --start_month 1 --end_month 12
"""

import zstandard
import json
import os
import argparse
from collections import Counter
import multiprocessing
import csv
import logging

# Setup logging
log = logging.getLogger("bot")
log.setLevel(logging.INFO)
log_formatter = logging.Formatter('%(asctime)s - %(levelname)s: %(message)s')
log_str_handler = logging.StreamHandler()
log_str_handler.setFormatter(log_formatter)
log.addHandler(log_str_handler)

# Read compressed zst files and count how many posts/comments each author made
def count_authors_in_month(files):
    author_counter = Counter()
    for file_path in files:
        try:
            with open(file_path, 'rb') as file_handle:
                # Initialize decompressor
                reader = zstandard.ZstdDecompressor().stream_reader(file_handle)
                buffer = ''
                line_count = 0

                while True:
                    chunk = reader.read(2 ** 27)
                    if not chunk:
                        break
                    lines = (buffer + chunk.decode('utf-8', errors='ignore')).split("\n")
                    buffer = lines[-1]
                    for line in lines[:-1]:
                        line_count += 1
                        try:
                            obj = json.loads(line)
                            author_counter[obj['author']] += 1
                        except (json.JSONDecodeError, KeyError):
                            continue

                        if line_count % 100000 == 0:
                            log.info(f"Processed {line_count:,} lines from {file_path}")

        except FileNotFoundError:
            log.error(f"File not found: {file_path}")
    return author_counter

# Write per-user counts to a CSV file
def write_author_counts_to_csv(author_counts, output_file_path):
    with open(output_file_path, "w", encoding='utf-8', newline="") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["author", "all"])
        for author, count in author_counts.items():
            writer.writerow([author, count])

# Process a single month: read data files, count posts/comments, save results
def process_month(year, month, input_folder, output_folder):
    process_name = multiprocessing.current_process().name
    log.info(f"[{process_name}] Processing {year}-{month:02d}")

    submission_file = os.path.join(input_folder, "submissions", f"p_RS_{year}-{month:02d}.zst")
    comment_file = os.path.join(input_folder, "comments", f"p_RC_{year}-{month:02d}.zst")

    # Only keep files that actually exist
    files = [f for f in [submission_file, comment_file] if os.path.exists(f)]

    if not files:
        log.warning(f"[{process_name}] No files found for {year}-{month:02d}. Skipping.")
        return None

    # Count the number of posts/comments per user
    author_counts = count_authors_in_month(files)

    # Write output CSV with p_postnum_ prefix
    output_file_path = os.path.join(output_folder, f"p_postnum_{year}-{month:02d}.csv")
    write_author_counts_to_csv(author_counts, output_file_path)
    log.info(f"[{process_name}] Saved to {output_file_path}")
    return output_file_path

# Process all selected months in parallel
def process_all_months(input_folder, output_folder, start_year, end_year, start_month, end_month, processes):
    months_to_process = []
    for year in range(start_year, end_year + 1):
        m_start = start_month if year == start_year else 1
        m_end = end_month if year == end_year else 12
        for month in range(m_start, m_end + 1):
            months_to_process.append((year, month))

    # Make sure output directory exists
    os.makedirs(output_folder, exist_ok=True)

    # Use multiprocessing to process multiple months simultaneously
    with multiprocessing.Pool(processes=processes) as pool:
        pool.starmap(process_month, [
            (year, month, input_folder, output_folder)
            for year, month in months_to_process
        ])

if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Count total number of posts/comments per user in geolocated Reddit data.")
    parser.add_argument("--input_folder", required=True, help="Folder containing submissions/ and comments/ subfolders with p_*.zst files")
    parser.add_argument("--output_folder", required=True, help="Folder to save output CSV files")
    parser.add_argument("--start_year", type=int, required=True, help="Start year")
    parser.add_argument("--end_year", type=int, required=True, help="End year")
    parser.add_argument("--start_month", type=int, default=1, help="Start month")
    parser.add_argument("--end_month", type=int, default=12, help="End month")
    parser.add_argument("--processes", type=int, default=4, help="Number of parallel processes")

    args = parser.parse_args()

    # Start processing
    process_all_months(
        input_folder=args.input_folder,
        output_folder=args.output_folder,
        start_year=args.start_year,
        end_year=args.end_year,
        start_month=args.start_month,
        end_month=args.end_month,
        processes=args.processes
    )
