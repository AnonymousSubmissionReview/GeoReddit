"""
This script processes monthly Reddit submission and comment `.zst` files
to count the number of posts made by Reddit users in US state-related subreddits
(as listed in 'o_state_info.csv' crawled from r/LocationReddits).

Input:
- A base input folder containing:
    - submissions/    (with files named o_RS_YYYY-MM.zst)
    - comments/       (with files named o_RC_YYYY-MM.zst)
    - o_state_info.csv  (mapping subreddit names to U.S. states)

Output:
- For each month: a CSV file saved in the output folder:
    - p_YYYY-MM.csv     (aggregated counts of posts/comments per user per state)

Columns in p_YYYY-MM.csv:
- time:      Year and month in the format YYYYMM (e.g., "200802")
- author:    Reddit username (rows with deleted users are excluded)
- substate:  Two-letter abbreviation of the U.S. state related to the subreddit (e.g., "CA", "NY")
- num:       Total number of posts and comments made by the user in subreddits for that state during the month

Example usage:

# Example 1: Process all months from January 2005 to December 2006
python 111.py --input_folder "C:/Users/u2288/Downloads/reddit" --output_folder "C:/Users/u2288/Desktop" --start_year 2005 --end_year 2006

# Example 2: Process only June 2005
python c_01_locate_users.py --input_folder "C:/data/reddit_input" --output_folder "C:/data/reddit_output" --start_year 2005 --end_year 2005 --start_month 6 --end_month 6

# Example 3: Process March to August 2006
python c_01_locate_users.py --input_folder "C:/data/reddit_input" --output_folder "C:/data/reddit_output" --start_year 2006 --end_year 2006 --start_month 3 --end_month 8

# Example 4: Process the entire year 2005
python c_01_locate_users.py --input_folder "C:/data/reddit_input" --output_folder "C:/data/reddit_output" --start_year 2005 --end_year 2005 --start_month 1 --end_month 12

# Example 5: Process the entire year 2006
python c_01_locate_users.py --input_folder "C:/data/reddit_input" --output_folder "C:/data/reddit_output" --start_year 2006 --end_year 2006 --start_month 1 --end_month 12
"""



import zstandard
import json
import logging.handlers
import pandas as pd
import os
import argparse

# Set up logger for progress info
log = logging.getLogger("bot")
log.setLevel(logging.INFO)
log_formatter = logging.Formatter('%(asctime)s - %(levelname)s: %(message)s')
log_str_handler = logging.StreamHandler()
log_str_handler.setFormatter(log_formatter)
log.addHandler(log_str_handler)

# Read and decode a zst chunk recursively to handle partial frames
def read_and_decode(reader, chunk_size, max_window_size, previous_chunk=None, bytes_read=0):
    chunk = reader.read(chunk_size)
    bytes_read += chunk_size
    if previous_chunk is not None:
        chunk = previous_chunk + chunk
    try:
        return chunk.decode()
    except UnicodeDecodeError:
        if bytes_read > max_window_size:
            raise UnicodeError(f"Unable to decode frame after reading {bytes_read:,} bytes")
        return read_and_decode(reader, chunk_size, max_window_size, chunk, bytes_read)

# Generator to stream JSON lines from a .zst file
def read_lines_zst(file_name):
    with open(file_name, 'rb') as file_handle:
        buffer = ''
        reader = zstandard.ZstdDecompressor(max_window_size=2 ** 31).stream_reader(file_handle)
        while True:
            chunk = read_and_decode(reader, 2 ** 27, (2 ** 29) * 2)
            if not chunk:
                break
            lines = (buffer + chunk).split("\n")
            for line in lines[:-1]:
                yield line
            buffer = lines[-1]
        reader.close()

# Process data for a given month
def process_data(submissions_folder, comments_folder, state_info_file, output_folder, year, month):
    year_month = f"{year}{month:02d}"

    os.makedirs(output_folder, exist_ok=True)

    # Prepare input file paths for this month
    input_files = [
        os.path.join(submissions_folder, f'o_RS_{year}-{month:02d}.zst'),
        os.path.join(comments_folder, f'o_RC_{year}-{month:02d}.zst')
    ]

    log.info(f"Processing files for {year}-{month:02d}: {input_files}")

    rows = []
    file_lines, bad_lines = 0, 0

    # Read all files and collect author/subreddit pairs
    for file in input_files:
        if not os.path.exists(file):
            log.warning(f"File not found, skipping: {file}")
            continue
        for line in read_lines_zst(file):
            try:
                obj = json.loads(line)
                # Skip entries without author or subreddit
                if 'author' not in obj or obj['author'] == '[deleted]' or 'subreddit' not in obj:
                    continue
                rows.append({
                    "author": obj["author"],
                    "subreddit": obj["subreddit"]
                })
            except json.JSONDecodeError:
                bad_lines += 1
            file_lines += 1
            # Log progress every 100,000 lines
            if file_lines % 100000 == 0:
                log.info(f"Processed {file_lines:,} lines | Bad lines: {bad_lines:,}")

    log.info(f"Completed reading for {year}-{month:02d}: {file_lines:,} lines, {bad_lines:,} bad lines")

    if not rows:
        log.warning("No data extracted for this period.")
        return

    # Convert to DataFrame
    extracted_df = pd.DataFrame(rows)
    # Load state mapping
    state_df = pd.read_csv(state_info_file, encoding='utf-8')

    try:
        # Merge subreddit with state info
        final_df = (
            pd.merge(extracted_df, state_df, left_on='subreddit', right_on='place', how='inner')
            .groupby(['author', 'substate'])
            .size()
            .reset_index(name='num')
            .assign(time=year_month)
        )
        # Rearrange columns and sort
        final_df = final_df[['time', 'author', 'substate', 'num']].sort_values(by='author')

        # Write output CSV
        final_output_file_path = os.path.join(output_folder, f'p_{year}-{month:02d}.csv')
        final_df.to_csv(final_output_file_path, index=False, encoding='utf-8')
        log.info(f"Aggregated CSV saved to {final_output_file_path}")
    except FileNotFoundError:
        log.error(f"State info file not found: {state_info_file}")
    except Exception as e:
        log.error(f"Error during processing: {e}")

if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Locate and count Reddit users' posts/comments in state-related subreddits.")
    parser.add_argument("--input_folder", required=True, help="Base input folder containing submissions/, comments/, and state_info.csv")
    parser.add_argument("--output_folder", required=True, help="Folder for all output CSV files")
    parser.add_argument("--start_year", type=int, required=True)
    parser.add_argument("--end_year", type=int)
    parser.add_argument("--start_month", type=int, default=1)
    parser.add_argument("--end_month", type=int, default=12)

    args = parser.parse_args()

    # Determine end year (defaults to start year if not specified)
    end_year = args.end_year if args.end_year else args.start_year

    # Build input paths
    submissions_folder = os.path.join(args.input_folder, "submissions")
    comments_folder = os.path.join(args.input_folder, "comments")
    state_info_file = os.path.join(args.input_folder, "o_state_info.csv")

    # Loop over all months in the specified range
    for year in range(args.start_year, end_year + 1):
        start_m = args.start_month if year == args.start_year else 1
        end_m = args.end_month if year == end_year else 12
        for month in range(start_m, end_m + 1):
            process_data(
                submissions_folder=submissions_folder,
                comments_folder=comments_folder,
                state_info_file=state_info_file,
                output_folder=args.output_folder,
                year=year,
                month=month
            )
