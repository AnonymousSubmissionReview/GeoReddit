"""
This script aggregates geolocated Reddit users’ activity into “types” (e.g., political orientation, religion,
socioeconomic group) via a subreddit→type mapping, and outputs per-user post counts by time period (month or day level).
What it does:
  - Time granularity is configurable:
      * --time month  : counts per "YYYY-MM" (default; derived from filename)
      * --time day    : counts per "YYYY-MM-DD" (from 'created_utc')
  - Output schema is configurable:
      * By default, includes the subreddit column (one more grouping key)
      * Use --include_subreddit to turn ON subreddit in outputs
        (if not set, the subreddit column will be omitted)

Inputs:
  - o_2005-06to2023-12_filtered_authors.csv
    Author-to-state mapping with columns: author, state. （ratio2 > 1)
  - submissions/
      * .zst files, e.g. p_RS_YYYY-MM.zst
  - comments/
      * .zst files, e.g. p_RC_YYYY-MM.zst
  - o_types.csv
      * CSV with columns: 'subreddit','type', like 'r/Baptist',baptist
      * Maps each subreddit to a type/category

Outputs:
  - p_user_type.csv  (final merged table)
      If --include_subreddit is set:
        columns: time,author,type,subreddit,num
      Otherwise:
        columns: time,author,type,num

Usage examples:
  # 1) Monthly counts, include subreddit in output
  python v_02a_vali_type.py \
      --input_folder /path/to/input \
      --output_folder /path/to/output \
      --processes 16 \
      --time month \
      --include_subreddit

  # 2) Daily counts, do NOT include subreddit
  python v_02a_vali_type.py \
      --input_folder /path/to/input \
      --output_folder /path/to/output \
      --processes 16 \
      --time day
"""

import zstandard
import json
import os
import csv
from collections import defaultdict
import logging.handlers
import multiprocessing
import argparse
import datetime

# Set up logging (keep original configuration)
log = logging.getLogger("bot")
log.setLevel(logging.INFO)
log_formatter = logging.Formatter('%(asctime)s - %(levelname)s: %(message)s')
log_str_handler = logging.StreamHandler()
log_str_handler.setFormatter(log_formatter)
log.addHandler(log_str_handler)

if not os.path.exists("logs"):
    os.makedirs("logs")

log_file_handler = logging.handlers.RotatingFileHandler(
    os.path.join("logs", "bot.log"), maxBytes=1024 * 1024 * 16, backupCount=5)
log_file_handler.setFormatter(log_formatter)
log.addHandler(log_file_handler)

# Load the set of filtered authors from authors.csv.
def load_filtered_authors(authors_csv):
    log.info(f"Loading filtered authors from {authors_csv}")
    authors = set()
    with open(authors_csv, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            authors.add(row['author'])
    return authors

# Load the mapping of subreddit to type from types.csv.
def load_subreddit_types(type_csv):
    log.info(f"Loading subreddit types from {type_csv}")
    mapping = {}
    with open(type_csv, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        reader.fieldnames = [fn.lstrip('\ufeff') for fn in reader.fieldnames]
        for row in reader:
            mapping[row['subreddit']] = row['type']
    return mapping

# Process one .zst file, count posts/comments per (time, author, type).
def process_single_file(
    file_path,
    authors,
    subreddit_types,
    output_dir,
    time: str = "month",   # "month" or "day"
    include_subreddit: bool = True,
):
    base_name = os.path.basename(file_path)
    year_month = base_name[5:-4]
    stats = defaultdict(int)

    input_handle = open(file_path, 'rb')
    reader = zstandard.ZstdDecompressor(max_window_size=2 ** 31).stream_reader(input_handle)

    try:
        buffer = ''
        while True:
            chunk = reader.read(2 ** 27)
            if not chunk:
                break

            lines = (buffer + chunk.decode('utf-8', errors='ignore')).split("\n")
            buffer = lines[-1]

            for line in lines[:-1]:
                try:
                    obj = json.loads(line)
                    author = obj.get("author", "")
                    if author not in authors:
                        continue
                    subreddit = obj.get("subreddit", "")
                    if not subreddit or subreddit not in subreddit_types:
                        continue

                    if time == "month":
                        time_key = year_month
                    else:
                        ts = obj.get("created_utc")
                        if ts is None:
                            continue
                        try:
                            time_key = datetime.datetime.utcfromtimestamp(int(ts)).strftime("%Y-%m-%d")
                        except Exception:
                            continue

                    sr_type = subreddit_types[subreddit]
                    if include_subreddit:
                        key = (time_key, author, sr_type, subreddit)
                    else:
                        key = (time_key, author, sr_type)

                    stats[key] += 1

                except json.JSONDecodeError:
                    log.warning(f"JSON decode error: {line[:50]}...")
                except Exception as e:
                    log.error(f"Error processing line: {line[:50]}... Error: {e}")

    finally:
        reader.close()

    # Write temporary per-file results
    output_path = os.path.join(output_dir, f"temp_{base_name[:-4]}.csv")
    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        if include_subreddit:
            writer.writerow(['time', 'author', 'type', 'subreddit', 'num'])
            for (time_key, author, post_type, subreddit), count in stats.items():
                writer.writerow([time_key, author, post_type, subreddit, count])
        else:
            writer.writerow(['time', 'author', 'type', 'num'])
            for (time_key, author, post_type), count in stats.items():
                writer.writerow([time_key, author, post_type, count])

# Merge all temporary CSVs into one history file.
def merge_results(output_dir, final_output):
    stats = defaultdict(int)
    has_subreddit = None

    for temp_file in os.listdir(output_dir):
        if not temp_file.startswith('temp_'):
            continue
        with open(os.path.join(output_dir, temp_file), 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            if has_subreddit is None:
                has_subreddit = ('subreddit' in reader.fieldnames)
            for row in reader:
                if has_subreddit:
                    key = (row['time'], row['author'], row['type'], row['subreddit'])
                else:
                    key = (row['time'], row['author'], row['type'])
                stats[key] += int(row['num'])
        os.remove(os.path.join(output_dir, temp_file))

    with open(final_output, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        if has_subreddit:
            writer.writerow(['time', 'author', 'type', 'subreddit', 'num'])
            for (t, a, ty, sr), cnt in stats.items():
                writer.writerow([t, a, ty, sr, cnt])
        else:
            writer.writerow(['time', 'author', 'type', 'num'])
            for (t, a, ty), cnt in stats.items():
                writer.writerow([t, a, ty, cnt])

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="This script extracts geolocated Reddit users’ monthly history of their “type” under a given attribute.")
    parser.add_argument('--input_folder', required=True,
                        help='Input directory containing submissions/, comments/, o_authors.csv')
    parser.add_argument('--output_folder', required=True,
                        help='o_types.csv,Output directory for results')
    parser.add_argument('--processes', type=int, default=4,
                        help='Number of parallel processes')
    parser.add_argument("--time", choices=["month", "day"], default="month",
                        help="Time granularity for statistics: 'month' or 'day'. Default is month.")
    parser.add_argument("--include_subreddit", action="store_true",
                        help="Include subreddit column in output if set (default: off).")
    args = parser.parse_args()

    # Load filtered authors and subreddit→type mapping
    authors = load_filtered_authors(os.path.join(args.input_folder, 'o_2005-06to2023-12_filtered_authors.csv'))
    subreddit_types = load_subreddit_types(os.path.join(args.output_folder, 'o_types.csv'))

    # Prepare output directory
    os.makedirs(args.output_folder, exist_ok=True)

    # Gather .zst files from submissions/ and comments/
    zst_files = []
    for subdir in ['submissions', 'comments']:
        dir_path = os.path.join(args.input_folder, subdir)
        if os.path.exists(dir_path):
            zst_files += [
                os.path.join(dir_path, f)
                for f in os.listdir(dir_path)
                if f.endswith('.zst') and (f.startswith('p_RS_') or f.startswith('p_RC_'))
            ]

    log.info(f"Starting processing of {len(zst_files)} files using {args.processes} processes")
    pool = multiprocessing.Pool(args.processes)
    for file_path in zst_files:
        pool.apply_async(
            process_single_file,
            args=(file_path, authors, subreddit_types, args.output_folder, args.time, args.include_subreddit)
        )
    pool.close()
    pool.join()

    # Merge all temporary results into the final history
    final_output = os.path.join(args.output_folder, 'p_user_type.csv')
    merge_results(args.output_folder, final_output)
    log.info("Processing completed successfully")
