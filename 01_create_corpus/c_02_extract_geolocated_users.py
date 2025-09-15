"""
This script filters Reddit submission and comment `.zst` files to extract only records belonging to a specified list of geolocated users.

Features:
1. Uses multiprocessing to process multiple `.zst` files in parallel.
2. Automatically writes output files to corresponding submissions/ and comments/ subfolders with filenames prefixed by "p_".
3. Provides an optional inspection mode to preview output.

Input:
- A base input folder containing:
    - submissions/    (o_RS_YYYY-MM.zst)
    - comments/       (o_RC_YYYY-MM.zst)
    - o_2005-06to2023-12_unique_authors.csv (A CSV file listing target usernames and formed by the unique user of 'p_YYYY-MM.csv' files)

Output:
    - submissions/    (p_RS_YYYY-MM.zst)
    - comments/       (p_RC_YYYY-MM.zst)
    - (Optional) Printed inspection preview of the first few lines from each output file

Example usage:

# Example 1: Use 4 processes to process all data
python c_02_extract_geolocated_users.py --input_folder "C:/data/reddit_input" --output_folder "C:/data/reddit_output" --processes 4

# Example 2: Use 8 processes and inspect the first 10 lines of each output file
python c_02_extract_geolocated_users.py --input_folder "C:/data/reddit_input" --output_folder "C:/data/reddit_output" --processes 8 --inspect --num_lines 10

# Example 3: Process with default 4 processes and inspect first 5 lines
python c_02_extract_geolocated_users.py --input_folder "C:/data/reddit_input" --output_folder "C:/data/reddit_output" --inspect

# Example 4: Process all files without inspection
python c_02_extract_geolocated_users.py --input_folder "C:/data/reddit_input" --output_folder "C:/data/reddit_output"
"""

import zstandard
import os
import json
import csv
import argparse
import re
import multiprocessing
import logging.handlers

# Set up logging
log = logging.getLogger("bot")
log.setLevel(logging.INFO)
log_formatter = logging.Formatter('%(asctime)s - %(levelname)s: %(message)s')
log_str_handler = logging.StreamHandler()
log_str_handler.setFormatter(log_formatter)
log.addHandler(log_str_handler)

# Rotating log file handler
if not os.path.exists("logs"):
    os.makedirs("logs")
log_file_handler = logging.handlers.RotatingFileHandler(
    os.path.join("logs", "bot.log"), maxBytes=1024 * 1024 * 16, backupCount=5)
log_file_handler.setFormatter(log_formatter)
log.addHandler(log_file_handler)

# Load unique geolocated authors from CSV
def load_authors_from_csv(csv_file):
    csv_file_path = os.path.abspath(csv_file)
    log.info(f"Loading authors list file: {csv_file_path}")
    authors = set()
    with open(csv_file_path, mode='r', encoding='utf-8') as file:
        reader = csv.reader(file)
        header = next(reader)
        if "author" not in header:
            raise ValueError("Missing 'author' column in CSV file")
        author_index = header.index("author")
        for row in reader:
            if row[author_index].strip():
                authors.add(row[author_index])
    return authors

# Process a single .zst file: decompress, filter by author, recompress
def process_file(file_path, output_subfolder, authors):
    basename = os.path.basename(file_path)
    if basename.startswith("o_"):
        basename = basename[2:]
    output_filename = "p_" + basename
    output_path = os.path.join(output_subfolder, output_filename)

    # Ensure output subfolder exists
    os.makedirs(output_subfolder, exist_ok=True)

    # Initialize decompressor and compressor
    input_handle = open(file_path, 'rb')
    reader = zstandard.ZstdDecompressor(max_window_size=2 ** 31).stream_reader(input_handle)
    writer = zstandard.ZstdCompressor().stream_writer(open(output_path, 'wb'))

    try:
        buffer = ''
        line_count = 0
        while True:
            # Read a chunk
            chunk = reader.read(2 ** 27)
            if not chunk:
                break
            # Split into lines
            lines = (buffer + chunk.decode('utf-8', errors='ignore')).split("\n")
            buffer = lines[-1]

            for line in lines[:-1]:
                line_count += 1
                try:
                    obj = json.loads(line)
                    # Keep only if author is in geolocated list
                    if obj.get("author", "") in authors:
                        writer.write((line + "\n").encode('utf-8'))
                except json.JSONDecodeError:
                    log.warning(f"JSON decode error: {line[:50]}...")

                if line_count % 100000 == 0:
                    log.info(f"Processed {line_count:,} lines from file {file_path}")
    except Exception as e:
        log.error(f"Error processing file {file_path}: {str(e)}")
    finally:
        reader.close()
        writer.close()

# Inspect compressed output file and print preview
def inspect_zst_file(file_path, num_lines=5):
    author_set = set()
    line_count = 0

    try:
        with open(file_path, 'rb') as file_handle:
            reader = zstandard.ZstdDecompressor(max_window_size=2 ** 31).stream_reader(file_handle)
            buffer = ""
            while True:
                chunk = reader.read(2 ** 27)
                if not chunk:
                    break
                lines = (buffer + chunk.decode('utf-8', errors='ignore')).split("\n")
                buffer = lines[-1]

                for line in lines[:-1]:
                    line_count += 1
                    if line_count <= num_lines:
                        print(f"Line {line_count}: {line}")

                    try:
                        obj = json.loads(line)
                        if "author" in obj:
                            author_set.add(obj["author"])
                    except json.JSONDecodeError:
                        print(f"Failed to parse JSON at line {line_count}")

        print("\nTotal lines:", line_count)
        print("Total unique authors:", len(author_set))
        print("Author list:", list(author_set)[:10], "..." if len(author_set) > 10 else "")
    except Exception as e:
        print(f"Error processing file: {e}")

if __name__ == '__main__':
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Extract data for geolocated Reddit users.")
    parser.add_argument("--input_folder", required=True, help="Input folder containing submissions/, comments/, and authors CSV")
    parser.add_argument("--output_folder", required=True, help="Base output folder for processed files")
    parser.add_argument("--processes", type=int, default=4, help="Number of processes to use")
    parser.add_argument("--inspect", action='store_true', help="Inspect first lines of output")
    parser.add_argument("--num_lines", type=int, default=5, help="Lines to inspect")
    args = parser.parse_args()

    # Resolve paths to input subfolders and authors CSV
    submissions_input = os.path.join(args.input_folder, "submissions")
    comments_input = os.path.join(args.input_folder, "comments")
    authors_csv = os.path.join(args.input_folder, "o_2005-06to2023-12_unique_authors.csv")

    # Collect all matching .zst files
    input_files = []
    for folder in [submissions_input, comments_input]:
        if os.path.exists(folder):
            input_files.extend(
                os.path.join(folder, f)
                for f in os.listdir(folder)
                if re.match(r"^o_(RS_|RC_).*\.zst$", f)
            )

    log.info(f"Found {len(input_files)} files to process")

    # Load authors
    authors = load_authors_from_csv(authors_csv)

    # Build processing jobs (assign each file to appropriate output subfolder)
    jobs = []
    for file_path in input_files:
        if "/submissions/" in file_path.replace("\\", "/"):
            out_subfolder = os.path.join(args.output_folder, "submissions")
        elif "/comments/" in file_path.replace("\\", "/"):
            out_subfolder = os.path.join(args.output_folder, "comments")
        else:
            out_subfolder = args.output_folder
        jobs.append((file_path, out_subfolder, authors))

    # Process in parallel
    with multiprocessing.Pool(processes=args.processes) as pool:
        pool.starmap(process_file, jobs)

    log.info("All files processed")

    # Optional inspection
    if args.inspect:
        for folder in ["submissions", "comments"]:
            out_dir = os.path.join(args.output_folder, folder)
            if os.path.exists(out_dir):
                for file_name in os.listdir(out_dir):
                    if file_name.endswith(".zst"):
                        print(f"\nInspecting file: {file_name}")
                        inspect_zst_file(os.path.join(out_dir, file_name), num_lines=args.num_lines)
