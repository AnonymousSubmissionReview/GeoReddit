"""
This script is a faster version of `a_01a_extract_topic_post.py`.
Unlike `a_01a_extract_topic_post.py`, it does NOT count how many times each keyword appears in a post.
By default, it extracts only topic-matched posts with basic metadata.
If `--full_output` is enabled, it will additionally include:
- `keyword`: list of distinct keywords matched in each post
- `total_keyword_num`: count of distinct keyword types matched

Input:
- A base input folder containing:
    - submissions/       (with files named p_RS_YYYY-MM.zst)
    - comments/          (with files named p_RC_YYYY-MM.zst)
    - o_2005-06to2023-12_filtered_authors.csv
        (it's formed by unique users with ratio2 > 1 from p_2005-06to2023-12_ratios_postnum.csv)
        (mapping Reddit usernames to U.S. states; must have columns 'author' and 'state')
    - p_dictionary.txt
        (dictionary of regex patterns and their keyword labels, one per line: regex,keyword, from c_00)

Output:
- For each month: a CSV file saved in the output folder:
    - p_{topic}_{YYYY-MM}.csv
      (all matched posts with detailed metadata)

Additional summary files saved in the output folder:
- p_{topic}_total_post.csv
    (monthly counts and proportions of processed and matched posts)
- p_{topic}_total_author.csv
    (monthly counts and proportions of unique authors)
- p_{topic}_author_month.csv
    (number of unique authors per state per month)
- p_{topic}_post_month.csv
    (number of matched posts per state per month)

Columns in p_{topic}_{YYYY-MM}.csv:
- author: Reddit username
- subreddit: Subreddit where the post/comment was published
- body: Text content of the post or comment
- keyword: JSON list of distinct keywords matched (only if --full_output is set)
- total_keyword_num: Count of distinct keyword types matched (only if --full_output is set)
- state: U.S. state associated with the author

Columns in p_{topic}_total_post.csv:
- time: Year and month (e.g., "2023-06")
- raw_posts: Total number of posts processed (before filtering)
- filtered_posts: Number of posts excluded because the author had ratio2 <= 1
- total_posts: Remaining posts after filtering
- matched_posts: Number of posts where at least one keyword was matched
- post_proportion: Proportion of matched posts (matched_posts / total_posts)

Columns in p_{topic}_total_author.csv:
- time: Year and month
- unique_authors_raw: Total number of unique authors before filtering
- unique_authors_filtered: Number of authors excluded because ratio2 <= 1
- unique_authors_total: Remaining unique authors after filtering
- unique_authors_matched: Number of unique authors with at least one matched post
- user_proportion: Proportion of matched authors (unique_authors_matched / unique_authors_total)

Columns in p_{topic}_author_month.csv:
- time: Year and month
- AL - WY: For each U.S. state, number of unique authors with matched posts

Columns in p_{topic}_post_month.csv:
- time: Year and month
- AL - WY: For each U.S. state, total number of matched posts

Additional diagnostic outputs:
- p_timing_{YYYY-MM}.json:
     (stores per-month processing time and regex compile time, with keys:
      'month', 'compile_time', 'process_time')
 - p_summary_timing.csv:
    (a table showing compile and process time for all months processed)

Example usage:
python a_01b_extract_topic_post.py --input_folder "C:/data/reddit_input" --output_folder "C:/data/reddit_output" --topic AI --full_output

"""


import argparse
import csv
import json
import logging.handlers
import multiprocessing
import os
import zstandard
import hyperscan
from collections import Counter
import re
from langdetect import detect, LangDetectException
import pandas as pd
import time


# Setup logging
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

# Load geolocated authors and their US states
def load_filtered_authors(authors_csv):
    log.info(f"Loading filtered authors from {authors_csv}")
    authors = {}
    with open(authors_csv, 'r', encoding='utf-8') as file:
        reader = csv.DictReader(file)
        for row in reader:
            authors[row['author']] = row['state']
    return authors

# Load regex patterns from dictionary file
def load_regex_keywords(dictionary_file):
    regex_keywords = {}
    with open(dictionary_file, 'r', encoding='utf-8') as f:
        for line_number, line in enumerate(f, start=1):
            line = line.strip()
            if line:
                try:
                    regex, keyword = line.split(',', 1)
                    regex = regex.strip()
                    keyword = keyword.strip()
                    re.compile(regex)
                    regex_keywords[regex] = keyword
                except re.error as e:
                    log.error(f"Invalid regex at line {line_number}: {regex} - Error: {e}")
                except ValueError:
                    log.error(f"Invalid format at line {line_number}: {line}")
    return regex_keywords

# Initialize Hyperscan engine
def initialize_hyperscan(regex_keywords):
    start_compile = time.time()
    db = hyperscan.Database()
    expressions = [regex.encode('utf-8') for regex in regex_keywords.keys()]
    ids = list(range(1, len(expressions) + 1))
    db.compile(
        expressions=expressions,
        ids=ids,
        elements=len(expressions),
        flags=[hyperscan.HS_FLAG_CASELESS] * len(expressions)
    )
    elapsed_compile = time.time() - start_compile
    return db, elapsed_compile

# Run Hyperscan matching on text
def match_keywords(text, db, regex_keywords):
    matched = []

    def on_match(id, from_, to_, flags, context):
        keyword = list(regex_keywords.values())[id - 1]
        matched.append(keyword)
        return hyperscan.HS_SUCCESS

    db.scan(text.encode('utf-8'), match_event_handler=on_match)
    return matched

# Write a single line of statistics to a CSV file
def write_statistics(csv_file, headers, data):
    file_exists = os.path.exists(csv_file)
    with open(csv_file, 'a', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        if not file_exists:
            writer.writerow(headers)
        writer.writerow(data)

# Process all files for one specific month
def process_month_files(args):
    files, output_folder, regex_keywords, filtered_authors, topic, full_output = args

    year_month = os.path.basename(files[0])[5:-4]
    log.info(f"Processing files for month: {year_month} with topic '{topic}'")

    # Initialize regex matcher
    db, compile_time = initialize_hyperscan(regex_keywords)
    log.info(f"Hyperscan database compiled in {compile_time:.2f} seconds.")

    start_month = time.time()

    output_csv_path = os.path.join(output_folder, f"p_{topic}_{year_month}.csv")
    statistics_total_author_file = os.path.join(output_folder, f"p_{topic}_total_author.csv")
    statistics_total_post_file = os.path.join(output_folder, f"p_{topic}_total_post.csv")
    file_exists = os.path.exists(output_csv_path)

    csv_fields = ['author', 'subreddit', 'body', 'state']
    
    if full_output:
        csv_fields.insert(3, 'keyword')
        csv_fields.insert(4, 'total_keyword_num')

    us_states = [
        'AL','AK','AZ','AR','CA','CO','CT','DC','DE','FL','GA','HI','ID','IL','IN','IA','KS','KY','LA',
        'ME','MD','MA','MI','MN','MS','MO','MT','NE','NV','NH','NJ','NM','NY','NC','ND','OH','OK',
        'OR','PA','RI','SC','SD','TN','TX','UT','VT','VA','WA','WV','WI','WY'
    ]

    raw_posts, filtered_posts, matched_posts = 0, 0, 0
    raw_authors, filtered_authors_set, valid_authors, matched_authors = set(), set(), set(), set()

    with open(output_csv_path, 'a', newline='', encoding='utf-8') as csv_file:
        csv_writer = csv.DictWriter(csv_file, fieldnames=csv_fields)
        if not file_exists:
            csv_writer.writeheader()

        for file_path in files:
            base_name = os.path.basename(file_path)
            #output_zst_path = os.path.join(output_folder, f"p_{topic}_{base_name}")
            input_handle = open(file_path, 'rb')
            reader = zstandard.ZstdDecompressor(max_window_size=2**31).stream_reader(input_handle)
            #writer = zstandard.ZstdCompressor().stream_writer(open(output_zst_path, 'wb'))

            try:
                buffer = ''
                while True:
                    # Read chunk from compressed file
                    chunk = reader.read(2**27)
                    if not chunk:
                        break
                    lines = (buffer + chunk.decode('utf-8', errors='ignore')).split("\n")
                    buffer = lines[-1]

                    for line in lines[:-1]:
                        raw_posts += 1
                        try:
                            obj = json.loads(line)
                            author = obj.get("author", "")
                            raw_authors.add(author)
                            
                            # Skip authors not in filtered set
                            if author not in filtered_authors:
                                filtered_authors_set.add(author)
                                filtered_posts += 1
                                continue

                            valid_authors.add(author)
                            
                            # Extract body text
                            body = ""
                            if "RS_" in base_name:
                                title = obj.get("title", "").strip()
                                selftext = obj.get("selftext", "").strip()
                                if title in {"", "[removed]", "[deleted]"}:
                                    title = ""
                                if selftext in {"", "[removed]", "[deleted]"}:
                                    selftext = ""
                                body = (title + " " + selftext).strip()
                            elif "RC_" in base_name:
                                body = obj.get("body", "").strip()
                                if body in {"", "[removed]", "[deleted]"}:
                                    continue

                            if not body:
                                continue
                            
                            # Run keyword matching
                            matched_keywords = match_keywords(body, db, regex_keywords)
                            if not matched_keywords:
                                continue
                            
                            # Filter out non-English posts
                            try:
                                if detect(body) != 'en':
                                    continue
                            except LangDetectException:
                                log.warning(f"Language detection failed for body: {body[:50]}...")
                            
                            # Write matching post to CSV
                            state = filtered_authors[author]
                            subreddit = obj.get("subreddit", None)

                            matched_posts += 1
                            matched_authors.add(author)
                            
                            output_row = {"author": author,"subreddit": subreddit,"body": body,"state": state}

                            if full_output:
                                output_row["keyword"] = json.dumps(list(set(matched_keywords)))
                                output_row["total_keyword_num"] = len(set(matched_keywords))

                            csv_writer.writerow(output_row)

                            
                            # Write to the zst file
                            #obj["keyword"] = json.dumps(list(set(matched_keywords)))
                            #obj["total_keyword_num"] = len(set(matched_keywords))
                            #writer.write((json.dumps(obj) + "\n").encode('utf-8'))

                        except json.JSONDecodeError:
                            log.warning(f"JSON decode error: {line[:50]}...")
                        except Exception as e:
                            log.error(f"Error processing line: {line[:50]}... Error: {e}")
            finally:
                reader.close()

    elapsed_month = time.time() - start_month
    log.info(f"Finished processing {year_month} in {elapsed_month:.2f} seconds.")

    #calculate running time for each month and saved in json file
    timing = {
        "month": year_month,
        "compile_time": compile_time,
        "process_time": elapsed_month
    }
    with open(os.path.join(output_folder, f"p_timing_{year_month}.json"), "w") as f:
        json.dump(timing, f)
    
    # Log summary stats
    total_posts = raw_posts - filtered_posts
    log.info(f"Completed processing for month: {year_month}")
    log.info(f"Raw posts: {raw_posts}, Filtered: {filtered_posts}, Matched: {matched_posts}")
    log.info(
        f"Unique authors (raw): {len(raw_authors)}, Filtered: {len(filtered_authors_set)}, "
        f"Total: {len(valid_authors)}, Matched: {len(matched_authors)}"
    )

    # Write total post statistics
    write_statistics(statistics_total_post_file,
        ["time", "raw_posts", "filtered_posts", "total_posts", "matched_posts", "post_proportion"],
        [year_month, raw_posts, filtered_posts, total_posts, matched_posts,
         matched_posts / total_posts if total_posts else 0])

    # Write total author statistics
    write_statistics(statistics_total_author_file,
        ["time", "unique_authors_raw", "unique_authors_filtered", "unique_authors_total",
         "unique_authors_matched", "user_proportion"],
        [year_month, len(raw_authors), len(filtered_authors_set), len(valid_authors), len(matched_authors),
         len(matched_authors) / len(valid_authors) if len(valid_authors) else 0])

    # Compute state-level stats
    df = pd.read_csv(output_csv_path)
    state_author_counts = df.groupby('state')['author'].nunique().reindex(us_states, fill_value=0).to_dict()
    state_post_counts = df['state'].value_counts().reindex(us_states, fill_value=0).to_dict()

    # Write per-state statistics
    write_statistics(os.path.join(output_folder, f"p_{topic}_author_month.csv"),
        ["time"] + us_states,
        [year_month] + [state_author_counts[state] for state in us_states])

    write_statistics(os.path.join(output_folder, f"p_{topic}_post_month.csv"),
        ["time"] + us_states,
        [year_month] + [state_post_counts[state] for state in us_states])


    log.info("State-level statistics written successfully.")
    

if __name__ == '__main__':
    script_start = time.time()
    parser = argparse.ArgumentParser(
        description="Process GeoReddit files to extract topic-related posts and generate statistics."
    )
    parser.add_argument("--input_folder", required=True, help="Input folder containing submissions/, comments/, authors CSV, and dictionary file.")
    parser.add_argument("--output_folder", required=True, help="Output folder for CSVs and logs.")
    parser.add_argument("--topic", required=True, help="Topic name for output file prefixes, e.g., 'AI'")
    parser.add_argument("--processes", type=int, default=4, help="Number of parallel processes.")
    parser.add_argument("--full_output", action="store_true", help="Enable full output with counting the number of distinct keyword categories detected in each post.")

    args = parser.parse_args()

    # Load supporting files
    authors_csv = os.path.join(args.input_folder, "o_2005-06to2023-12_filtered_authors.csv")
    dictionary_file = os.path.join(args.input_folder, "p_dictionary.txt")

    if not os.path.exists(authors_csv):
        log.error("Missing o_2005-06to2023-12_filtered_authors.csv")
        exit(1)
    if not os.path.exists(dictionary_file):
        log.error("Missing o_dictionary.txt")
        exit(1)

    filtered_authors = load_filtered_authors(authors_csv)
    regex_keywords = load_regex_keywords(dictionary_file)
    
    # Gather all ZST files
    zst_files = []
    for subfolder in ["submissions", "comments"]:
        folder = os.path.join(args.input_folder, subfolder)
        if os.path.exists(folder):
            zst_files += [os.path.join(folder, f) for f in os.listdir(folder) if f.startswith("p_") and f.endswith(".zst")]
        else:
            log.warning(f"Subfolder not found: {folder}")

    if not zst_files:
        log.error("No .zst files found.")
        exit(1)
    
    # Group files by month
    process_args = {}
    for file in zst_files:
        year_month = os.path.basename(file)[5:-4]
        process_args.setdefault(year_month, []).append(file)

    # Process in parallel
    with multiprocessing.Pool(processes=args.processes) as pool:
        pool.map(process_month_files, [
            (files, args.output_folder, regex_keywords, filtered_authors, args.topic, args.full_output)
            for files in process_args.values()
        ])

    log.info("All files processed successfully.")

    # gather all timing
    timing_files = [
        f for f in os.listdir(args.output_folder)
        if f.startswith("p_timing_") and f.endswith(".json")
    ]

    if timing_files:
        log.info("========== Runtime Summary ==========")
        timings = []
        for tf in sorted(timing_files):
            with open(os.path.join(args.output_folder, tf), "r") as f:
                data = json.load(f)
                timings.append(data)
        print("{:<10} {:>15} {:>15}".format("Month", "Compile(s)", "Process(s)"))
        for t in timings:
            print("{:<10} {:>15.2f} {:>15.2f}".format(
                t["month"], t["compile_time"], t["process_time"]
            ))

    total_compile = sum(t["compile_time"] for t in timings)
    total_process = sum(t["process_time"] for t in timings)

    final_runtime = time.time() - script_start

    log.info(f"Total script runtime: {final_runtime:.2f} seconds.")

    csv_path = os.path.join(args.output_folder, "p_summary_timing.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["month", "compile_time", "process_time"])
        for t in timings:
            writer.writerow([t["month"], f"{t['compile_time']:.2f}", f"{t['process_time']:.2f}"])
        writer.writerow([])
        writer.writerow(["TOTAL", f"{total_compile:.2f}", f"{total_process:.2f}"])
        writer.writerow(["SCRIPT_RUNTIME(s)", f"{final_runtime:.2f}"])

    log.info(f"Timing summary written to: {csv_path}")
