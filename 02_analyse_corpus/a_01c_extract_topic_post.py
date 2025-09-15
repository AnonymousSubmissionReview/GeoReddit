'''
    This script filters Reddit `.zst` compressed data files by:
      1. Keywords (from p_dictionary.txt, regex + label format)
      2. Subreddits (from o_subreddits.txt, case-sensitive exact match)
      3. Or both (union mode: match if either keywords or subreddits)

Input:

    ─ config_folder (required) should contain:
    p_dictionary.txt  (if filter_mode includes keywords)
        - One regex,label per line
        - Regex: valid Python regular expression
        - Label: descriptive keyword label (no commas inside label)
        - Example:
              (?i)\bAI\b,AI
              (?i)\bArtificial Intelligence\b,Artificial Intelligence
              deep\s*learning,Deep Learning

    o_subreddits.txt  (if filter_mode includes subreddits)
        - One subreddit name per line
        - Case-sensitive, must exactly match the subreddit in the dataset
        - Can be written as `r/Name` or just `Name` (script strips `r/`)
        - Example:
              r/Conservative
              technology
              MachineLearning

    ─ input_folder (required) should contain:

    (A) For All_Reddit data (no geolocation mapping):
        - submissions/ and comments/ folders
        - `.zst` files named like o_RS_YYYY-MM.zst or o_RC_YYYY-MM.zst
        - p_dictionary.txt  (if filter_mode includes keywords)
        - o_subreddits.txt  (if filter_mode includes subreddits)
        ❗ DO NOT place the author→state mapping CSV here for All_Reddit.

    (B) For GeoReddit data (with geolocation mapping):
        - submissions/ and comments/ folders
        - `.zst` files named like p_RS_YYYY-MM.zst or p_RC_YYYY-MM.zst
        - p_dictionary.txt  (if filter_mode includes keywords)
        - o_subreddits.txt  (if filter_mode includes subreddits)
        - o_2005-06to2023-12_filtered_authors.csv (author→state map)
          (must match the dataset's authors to enable state-level aggregation)

Output:
    - Monthly CSVs with filtered results: p_{topic}_{YYYY-MM}.csv
    - Total monthly statistics:
        p_{topic}_total_post.csv
        p_{topic}_total_author.csv
    - If author→state map provided:
        p_{topic}_author_month.csv  (state-level unique authors)
        p_{topic}_post_month.csv    (state-level posts)
    - Timing summary: p_summary_timing.csv

Notes:
    - In 'subreddits'-only mode, full_output is ignored (no keyword matching).
    - Language filtering: only keeps English posts (detected via langdetect).
    - State mapping is optional, but without it, state-level CSVs are skipped.

Command-line usage examples:

    # 1) Filter by keywords only
    python a_01c_extract_topic_post.py --input_folder /path/to/input \
        --output_folder /path/to/output \
        --topic AI \
        --filter_mode keywords \
        --full_output \
        --processes 32

    # 2) Filter by subreddits only
    python a_01c_extract_topic_post.py --input_folder /path/to/input \
        --output_folder /path/to/output \
        --topic politics \
        --filter_mode subreddits \
        --processes 32

    # 3) Filter by both keywords and subreddits (union logic)
    python a_01c_extract_topic_post.py --input_folder /path/to/input \
        --output_folder /path/to/output \
        --topic mixed \
        --filter_mode both \
        --processes 32
'''
from collections import defaultdict
import sys
import argparse
import csv
import json
import logging.handlers
import multiprocessing
import os
import zstandard
import hyperscan
import re
from langdetect import detect, LangDetectException
import pandas as pd
import time
import logging

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

US_STATES = [
    'AL','AK','AZ','AR','CA','CO','CT','DC','DE','FL','GA','HI','ID','IL','IN','IA','KS','KY','LA',
    'ME','MD','MA','MI','MN','MS','MO','MT','NE','NV','NH','NJ','NM','NY','NC','ND','OH','OK',
    'OR','PA','RI','SC','SD','TN','TX','UT','VT','VA','WA','WV','WI','WY'
]

# Load geolocated authors and their US states
def load_filtered_authors(authors_csv):
    if not os.path.exists(authors_csv):
        log.info(f"Author->state map not found: {authors_csv}")
        return None
    log.info(f"Loading filtered authors from {authors_csv}")
    authors = {}
    with open(authors_csv, 'r', encoding='utf-8') as file:
        reader = csv.DictReader(file)
        for row in reader:
            authors[row['author']] = row['state']
    return authors


# regex -> keyword label (optional). Validates regex compilation.
def load_regex_keywords(dictionary_file):
    if not os.path.exists(dictionary_file):
        log.info(f"Keyword dictionary not found: {dictionary_file}")
        return {}
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
    if not regex_keywords:
        return None, 0.0
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

#Return DISTINCT keyword labels matched in text, using Hyperscan callback.
#Safe when db is None (returns []).
#De-duplicates labels before returning.
def match_keywords(text, db, regex_keywords):
    if db is None:
        return []
    matched = []

    def on_match(id, from_, to_, flags, context):
        keyword = list(regex_keywords.values())[id - 1]
        matched.append(keyword)
        return hyperscan.HS_SUCCESS

    db.scan(text.encode('utf-8'), match_event_handler=on_match)
    return list(set(matched))

#Load subreddit names (one per line), case-sensitive, exact match.
#Allows 'r/Name' or 'Name' in file; internally store without 'r/'.
def load_subreddit_list(path):
    if not os.path.exists(path):
        log.info(f"No subreddit list found at {path}")
        return set()
    s = set()
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            name = line.strip()
            if not name:
                continue
            if name.startswith('r/'):
                name = name[2:]
            s.add(name)  # case preserved
    log.info(f"Loaded {len(s)} subreddits from {path}")
    return s

# Write a single line of statistics to a CSV file
def write_statistics(csv_file, headers, data):
    file_exists = os.path.exists(csv_file)
    with open(csv_file, 'a', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        if not file_exists:
            writer.writerow(headers)
        writer.writerow(data)


def is_post_file(base_name):
    return "_RS_" in base_name  # submissions; comments are _RC_

# Process all files for one specific month
def process_month_files(args):
    (files, output_folder, regex_keywords, subreddit_set, filter_mode,
     author_state_map, topic, full_output) = args

    year_month = os.path.basename(files[0])[5:-4]
    log.info(f"Processing {year_month} topic='{topic}' mode='{filter_mode}'")

    db, compile_time = initialize_hyperscan(regex_keywords)
    if db:
        log.info(f"Hyperscan compiled in {compile_time:.2f}s for {year_month}")

    start_month = time.time()

    # dynamic csv fields
    csv_fields = ['author', 'subreddit', 'body']
    has_state = author_state_map is not None and len(author_state_map) > 0
    if has_state:
        csv_fields.append('state')
    if full_output:
        csv_fields += ['keyword', 'total_keyword_num']

    output_csv_path = os.path.join(output_folder, f"p_{topic}_{year_month}.csv")
    statistics_total_author_file = os.path.join(output_folder, f"p_{topic}_total_author.csv")
    statistics_total_post_file = os.path.join(output_folder, f"p_{topic}_total_post.csv")
    file_exists = os.path.exists(output_csv_path)

    raw_posts, filtered_posts, matched_posts = 0, 0, 0
    raw_authors, filtered_authors_set, valid_authors, matched_authors = set(), set(), set(), set()

    with open(output_csv_path, 'a', newline='', encoding='utf-8') as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=csv_fields)
        if not file_exists:
            writer.writeheader()

        for file_path in files:
            base_name = os.path.basename(file_path)
            input_handle = open(file_path, 'rb')
            reader = zstandard.ZstdDecompressor(max_window_size=2 ** 31).stream_reader(input_handle)

            try:
                buffer = ''
                while True:
                    # Read chunk from compressed file
                    try:
                        # Read chunk from compressed file
                        chunk = reader.read(2 ** 27)
                    except zstandard.ZstdError as e:
                        log.error(f"[ZSTD] Decompression failed for {base_name}: {e}")
                        break

                    if not chunk:
                        break
                    lines = (buffer + chunk.decode('utf-8', errors='ignore')).split("\n")
                    buffer = lines[-1]

                    for line in lines[:-1]:
                        raw_posts += 1
                        try:
                            obj = json.loads(line)
                            author = obj.get("author", "")
                            if not author or author in {"[deleted]", "[removed]"}:
                                continue
                            raw_authors.add(author)

                            # if state map provided, filter authors not in map (keep behavior consistent)
                            if has_state:
                                if author not in author_state_map:
                                    filtered_authors_set.add(author)
                                    filtered_posts += 1
                                    continue
                                valid_authors.add(author)

                            # assemble body
                            if is_post_file(base_name):
                                title = (obj.get("title") or "").strip()
                                selftext = (obj.get("selftext") or "").strip()
                                if title in {"", "[removed]", "[deleted]"}:
                                    title = ""
                                if selftext in {"", "[removed]", "[deleted]"}:
                                    selftext = ""
                                body = (title + " " + selftext).strip()
                            else:
                                body = (obj.get("body") or "").strip()
                                if body in {"", "[removed]", "[deleted]"}:
                                    continue
                            if not body:
                                continue

                            do_kw = (filter_mode in ("keywords", "both"))
                            do_sr = (filter_mode in ("subreddits", "both"))

                            m_sr = False
                            m_kw = False
                            matched_keywords = []

                            if do_sr:
                                subr = obj.get("subreddit", "")
                                if subr and subr in subreddit_set:
                                    m_sr = True

                            if filter_mode == "keywords":
                                matched_keywords = match_keywords(body, db, regex_keywords)
                                m_kw = bool(matched_keywords)
                                matched = m_kw

                            elif filter_mode == "subreddits":
                                matched = m_sr

                            else:
                                if m_sr:
                                    matched = True
                                    if full_output and do_kw and db is not None:
                                        matched_keywords = match_keywords(body, db, regex_keywords)
                                else:
                                    if do_kw and db is not None:
                                        matched_keywords = match_keywords(body, db, regex_keywords)
                                        m_kw = bool(matched_keywords)
                                    matched = m_kw

                            if not matched:
                                continue

                            # english only
                            try:
                                if detect(body) != 'en':
                                    continue
                            except LangDetectException:
                                pass

                            matched_posts += 1
                            matched_authors.add(author)

                            row = {
                                "author": author,
                                "subreddit": obj.get("subreddit"),
                                "body": body
                            }
                            if has_state:
                                row["state"] = author_state_map.get(author)

                            if full_output and matched_keywords:
                                row["keyword"] = json.dumps(sorted(matched_keywords))
                                row["total_keyword_num"] = len(matched_keywords)

                            writer.writerow(row)

                        except json.JSONDecodeError:
                            log.warning(f"JSON decode error: {line[:50]}...")
                        except Exception as e:
                            log.error(f"Error processing line: {line[:50]}... Error: {e}")
            finally:
                reader.close()

    elapsed_month = time.time() - start_month
    total_posts = raw_posts - filtered_posts
    log.info(f"Finished {year_month}: raw={raw_posts}, filtered={filtered_posts}, matched={matched_posts}, time={elapsed_month:.2f}s")

    # per-month timing
    with open(os.path.join(output_folder, f"p_timing_{year_month}.json"), "w") as f:
        json.dump({"month": year_month, "compile_time": compile_time, "process_time": elapsed_month}, f)

    # totals
    write_statistics(
        statistics_total_post_file,
        ["time", "raw_posts", "filtered_posts", "total_posts", "matched_posts", "post_proportion"],
        [year_month, raw_posts, filtered_posts, total_posts, matched_posts,
         matched_posts / total_posts if total_posts else 0]
    )
    write_statistics(
        statistics_total_author_file,
        ["time", "unique_authors_raw", "unique_authors_filtered", "unique_authors_total",
         "unique_authors_matched", "user_proportion"],
        [year_month, len(raw_authors), len(filtered_authors_set), len(valid_authors), len(matched_authors),
         len(matched_authors) / len(valid_authors) if len(valid_authors) else len(matched_authors) / len(raw_authors)]
    )

    # per-state monthly (only if state available)
    has_state = author_state_map is not None and len(author_state_map) > 0
    if has_state:
        try:
            df = pd.read_csv(output_csv_path)
            state_author_counts = (
                df.groupby('state')['author'].nunique()
                  .reindex(US_STATES, fill_value=0).to_dict()
            )
            state_post_counts = (
                df['state'].value_counts()
                  .reindex(US_STATES, fill_value=0).to_dict()
            )
            write_statistics(os.path.join(output_folder, f"p_{topic}_author_month.csv"),
                ["time"] + US_STATES,
                [year_month] + [state_author_counts[s] for s in US_STATES])
            write_statistics(os.path.join(output_folder, f"p_{topic}_post_month.csv"),
                ["time"] + US_STATES,
                [year_month] + [state_post_counts[s] for s in US_STATES])
            log.info("State-level statistics written.")
        except Exception as e:
            log.warning(f"State-level aggregation failed for {year_month}: {e}")
    else:
        log.info("No author->state map provided. Skip state-level statistics.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Filter AllReddit/GeoReddit by keywords and/or subreddits, output monthly CSVs.")
    parser.add_argument("--input_folder", required=True,
                        help="Folder containing submissions/, comments/. "
                             "If available, o_2005-06to2023-12_filtered_authors.csv for state mapping.")
    parser.add_argument("--config_folder", required=True,
                        help="Folder containing p_dictionary.txt and/or o_subreddits.txt")
    parser.add_argument("--output_folder", required=True, help="Output folder for CSVs and logs.")
    parser.add_argument("--topic", required=True, help="Topic name for output file prefixes, e.g., 'AI'")
    parser.add_argument("--processes", type=int, default=4, help="Number of parallel processes.")
    parser.add_argument("--full_output", action="store_true",
                        help="Write matched keyword list + distinct count per post.")
    parser.add_argument("--filter_mode", default="keywords",
                        choices=["keywords", "subreddits", "both"],
                        help="Use keywords only, subreddits only, or both (OR logic).")

    args = parser.parse_args()
    if args.filter_mode == "subreddits" and args.full_output:
        log.warning("full_output ignored in subreddits-only mode (no keyword matching).")
        args.full_output = False

    os.makedirs(args.output_folder, exist_ok=True)

    # Load inputs from input_folder
    dictionary_file = os.path.join(args.config_folder, "p_dictionary.txt")
    regex_keywords = load_regex_keywords(dictionary_file) if args.filter_mode in ("keywords", "both") else {}

    subreddit_file = os.path.join(args.config_folder, "o_subreddits.txt")
    subreddit_set = load_subreddit_list(subreddit_file) if args.filter_mode in ("subreddits", "both") else set()

    authors_csv = os.path.join(args.input_folder, "o_2005-06to2023-12_filtered_authors.csv")
    author_state_map = load_filtered_authors(authors_csv)

    # Collect .zst files
    zst_files = []
    for subfolder in ["submissions", "comments"]:
        folder = os.path.join(args.input_folder, subfolder)
        if not os.path.exists(folder):
            log.warning(f"Subfolder not found: {folder}")
            continue
        for f in os.listdir(folder):
            if (f.startswith("p_RS_") or f.startswith("p_RC_") or
                f.startswith("o_RS_") or f.startswith("o_RC_")) and f.endswith(".zst"):
                zst_files.append(os.path.join(folder, f))

    if not zst_files:
        log.error("No .zst files found under submissions/ or comments/.")
        sys.exit(1)

    # Group files by month
    process_args = defaultdict(list)
    for file in zst_files:
        base = os.path.basename(file)
        year_month = base[5:-4]  # works for both p_* and o_* names
        process_args[year_month].append(file)

    # Validate filters
    if args.filter_mode == "keywords" and not regex_keywords:
        log.error("filter_mode=keywords but no p_dictionary.txt found or empty.")
        sys.exit(1)
    if args.filter_mode == "subreddits" and not subreddit_set:
        log.error("filter_mode=subreddits but no o_subreddits.txt found or empty.")
        sys.exit(1)
    if args.filter_mode == "both" and not (regex_keywords or subreddit_set):
        log.error("filter_mode=both but neither p_dictionary.txt nor o_subreddits.txt is available.")
        sys.exit(1)

    script_start = time.time()

    # Process in parallel by month
    work_items = [
        (files, args.output_folder, regex_keywords, subreddit_set, args.filter_mode,
         author_state_map, args.topic, args.full_output)
        for _, files in sorted(process_args.items())
    ]
    with multiprocessing.Pool(processes=args.processes) as pool:
        pool.map(process_month_files, work_items)

    # timing summary
    timing_files = [
        f for f in os.listdir(args.output_folder)
        if f.startswith("p_timing_") and f.endswith(".json")
    ]
    timings = []
    for tf in sorted(timing_files):
        with open(os.path.join(args.output_folder, tf), "r") as f:
            timings.append(json.load(f))
    if timings:
        total_compile = sum(t.get("compile_time", 0.0) for t in timings)
        total_process = sum(t.get("process_time", 0.0) for t in timings)
        final_runtime = time.time() - script_start

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

    log.info("All files processed successfully.")

    for tf in timing_files:
        try:
            os.remove(os.path.join(args.output_folder, tf))
            log.info(f"Deleted temporary timing file: {tf}")
        except Exception as e:
            log.warning(f"Failed to delete {tf}: {e}")
