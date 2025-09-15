r"""
This script processes monthly Reddit submissions and comments from topic based `.csv` files.
It conducts sentiment analysis using the National Research Council Emotion Lexicon (NCRLex).

Input:
- A base input folder containing by topic filtered .csv files:
    - files named p_{topic}_YYYY-MM.csv
    - (no author filtering file needed anymore)

Output:
- For each month: a CSV file saved in the output folder:
    - p_{topic}_sentiment_{YYYY-MM}.csv

Additional summary files and filtering steps removed.

Columns in p_{topic}_sentiment_{YYYY-MM}.csv:
- author: Reddit username
- subreddit: Subreddit where the post/comment was published
- body: Text content of the post or comment
- keyword: list of distinct keywords matched in each post
- total_keyword_num: Total number of keyword matches in the text
- total_word_count: Total number of words in body (i.e., comment)
- state: U.S. state associated with the author
- NRCL_count_*: counts of words in a comment associated with a specific emotion or sentiment (i.e., positive or negative) based on NCRLex (National Research Council Emotion Lexicon)
- NRCL_prop_*: proportion of words in a comment associated with a specific emotion or sentiment (i.e., positive or negative) in comparison to the total count of emotional words ranging form 0 to 1 based on NCRLex (National Research Council Emotion Lexicon)
- NRCL_freq_*: proportion of words in a comment associated with a specific emotion or sentiment (i.e., positive or negative) in comparison to the total number of words ranging form 0 to 1 based on NCRLex (National Research Council Emotion Lexicon)

Example usage:
python a_08_topicspecific_sentiment_NRCL.py `
--input_folder "C:/.../{topic}_output" `
--output_folder "C:/.../{topic}_sentiment" `
--topic {topic}

"""

import argparse
import csv
import logging.handlers
import os
import re
import pandas as pd
from langdetect import detect, LangDetectException
import nltk
from nrclex import NRCLex  # Added import for NRCLex
nltk.download('punkt', quiet=True)

# Define NRCLex emotions and sentiments
NRCL_EMOTIONS = ['anger', 'anticipation', 'disgust', 'fear', 'joy', 'sadness', 'surprise', 'trust']
NRCL_SENTIMENTS = ['positive', 'negative']

log = logging.getLogger("sentiment_logger")
log.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(levelname)s: %(message)s')
stream_handler = logging.StreamHandler()
stream_handler.setFormatter(formatter)
log.addHandler(stream_handler)

if not os.path.exists("logs"):
    os.makedirs("logs")

file_handler = logging.handlers.RotatingFileHandler(
    os.path.join("logs", "sentiment.log"), maxBytes=1024 * 1024 * 16, backupCount=5
)
file_handler.setFormatter(formatter)
log.addHandler(file_handler)


def process_month_files(args):
    files, output_folder, topic, _ = args

    year_month = os.path.basename(files[0])[len(f"p_{topic}_"):-4]
    log.info(f"Processing files for month: {year_month} and topic: {topic}")

    output_csv_path = os.path.join(output_folder, f"p_{topic}_sentiment_{year_month}.csv")

    csv_fields = [
        'author', 'subreddit', 'body', 'keyword', 'total_keyword_num', 'total_word_count', 'state',
        *[f'NRCL_count_{e}' for e in NRCL_EMOTIONS + NRCL_SENTIMENTS],
        *[f'NRCL_prop_{e}' for e in NRCL_EMOTIONS + NRCL_SENTIMENTS],
        *[f'NRCL_freq_{e}' for e in NRCL_EMOTIONS + NRCL_SENTIMENTS],
    ]

    raw_posts, matched_posts = 0, 0

    file_exists = os.path.exists(output_csv_path)
    with open(output_csv_path, 'a', newline='', encoding='utf-8') as out_csv:
        writer = csv.DictWriter(out_csv, fieldnames=csv_fields)
        if not file_exists:
            writer.writeheader()

        for file_path in files:
            try:
                df = pd.read_csv(file_path)
            except Exception as e:
                log.error(f"Could not read file {file_path}: {e}")
                continue

            for _, obj in df.iterrows():
                raw_posts += 1
                author = obj.get("author", "")
                body = obj.get("body", "")

                if not isinstance(body, str) or body.strip() in {"", "[removed]", "[deleted]"}:
                    continue

                total_word_count = len(body.split())
                
                try:
                    if detect(body) != 'en':
                        continue
                except LangDetectException:
                    continue

                # NRCLex emotion and sentiment analysis
                nrc_obj = NRCLex(body)
                counts = nrc_obj.raw_emotion_scores
                freqs = nrc_obj.affect_frequencies
                nrc_freqs = {e: freqs.get(e, 0) for e in NRCL_EMOTIONS + NRCL_SENTIMENTS}
                nrc_counts = {e: counts.get(e, 0) for e in NRCL_EMOTIONS + NRCL_SENTIMENTS}
                total_emotions = sum(nrc_counts[e] for e in NRCL_EMOTIONS) or 1
                total_sentiments = sum(nrc_counts[e] for e in NRCL_SENTIMENTS) or 1
                nrc_props = {}

                for e in NRCL_EMOTIONS:
                    nrc_props[e] = nrc_counts[e] / total_emotions
                for e in NRCL_SENTIMENTS:
                    nrc_props[e] = nrc_counts[e] / total_sentiments

                matched_posts += 1

                output_row = {
                    "author": author,
                    "subreddit": obj.get("subreddit", ""),
                    "body": body,
                    "keyword": obj.get("keyword", ""),
                    "total_keyword_num": obj.get("total_keyword_num", ""),
                    "total_word_count": total_word_count,
                    "state": obj.get("state", ""),
                }

                for e in NRCL_EMOTIONS + NRCL_SENTIMENTS:
                    output_row[f'NRCL_count_{e}'] = nrc_counts[e]
                    output_row[f'NRCL_prop_{e}'] = nrc_props[e]
                    output_row[f'NRCL_freq_{e}'] = nrc_freqs[e] 

                writer.writerow(output_row)

    log.info(f"Finished processing {year_month}: Matched posts = {matched_posts}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract sentiment-labeled posts from Reddit CSV files.")
    parser.add_argument("--input_folder", required=True, help="Folder containing CSV files.")
    parser.add_argument("--output_folder", required=True, help="Where to save the output.")
    parser.add_argument("--topic", required=True, help="Topic name to use as file prefix, e.g., 'Kendall'")
    parser.add_argument("--processes", type=int, default=4, help="Number of processes to run in parallel.")

    args = parser.parse_args()

    if not os.path.exists(args.output_folder):
        os.makedirs(args.output_folder)

    pattern = re.compile(rf"^p_{args.topic}_\d{{4}}-\d{{2}}\.csv$")

    csv_files = [
        os.path.join(args.input_folder, f)
        for f in os.listdir(args.input_folder)
        if pattern.match(f)
    ]

    if not csv_files:
        log.error(f"No matching CSV files with prefix p_{args.topic}_ found.")
        exit(1)

    grouped_files = {}
    for f in csv_files:
        year_month = os.path.basename(f)[len(f"p_{args.topic}_"):-4]
        grouped_files.setdefault(year_month, []).append(f)

    import multiprocessing
    with multiprocessing.Pool(processes=args.processes) as pool:
        pool.map(process_month_files, [
            (files, args.output_folder, args.topic, args.input_folder)
            for files in grouped_files.values()
        ])

    log.info("All processing completed.")