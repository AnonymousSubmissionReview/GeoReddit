"""
Count monthly keyword occurrences and export co‐occurrence network and detailed summaries
with case-preserving dictionary labels.

Input:
  - o_dictionary.txt
      - one keyword or phrase per line (may include “[ENTITY]” prefixes).
  - p_{topic}_YYYY-MM.csv files
      - one per month, each row has a “keyword” column storing a Python list of matched keywords.

Output:
  - p_keyword_counts.csv
      - rows： months (sorted, plus a final "TOTAL" row);
      - columns: each original dictionary phrase;
      -  values: number of posts in that month matching the phrase.
  - p_cooccurrence_edges.csv
      - columns [Source, Target, Weight];
      - each row = undirected edge between two phrases that co-occurred in the same post,
      - Weight = number of posts where both appear.
  - p_keyword_totals_summary.csv
      - rows = five meta rows then one per phrase;
      - columns = [phrase, total, percentage];
      - meta rows:
          • Average    : mean of all "total" values
          • Median     : median of all "total" values
          • Keyword count : number of phrases with total > 0
          • Zero count : number of phrases with total = 0
          • Zero list  : semicolon-separated list of phrases with total = 0
        data rows:
          • phrase     : dictionary phrase
          • total      : total matched posts across all months
          • percentage : total / (TOTAL posts count)
  - p_keyword_degree_summary.csv
      : rows = five meta rows then one per phrase;
        columns = [phrase, degree];
        meta rows as above (but for "degree");
        data rows:
          • phrase : dictionary phrase
          • degree : number of distinct co-occurrence partners

Example usage:
    python a_06_dic_stat.py --input_folder /path/to/input_dir --output_folder /path/to/output_dir --topic AI
"""

import os
import re
import csv
import sys
import argparse
import ast
from itertools import combinations
from collections import Counter
import pandas as pd

# work around CSV field size limits
csv.field_size_limit(sys.maxsize)

#Load dictionary file and create case-preserving mappings
def load_dictionary(dict_path):

    kws_orig = []  # Original phrases in dictionary order
    seen = set()  # Track lowercase versions for deduplication
    lower2orig = {}  # Map lowercase to original case

    with open(dict_path, 'r', encoding='utf-8') as f:
        for line in f:
            kw = line.strip()
            # Remove [ENTITY] prefix if present
            if kw.startswith('[ENTITY]'):
                kw = kw[len('[ENTITY]'):].strip()
            if not kw:
                continue
            lc = kw.lower()
            if lc not in seen:
                seen.add(lc)
                kws_orig.append(kw)
                lower2orig[lc] = kw
        return kws_orig, lower2orig

#Count keyword occurrences and co-occurrences in a monthly CSV file
def count_and_cooccurrence(csv_path, dict_set, cooc_ctr):

    ctr = Counter()  # Count individual keyword occurrences
    with open(csv_path, 'r', encoding='utf-8', newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                # Parse the keyword list from string representation
                kws_raw = ast.literal_eval(row.get('keyword', '[]'))
            except Exception:
                continue
            # Convert to lowercase for matching
            kws_lc = [w.lower() for w in kws_raw if isinstance(w, str)]
            # Find matches with dictionary
            matched = [w for w in kws_lc if w in dict_set]
            # Update individual counts
            ctr.update(matched)
            # Count co-occurrences for pairs
            if len(matched) > 1:
                # Generate all unique sorted pairs
                for a, b in combinations(sorted(set(matched)), 2):
                    cooc_ctr[(a, b)] += 1
    return ctr

#Create metadata rows for summary files
def write_meta(meta_rows, stat_name):
    # meta_rows: dict of stat name -> value
    entries = [
        ('Average', meta_rows['avg']),
        ('Median', meta_rows['med']),
        ('Keyword count', meta_rows['n_kw']),
        ('Zero count', meta_rows['n_zero']),
        ('Zero list', ";".join(meta_rows['zero_list']))
    ]
    df = pd.DataFrame(entries, columns=['phrase', stat_name])
    return df


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="Count monthly keywords and export co‐occurrence network"
    )
    parser.add_argument('--input_folder', required=True,
                        help="Directory containing p_{topic}_YYYY-MM.csv and o_dictionary.txt")
    parser.add_argument('--output_folder', required=True, help="Directory to write summary and co-occurrence files")
    parser.add_argument('--topic', required=True, help="Topic prefix in filenames, e.g. AI")
    args = parser.parse_args()

    # Load dictionary file
    dict_path = os.path.join(args.input_folder, 'o_dictionary.txt')
    if not os.path.isfile(dict_path):
        print(f"Cannot find dictionary file: {dict_path}")
        return
    kws_orig, lower2orig = load_dictionary(dict_path)
    dict_set = set(lower2orig.keys())  # Set of lowercase keywords for fast lookup

    # Find all monthly data files matching pattern
    pat = re.compile(rf"^p_{re.escape(args.topic)}_(\d{{4}}-\d{{2}})\.csv$")
    month_files = {m.group(1): os.path.join(args.input_folder, fn)
                   for fn in os.listdir(args.input_folder)
                   if (m := pat.match(fn))}
    if not month_files:
        print("No matching p_{topic}_YYYY-MM.csv files found")
        return

    # Initialize counters
    month_counts = {}  # Monthly counts per keyword
    cooc_ctr = Counter()  # Co-occurrence counts for pairs

    # Process each monthly file
    for month in sorted(month_files):
        ctr = count_and_cooccurrence(month_files[month], dict_set, cooc_ctr)
        month_counts[month] = ctr

    # Calculate totals across all months
    total_ctr = Counter()
    for ctr in month_counts.values():
        total_ctr.update(ctr)
    month_counts['TOTAL'] = total_ctr

    # Ensure output directory exists
    os.makedirs(args.output_folder, exist_ok=True)

    # 1) Write monthly summary counts
    summary_path = os.path.join(args.output_folder, 'p_keyword_counts.csv')
    with open(summary_path, 'w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        # Header row with original phrases
        writer.writerow(['month'] + kws_orig)
        # Write data for each month (with TOTAL last)
        for month in sorted(month_counts, key=lambda x: (x != 'TOTAL', x)):
            ctr = month_counts[month]
            # Create row with counts for all keywords (0 if not present)
            row = [month] + [ctr.get(kw.lower(), 0) for kw in kws_orig]
            writer.writerow(row)
    print(f"Keyword summary written to: {summary_path}")

    # 2) Write co-occurrence edges (network data)
    graph_path = os.path.join(args.output_folder, 'p_cooccurrence_edges.csv')
    with open(graph_path, 'w', encoding='utf-8', newline='') as gf:
        writer = csv.writer(gf)
        writer.writerow(['Source', 'Target', 'Weight'])
        # Write each co-occurrence pair with original case
        for (a_lc, b_lc), w in cooc_ctr.items():
            writer.writerow([lower2orig[a_lc], lower2orig[b_lc], w])
    print(f"Co‐occurrence edges written to: {graph_path}")

    # 3) Generate totals summary with metadata
    # Load counts data
    df = pd.read_csv(summary_path)
    # Get just the TOTAL row and transpose (phrases become rows)
    tot = df[df['month'] == 'TOTAL'].drop(columns=['month']).T
    tot.columns = ['total']
    tot.index.name = 'phrase'
    tot = tot.reset_index()

    # Calculate degree (number of co-occurrence partners)
    partners = Counter()
    for (a_lc, b_lc) in cooc_ctr:
        partners[a_lc] += 1
        partners[b_lc] += 1
    tot['degree'] = tot['phrase'].str.lower().map(lambda x: partners.get(x, 0))

    # Create metadata for totals
    meta_total = {
        'avg': tot['total'].mean(),
        'med': tot['total'].median(),
        'n_kw': len(tot),
        'n_zero': (tot['total'] == 0).sum(),
        'zero_list': tot.loc[tot['total'] == 0, 'phrase'].tolist()
    }
    df_meta_total = write_meta(meta_total, 'total')
    # Calculate percentage of total posts (hardcoded denominator)
    tot['percentage'] = tot['total'] / 5917277590
    # Sort by total count descending
    df_totals_sorted = tot.sort_values('total', ascending=False)[['phrase', 'total', 'percentage']]
    # Combine metadata and data
    df_totals_summary = pd.concat([df_meta_total, df_totals_sorted], ignore_index=True)
    out_totals = os.path.join(args.output_folder, 'p_keyword_totals_summary.csv')
    df_totals_summary.to_csv(out_totals, index=False, encoding='utf-8-sig')
    print(f"✅ Totals summary written to: {out_totals}")

    # 4) Generate degree summary with metadata
    meta_degree = {
        'avg': tot['degree'].mean(),
        'med': tot['degree'].median(),
        'n_kw': len(tot),
        'n_zero': (tot['degree'] == 0).sum(),
        'zero_list': tot.loc[tot['degree'] == 0, 'phrase'].tolist()
    }
    df_meta_degree = write_meta(meta_degree, 'degree')
    # Sort by degree descending
    df_degree_sorted = tot.sort_values('degree', ascending=False)[['phrase', 'degree']]
    # Combine metadata and data
    df_degree_summary = pd.concat([df_meta_degree, df_degree_sorted], ignore_index=True)
    out_degree = os.path.join(args.output_folder, 'p_keyword_degree_summary.csv')
    df_degree_summary.to_csv(out_degree, index=False, encoding='utf-8-sig')
    print(f"Degree summary written to: {out_degree}")


if __name__ == '__main__':
    main()