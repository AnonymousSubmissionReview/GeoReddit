r"""
This script computes dummy variables when marijuana was voted legal and when possession was legal for each state in the US. 
This script creates several dummy variables reflecting different coding schemes reflecting the month that marijuana was voted legal and when possession itself was legal.
Coding can be month or quarterwise.

## Overview dummy coding

vote_01:
  0 - Before vote or no vote
  1 - After vote (vote month or later)

vote_012:
  0 - Before vote
  1 - Vote month
  2 - After vote

poss_01:
  0 - Before possession or no possession date
  1 - After possession becomes legal (possession month or later)

poss_012:
  0 - Before possession
  1 - Possession month
  2 - After possession

mix_combo:
  0 - No legalization event yet
  1 - Vote month only (possession not yet legal)
  2 - After vote, before possession
  3 - Possession month  # if possesion is same month as vote possession is prioritized
  4 - After possession

vote_poss_012:
  0 - Before vote
  1 - After vote, before possession ## including month of voting
  2 - After possession   # including month of possession legal, # if possesion is same month as vote possession is prioritized

- ouput files
    - monthly and quarterly .csv-file for each coding scheme (monthly/quarterly_{coding}.csv) and a merged file including all coding schemes (monthly/quarterly_merged.csv)

- source dates of elections for Recreational (adult use) legalization: https://www.leafly.com/news/politics/when-did-your-state-legalize-marijuana
  Barcott, B. (2023, January 10). When did your state legalize marijuana? Leafly. Retrieved September 9, 2025, from https://www.leafly.com/news/politics/when-did-your-state-legalize-marijuana
  and https://ballotpedia.org/History_of_marijuana_ballot_measures_and_laws?utm_source=chatgpt.com for recent legalization efforts
  History of marijuana ballot measures and laws. (2025, June). Ballotpedia. Retrieved September 9, 2025, from https://ballotpedia.org/History_of_marijuana_ballot_measures_and_laws 

Example usage:
python "...\e_v2_00_legalization_dummy_variables.py" `
  --output_folder "C:\...\Dummy_coding_legalization" `

"""

import csv
from datetime import datetime
from dateutil.relativedelta import relativedelta
import argparse
import os
import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument('--output_folder', type=str, required=True, help="Path to output folder for saving CSVs")
args = parser.parse_args()

# All US states + DC with election and possession dates
state_info = {
    "Alabama": {"abbr":"AL", "election":None,"possession":None},
    "Alaska": {"abbr":"AK", "election":"2014-11-04","possession":"2014-11-05"},
    "Arizona": {"abbr":"AZ", "election":"2020-11-03","possession":"2020-11-30"},
    "Arkansas": {"abbr":"AR", "election":None,"possession":None},
    "California": {"abbr":"CA", "election":"2016-11-08","possession":"2016-11-09"},
    "Colorado": {"abbr":"CO", "election":"2012-11-06","possession":"2012-12-10"},
    "Connecticut": {"abbr":"CT","election":"2021-06-22","possession":"2021-07-01"},
    "Delaware": {"abbr":"DE","election":"2023-04-23","possession":"2023-04-23"},
    "Florida": {"abbr":"FL", "election":None,"possession":None},
    "Georgia": {"abbr":"GA","election":None,"possession":None},
    "Hawaii": {"abbr":"HI","election":None,"possession":None},
    "Idaho": {"abbr":"ID","election":None,"possession":None},
    "Illinois": {"abbr":"IL","election":"2019-05-31","possession":"2019-05-31"},
    "Indiana": {"abbr":"IN","election":None,"possession":None},
    "Iowa": {"abbr":"IA","election":None,"possession":None},
    "Kansas": {"abbr":"KS","election":None,"possession":None},
    "Kentucky": {"abbr":"KY","election":None,"possession":None},
    "Louisiana": {"abbr":"LA","election":None,"possession":None},
    "Maine": {"abbr":"ME","election":"2016-11-10","possession":"2017-01-30"},
    "Maryland": {"abbr":"MD","election":"2022-11-08","possession":"2023-07-01"},
    "Massachusetts": {"abbr":"MA","election":"2016-11-08","possession":"2016-12-15"},
    "Michigan": {"abbr":"MI","election":"2018-11-06","possession":"2018-11-16"},
    "Minnesota": {"abbr":"MN","election":"2023-05-30","possession":"2023-08-01"},
    "Mississippi": {"abbr":"MS","election":None,"possession":None},
    "Missouri": {"abbr":"MO","election":"2022-11-08","possession":"2022-12-08"},
    "Montana": {"abbr":"MT","election":"2020-11-03","possession":"2021-01-01"},
    "Nebraska": {"abbr":"NE","election":None,"possession":None},
    "Nevada": {"abbr":"NV","election":"2016-11-08","possession":"2017-01-01"},
    "New Hampshire": {"abbr":"NH","election":None,"possession":None},
    "New Jersey": {"abbr":"NJ","election":"2020-11-03","possession":"2021-01-01"},
    "New Mexico": {"abbr":"NM","election":"2021-04-12","possession":"2021-04-12"},
    "New York": {"abbr":"NY","election":"2021-03-31","possession":"2021-03-31"},
    "North Carolina": {"abbr":"NC","election":None,"possession":None},
    "North Dakota": {"abbr":"ND","election":None,"possession":None},
    "Ohio": {"abbr":"OH","election":"2023-11-07","possession":"2023-12-07"},
    "Oklahoma": {"abbr":"OK","election":None,"possession":None},
    "Oregon": {"abbr":"OR","election":"2014-11-04","possession":"2014-11-05"},
    "Pennsylvania": {"abbr":"PA","election":None,"possession":None},
    "Rhode Island": {"abbr":"RI","election":"2022-05-25","possession":"2022-05-25"},
    "South Carolina": {"abbr":"SC","election":None,"possession":None},
    "South Dakota": {"abbr":"SD","election":None,"possession":None},
    "Tennessee": {"abbr":"TN","election":None,"possession":None},
    "Texas": {"abbr":"TX","election":None,"possession":None},
    "Utah": {"abbr":"UT","election":None,"possession":None},
    "Vermont": {"abbr":"VT","election":"2018-01-22","possession":"2018-07-01"},
    "Virginia": {"abbr":"VA","election":"2020-04-11","possession":"2021-07-01"},
    "Washington": {"abbr":"WA","election":"2012-11-06","possession":"2012-12-06"},
    "West Virginia": {"abbr":"WV","election":None,"possession":None},
    "Wisconsin": {"abbr":"WI","election":None,"possession":None},
    "Wyoming": {"abbr":"WY","election":None,"possession":None},
    "District of Columbia": {"abbr":"DC","election":"2024-11-04","possession":"2025-02-26"}
}

# Generate monthly and quarterly labels
start_date = datetime(2010, 1, 1)
end_date = datetime(2025, 12, 1)

months = []
quarters = set()

current = start_date
while current <= end_date:
    month_str = current.strftime("%Y-%m")
    quarter_str = f"{current.year}-Q{((current.month - 1) // 3 + 1)}"
    months.append(month_str)
    quarters.add(quarter_str)
    current += relativedelta(months=1)

# Sort quarters for consistent ordering
quarters = sorted(list(quarters))

# Build list of monthly and quarterly labels
def date_to_quarter(datestr):
    year, month = map(int, datestr.split("-"))
    q = (month - 1) // 3 + 1
    return f"{year}-Q{q}"

# Determine dummy values
def make_records(record_type, labels):
    rec = {}
    for state, info in state_info.items():
        ed = info["election"][:7] if info["election"] else None  # 'YYYY-MM'
        pd = info["possession"][:7] if info["possession"] else None
        ed_q = date_to_quarter(ed) if ed else None
        pd_q = date_to_quarter(pd) if pd else None

        vals = []
        for label in labels:
            val = 0

            if record_type == "vote_01":
                if "Q" in label:  
                    val = 1 if ed_q and label >= ed_q else 0
                else:  
                    val = 1 if ed and label >= ed else 0

            elif record_type == "vote_012":
                if "Q" in label:
                    if ed_q:
                        if label == ed_q:
                            val = 1
                        elif label > ed_q:
                            val = 2
                else:
                    if ed:
                        if label == ed:
                            val = 1
                        elif label > ed:
                            val = 2

            if record_type == "poss_01":
                if "Q" in label:
                    val = 1 if pd_q and label >= pd_q else 0
                else:
                    val = 1 if pd and label >= pd else 0

            elif record_type == "poss_012":
                if "Q" in label:
                    if pd_q:
                        if label == pd_q:
                            val = 1
                        elif label > pd_q:
                            val = 2
                else:
                    if pd:
                        if label == pd:
                            val = 1
                        elif label > pd:
                            val = 2

            elif record_type == "mix_combo":
                if "Q" in label:
                    if ed_q and label == ed_q and (not pd_q or pd_q > label):
                        val = 1
                    elif ed_q and ed_q < label and (not pd_q or pd_q > label):
                        val = 2
                    elif pd_q and label == pd_q:
                        val = 3
                    elif pd_q and pd_q < label:
                        val = 4
                else:
                    if ed and label == ed:
                        val = 1
                    elif ed and ed < label and (not pd or pd > label):
                        val = 2
                    elif pd and label == pd:
                        val = 3
                    elif pd and pd < label:
                        val = 4

            elif record_type == "vote_poss_012":
                if "Q" in label:
                    if ed_q and label < ed_q:
                        val = 0
                    elif ed_q and label >= ed_q and (not pd_q or label < pd_q):
                        val = 1
                    elif pd_q and label >= pd_q:
                        val = 2
                else:
                    if ed and label < ed:
                        val = 0
                    elif ed and label >= ed and (not pd or label < pd):
                        val = 1
                    elif pd and label >= pd:
                        val = 2

            vals.append(val)
        rec[state] = vals
    return rec

# Define output files
versions = {
    "monthly_vote_01.csv": ("vote_01", months),
    "monthly_vote_012.csv": ("vote_012", months),
    "monthly_poss_01.csv": ("poss_01", months),
    "monthly_poss_012.csv": ("poss_012", months),
    "monthly_mix_combo.csv": ("mix_combo", months),
    "monthly_vote_poss_012.csv": ("vote_poss_012", months),
    "quarterly_vote_01.csv": ("vote_01", quarters),
    "quarterly_vote_012.csv": ("vote_012", quarters),
    "quarterly_poss_01.csv": ("poss_01", quarters),
    "quarterly_poss_012.csv": ("poss_012", quarters),
    "quarterly_mix_combo.csv": ("mix_combo", quarters),
    "quarterly_vote_poss_012.csv": ("vote_poss_012", quarters),
}

# Write long format CSVs
for filename, (rtype, labels) in versions.items():
    data = make_records(rtype, labels)
    output_path = os.path.join(args.output_folder, filename)
    os.makedirs(args.output_folder, exist_ok=True)
    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["state", "time", "event"])
        for state in sorted(data):
            abbr = state_info[state]["abbr"]
            for time, value in zip(labels, data[state]):
                writer.writerow([abbr, time, value])

# Merge logic
def merge_versions(output_folder, prefix, label_list):
    base_labels = ["vote_01", "vote_012", "poss_01", "poss_012", "mix_combo", "vote_poss_012"]
    merged = None
    dfs = {}

    for label in base_labels:
        filename = f"{prefix}_{label}.csv"
        filepath = os.path.join(output_folder, filename)
        df = pd.read_csv(filepath)
        df = df.rename(columns={"event": label})
        dfs[label] = df
        merged = df.copy() if merged is None else pd.merge(merged, df, on=["state", "time"], how="outer")

        monthly_data = {}
        for label in ["vote_01", "vote_012", "poss_01", "poss_012", "mix_combo", "vote_poss_012"]:
            filepath = os.path.join(output_folder, f"monthly_{label}.csv")
            df = pd.read_csv(filepath)
            df = df.rename(columns={"event": label})
            monthly_data[label] = df


    merged_filename = os.path.join(output_folder, f"{prefix}_merged.csv")
    merged.to_csv(merged_filename, index=False)
    print(f"✅ Merged file created: {merged_filename}")

# Execute merging
merge_versions(args.output_folder, "monthly", months)
merge_versions(args.output_folder, "quarterly", quarters)

print("✅ All long-format CSVs with state abbreviations generated successfully.")