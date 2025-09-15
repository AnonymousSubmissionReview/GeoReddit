"""
This script generates a regex-based dictionary (p_dictionary.txt) for fuzzy keyword matching,
based on a clean dictionary file (o_dictionary.txt) that contains phrases and entities.
This code ensures accurate keyword extraction by using strict matching for short or ambiguous terms,
while allowing edit-distance tolerance for longer, semantically unique words.

Features:
- Handles both labeled entities (marked with [ENTITY]) and general keyword phrases
- For each entity:
    - These words are short or easily confused with other common terms, so strict matching is
      recommended to avoid extracting irrelevant content.
    - Generates a regex with word boundaries and optional hyphenated suffixes (e.g., "(?:^|[^A-Za-z]){Pepsi}(?:s|[^\w\s]\S*)?\b“)
- For each phrase:
    - Words that are longer and semantically more distinctive are less likely to be misidentified
      as unrelated terms even with an edit distance of 1, so it’s generally safe to allow for minor typos.
    - Generates regex for exact match with flexible separators (e.g., punctuation or whitespace)
    - Also generates edit-distance-1 variants with the same spacing flexibility

Input:
- o_dictionary.txt (in the input folder)
    Format: one entry per line
        - [ENTITY] entity_name (e.g., [ENTITY] GPT)
        - phrase (e.g., artificial intelligence)

Output:
- p_dictionary.txt (in the output folder)
    Format: regex_pattern,original_phrase
      - regex_pattern: compiled regular expression (e.g., \bartificial[\W\s]*intelligence\b)
      - original_label: the original canonical form (e.g., artificial intelligence)

Example usage:
    python a_00_build_dictionary.py --input_folder "/path/to/input" --output_folder "/path/to/output"
"""

import os
import argparse
import string
import re


#Generate all variants of a word with edit distance = 1.
#Includes deletion, insertion, and substitution.
def generate_distance_one_variants(word):

    variants = set()
    letters = string.ascii_lowercase

    # Deletion
    for i in range(len(word)):
        variants.add(word[:i] + word[i+1:])

    # Insertion
    for i in range(len(word)+1):
        for c in letters:
            variants.add(word[:i] + c + word[i:])

    # Substitution
    for i in range(len(word)):
        for c in letters:
            if word[i] != c:
                variants.add(word[:i] + c + word[i+1:])

    return variants

#Convert a phrase into a regex pattern that allows flexible separators (e.g., punctuation, space).
def build_regex_phrase(word):
    parts = word.strip().split()
    return r"%s" % r"[\W\s]*".join([re.escape(p) for p in parts])


#Create a regex pattern for an entity using word boundaries and optional hyphenated suffix.
def build_regex_entity(word):
    escaped = re.escape(word)
    # (?:^|[^A-Za-z])   → start of string or preceded by a non-letter character
    # WORD               → exact match of the keyword (escaped)
    # (?:s|[^\w\s]\S*)?  → optionally followed by an 's' (e.g., AIs), or a punctuation character plus any non-space chars (e.g., AI!foo, AI-q)
    # \b                 → word boundary to ensure we don't consume trailing letters, digits, or underscores
    return rf"(?:^|[^A-Za-z]){escaped}(?:s|[^\w\s]\S*)?\b"

def main():
    parser = argparse.ArgumentParser(description="Generate regex dictionary with variants.")
    parser.add_argument("--input_folder", required=True, help="Folder containing o_dictionary.txt")
    parser.add_argument("--output_folder", required=True, help="Folder to save p_dictionary.txt")
    args = parser.parse_args()

    input_file = os.path.join(args.input_folder, "o_dictionary.txt")
    output_file = os.path.join(args.output_folder, "p_dictionary.txt")
    os.makedirs(args.output_folder, exist_ok=True)

    # Read input dictionary
    with open(input_file, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]

    regex_lines = []

    # Process each line in dictionary
    for line in lines:
        if line.startswith("[ENTITY]"):
            # For entities:

            entity = line.replace("[ENTITY]", "").strip()
            regex = build_regex_entity(entity)
            regex_lines.append(f"{regex},{entity}")

        else:
            # For phrases: case-insensitive and flexible spacing
            phrase = line.strip().lower()

            # Original correct version
            regex = build_regex_phrase(phrase)
            regex_lines.append(f"{regex},{phrase}")

            # Generate typo variants (edit distance = 1)
            variants = generate_distance_one_variants(phrase)
            for var in variants:
                var_clean = var.strip()
                if var_clean == phrase or len(var_clean) < 3:
                    continue
                regex = build_regex_phrase(var_clean)
                regex_lines.append(f"{regex},{phrase}")

    # Save output regex rules
    with open(output_file, "w", encoding="utf-8") as f:
        for reg in regex_lines:
            f.write(reg + "\n")

    print(f"Generated {len(regex_lines)} regex rules into {output_file}")

if __name__ == "__main__":
    main()
