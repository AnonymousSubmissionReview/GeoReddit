"""
Step 1:
Download qBittorrent to your computer, and import the torrent file
`reddit-9c263fc85366c1ef8f5bb9da0203f4c8c8db75f4.torrent`
from the `Reddit_Result` folder. This will allow you to download the original Reddit dataset.

Step 2:
Rename the original dataset files by adding the prefix `o_`.
The following script performs this renaming operation.

Input:
- One or more folders containing Reddit monthly dataset files.
- Each file should be named in the format:
    RS_YYYY-MM.zst
    RC_YYYY-MM.zst

Output:
- The same files renamed with the prefix 'o_'.
  For example:
    RS_2020-01.zst  -->  o_RS_2020-01.zst
    RC_2020-01.zst  -->  o_RC_2020-01.zst

Example usage:
python c_00_dr_o_Reddit.py --folders "C:\\data\\RS_folder" "C:\\data\\RC_folder"
"""


import os
import glob

def rename_files(folders):
    renamed_count = 0
    skipped_count = 0
    all_matched_files = []

    for folder in folders:
        # Make sure path ends without trailing slash
        folder = folder.rstrip("\\/")
        # Find RS_*.zst and RC_*.zst in this folder
        patterns = ["RS_*.zst", "RC_*.zst"]
        matched_files = []
        for pattern in patterns:
            matched_files.extend(glob.glob(os.path.join(folder, pattern)))
        all_matched_files.extend(matched_files)

    if not all_matched_files:
        print("No matching files found.")
        return

    for file_path in sorted(all_matched_files):
        dirname = os.path.dirname(file_path)
        filename = os.path.basename(file_path)
        if filename.startswith("o_"):
            print(f"Skipping already renamed: {filename}")
            skipped_count += 1
            continue
        new_filename = "o_" + filename
        new_path = os.path.join(dirname, new_filename)
        os.rename(file_path, new_path)
        renamed_count += 1
        print(f"Renamed: {filename} -> {new_filename}")

    print("\nDone.")
    print(f"Total files renamed: {renamed_count}")
    print(f"Total files skipped: {skipped_count}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
        description="Rename RS_*.zst and RC_*.zst files by adding 'o_' prefix."
    )
    parser.add_argument(
        "--folders",
        nargs="+",
        required=True,
        help="One or more folders to process (separate by space)."
    )
    args = parser.parse_args()
    rename_files(args.folders)
