import pandas as pd
import os
import argparse
import shutil
from pathlib import Path
from tqdm import tqdm


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Copy images from a directory and subdirectories based on a csv")
    parser.add_argument("csv_path", help="path to the csv with labels")
    parser.add_argument("imgs_path", help="path to the root directory with images")
    parser.add_argument("dst_dir", help="directory to copy the images into")
    args = parser.parse_args()

    labels = pd.read_csv(args.csv_path)
    root = Path(args.imgs_path)
    os.makedirs(args.dst_dir, exist_ok=True)
    dst = Path(args.dst_dir)
    
    multi_match = []
    no_match = []
    for i, row in tqdm(labels.iterrows(), total=labels.shape[0]):
        
        matches = list(root.rglob(row.iloc[0]))
        if len(matches) == 1:
            shutil.copy2(matches[0], dst)
        elif len(matches) > 1:
            multi_match.append(row.iloc[0])
        else:
            no_match.append(row.iloc[0])

    if multi_match:
        with open("multi_match.log", "w")  as f:
            f.write("\n".join(multi_match))

    if no_match:
        with open("no_match.log", "w")  as f:
            f.write("\n".join(no_match))


