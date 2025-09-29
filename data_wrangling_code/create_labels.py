import os
import shutil
import argparse
import pandas as pd
import numpy as np

'''
The script is made for taking images from 2 folders:
1) contains images of both pseudo-scrap and scrap (bigger folder)
2) contains only images of products deemed as scrap.

It copies all the images into a single folder and also creates a csv in that folder that keeps the information on
which images are scrap/pseudo-scrap
'''

def copy_images(big_dir: str, small_dir: str, dst_dir: str, scrap: set[str], pseudo_scrap: set[str]):

    shutil.copytree(
    small_dir,
    dst_dir,
    dirs_exist_ok=True  # Allows copying into an existing folder (Python 3.8+)
    )


    for file in pseudo_scrap:
        # Build full paths
        src_file = os.path.join(big_dir, file)
        dst_file = os.path.join(dst_dir, file)

        shutil.copy2(src_file, dst_file)


def create_labels(dst_dir: str, scrap: set[str], pseudo_scrap: set[str]):
    scrap_labels = pd.DataFrame({"filename": list(scrap), "scrap": np.ones(len(scrap), dtype=np.int8)})
    pseudo_scrap_labels = pd.DataFrame({"filename": list(pseudo_scrap), "scrap": np.zeros(len(pseudo_scrap), dtype=np.int8)})

    labels = pd.concat([scrap_labels, pseudo_scrap_labels])

    labels.to_csv(os.path.join(dst_dir, "labels.csv"), index=False)
    


def main():
    parser = argparse.ArgumentParser(description="Put all images into a single folder and create labels")
    parser.add_argument("big_dir", help="Path to the directory of pseudo-scrap and scrap")
    parser.add_argument("small_dir", help="Path to the directory of only scrap")
    parser.add_argument("dst_dir", help="Path to the destination directory")
    args = parser.parse_args()

    # Validate source directories
    assert os.path.isdir(args.big_dir)
    assert os.path.isdir(args.small_dir)

    # Create destination directory if missing
    os.makedirs(args.dst_dir, exist_ok=True)
    os.makedirs(os.path.join(args.dst_dir, "images"), exist_ok=True)
    
    scrap = set(os.listdir(args.small_dir))
    scrap_plus_pseudo = set(os.listdir(args.big_dir))

    pseudo_scrap = scrap_plus_pseudo.difference(scrap)
    
    copy_images(args.big_dir, args.small_dir, os.path.join(args.dst_dir, "images"), scrap, pseudo_scrap)

    create_labels(args.dst_dir, scrap, pseudo_scrap)

    

if __name__ == "__main__":
    
    main()