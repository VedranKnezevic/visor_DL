import os
import pandas as pd
import argparse
from tqdm import tqdm
import shutil
from sklearn.model_selection import train_test_split



if __name__=="__main__":
    parser = argparse.ArgumentParser(description="Split the data into train, val and test sets")
    parser.add_argument("data_dir", help="data directory")
    parser.add_argument("train_size", help="fraction of original data for training", type=float)
    parser.add_argument("val_size", help="fraction of original data for validation", type=float)
    parser.add_argument("test_size", help="fraction of original data for testing", type=float)
    args = parser.parse_args()

    assert os.path.isdir(args.data_dir)
    assert args.train_size + args.val_size + args.test_size == 1

    data = pd.read_csv(os.path.join(args.data_dir, "labels.csv"))
    
    scrap = data[data["scrap"] == 1]
    not_scrap = data[data["scrap"] == 0]

    
    train_scrap, temp = train_test_split(scrap, train_size=args.train_size, random_state=42)
    val_scrap, test_scrap = train_test_split(temp, train_size=args.val_size / (args.val_size + args.test_size), random_state=42)
    train_not_scrap, temp = train_test_split(not_scrap, train_size=args.train_size, random_state=42)
    val_not_scrap, test_not_scrap = train_test_split(temp, train_size=args.val_size / (args.val_size + args.test_size), random_state=42)
    
    train = pd.concat([train_scrap, train_not_scrap])
    val = pd.concat([val_scrap, val_not_scrap])
    test = pd.concat([test_scrap, test_not_scrap])

    train_path = os.path.join(args.data_dir, "train")
    os.makedirs(train_path)
    val_path = os.path.join(args.data_dir, "val")
    os.makedirs(val_path)
    test_path = os.path.join(args.data_dir, "test")
    os.makedirs(test_path)

    train.to_csv(os.path.join(train_path, "labels.csv"), index=False)
    val.to_csv(os.path.join(val_path, "labels.csv"), index=False)
    test.to_csv(os.path.join(test_path, "labels.csv"), index=False)

    os.makedirs(os.path.join(args.data_dir, "train", "images"))
    os.makedirs(os.path.join(args.data_dir, "val", "images"))
    os.makedirs(os.path.join(args.data_dir, "test", "images"))

    
    for i, row in tqdm(train.iterrows(), total=train.shape[0], desc="copying train images"):
        src_file = os.path.join(args.data_dir, "images", row["filename"])
        dst_file = os.path.join(train_path, "images", row["filename"])

        shutil.copy2(src_file, dst_file)

    for i, row in tqdm(val.iterrows(), total=val.shape[0], desc="copying validation images"):
        src_file = os.path.join(args.data_dir, "images", row["filename"])
        dst_file = os.path.join(val_path, "images", row["filename"])

        shutil.copy2(src_file, dst_file)

    for i, row in tqdm(test.iterrows(), total=test.shape[0], desc="copying test images"):
        src_file = os.path.join(args.data_dir, "images", row["filename"])
        dst_file = os.path.join(test_path, "images", row["filename"])

        shutil.copy2(src_file, dst_file)
