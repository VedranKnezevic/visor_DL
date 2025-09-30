import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, PrecisionRecallDisplay
from tqdm import tqdm
import argparse
from train import ConvModel, LogitsConvModel
from dataset import TWADataset
from pathlib import Path

BASE_DIR = Path(__file__).parent.resolve()
RUNS_DIR = BASE_DIR / "runs"


def evaluate(model, trainset, valset, testset, save_dir, exp_num, criterion):
    model.eval()
    filenames = []
    subset = []
    ground_truth = []
    scores = []
    losses = []
    # pr_curve_numbers = pd.DataFrame([], columns=["precision", "recall", "thresholds"])
    # loss_per_image = pd.DataFrame([], columns=["filename", "score", "label", "loss"])
            
    with tqdm(range(len(testset)), total=len(testset), unit="img", desc="eval on test") as t:
        for i in t:
            image, label, filename = testset[i]
            filenames.append(filename)
            subset.append("test")
            image = image.unsqueeze(0)
            ground_truth.append(label.item())
            score = model(image)
            scores.append(score.item())
            label = label.unsuqeeze(0)
            score = score.unsqueeze(0)
            loss = criterion(score, label)
            losses.append(loss.item())


    precision, recall, thresholds = precision_recall_curve(ground_truth, scores)
    thresholds = np.hstack([np.array([0]), thresholds])
    pr_curve_numbers = pd.DataFrame({"precision": precision, 
                                     "recall": recall, 
                                     "thresholds": thresholds})
    
    PrecisionRecallDisplay(precision, recall).plot()
    plt.savefig(os.path.join(save_dir, f"exp{exp_num}_PR.png"), format="png")

    with tqdm(range(len(trainset)), total=len(trainset), unit="img", desc="eval on train") as t:
        for i in t:
            image, label, filename = testset[i]
            filenames.append(filename)
            subset.append("train")
            image = image.unsqueeze(0)
            ground_truth.append(label.item())
            score = model(image)
            scores.append(score.item())
            print(score)
            print(label)
            loss = criterion(score, label)
            losses.append(loss.item())


    with tqdm(range(len(valset)), total=len(valset), unit="img", desc="eval on val") as t:
        for i in t:
            image, label, filename = testset[i]
            filenames.append(filename)
            subset.append("val")
            image = image.unsqueeze(0)
            ground_truth.append(label.item())
            score = model(image)
            scores.append(score.item())
            loss = criterion(score, label)
            losses.append(loss.item())


    loss_per_image = pd.DataFrame({
                                "filename": filenames,
                                "score" : scores,
                                "ground_truth": ground_truth,
                                "loss": losses
                                })
    loss_per_image = loss_per_image.sort_values('loss')
    pr_curve_numbers.to_csv(os.path.join(save_dir, f"exp{exp_num}_PR_numbers.csv"),index=False)
    loss_per_image.to_csv(os.path.join(save_dir, f"exp{exp_num}_loss_per_image.csv"),index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate model from weights")
    parser.add_argument("weights_path", help="path to the .pt file with model weights")
    parser.add_argument("data_dir", help="Path to directory with images and labels")
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = LogitsConvModel(16, 32, 64)
    model = model.to(device)
    if model.__class__ == LogitsConvModel:
        criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([50], device=device))
    else:
        criterion = nn.BCELoss()
    model.load_state_dict(torch.load(args.weights_path, weights_only=True))
    model.eval()

    trainset = TWADataset(os.path.join(args.data_dir, "train", "labels.csv"), os.path.join(args.data_dir, "train", "images"), device)
    valset = TWADataset(os.path.join(args.data_dir, "val", "labels.csv"), os.path.join(args.data_dir, "val", "images"), device)
    testset = TWADataset(os.path.join(args.data_dir, "test", "labels.csv"), os.path.join(args.data_dir, "test", "images"), device)

    evaluate(model, trainset, valset, testset, save_dir=os.path.join(RUNS_DIR, "exp5"), exp_num=5, criterion=criterion)
