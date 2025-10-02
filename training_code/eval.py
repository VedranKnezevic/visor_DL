import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, PrecisionRecallDisplay
from sklearn.metrics import recall_score, precision_score
from tqdm import tqdm
import argparse
from models import LogitsConvModel
from dataset import TWADataset
from pathlib import Path

BASE_DIR = Path(__file__).parent.resolve()
RUNS_DIR = BASE_DIR / "runs"

def plot_pos_vs_neg_recall(recall_pos, recall_neg):
    # Plot
    plt.figure(figsize=(6,6))
    plt.plot(recall_pos, recall_neg, marker='o', markersize=3, linestyle='-')
    plt.xlabel("Positive Recall (Sensitivity / TPR)")
    plt.ylabel("Negative Recall (Specificity / TNR)")
    plt.title("Negative vs Positive Recall Trade-off")
    plt.grid(True)


def recalls_at_thresholds(y_true, y_scores, thresholds):
    recalls_pos = []
    recalls_neg = []
    precisions = []
    for t in thresholds:
        preds = (y_scores >= t).astype(int)

        # Positive-class recall (TPR)
        recalls_pos.append(recall_score(y_true, preds, pos_label=1))

        # Negative-class recall (specificity)
        recalls_neg.append(recall_score(1 - y_true, 1 - preds, pos_label=1))

        # Precision (for PR curve)
        precisions.append(precision_score(y_true, preds, zero_division=0))

    return np.array(precisions), np.array(recalls_pos), np.array(recalls_neg)


def evaluate(model, trainset, valset, save_dir, exp_num, criterion):
    model.eval()
    filenames = []
    subset = []
    ground_truth = []
    scores = []
    losses = []
    # pr_curve_numbers = pd.DataFrame([], columns=["precision", "recall", "thresholds"])
    # loss_per_image = pd.DataFrame([], columns=["filename", "score", "label", "loss"])
    with tqdm(range(len(valset)), total=len(valset), unit="img", desc="eval on val") as t:
        for i in t:
            image, label, filename = valset[i]
            filenames.append(filename)
            subset.append("val")
            image = image.unsqueeze(0)
            ground_truth.append(label.item())
            score = model(image)
            scores.append(score.item())
            label = label.unsqueeze(0)
            score = score.unsqueeze(0)
            loss = criterion(score, label)
            losses.append(loss.item())

    thresholds = np.linspace(0, 1, 1001)
    precision, recall_pos, recall_neg = recalls_at_thresholds(np.array(ground_truth), np.array(scores), thresholds)
    
    pr_curve_numbers = pd.DataFrame({"precision": precision, 
                                     "recall_pos": recall_pos,
                                     "recall_neg": recall_neg, 
                                     "thresholds": thresholds})
    
    # PrecisionRecallDisplay(precision, recall_pos).plot()
    # plt.savefig(os.path.join(save_dir, f"exp{exp_num}_PR.png"), format="png")
    plot_pos_vs_neg_recall(recall_pos, recall_neg)
    plt.savefig(os.path.join(save_dir, f"exp{exp_num}_recalls.png"), format="png")

    with tqdm(range(len(trainset)), total=len(trainset), unit="img", desc="eval on train") as t:
        for i in t:
            image, label, filename = trainset[i]
            filenames.append(filename)
            subset.append("train")
            image = image.unsqueeze(0)
            ground_truth.append(label.item())
            score = model(image)
            scores.append(score.item())
            label = label.unsqueeze(0)
            score = score.unsqueeze(0)
            loss = criterion(score, label)
            losses.append(loss.item())

    loss_per_image = pd.DataFrame({
                                "filename": filenames,
                                "subset": subset,
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
    parser.add_argument("save_dir", help="Directory to save the results to")
    args = parser.parse_args()
    exp_num = int(args.save_dir[-1])


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

    evaluate(model, trainset, valset, args.save_dir, exp_num, criterion)
