import torch
import torch.functional as F
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, PrecisionRecallDisplay
from tqdm import tqdm
import argparse
from train import ConvModel
from dataset import TWADataset


def evaluate(model, testset, save_dir, exp_num):
    model.eval()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    filenames = []
    ground_truth = []
    scores = []
    losses = []
    # pr_curve_numbers = pd.DataFrame([], columns=["precision", "recall", "thresholds"])
    # loss_per_image = pd.DataFrame([], columns=["filename", "score", "label", "loss"])
            

    with tqdm(range(len(testset)), total=len(testset), unit="img",) as t:
        for i in t:
            image, label, filename = testset[i]
            filenames.append(filename)
            image = image.unsqueeze(0).to(device)
            ground_truth.append(label.item())
            score = model(image)
            scores.append(score.item())
            loss = F.binary_cross_entropy(score, label.to(device))
            losses.append(loss.item())

    precision, recall, thresholds = precision_recall_curve(ground_truth, scores)
    thresholds = np.hstack([np.array([0]), thresholds])
    loss_per_image = pd.DataFrame({
                                "filename": filenames,
                                "score" : scores,
                                "ground_truth": ground_truth,
                                "loss": losses
                                })
    loss_per_image = loss_per_image.sort_values('loss')
    pr_curve_numbers = pd.DataFrame({"precision": precision, 
                                     "recall": recall, 
                                     "thresholds": thresholds})
    
    PrecisionRecallDisplay(precision, recall).plot()
    plt.savefig(os.path.join(save_dir, f"exp{exp_num}_PR.png"), format="png")

    pr_curve_numbers.to_csv(os.path.join(save_dir, f"exp{exp_num}_PR_numbers.csv"),index=False)
    loss_per_image.to_csv(os.path.join(save_dir, f"exp{exp_num}_loss_per_image.csv"),index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate model from weights")
    parser.add_argument("weights_path", help="path to the .pt file with model weights")
    parser.add_argument("data_dir", help="Path to directory with images and labels")
    args = parser.parse_args()

    model = ConvModel(16, 32, 64)
    model.load_state_dict(torch.load(args.weights_path, weights_only=True))
    model.eval()


    dataset = TWADataset(os.path.join(args.data_dir, "labels.csv"), os.path.join(args.data_dir, "images"))

    evaluate(model, dataset, save_dir="runs/exp3", exp_num=3)
