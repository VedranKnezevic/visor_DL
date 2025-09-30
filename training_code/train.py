from dataset import TWADataset
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
import os
from pathlib import Path
import pandas as pd
import numpy as np
import time
import datetime
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, PrecisionRecallDisplay
import argparse
from models import ConvModel, LogitsConvModel
from pathlib import Path

BASE_DIR = Path(__file__).parent.resolve()
RUNS_DIR = BASE_DIR / "runs"



def find_highest_experiment(directory: str):
    """
    Find the highest numbered experiment in a directory containing subdirectories named exp1, exp2, etc.
    """
    dir_path = Path(directory)
    
    max_exp = 0
    for subdir in dir_path.iterdir():
        if subdir.is_dir() and subdir.name.startswith("exp"):
            try:
                exp_num = int(subdir.name[3:])
                max_exp = max(max_exp, exp_num)
            except ValueError:
                continue
    
    return max_exp

def initialize_experiment():
    if not os.path.exists(RUNS_DIR):
        os.makedirs(RUNS_DIR, exist_ok=True)
    exp_num = find_highest_experiment(RUNS_DIR) + 1

    save_dir = os.path.join(RUNS_DIR, f"exp{exp_num}")
    os.makedirs(save_dir)
    

    return save_dir, exp_num

    
def train(model, train_dataloader, val_dataloader, save_dir,  exp_num, param_niter=10000, 
          param_delta=1e-8, param_lambda=1e-2, criterion=None):
    optimizer = torch.optim.SGD(model.parameters(), lr = param_delta, weight_decay=param_lambda)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.1)
    
    
    hiperparameter_string = f"{model}\nn_epoch: {param_niter}\noptimizer: {optimizer.__class__.__name__}\n" \
    + f"lr: {param_delta}\nweight_decay: {param_niter}\n"
    with open(os.path.join(save_dir, f"exp{exp_num}_info.txt"), "a") as f:
        f.write(hiperparameter_string)
        if criterion.__class__ == nn.BCEWithLogitsLoss:
            f.write(f"class weight: {criterion.pos_weight.item()}\n")


    results_table = pd.DataFrame([],
        columns=["epoch",
                "train_loss",
                "val_loss",
                "duration" 
                ])
    for epoch in range(1, param_niter+1):
        start = time.time()
        model.train()
        train_loss = 0
        with tqdm(enumerate(train_dataloader), total=len(train_dataloader), unit="batch",
                  desc=f'Training (epoch={epoch}/{param_niter})', mininterval=1, miniters=1) as t:   
            for i, (X, Y_, _) in t:
                X = X
                Y_ = Y_
                Y = model(X)
                loss = criterion(Y, Y_)
                train_loss += loss.item()
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                

            scheduler.step()
            model.eval()
            with torch.no_grad():
                for X, Y_, _ in val_dataloader:
                    X = X
                    Y_ = Y_
                    Y = model(X)
                    loss = criterion(Y, Y_)
                    val_loss = loss.item()
        
        train_loss /= len(train_dataloader)
        end = time.time()
        duration = end - start
        results_table.loc[len(results_table)] = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "duration": duration
        }
    
        results_table.to_csv(os.path.join(save_dir, "epoch_loss.csv"), index=False)

    plt.plot(results_table.iloc[:, 0], results_table.iloc[:, 1], label="training loss")
    plt.plot(results_table.iloc[:, 0], results_table.iloc[:, 2], label="validation loss")

    plt.grid()
    plt.legend(loc="best")
    plt.savefig(os.path.join(save_dir, f"exp{exp_num}_epoch_loss.png"), format="png")
        
    torch.save(model.state_dict(), os.path.join(save_dir, f"exp{exp_num}_weights.pt"))
    

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
            label = label.unsqueeze(0)
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


if __name__=="__main__":
    parser = argparse.ArgumentParser(description="Train model and log the results")
    parser.add_argument("data_dir", help="Path to directory with images and labels")
    args = parser.parse_args()


    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = LogitsConvModel(32, 32, 64)
    model = model.to(device)
    if model.__class__ == LogitsConvModel:
        criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([50], device=device))
    else:
        criterion = nn.BCELoss()

    
    trainset = TWADataset(os.path.join(args.data_dir, "train", "labels.csv"), os.path.join(args.data_dir, "train", "images"), device)
    valset = TWADataset(os.path.join(args.data_dir, "val", "labels.csv"), os.path.join(args.data_dir, "val", "images"), device)
    testset = TWADataset(os.path.join(args.data_dir, "test", "labels.csv"), os.path.join(args.data_dir, "test", "images"), device)

    train_dataloader = DataLoader(trainset, batch_size=32, shuffle=True)
    val_dataloader = DataLoader(valset, batch_size=32, shuffle=False)
    
    
    save_dir, exp_num = initialize_experiment()
    

    with open(os.path.join(save_dir, f"exp{exp_num}_info.txt"), "a") as f:
        f.write(f"start: {datetime.datetime.now()}\n")

    train(model, train_dataloader, val_dataloader, param_niter=7, save_dir=save_dir, exp_num=exp_num, criterion=criterion)


    evaluate(model, trainset, valset, testset, save_dir, exp_num, criterion)

    with open(os.path.join(save_dir, f"exp{exp_num}_info.txt"), "a") as f:
        f.write(f"end: {datetime.datetime.now()}\n")