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


class ConvModel(nn.Module):
    def __init__(self, conv1_width, conv2_width, conv3_width):
        super(ConvModel, self).__init__()
        self.conv1 = nn.Conv2d(1, conv1_width, kernel_size=7, stride=2)
        self.relu1 = nn.ReLU()
        self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(conv1_width, conv2_width, kernel_size=3, stride=2)
        self.relu2 = nn.ReLU()
        self.maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv3 = nn.Conv2d(conv2_width, conv3_width, kernel_size=3)
        self.relu3 = nn.ReLU()
        self.maxpool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc = nn.Linear(conv3_width*28*55, 1)

    def forward(self, x):
        h = self.conv1(x)
        h = self.relu1(h)
        h = self.maxpool1(h)
        h = self.conv2(h)
        h = self.relu2(h)
        h = self.maxpool2(h)
        h = self.conv3(h)
        h = self.relu3(h)
        h = self.maxpool3(h)

        h = h.view(h.shape[0], -1)
        h = self.fc(h)
        return torch.sigmoid(h).squeeze()
    

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
    if not os.path.exists("runs"):
        os.makedirs("runs", exist_ok=True)
    exp_num = find_highest_experiment("runs") + 1

    save_dir = os.path.join("runs", f"exp{exp_num}")
    os.makedirs(save_dir)

    return save_dir, exp_num

    
def train(model, train_dataloader, val_dataloader, save_dir,  exp_num, param_niter=10000, 
          param_delta=1e-8, param_lambda=1e-2,):
    optimizer = torch.optim.SGD(model.parameters(), lr = param_delta, weight_decay=param_lambda)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.1)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    hiperparameter_string = f"{model}\nn_epoch: {param_niter}\noptimizer: {optimizer.__class__.__name__}\n" \
    + f"lr: {param_delta}\nweight_decay: {param_niter}\n"
    with open(os.path.join(save_dir, f"exp{exp_num}_info.txt"), "a") as f:
        f.write(hiperparameter_string)


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
                X = X.to(device)
                Y_ = Y_.to(device)
                Y = model(X)
                loss = F.binary_cross_entropy(Y, Y_)
                train_loss += loss.item()
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                # t.set_postfix(loss=loss.item())

            scheduler.step()
            model.eval()
            with torch.no_grad():
                for X, Y_, _ in val_dataloader:
                    X = X.to(device)
                    Y_ = Y_.to(device)
                    Y = model(X)
                    loss = F.binary_cross_entropy(Y, Y_)
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


if __name__=="__main__":
    parser = argparse.ArgumentParser(description="Train model and log the results")
    parser.add_argument("data_dir", help="Path to directory with images and labels")
    args = parser.parse_args()


    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = ConvModel(64, 126, 256)
    model.to(device)
    dataset = TWADataset(os.path.join(args.data_dir, "labels.csv"), os.path.join(args.data_dir, "images"))

    n = len(dataset)
    split = random_split(dataset, [0.7, 0.15, 0.15])
    trainset = split[0]
    valset = split[1]
    testset = split[2]

    train_dataloader = DataLoader(trainset, batch_size=16, shuffle=True)
    val_dataloader = DataLoader(valset, batch_size=16, shuffle=False)
    
    save_dir, exp_num = initialize_experiment()

    with open(os.path.join(save_dir, f"exp{exp_num}_info.txt"), "a") as f:
        f.write(f"start: {datetime.datetime.now()}\n")

    train(model, train_dataloader, val_dataloader, param_niter=20, save_dir=save_dir, exp_num=exp_num)

    ratios = []
    for set in [trainset, valset, testset]:
        pseudo_scrap = dataset.img_labels.iloc[set.indices, 1].sum()
        scrap = (1- dataset.img_labels.iloc[trainset.indices, 1]).sum()
        ratios.append(pseudo_scrap/scrap)


    with open(os.path.join(save_dir, f"exp{exp_num}_info.txt"), "a") as f:
        f.write(f"train_images: {len(trainset)}\ntrain_ratio: 1:{ratios[0]:.3f}\n" + \
                f"val_images: {len(valset)}\nval_ratio: 1:{ratios[1]:.3f}\n" + \
                f"test_images: {len(testset)}\ntest_ratio: 1:{ratios[2]:.3f}\n")

    evaluate(model, testset, save_dir, exp_num)

    with open(os.path.join(save_dir, f"exp{exp_num}_info.txt"), "a") as f:
        f.write(f"end: {datetime.datetime.now()}\n")