from torch.utils.data import Dataset
import os
import pandas as pd
from torchvision.io import read_image, ImageReadMode
import torch
import random
import matplotlib.pyplot as plt


class TWADataset(Dataset):
    def __init__(self, annotations_file: str, img_dir: str, device):
        self.img_labels = pd.read_csv(annotations_file)
        self.img_dir = img_dir
        self.device = device

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
        image = read_image(img_path, mode=ImageReadMode.GRAY).float() / 255.0
        label = self.img_labels.iloc[idx, 1]

        return image.to(self.device).unsqueeze(0), torch.tensor(label, dtype=torch.float, device=self.device), self.img_labels.iloc[idx, 0]
    

if __name__=="__main__":
    dataset = TWADataset("../data/labels.csv", "../data/images")

    random_index = random.randint(0, len(dataset))
    image, label = dataset[random_index]
    print(f"index: {random_index}")
    print(image.shape)
    if label:
        print("scrap")
    else:
        print("pseudo-scrap")
    
    plt.imshow(image, cmap="grey")
    plt.show()

