import torch
import torch.nn as nn
import torchvision
from dataset import TWADataset
import os
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
    
class LogitsConvModel(nn.Module):
    def __init__(self, conv1_width, conv2_width, conv3_width):
        super(LogitsConvModel, self).__init__()
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
        return h.squeeze()
    

class ResizeConvModel(nn.Module):
    def __init__(self, conv1_width, conv2_width, conv3_width, conv4_width):
        super(ResizeConvModel, self).__init__()
        self.resize = torchvision.transforms.Resize((960, 580))
        self.conv1 = nn.Conv2d(1, conv1_width, kernel_size=7, stride=2)
        self.relu1 = nn.ReLU()
        self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(conv1_width, conv2_width, kernel_size=3, stride=2)
        self.relu2 = nn.ReLU()
        self.maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv3 = nn.Conv2d(conv2_width, conv3_width, kernel_size=3)
        self.relu3 = nn.ReLU()
        self.maxpool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv4 = nn.Conv2d(conv2_width, conv4_width, kernel_size=3)
        self.relu4 = nn.ReLU()
        self.maxpool4 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc = nn.Linear(conv4_width*13*7, 1)

    def forward(self, x):
        print(x.shape)
        h = self.resize(x)
        h = self.conv1(h)
        h = self.relu1(h)
        h = self.maxpool1(h)
        print(h.shape)
        h = self.conv2(h)
        h = self.relu2(h)
        h = self.maxpool2(h)
        print(h.shape)
        h = self.conv3(h)
        h = self.relu3(h)
        h = self.maxpool3(h)
        print(h.shape)
        h = self.conv4(h)
        h = self.relu4(h)
        h = self.maxpool4(h)
        print(h.shape)
        
        h = h.view(h.shape[0], -1)
        h = self.fc(h)
        return h.squeeze()
    

if __name__=="__main__":
    parser = argparse.ArgumentParser(description="Test out models")
    parser.add_argument("data_dir", help="Path to directory with images and labels")
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = ResizeConvModel(10, 10, 10, 10)
    model.to(device)
    dataset = TWADataset(os.path.join(args.data_dir, "labels.csv"), os.path.join(args.data_dir, "images"), device)
    image, label, filename = dataset[30000]

    model.eval()
    criterion = nn.BCEWithLogitsLoss()
    weighted_criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([0.01]))
    logit = model(image.unsqueeze(0))

    loss = criterion(logit, label.reshape((1, 1)))
    weighted_loss = weighted_criterion(logit, label.reshape((1, 1)))

    print(logit.item())
    print(loss.item())
    print(weighted_loss.item())
