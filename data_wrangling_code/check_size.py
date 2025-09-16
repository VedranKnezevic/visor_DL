import os
import cv2
import pandas as pd
from tqdm import tqdm

labels = pd.read_csv("../data/labels.csv")


cnt_misshaped = 0
cnt_not_there = 0
not_there = []
for _, row in tqdm(labels.iterrows(), total=len(labels), desc="Checking images"):
    image = cv2.imread(f"../data/images/{row['filename']}", cv2.IMREAD_GRAYSCALE)
    try:
        if image.shape != (966, 1832):
            print(f"misshaped image: {row['filename']}")
            cnt_misshaped += 1
    except(AttributeError):
        # print(f"No such image in the folder: {row['filename']}")
        cnt_not_there += 1
        not_there.append(row['filename'])

print(not_there)

