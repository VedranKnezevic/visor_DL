import cv2
import os
import argparse
from tqdm import tqdm


if __name__=="__main__":
    parser = argparse.ArgumentParser(description="do reprocessing of images by resizing them so they don't need to be resized during training")
    parser.add_argument("data_dir", help="directory with images you want to resize inside it")
    parser.add_argument("width")
    parser.add_argument("height")
    args = parser.parse_args()
    

    assert os.path.isdir(args.data_dir)
    
    width = int(args.width)
    height = int(args.height)

    target_size = (width, height)
    for filename in tqdm(os.listdir(args.data_dir), total=len(os.listdir(args.data_dir))):
        if filename.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".webp")):
            img_path = os.path.join(args.data_dir, filename)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                print(f"⚠️ Skipping {filename}, couldn't read the file.")
                continue
            
            resized_img = cv2.resize(img, target_size, interpolation=cv2.INTER_AREA)
            
            cv2.imwrite(img_path, resized_img)
            exit()

