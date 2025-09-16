import sys
import argparse
import pandas as pd
import cv2 as cv
import os


def get_image_info(image_path):
    """Return image dimensions and size in KB using OpenCV."""
    try:
        img = cv.imread(image_path)
        if img is None:
            return None, None, None  # Skip unreadable images
        height, width = img.shape[:2]
        size_kb = os.path.getsize(image_path) / 1024
        return width, height, round(size_kb, 2)
    except Exception:
        return None, None, None

def main():
    # Create argument parser
    parser = argparse.ArgumentParser(
        description="Process a path to a directory."
    )
    
    # Add directory argument
    parser.add_argument(
        "directory",
        type=str,
        help="Path to the directory"
    )
    
    # Parse arguments
    args = parser.parse_args()
    directory_path = args.directory

    # Validate the path
    if not os.path.exists(directory_path):
        print(f"Error: The path '{directory_path}' does not exist.")
        sys.exit(1)
    if not os.path.isdir(directory_path):
        print(f"Error: The path '{directory_path}' is not a directory.")
        sys.exit(1)


    # Supported image extensions
    image_extensions = (".png", ".jpg", ".jpeg", ".gif", ".bmp", ".tiff", ".webp")

    # Iterate through images
    # print(f"\nChecking images in: {directory_path}\n")
    # print(f"{'Filename':40} {'Width':>8} {'Height':>8} {'Size (KB)':>12}")
    # print("-" * 70)

    found_images = False
    image_formats = {}
    # for filename in os.listdir(directory_path):
    #     filepath = os.path.join(directory_path, filename)

    #     if os.path.isfile(filepath) and filename.lower().endswith(image_extensions):
    #         found_images = True
    #         width, height, size_kb = get_image_info(filepath)
    #         if width is not None:
    #             image_formats[(width, height)] = image_formats.get((width, height), 0) + 1
                
    #             print(f"{filename:40} {width:8} {height:8} {size_kb:12}")
    #         else:
    #             print(f"{filename:40} {'Unreadable':>30}")

    for root, dirs, files in os.walk(directory_path):
        for filename in files:
            if filename.lower().endswith(image_extensions):
                filepath = os.path.join(root, filename)
                print(filepath)
                if os.path.isfile(filepath):
                    found_images = True
                    width, height, size_kb = get_image_info(filepath)
                    if width is not None:
                        image_formats[(width, height)] = image_formats.get((width, height), 0) + 1
                        # print(f"{filename:40} {width:8} {height:8} {size_kb:12}")
                    else:
                        print(f"{filename:40} {'Unreadable':>30}")

    print(image_formats)
    if not found_images:
        print("No images found in the specified directory.")

if __name__ == "__main__":
    main()

