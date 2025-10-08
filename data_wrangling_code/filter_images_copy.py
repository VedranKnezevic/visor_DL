import os
import shutil
import argparse
import logging
from tqdm import tqdm

# --- Supported image extensions ---
IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".gif", ".bmp", ".tiff", ".webp")

# --- Configure logging ---
logging.basicConfig(
    filename="filtered_copy.log",
    filemode="a",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
console = logging.StreamHandler()
console.setLevel(logging.INFO)
logging.getLogger().addHandler(console)

def copy_filtered_images(src_dir, dst_dir, name_filter, invert_match=False, preserve_structure=False):
    """
    Recursively copy images from src_dir to dst_dir if filename contains name_filter.
    """
    copied_count = 0
    skipped_count = 0

    for root, _, files in os.walk(src_dir):
        # Determine destination root
        if preserve_structure:
            rel_path = os.path.relpath(root, src_dir)
            dst_root = os.path.join(dst_dir, rel_path)
        else:
            dst_root = dst_dir

        os.makedirs(dst_root, exist_ok=True)

        for filename in files:
            # Check if it's an image
            if not filename.lower().endswith(IMAGE_EXTENSIONS):
                continue

            # Decide whether this file matches or not
            contains_pattern = name_filter.lower() in filename.lower()

            # If invert_match is ON, we skip matching files; otherwise, we skip non-matching files
            if (not invert_match and not contains_pattern) or (invert_match and contains_pattern):
                skipped_count += 1
                continue

            # Build full paths
            src_file = os.path.join(root, filename)
            dst_file = os.path.join(dst_root, filename)

            try:
                shutil.copy2(src_file, dst_file)
                logging.info(f"Copied: {src_file} -> {dst_file}")
                copied_count += 1
            except Exception as e:
                logging.error(f"Failed to copy {src_file}: {e}")

    logging.info(f"Finished. Copied {copied_count} files. Skipped {skipped_count} files with unwanted name pattern.")

def main():
    parser = argparse.ArgumentParser(description="Copy filtered images from one directory to another.")
    parser.add_argument("src_dir", help="Path to the source directory")
    parser.add_argument("dst_dir", help="Path to the destination directory")
    parser.add_argument("name_filter", help="Substring for filter")
    parser.add_argument(
        "-v", "--invert-match",
        action="store_true",
        dest="invert_match",
        help="Invert the match: copy files that DO NOT contain the pattern (like grep -v)"
    )
    parser.add_argument(
        "--preserve-structure",
        action="store_true",
        help="Preserve folder structure in destination (default: flat copy)"
    )
    args = parser.parse_args()

    # Validate source directory
    if not os.path.exists(args.src_dir):
        logging.error(f"Source directory does not exist: {args.src_dir}")
        return
    if not os.path.isdir(args.src_dir):
        logging.error(f"Source path is not a directory: {args.src_dir}")
        return

    # Create destination directory if missing
    os.makedirs(args.dst_dir, exist_ok=True)

    # Start copying
    copy_filtered_images(args.src_dir, args.dst_dir, args.name_filter, args.invert_match, args.preserve_structure)

if __name__ == "__main__":
    main()
