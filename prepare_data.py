import os
import glob
import shutil
import random
import sys
import scipy.io
import cv2
import numpy as np

ROOT_DIR = "data"
TRAIN_IMG_DIR = os.path.join(ROOT_DIR, "train/images")
TRAIN_MASK_DIR = os.path.join(ROOT_DIR, "train/masks")
TEST_IMG_DIR = os.path.join(ROOT_DIR, "test/images")
TEST_MASK_DIR = os.path.join(ROOT_DIR, "test/masks")
EGOHANDS_ROOT = "egohands_data/_LABELLED_SAMPLES/"

def process_dataset(train_ratio):
    all_pairs = []  

    subfolders = sorted(os.listdir(EGOHANDS_ROOT))
    for subf in subfolders:
        folder_path = os.path.join(EGOHANDS_ROOT, subf)
        if not os.path.isdir(folder_path):
            continue

        mat_files = glob.glob(os.path.join(folder_path, "*.mat"))
        if len(mat_files) == 0:
            print(f"No .mat file in {folder_path}, skipping...")
            continue

        mat_path = mat_files[0]
        mat = scipy.io.loadmat(mat_path)
        polygons = mat['polygons']
        record = polygons[0] if polygons.shape[0] == 1 else None

        jpgs = sorted(glob.glob(os.path.join(folder_path, "*.jpg")))
        print(f"Processing '{subf}': {len(jpgs)} images")

        for i, img_path in enumerate(jpgs):
            img = cv2.imread(img_path)
            if img is None:
                continue

            mask = np.zeros(img.shape[:2], dtype=np.uint8)

            if record is not None:
                for hand_field in ["myleft", "myright", "yourleft", "yourright"]:
                    all_polys = record[hand_field]
                    if i >= len(all_polys):
                        break
                    coords = all_polys[i]
                    if coords is None or len(coords) == 0:
                        continue
                    coords = np.array(coords, dtype=np.float32)
                    coords = np.round(coords).astype(np.int32)
                    if coords.shape[0] >= 3:
                        cv2.fillPoly(mask, [coords], 255)
            else:
                if i >= len(polygons):
                    break
                item = polygons[i]
                for hand_field in ["myleft", "myright", "yourleft", "yourright"]:
                    coords = item[hand_field]
                    if coords is None or len(coords) == 0:
                        continue
                    coords = np.array(coords, dtype=np.float32)
                    coords = np.round(coords).astype(np.int32)
                    if coords.shape[0] >= 3:
                        cv2.fillPoly(mask, [coords], 255)

            base_name = os.path.basename(img_path)
            mask_name = base_name.replace(".jpg", "_mask.png")
            all_pairs.append((os.path.abspath(img_path), mask, mask_name))

    random.seed(42)
    random.shuffle(all_pairs)

    train_count = int(len(all_pairs) * train_ratio)
    train_pairs = all_pairs[:train_count]
    test_pairs = all_pairs[train_count:]

    os.makedirs(TRAIN_IMG_DIR, exist_ok=True)
    os.makedirs(TRAIN_MASK_DIR, exist_ok=True)
    os.makedirs(TEST_IMG_DIR, exist_ok=True)
    os.makedirs(TEST_MASK_DIR, exist_ok=True)

    for img_path, mask, mask_name in train_pairs:
        shutil.copy(img_path, TRAIN_IMG_DIR)
        cv2.imwrite(os.path.join(TRAIN_MASK_DIR, mask_name), mask)

    for img_path, mask, mask_name in test_pairs:
        shutil.copy(img_path, TEST_IMG_DIR)
        cv2.imwrite(os.path.join(TEST_MASK_DIR, mask_name), mask)

    print(f"Train: {len(train_pairs)}, Test: {len(test_pairs)}")
    print("Train/test split done.")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python prepare_data.py <train_ratio>")
        sys.exit(1)

    try:
        train_ratio = float(sys.argv[1])
        if not 0 < train_ratio < 1:
            raise ValueError()
    except ValueError:
        print("Please provide a valid train ratio between 0 and 1.")
        sys.exit(1)

    process_dataset(train_ratio)
