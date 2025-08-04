import os
import shutil
from sklearn.model_selection import train_test_split


SOURCE_DIR = "./PBC_dataset_normal_DIB"
TARGET_DIR = "./PBC_dataset_split"
VAL_SPLIT = 0.2
SEED = 42
COPY = True  

for split in ['train', 'val']:
    split_dir = os.path.join(TARGET_DIR, split)
    os.makedirs(split_dir, exist_ok=True)

class_names = sorted([d for d in os.listdir(SOURCE_DIR) if os.path.isdir(os.path.join(SOURCE_DIR, d))])

for class_name in class_names:
    class_dir = os.path.join(SOURCE_DIR, class_name)
    images = [f for f in os.listdir(class_dir) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]

    train_files, val_files = train_test_split(
        images,
        test_size=VAL_SPLIT,
        random_state=SEED,
        stratify=[class_name] * len(images)  
    )

    for split, files in [('train', train_files), ('val', val_files)]:
        split_class_dir = os.path.join(TARGET_DIR, split, class_name)
        os.makedirs(split_class_dir, exist_ok=True)
        for fname in files:
            src = os.path.join(class_dir, fname)
            dst = os.path.join(split_class_dir, fname)
            if COPY:
                shutil.copy2(src, dst)
            else:
                shutil.move(src, dst)

print(f"Done. Train/val split saved in: {TARGET_DIR}")