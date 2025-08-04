import os
import shutil
from sklearn.model_selection import train_test_split

# === CONFIG ===
SOURCE_DIR = "./PBC_dataset_normal_DIB"
TARGET_DIR = "./new_split"
SPLIT_RATIO = {
    'train': 0.7,
    'val': 0.15,
    'test': 0.15
}
SEED = 42
COPY = True  # Set to False to move files instead of copying

# === Make output dirs ===
for split in ['train', 'val', 'test']:
    split_dir = os.path.join(TARGET_DIR, split)
    os.makedirs(split_dir, exist_ok=True)

# === Process each class ===
class_names = sorted([d for d in os.listdir(SOURCE_DIR) if os.path.isdir(os.path.join(SOURCE_DIR, d))])

for class_name in class_names:
    class_dir = os.path.join(SOURCE_DIR, class_name)
    images = [f for f in os.listdir(class_dir) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]

    # Step 1: Train split
    train_files, temp_files = train_test_split(
        images,
        test_size=1 - SPLIT_RATIO['train'],
        random_state=SEED,
        stratify=[class_name] * len(images)
    )

    # Step 2: Validation and Test split (50/50 of remaining 30%)
    val_files, test_files = train_test_split(
        temp_files,
        test_size=0.5,
        random_state=SEED,
        stratify=[class_name] * len(temp_files)
    )

    # === Save files to respective folders ===
    for split, files in [('train', train_files), ('val', val_files), ('test', test_files)]:
        split_class_dir = os.path.join(TARGET_DIR, split, class_name)
        os.makedirs(split_class_dir, exist_ok=True)
        for fname in files:
            src = os.path.join(class_dir, fname)
            dst = os.path.join(split_class_dir, fname)
            if COPY:
                shutil.copy2(src, dst)
            else:
                shutil.move(src, dst)

print(f"✅ Done. Train/val/test split saved in: {TARGET_DIR}")
