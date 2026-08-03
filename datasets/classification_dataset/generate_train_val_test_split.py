# =============================================================================
# Before running this script:
# 1. Download the original classification dataset from:
#    https://www.kaggle.com/datasets/memohussein/plant-leaf-healthy-and-disease-dataset/data
# 2. Extract the dataset.
# 3. Update the input and output directory paths if necessary.
# =============================================================================
import os, shutil, random, hashlib
from pathlib import Path

# ====== مسار الداتا  ======
healthy_dir  = Path("plantvillage_two_folder_healthy_disease/healthy")
diseases_dir = Path("plantvillage_two_folder_healthy_disease/diseases")

out_dir = Path("/content/plantVillage_split_v2")

train_ratio = 0.70
val_ratio   = 0.20
test_ratio  = 0.10
seed = 42

assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6
random.seed(seed)

# ====== فحص هل الملف صورة  ======#
def is_image(p: Path):
    return p.suffix.lower() in [".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"]

# ====== لتفادي overwrite إذا نفس الاسم يتكرر ======#
def short_hash(p: Path, block=1024*1024):
    h = hashlib.md5()
    with open(p, "rb") as f:
        while True:
            b = f.read(block)
            if not b: break
            h.update(b)
    return h.hexdigest()[:8]

def safe_copy(src: Path, dst_folder: Path):
    dst_folder.mkdir(parents=True, exist_ok=True)
    h = short_hash(src)
    dst_path = dst_folder / f"{src.stem}_{h}{src.suffix.lower()}"
    shutil.copy2(src, dst_path)

def split_class_images(images):
    random.shuffle(images)
    n = len(images)
    n_train = int(n * train_ratio)
    n_val   = int(n * val_ratio)

    train_imgs = images[:n_train]
    val_imgs   = images[n_train:n_train+n_val]
    test_imgs  = images[n_train+n_val:]
    return train_imgs, val_imgs, test_imgs

# ====== دالة تعالج root واحد (healthy أو diseases) ======
def process_root(src_root: Path, title: str):
    classes = [d for d in src_root.iterdir() if d.is_dir()]
    print(f"\n{title} | Classes found: {len(classes)}")

    for cls in classes:
        images = [p for p in cls.iterdir() if p.is_file() and is_image(p)]
        if len(images) == 0:
            print(f"[Skip] {cls.name} (no images)")
            continue

        train_imgs, val_imgs, test_imgs = split_class_images(images)

        # إنشاء فولدر الكلاس داخل كل split
        for split in ["train", "val", "test"]:
            (out_dir / split / cls.name).mkdir(parents=True, exist_ok=True)

        # نسخ الملفات (مع rename لمنع overwrite)
        for p in train_imgs:
            safe_copy(p, out_dir / "train" / cls.name)
        for p in val_imgs:
            safe_copy(p, out_dir / "val" / cls.name)
        for p in test_imgs:
            safe_copy(p, out_dir / "test" / cls.name)

        print(f"{cls.name}: total={len(images)}, train={len(train_imgs)}, val={len(val_imgs)}, test={len(test_imgs)}")

# ====== إنشاء الفولدرات ======
if(
    (out_dir / "train").exists()
    and (out_dir / "val").exists()
    and (out_dir / "test").exists()
):
    print("Split already exists. Using existing split.")
else:
  print("Creating train/validation/test split...")
  for split in ["train", "val", "test"]:
    (out_dir / split).mkdir(parents=True, exist_ok=True)

  # ====== تنفيذ الدمج ======
  process_root(healthy_dir,  "HEALTHY")
  process_root(diseases_dir, "DISEASES")
  print("\n Done. Splits saved in:", out_dir)


# ====== إحصائيات ======
def count_images(root: Path):
    return sum(1 for p in root.rglob("*") if p.is_file() and is_image(p))

total_healthy  = count_images(healthy_dir)
total_diseases = count_images(diseases_dir)
total_before   = total_healthy + total_diseases

train_count = count_images(out_dir / "train")
val_count   = count_images(out_dir / "val")
test_count  = count_images(out_dir / "test")

print("\n Dataset Statistics")
print("-" * 35)
print(f"Total images (healthy):  {total_healthy}")
print(f"Total images (diseases): {total_diseases}")
print(f"Total images (before):   {total_before}")
print("No. of images after split")
print(f"Train images: {train_count}")
print(f"Validation images: {val_count}")
print(f"Test images: {test_count}")
