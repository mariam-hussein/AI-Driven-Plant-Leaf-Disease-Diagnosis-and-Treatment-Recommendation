import os
import re
import json
import numpy as np
from collections import defaultdict
from PIL import Image

import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
import timm


# =========================================================
# 1) SETTINGS
# =========================================================

# مسار فولدر الصور الصحية
# مثال:
# healthy/
#   Tomato___healthy/
#   Potato___healthy/
#   Corn_(maize)___healthy/
HEALTHY_ROOT = r"C:\Users\DELL-MT\Desktop\plantvillage_two_folder_healthy_disease\plantvillage_two_folder_healthy_disease\healthy"

# مسار وزنات موديل التصنيف
MODEL_WEIGHTS_PATH = r"C:\Users\DELL-MT\Desktop\prototype_gui\workflow_4_6\best_weight_model.pth"

# الفولدر الجديد الذي ستُحفظ فيه النتائج
OUTPUT_DIR = r"C:\Users\DELL-MT\Desktop\prototype_gui\workflow_4_6\prototype healthy features"

# عدد كلاسات موديل التصنيف
NUM_CLASSES = 37

# حجم الصورة
IMAGE_SIZE = 224

# الامتدادات المسموح بها
ALLOWED_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


# =========================================================
# 2) DEVICE
# =========================================================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)


# =========================================================
# 3) MODEL DEFINITION
# =========================================================

class EffB0_SwinTiny_Fusion(nn.Module):
    def __init__(self, num_classes: int, eff_weights=None):
        super().__init__()

        # EfficientNet-B0 branch
        eff = efficientnet_b0(weights=eff_weights)
        self.eff_features = eff.features
        self.eff_pool = eff.avgpool
        self.eff_out_dim = eff.classifier[1].in_features  # 1280

        # Swin-Tiny branch
        self.swin = timm.create_model(
            "swin_tiny_patch4_window7_224",
            pretrained=True,
            num_classes=0
        )
        self.swin_out_dim = self.swin.num_features  # 768

        # Final classifier
        self.fusion_dim = self.eff_out_dim + self.swin_out_dim  # 2048
        self.classifier = nn.Linear(self.fusion_dim, num_classes)

    def extract_features(self, x):
        # EfficientNet features
        eff_feat = self.eff_features(x)
        eff_feat = self.eff_pool(eff_feat)
        eff_feat = torch.flatten(eff_feat, 1)  # [B, 1280]

        # Swin features
        swin_feat = self.swin(x)  # [B, 768]

        # Feature fusion
        fused = torch.cat([eff_feat, swin_feat], dim=1)  # [B, 2048]
        return fused

    def forward(self, x):
        fused = self.extract_features(x)
        logits = self.classifier(fused)
        return logits


# =========================================================
# 4) LOAD MODEL
# =========================================================

def load_model(weights_path, num_classes, device):
    model = EffB0_SwinTiny_Fusion(
        num_classes=num_classes,
        eff_weights=EfficientNet_B0_Weights.DEFAULT
    )

    checkpoint = torch.load(weights_path, map_location=device)

    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    else:
        state_dict = checkpoint

    model.load_state_dict(state_dict, strict=True)
    model.to(device)
    model.eval()
    return model


# =========================================================
# 5) TRANSFORM
# =========================================================

def get_transform():
    weights = EfficientNet_B0_Weights.DEFAULT
    mean = weights.transforms().mean
    std = weights.transforms().std

    transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])
    return transform


# =========================================================
# 6) HELPERS
# =========================================================

def is_image_file(filename):
    return filename.lower().endswith(ALLOWED_EXTENSIONS)


def clean_plant_name(folder_name):
    """
    أمثلة:
    Tomato___healthy        -> Tomato
    Apple___healthy         -> Apple
    Corn_(maize)___healthy  -> Corn_(maize)
    Pepper_bell___healthy   -> Pepper_bell
    """

    name = folder_name.strip()

    # حذف healthy من النهاية
    name = re.sub(r'___healthy$', '', name, flags=re.IGNORECASE)
    name = re.sub(r'_healthy$', '', name, flags=re.IGNORECASE)
    name = re.sub(r'\s+healthy$', '', name, flags=re.IGNORECASE)

    # تحويل المسافات إلى underscore
    name = name.replace(" ", "_")

    # الإبقاء على الحروف والأرقام و underscore والأقواس فقط
    name = re.sub(r'[^\w\(\)_]', '', name)

    return name


def collect_healthy_images(healthy_root):
    """
    يرجع:
    {
        plant_name: [image1, image2, ...]
    }
    """

    plant_to_images = defaultdict(list)

    if not os.path.exists(healthy_root):
        raise FileNotFoundError(f"HEALTHY_ROOT not found: {healthy_root}")

    for item in os.listdir(healthy_root):
        subfolder = os.path.join(healthy_root, item)

        if not os.path.isdir(subfolder):
            continue

        plant_name = clean_plant_name(item)

        for root, _, files in os.walk(subfolder):
            for file in files:
                if is_image_file(file):
                    img_path = os.path.join(root, file)
                    plant_to_images[plant_name].append(img_path)

    return dict(plant_to_images)


def extract_feature_vector(model, image_path, transform, device):
    image = Image.open(image_path).convert("RGB")
    x = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        feat = model.extract_features(x)
        feat = feat.squeeze(0).cpu().numpy().astype(np.float32)

    return feat


# =========================================================
# 7) BUILD PROTOTYPES
# =========================================================

def build_prototypes(healthy_root, output_dir, model, transform, device):
    os.makedirs(output_dir, exist_ok=True)

    plant_to_images = collect_healthy_images(healthy_root)

    if not plant_to_images:
        raise ValueError("No healthy images were found inside the specified folder.")

    summary = {}

    for plant_name, image_paths in plant_to_images.items():
        print("\n" + "=" * 60)
        print(f"Processing plant: {plant_name}")
        print(f"Number of images: {len(image_paths)}")

        features = []
        failed_images = []

        for img_path in image_paths:
            try:
                feat = extract_feature_vector(model, img_path, transform, device)
                features.append(feat)
            except Exception as e:
                failed_images.append({
                    "image": img_path,
                    "error": str(e)
                })

        if len(features) == 0:
            print(f"[WARNING] No valid features extracted for {plant_name}")
            summary[plant_name] = {
                "num_images": len(image_paths),
                "num_valid_features": 0,
                "num_failed": len(failed_images),
                "prototype_path": None
            }
            continue

        features = np.stack(features, axis=0)   # [N, 2048]
        prototype = np.mean(features, axis=0)   # [2048]

        # اسم الملف بالشكل المطلوب
        prototype_path = os.path.join(output_dir, f"{plant_name}_prototype.npy")
        all_features_path = os.path.join(output_dir, f"{plant_name}_all_features.npy")

        # حفظ الملفات
        np.save(prototype_path, prototype)
        np.save(all_features_path, features)

        # الصور الفاشلة إن وجدت
        failed_path = None
        if failed_images:
            failed_path = os.path.join(output_dir, f"{plant_name}_failed_images.json")
            with open(failed_path, "w", encoding="utf-8") as f:
                json.dump(failed_images, f, indent=4, ensure_ascii=False)

        summary[plant_name] = {
            "num_images": len(image_paths),
            "num_valid_features": int(features.shape[0]),
            "num_failed": len(failed_images),
            "feature_dim": int(features.shape[1]),
            "prototype_path": prototype_path,
            "all_features_path": all_features_path,
            "failed_images_path": failed_path
        }

        print(f"Saved prototype: {prototype_path}")
        print(f"Prototype shape: {prototype.shape}")

    summary_path = os.path.join(output_dir, "prototype_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=4, ensure_ascii=False)

    print("\n" + "=" * 60)
    print("All prototypes created successfully.")
    print(f"Saved in folder: {output_dir}")
    print(f"Summary file: {summary_path}")
    print("=" * 60)


# =========================================================
# 8) MAIN
# =========================================================

if __name__ == "__main__":
    model = load_model(MODEL_WEIGHTS_PATH, NUM_CLASSES, device)
    transform = get_transform()

    build_prototypes(
        healthy_root=HEALTHY_ROOT,
        output_dir=OUTPUT_DIR,
        model=model,
        transform=transform,
        device=device
    )