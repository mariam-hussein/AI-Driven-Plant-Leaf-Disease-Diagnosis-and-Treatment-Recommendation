
import os
import re
import json
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
import timm


# =========================================================
# 1) SETTINGS
# =========================================================

MODEL_WEIGHTS_PATH = r"C:\Users\DELL-MT\Desktop\prototype_gui\workflow_4_6\best_weight_model.pth"
PROTOTYPE_DIR = r"C:\Users\DELL-MT\Desktop\prototype_gui\workflow_4_6\prototype_healthy_features"
TEST_IMAGE_PATH = r"C:\Users\DELL-MT\Desktop\plantvillage_two_folder_healthy_disease\plantvillage_two_folder_healthy_disease\healthy\Apple___healthy\00fca0da-2db3-481b-b98a-9b67bb7b105c___RS_HL 7708.JPG"

NUM_CLASSES = 37
IMAGE_SIZE = 224


# =========================================================
# 2) CLASS NAMES
#    لازم يكون الترتيب مطابق لترتيب التدريب
# =========================================================

CLASS_NAMES = [
    "Apple___Apple_scab",
    "Apple___Black_rot",
    "Apple___Cedar_apple_rust",
    "Apple___healthy",
    "Cherry_(including_sour)___Powdery_mildew",
    "Cherry_(including_sour)___healthy",
    "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot",
    "Corn_(maize)___Common_rust_",
    "Corn_(maize)___Northern_Leaf_Blight",
    "Corn_(maize)___healthy",
    "Grape___Black_rot",
    "Grape___Esca_(Black_Measles)",
    "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)",
    "Grape___healthy",
    "Orange___Haunglongbing_(Citrus_greening)",
    "Peach___Bacterial_spot",
    "Peach___healthy",
    "Pepper,_bell___Bacterial_spot",
    "Pepper,_bell___healthy",
    "Potato___Early_blight",
    "Potato___Late_blight",
    "Potato___healthy",
    "Squash___Powdery_mildew",
    "Strawberry___Leaf_scorch",
    "Strawberry___healthy",
    "Tomato___Bacterial_spot",
    "Tomato___Early_blight",
    "Tomato___Late_blight",
    "Tomato___Leaf_Mold",
    "Tomato___Septoria_leaf_spot",
    "Tomato___Spider_mites Two-spotted_spider_mite",
    "Tomato___Target_Spot",
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus",
    "Tomato___Tomato_mosaic_virus",
    "Tomato___healthy",
    "Orange___healthy",
    "Squash___healthy"
]


# =========================================================
# 3) DEVICE
# =========================================================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)


# =========================================================
# 4) MODEL DEFINITION
# =========================================================

class EffB0_SwinTiny_Fusion(nn.Module):
    def __init__(self, num_classes: int, eff_weights=None):
        super().__init__()

        eff = efficientnet_b0(weights=eff_weights)
        self.eff_features = eff.features
        self.eff_pool = eff.avgpool
        self.eff_out_dim = eff.classifier[1].in_features  # 1280

        self.swin = timm.create_model(
            "swin_tiny_patch4_window7_224",
            pretrained=True,
            num_classes=0
        )
        self.swin_out_dim = self.swin.num_features  # 768

        self.fusion_dim = self.eff_out_dim + self.swin_out_dim  # 2048
        self.classifier = nn.Linear(self.fusion_dim, num_classes)

    def extract_features(self, x):
        eff_feat = self.eff_features(x)
        eff_feat = self.eff_pool(eff_feat)
        eff_feat = torch.flatten(eff_feat, 1)

        swin_feat = self.swin(x)

        fused = torch.cat([eff_feat, swin_feat], dim=1)
        return fused

    def forward(self, x):
        fused = self.extract_features(x)
        logits = self.classifier(fused)
        return logits


# =========================================================
# 5) LOAD MODEL
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
# 6) TRANSFORM
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
# 7) HELPERS
# =========================================================

def extract_plant_name(predicted_class_name):
    if "___" in predicted_class_name:
        plant_name = predicted_class_name.split("___")[0]
    else:
        plant_name = predicted_class_name

    plant_name = plant_name.replace(" ", "_")
    plant_name = re.sub(r'[^\w\(\)_]', '', plant_name)
    return plant_name


def get_prototype_path(plant_name, prototype_dir):
    return os.path.join(prototype_dir, f"{plant_name}_prototype.npy")


def load_image(image_path, transform, device):
    image = Image.open(image_path).convert("RGB")
    x = transform(image).unsqueeze(0).to(device)
    return x


def cosine_similarity_np(vec1, vec2):
    vec1 = vec1.astype(np.float32)
    vec2 = vec2.astype(np.float32)

    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)

    if norm1 == 0 or norm2 == 0:
        return 0.0

    sim = np.dot(vec1, vec2) / (norm1 * norm2)
    return float(sim)


def convert_similarity_to_scores(similarity):
    similarity = max(0.0, min(1.0, similarity))

    health_score = similarity * 100.0
    disease_score = (1.0 - similarity) * 100.0
    severity_percent = disease_score

    return health_score, disease_score, severity_percent


def convert_percent_to_severity_level(severity_percent):
    if severity_percent <= 20:
        return "EARLY"
    elif severity_percent <= 40:
        return "MODERATE"
    else:
        return "SEVERE"


def is_healthy_class(predicted_class):
    return "healthy" in predicted_class.lower()


# =========================================================
# 8) MAIN INFERENCE FUNCTION
# =========================================================

def predict_with_prototype_severity(
    image_path,
    model,
    transform,
    class_names,
    prototype_dir,
    device
):
    x = load_image(image_path, transform, device)

    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1)
        pred_idx = torch.argmax(probs, dim=1).item()
        pred_conf = probs[0, pred_idx].item()

        feature_vec = model.extract_features(x)
        feature_vec = feature_vec.squeeze(0).cpu().numpy().astype(np.float32)

    predicted_class = class_names[pred_idx]
    plant_name = extract_plant_name(predicted_class)
    prototype_path = get_prototype_path(plant_name, prototype_dir)

    if not os.path.exists(prototype_path):
        raise FileNotFoundError(
            f"Prototype not found for plant '{plant_name}'.\nExpected path:\n{prototype_path}"
        )

    # -----------------------------------------------------
    # HEALTHY CONDITION
    # -----------------------------------------------------
    if is_healthy_class(predicted_class):
        result = {
            "image_path": image_path,
            "predicted_class": predicted_class,
            "plant_name": plant_name,
            "classification_confidence": round(pred_conf * 100, 2),
            "prototype_path": prototype_path,
            "cosine_similarity": 1.0,
            "health_score": 100.0,
            "disease_score": 0.0,
            "severity_percent": 0.0,
            "severity_level": "HEALTHY"
        }
        return result

    # -----------------------------------------------------
    # DISEASED CONDITION
    # -----------------------------------------------------
    healthy_prototype = np.load(prototype_path).astype(np.float32)

    similarity = cosine_similarity_np(feature_vec, healthy_prototype)
    health_score, disease_score, severity_percent = convert_similarity_to_scores(similarity)
    severity_level = convert_percent_to_severity_level(severity_percent)

    result = {
        "image_path": image_path,
        "predicted_class": predicted_class,
        "plant_name": plant_name,
        "classification_confidence": round(pred_conf * 100, 2),
        "prototype_path": prototype_path,
        "cosine_similarity": round(similarity, 6),
        "health_score": round(health_score, 2),
        "disease_score": round(disease_score, 2),
        "severity_percent": round(severity_percent, 2),
        "severity_level": severity_level
    }

    return result


# =========================================================
# 9) RUN
# =========================================================

if __name__ == "__main__":
    model = load_model(MODEL_WEIGHTS_PATH, NUM_CLASSES, device)
    transform = get_transform()

    result = predict_with_prototype_severity(
        image_path=TEST_IMAGE_PATH,
        model=model,
        transform=transform,
        class_names=CLASS_NAMES,
        prototype_dir=PROTOTYPE_DIR,
        device=device
    )

    print("\n" + "=" * 60)
    print("PROTOTYPE-BASED SEVERITY RESULT")
    print("=" * 60)
    print(f"Image Path               : {result['image_path']}")
    print(f"Predicted Class          : {result['predicted_class']}")
    print(f"Plant Name               : {result['plant_name']}")
    print(f"Classification Confidence: {result['classification_confidence']}%")
    print(f"Prototype Path           : {result['prototype_path']}")
    print(f"Cosine Similarity        : {result['cosine_similarity']}")
    print(f"Health Score             : {result['health_score']}%")
    print(f"Disease Score            : {result['disease_score']}%")
    print(f"Severity Percent         : {result['severity_percent']}%")
    print(f"Severity Level           : {result['severity_level']}")
    print("=" * 60)

    save_path = os.path.join(PROTOTYPE_DIR, "last_prototype_inference_result.json")
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=4, ensure_ascii=False)

    print(f"\nResult saved to:\n{save_path}")