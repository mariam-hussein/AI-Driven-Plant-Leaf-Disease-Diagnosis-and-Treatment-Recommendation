import os
import re
import numpy as np
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk

import torch
import torch.nn as nn
from torchvision import models, transforms

try:
    import timm
    TIMM_AVAILABLE = True
except ImportError:
    TIMM_AVAILABLE = False

try:
    import customtkinter as ctk
    CTK_AVAILABLE = True
except ImportError:
    CTK_AVAILABLE = False

try:
    import supervision as sv
    SUPERVISION_AVAILABLE = True
except ImportError:
    SUPERVISION_AVAILABLE = False

try:
    from rfdetr import RFDETRSegNano
    RFDETR_AVAILABLE = True
except ImportError:
    RFDETR_AVAILABLE = False

try:
    from treatment_manager import TreatmentManager
    TM_AVAILABLE = True
except ImportError:
    TM_AVAILABLE = False

import warnings
# Suppress Torch Tracer warnings
warnings.filterwarnings("ignore", category=torch.jit.TracerWarning)
# Also common to ignore related UserWarnings during inference
warnings.filterwarnings("ignore", message="Converting a tensor to a Python boolean")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class UITheme:
    BG = "#131320"
    CARD = "#1C1C31"
    CARD_SOFT = "#252542"
    BORDER = "#2B2B4A"

    PRIMARY = "#5d4ef5"
    PRIMARY_HOVER = "#4f3fe0"

    TEXT = "#F4F4F9"
    MUTED = "#9da1ca"
    TITLE = "#f5c2e7"
    SECTION = "#f4a261"

    SUCCESS = "#10b981"
    SUCCESS_BG = "#183d33"

    WARNING = "#f59e0b"
    WARNING_BG = "#473616"

    DANGER = "#ef4444"
    DANGER_BG = "#4c1d1d"

    UNKNOWN = "#9da1ca"
    UNKNOWN_BG = "#252542"


class EffB0_SwinTiny_Fusion(nn.Module):
    def __init__(self, num_classes: int):
        super().__init__()

        eff = models.efficientnet_b0(weights=None)
        self.eff_features = eff.features
        self.eff_pool = eff.avgpool
        self.eff_out_dim = eff.classifier[1].in_features

        self.swin = timm.create_model(
            "swin_tiny_patch4_window7_224",
            pretrained=False,
            num_classes=0
        )
        self.swin_out_dim = self.swin.num_features

        fusion_dim = self.eff_out_dim + self.swin_out_dim
        self.classifier = nn.Linear(fusion_dim, num_classes)

    def extract_features(self, x):
        fe = self.eff_features(x)
        fe = self.eff_pool(fe)
        fe = torch.flatten(fe, 1)

        fs = self.swin(x)

        emb = torch.cat([fe, fs], dim=1)
        return emb

    def forward(self, x):
        emb = self.extract_features(x)
        logits = self.classifier(emb)
        return logits, emb


class PlantPrototypeLeafSegGUI:
    def __init__(self, root):
        if not CTK_AVAILABLE:
            raise ImportError("customtkinter is not installed. Run: pip install customtkinter")

        self.root = root
        self.root.title("Plant Disease Diagnosis & Treatment System")
        self.root.geometry("1500x920")
        self.root.minsize(1320, 820)
        self.root.configure(fg_color=UITheme.BG)

        self.base_path = os.path.dirname(os.path.abspath(__file__))

        self.tk_img_original = None
        self.tk_img_result = None
        self.current_original_pil = None
        self.current_result_pil = None
        self._resize_after_id = None

        self.current_result = None
        self.rank_btns = []
        self.active_rank = tk.StringVar(value="1")

        # لازم يطابق كود prototype extraction
        self.SEG_THRESHOLD = 0.2
        self.SEG_IOU_THRESHOLD = 0.5
        self.USE_BBOX_CROP = True
        self.classifier_classes = [] # Will be loaded dynamically
        self._load_models()
        self._setup_ui()
        self.root.bind("<Configure>", self._on_resize)

    def _load_models(self):
        self.cls_model = None
        self.leaf_model = None
        self.treatment_mgr = None

        self.prototype_dir = os.path.join(self.base_path, "prototype_healthy_features")
        cls_model_path = os.path.join(self.base_path, "best_weight_model.pth")
        treatments_path = os.path.join(self.base_path, "treatments2.json")
        leaf_model_path = os.path.join(self.base_path, "checkpoint_best_total.pth")
        
        # 0) Load/Sync Classes first
        loaded_classes = self._load_model_classes(cls_model_path)
        if loaded_classes:
            self.classifier_classes = loaded_classes
            print(f"[Classification] Loaded {len(self.classifier_classes)} classes dynamically.")
        else:
            # Fallback to current hardcoded list if everything fails
            self.classifier_classes = [
                "Apple___Apple_scab", "Apple___Black_rot", "Apple___Cedar_apple_rust", "Apple___healthy",
                "Cherry_(including_sour)___Powdery_mildew", "Cherry_(including_sour)___healthy",
                "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot", "Corn_(maize)___Common_rust_",
                "Corn_(maize)___Northern_Leaf_Blight", "Corn_(maize)___healthy",
                "Grape___Black_rot", "Grape___Esca_(Black_Measles)", "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)", "Grape___healthy",
                "Orange___Haunglongbing_(Citrus_greening)", "Peach___Bacterial_spot", "Peach___healthy",
                "Pepper,_bell___Bacterial_spot", "Pepper,_bell___healthy",
                "Potato___Early_blight", "Potato___Late_blight", "Potato___healthy",
                "Squash___Powdery_mildew", "Strawberry___Leaf_scorch", "Strawberry___healthy",
                "Tomato___Bacterial_spot", "Tomato___Early_blight", "Tomato___Late_blight",
                "Tomato___Leaf_Mold", "Tomato___Septoria_leaf_spot", "Tomato___Spider_mites Two-spotted_spider_mite",
                "Tomato___Target_Spot", "Tomato___Tomato_Yellow_Leaf_Curl_Virus", "Tomato___Tomato_mosaic_virus",
                "Tomato___healthy", "Orange___healthy", "Squash___healthy"
            ]
            print("[Classification] Using fallback hardcoded classes.")

        print(f"[LeafSeg] Expected leaf model path: {leaf_model_path}")

        if not TIMM_AVAILABLE:
            messagebox.showerror("Missing Library", "timm is not installed.\nRun: pip install timm")
            return

        if os.path.exists(cls_model_path):
            try:
                num_classes = len(self.classifier_classes)
                self.cls_model = EffB0_SwinTiny_Fusion(num_classes=num_classes)

                checkpoint = torch.load(cls_model_path, map_location=DEVICE)
                if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
                    state_dict = checkpoint["model_state_dict"]
                else:
                    state_dict = checkpoint

                self.cls_model.load_state_dict(state_dict, strict=True)
                self.cls_model.to(DEVICE)
                self.cls_model.eval()

                self.cls_preprocess = transforms.Compose([
                    transforms.Resize((224, 224)),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        [0.485, 0.456, 0.406],
                        [0.229, 0.224, 0.225]
                    )
                ])

                print("[Classification] Model loaded successfully")
            except Exception as e:
                messagebox.showerror("Model Error", f"Error loading classification model:\n{e}")
        else:
            messagebox.showerror("Missing File", f"Classification model not found:\n{cls_model_path}")

        if RFDETR_AVAILABLE and SUPERVISION_AVAILABLE and os.path.exists(leaf_model_path):
            try:
                self.leaf_model = RFDETRSegNano(pretrain_weights=leaf_model_path)

                torch_mod = self.leaf_model.model.model
                torch_mod.to(DEVICE)

                if DEVICE == "cuda":
                    torch_mod.half()
                    self.leaf_model.optimize_for_inference(dtype=torch.float16)
                else:
                    self.leaf_model.optimize_for_inference()

                print("[LeafSeg] Leaf segmentation model loaded successfully")
            except Exception as e:
                print(f"[LeafSeg] Error loading leaf model: {e}")
                self.leaf_model = None
        else:
            print("[LeafSeg] Leaf model not loaded")
            print(f"[LeafSeg] RFDETR_AVAILABLE={RFDETR_AVAILABLE}, SUPERVISION_AVAILABLE={SUPERVISION_AVAILABLE}")
            print(f"[LeafSeg] File exists? {os.path.exists(leaf_model_path)}")

        if TM_AVAILABLE and os.path.exists(treatments_path):
            try:
                self.treatment_mgr = TreatmentManager(treatments_path, top_k=2)
            except Exception as e:
                messagebox.showwarning("Treatment Error", f"Error initializing TreatmentManager:\n{e}")

    def _load_model_classes(self, model_path):
        """Attempts to load class names from the model checkpoint or a companion classes.txt file."""
        dynamic_classes = None
        
        # 1. Try reading from checkpoint metadata (if saved as a dict with 'classes' key)
        if os.path.exists(model_path):
            try:
                # Use CPU for quick metadata check
                checkpoint = torch.load(model_path, map_location="cpu")
                if isinstance(checkpoint, dict) and "classes" in checkpoint:
                    dynamic_classes = checkpoint["classes"]
            except Exception as e:
                print(f"[Classification] Could not read classes from checkpoint: {e}")

        # 2. Try reading from classes.txt (common companion file)
        if dynamic_classes is None:
            classes_txt_path = os.path.join(os.path.dirname(model_path), "classes.txt")
            if os.path.exists(classes_txt_path):
                try:
                    with open(classes_txt_path, "r", encoding="utf-8") as f:
                        dynamic_classes = [line.strip() for line in f if line.strip()]
                except Exception as e:
                    print(f"[Classification] Error reading classes.txt: {e}")

        return dynamic_classes

    def _clean_label(self, text):
        if not text:
            return "-"
        text = text.replace("_", " ").replace(",", " ")
        text = text.replace("(", "").replace(")", "")
        text = " ".join(text.split()).strip()
        return text.title()

    def split_plant_disease(self, disease_class):
        if not disease_class or disease_class in ["Unknown", "Ambiguous Diagnosis"]:
            return "-", disease_class

        raw = disease_class.strip()

        if "___" in raw:
            plant_part, disease_part = raw.split("___", 1)
        else:
            return "-", self._clean_label(raw)

        plant = self._clean_label(plant_part)
        disease = self._clean_label(disease_part)

        plant_norm = plant.lower().replace(" ", "")
        disease_norm = disease.lower().replace(" ", "")

        if disease_norm.startswith(plant_norm):
            disease = disease[len(plant):].strip()
            if not disease:
                disease = "Healthy"

        return plant, disease

    def extract_plant_name_for_prototype(self, predicted_class_name):
        if "___" in predicted_class_name:
            plant_name = predicted_class_name.split("___")[0]
        else:
            plant_name = predicted_class_name

        plant_name = plant_name.replace(" ", "_")
        plant_name = re.sub(r'[^\w\(\)_]', '', plant_name)
        return plant_name

    def is_healthy_class(self, predicted_class):
        return "healthy" in predicted_class.lower()

    def get_prototype_path(self, plant_name):
        """Finds the prototype file with flexible matching for case and punctuation."""
        if not plant_name:
            return ""
            
        # 1. Try exact match first
        exact_path = os.path.join(self.prototype_dir, f"{plant_name}_prototype.npy")
        if os.path.exists(exact_path):
            return exact_path
            
        # 2. Case-insensitive and fuzzy matching fallback
        if os.path.exists(self.prototype_dir):
            try:
                files = os.listdir(self.prototype_dir)
                proto_files = [f for f in files if f.endswith("_prototype.npy")]
                
                # Case-insensitive match
                for f in proto_files:
                    if f.lower() == f"{plant_name}_prototype.npy".lower():
                        return os.path.join(self.prototype_dir, f)
                
                # "Starts-with" match (handles cases like 'Squash' vs 'Squash -')
                for f in proto_files:
                    # Remove the suffix to get the base name in the file
                    base_in_file = f.replace("_prototype.npy", "").lower().strip()
                    search_base = plant_name.lower().strip()
                    
                    if search_base in base_in_file or base_in_file in search_base:
                        return os.path.join(self.prototype_dir, f)
            except Exception as e:
                print(f"[Prototype] Error during fuzzy search: {e}")

        return exact_path # Final fallback to predicted path

    def cosine_similarity_np(self, vec1, vec2):
        vec1 = vec1.astype(np.float32)
        vec2 = vec2.astype(np.float32)

        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)

        if norm1 == 0 or norm2 == 0:
            return 0.0

        return float(np.dot(vec1, vec2) / (norm1 * norm2))

    def convert_similarity_to_scores(self, similarity):
        similarity = max(0.0, min(1.0, similarity))
        health_score = similarity * 100.0
        disease_score = (1.0 - similarity) * 100.0
        severity_percent = disease_score
        return health_score, disease_score, severity_percent

    def convert_percent_to_severity_level(self, severity_percent):
        # Early from 1 to 20
        # Moderate from 21 to 40
        # Severe from 41 to up
        if severity_percent <= 20:
            return "EARLY"
        elif severity_percent <= 40:
            return "MODERATE"
        else:
            return "SEVERE"

    def severity_style(self, severity_level):
        severity_level = (severity_level or "").upper()

        if severity_level == "HEALTHY":
            return "🟢 Healthy", UITheme.SUCCESS, UITheme.SUCCESS_BG
        elif severity_level == "EARLY":
            return "🟢 Early", UITheme.SUCCESS, UITheme.SUCCESS_BG
        elif severity_level == "MODERATE":
            return "🟡 Moderate", UITheme.WARNING, UITheme.WARNING_BG
        elif severity_level == "SEVERE":
            return "🔴 Severe", UITheme.DANGER, UITheme.DANGER_BG
        else:
            return "⚪ Unknown", UITheme.UNKNOWN, UITheme.UNKNOWN_BG

    # =========================
    # HELPER FUNCTIONS FOR SEGMENTATION
    # =========================
    def get_bbox_from_mask(self, mask_bool):
        ys, xs = np.where(mask_bool)
        if len(xs) == 0 or len(ys) == 0:
            return None
        x1, x2 = xs.min(), xs.max()
        y1, y2 = ys.min(), ys.max()
        return x1, y1, x2, y2

    def apply_mask_black_background(self, img_np, mask_bool):
        out = np.zeros_like(img_np)
        out[mask_bool] = img_np[mask_bool]
        return out

    def crop_to_mask_bbox(self, img_np, mask_bool):
        bbox = self.get_bbox_from_mask(mask_bool)
        if bbox is None:
            return img_np
        x1, y1, x2, y2 = bbox
        return img_np[y1:y2 + 1, x1:x2 + 1]

    def _get_combined_leaf_mask(self, detections, image_shape_hw):
        h, w = image_shape_hw
        mask_combined = np.zeros((h, w), dtype=bool)

        if not hasattr(detections, "mask") or detections.mask is None:
            return None

        if len(detections.mask) == 0:
            return None

        for i in range(len(detections.mask)):
            try:
                mask_i = detections.mask[i].astype(bool)
                if mask_i.shape == (h, w):
                    mask_combined = np.logical_or(mask_combined, mask_i)
            except Exception as e:
                print(f"[LeafSeg] Failed to combine mask {i}: {e}")

        if not mask_combined.any():
            return None

        return mask_combined

    def create_leaf_segmentation_visualization(self, image, detections=None):
        if self.leaf_model is None:
            print("[LeafSeg] self.leaf_model is None")
            return image.copy()

        if not SUPERVISION_AVAILABLE:
            print("[LeafSeg] supervision library is not available")
            return image.copy()

        try:
            if detections is None:
                detections = self.leaf_model.predict(
                    image,
                    threshold=self.SEG_THRESHOLD,
                    iou_threshold=self.SEG_IOU_THRESHOLD
                )

            print(f"[LeafSeg] Number of detections: {len(detections)}")

            if len(detections) == 0:
                print("[LeafSeg] No leaf detections found")
                return image.copy()

            annotated = np.array(image).copy()

            try:
                detections.class_id = np.zeros(len(detections), dtype=int)
            except Exception as e:
                print(f"[LeafSeg] Could not set class_id: {e}")

            color_leaf = sv.Color(r=32, g=178, b=107)

            annotated = sv.MaskAnnotator(
                color=color_leaf,
                opacity=0.45
            ).annotate(annotated, detections)

            try:
                annotated = sv.BoxAnnotator(color=color_leaf).annotate(annotated, detections)
            except Exception as e:
                print(f"[LeafSeg] BoxAnnotator error: {e}")

            try:
                labels = []
                if hasattr(detections, "confidence") and detections.confidence is not None:
                    labels = [f"Leaf {conf:.2f}" for conf in detections.confidence]
                else:
                    labels = ["Leaf"] * len(detections)

                annotated = sv.LabelAnnotator(
                    color=color_leaf,
                    text_color=sv.Color.WHITE
                ).annotate(annotated, detections, labels=labels)
            except Exception as e:
                print(f"[LeafSeg] LabelAnnotator error: {e}")

            return Image.fromarray(annotated)

        except Exception as e:
            print(f"[LeafSeg] Prediction/visualization error: {e}")
            return image.copy()

    def segment_leaf_for_inference(self, image):
        """
        يطابق منطق prototype extraction:
        1) segmentation
        2) black background
        3) crop to bbox إذا USE_BBOX_CROP=True
        """
        if self.leaf_model is None:
            raise RuntimeError("Leaf segmentation model is not loaded.")

        image_rgb = image.convert("RGB")
        img_np = np.array(image_rgb)

        detections = self.leaf_model.predict(
            image_rgb,
            threshold=self.SEG_THRESHOLD,
            iou_threshold=self.SEG_IOU_THRESHOLD
        )

        if len(detections) == 0:
            raise RuntimeError("No leaf detected in the image.")

        mask_combined = self._get_combined_leaf_mask(detections, img_np.shape[:2])
        if mask_combined is None:
            raise RuntimeError("Leaf mask was not found in detections.")

        # apply black background
        full_segmented_np = self.apply_mask_black_background(img_np, mask_combined)
        
        # display_pil: Full image, black background, NO masks/labels/boxes
        display_pil = Image.fromarray(full_segmented_np.astype(np.uint8))

        # inference_pil: Cropped version for consistent classification input if enabled
        if self.USE_BBOX_CROP:
            inference_np = self.crop_to_mask_bbox(full_segmented_np, mask_combined)
        else:
            inference_np = full_segmented_np

        inference_pil = Image.fromarray(inference_np.astype(np.uint8))

        return inference_pil, display_pil

    def show_treatment_headers(self):
        if not self.treatment_header_frame.winfo_ismapped():
            self.treatment_header_frame.pack(fill="x", before=self.lbl_details_title)

        if not self.lbl_details_title.winfo_ismapped():
            self.lbl_details_title.pack(anchor="w", padx=20, pady=(0, 4))

    def hide_treatment_headers_for_healthy(self):
        if self.treatment_header_frame.winfo_ismapped():
            self.treatment_header_frame.pack_forget()

        if not self.lbl_details_title.winfo_ismapped():
            self.lbl_details_title.pack(anchor="w", padx=20, pady=(0, 4))

    def _setup_ui(self):
        COLOR_BG = UITheme.BG
        COLOR_CARD = UITheme.CARD
        COLOR_CARD_SOFT = UITheme.CARD_SOFT
        COLOR_PRIMARY = UITheme.PRIMARY
        COLOR_TEXT = UITheme.TEXT
        COLOR_MUTED = UITheme.MUTED
        COLOR_BORDER = UITheme.BORDER
        SECTION_COLOR = UITheme.SECTION

        self.root.grid_columnconfigure(0, weight=1)
        self.root.grid_rowconfigure(1, weight=1)

        header = ctk.CTkFrame(self.root, fg_color="transparent")
        header.grid(row=0, column=0, sticky="ew", padx=40, pady=(35, 10))
        header.grid_columnconfigure(0, weight=1)
        header.grid_columnconfigure(1, weight=0)

        title_frame = ctk.CTkFrame(header, fg_color="transparent")
        title_frame.grid(row=0, column=0, sticky="w")

        ctk.CTkLabel(
            title_frame,
            text="Plant Disease Diagnosis & Treatment Recommendation",
            font=("Roboto", 24, "bold"),
            text_color=COLOR_TEXT
        ).pack(anchor="w")

        status_inner = ctk.CTkFrame(title_frame, fg_color="transparent")
        status_inner.pack(anchor="w", pady=(5, 0))

        self.status_dot = ctk.CTkLabel(
            status_inner, text="✔", font=("Roboto", 18, "bold"), text_color=UITheme.SUCCESS
        )
        self.status_dot.pack(side="left", padx=(0, 8))

        self.lbl_status = ctk.CTkLabel(
            status_inner, text="Ready", font=("Roboto", 14), text_color=COLOR_MUTED
        )
        self.lbl_status.pack(side="left")

        self.btn_select = ctk.CTkButton(
            header,
            text="Upload Leaf Image",
            command=self.select_image,
            font=("Roboto", 15, "bold"),
            fg_color=COLOR_PRIMARY,
            hover_color=UITheme.PRIMARY_HOVER,
            text_color="white",
            corner_radius=8,
            height=42,
            width=160
        )
        self.btn_select.grid(row=0, column=1, sticky="e")

        main_view = ctk.CTkFrame(self.root, fg_color="transparent")
        main_view.grid(row=1, column=0, sticky="nsew", padx=35, pady=(20, 35))
        main_view.grid_rowconfigure(0, weight=1)

        main_view.grid_columnconfigure(0, weight=4, uniform="maincols", minsize=400)
        main_view.grid_columnconfigure(1, weight=2, uniform="maincols", minsize=320)
        main_view.grid_columnconfigure(2, weight=4, uniform="maincols", minsize=400)

        def create_card(parent, title):
            card = ctk.CTkFrame(
                parent,
                fg_color=COLOR_CARD,
                corner_radius=12,
                border_width=1,
                border_color=COLOR_BORDER
            )
            ctk.CTkLabel(
                card,
                text=title,
                font=("Roboto", 15, "bold"),
                text_color=COLOR_TEXT
            ).pack(anchor="w", padx=15, pady=(6, 3))
            return card

        col1 = ctk.CTkFrame(main_view, fg_color="transparent")
        col1.grid(row=0, column=0, sticky="nsew", padx=(0, 10))
        col1.grid_rowconfigure(0, weight=1)
        col1.grid_rowconfigure(1, weight=1)
        col1.grid_columnconfigure(0, weight=1)

        card_orig = create_card(col1, "Origin")
        card_orig.grid(row=0, column=0, sticky="nsew", pady=(0, 10))
        original_border = ctk.CTkFrame(card_orig, fg_color=COLOR_CARD_SOFT, corner_radius=8)
        original_border.pack(fill="both", expand=True, padx=20, pady=(5, 20))
        self.canvas_original = tk.Canvas(original_border, bg=COLOR_BG, highlightthickness=0, bd=0)
        self.canvas_original.pack(fill="both", expand=True, padx=3, pady=3)

        card_result = create_card(col1, "Leaf Segmentation")
        card_result.grid(row=1, column=0, sticky="nsew", pady=(10, 0))
        result_border = ctk.CTkFrame(card_result, fg_color=COLOR_CARD_SOFT, corner_radius=8)
        result_border.pack(fill="both", expand=True, padx=20, pady=(5, 20))
        self.canvas_result = tk.Canvas(result_border, bg=COLOR_BG, highlightthickness=0, bd=0)
        self.canvas_result.pack(fill="both", expand=True, padx=3, pady=3)

        col2 = ctk.CTkFrame(main_view, fg_color="transparent")
        col2.grid(row=0, column=1, sticky="nsew", padx=10)

        card_mid = create_card(col2, "Disease Information")
        card_mid.pack(fill="both", expand=True)

        disease_box = ctk.CTkFrame(card_mid, fg_color=COLOR_CARD_SOFT, corner_radius=8)
        disease_box.pack(fill="x", padx=20, pady=(15, 18))

        self.lbl_disease_value = ctk.CTkLabel(
            disease_box, text="-", font=("Roboto", 17, "bold"), text_color=COLOR_TEXT, anchor="w"
        )
        self.lbl_disease_value.pack(side="left", padx=15, pady=12)

        ctk.CTkLabel(card_mid, text="Severity", font=("Roboto", 14), text_color=COLOR_TEXT).pack(anchor="w", padx=20)
        self.lbl_severity = ctk.CTkLabel(
            card_mid,
            text="⚪ Unknown",
            font=("Roboto", 13, "bold"),
            fg_color=UITheme.UNKNOWN_BG,
            text_color=UITheme.UNKNOWN,
            corner_radius=6,
            anchor="w"
        )
        self.lbl_severity.pack(fill="x", padx=20, pady=(8, 18), ipady=8)

        ctk.CTkLabel(card_mid, text="Health Score", font=("Roboto", 14), text_color=COLOR_TEXT).pack(anchor="w", padx=20)
        self.lbl_health_score = ctk.CTkLabel(card_mid, text="-", font=("Roboto", 14, "bold"), text_color=COLOR_TEXT, anchor="w")
        self.lbl_health_score.pack(fill="x", padx=20, pady=(8, 10))

        ctk.CTkLabel(card_mid, text="Disease Score", font=("Roboto", 14), text_color=COLOR_TEXT).pack(anchor="w", padx=20)
        self.lbl_disease_score = ctk.CTkLabel(card_mid, text="-", font=("Roboto", 14, "bold"), text_color=COLOR_TEXT, anchor="w")
        self.lbl_disease_score.pack(fill="x", padx=20, pady=(8, 18))

        ctk.CTkLabel(card_mid, text="Treatment Options", font=("Roboto", 14), text_color=COLOR_TEXT).pack(anchor="w", padx=20)
        self.rank_buttons_frame = ctk.CTkFrame(card_mid, fg_color="transparent")
        self.rank_buttons_frame.pack(fill="x", padx=20, pady=(8, 20))

        col3 = ctk.CTkFrame(main_view, fg_color="transparent")
        col3.grid(row=0, column=2, sticky="nsew", padx=(10, 0))

        card_right = create_card(col3, "Recommended Treatment")
        card_right.pack(fill="both", expand=True)

        self.treatment_header_frame = ctk.CTkFrame(card_right, fg_color="transparent")
        self.treatment_header_frame.pack(fill="x", anchor="w")

        self.lbl_name_title = ctk.CTkLabel(self.treatment_header_frame, text="Name:", font=("Roboto", 14, "bold"), text_color=SECTION_COLOR, anchor="w")
        self.lbl_name_title.pack(anchor="w", padx=20, pady=(12, 4))

        self.lbl_treatment_title = ctk.CTkLabel(
            self.treatment_header_frame, text="", font=("Roboto", 14, "bold"), text_color="#cbd5e1", justify="left", anchor="w", wraplength=420
        )
        self.lbl_treatment_title.pack(anchor="w", fill="x", padx=20, pady=(0, 14))

        self.lbl_type_title = ctk.CTkLabel(self.treatment_header_frame, text="Type:", font=("Roboto", 14, "bold"), text_color=SECTION_COLOR, anchor="w")
        self.lbl_type_title.pack(anchor="w", padx=20, pady=(0, 4))

        self.lbl_selected_type = ctk.CTkLabel(
            self.treatment_header_frame, text="-", font=("Roboto", 14), text_color="#cbd5e1", justify="left", anchor="w", wraplength=420
        )
        self.lbl_selected_type.pack(anchor="w", fill="x", padx=20, pady=(0, 14))

        self.lbl_details_title = ctk.CTkLabel(card_right, text="Details:", font=("Roboto", 14, "bold"), text_color=SECTION_COLOR, anchor="w")
        self.lbl_details_title.pack(anchor="w", padx=20, pady=(0, 4))

        self.txt_treatment = tk.Text(
            card_right,
            bg=COLOR_CARD,
            fg=COLOR_MUTED,
            font=("Roboto", 14),
            wrap="word",
            bd=0,
            highlightthickness=0,
            padx=20,
            pady=8,
            spacing1=4,
            spacing2=4,
            spacing3=10,
            insertbackground=COLOR_MUTED
        )
        self.txt_treatment.pack(fill="both", expand=True, padx=2, pady=(0, 2))

        self._reset_ui_placeholders()

    def _reset_ui_placeholders(self):
        self.lbl_disease_value.configure(text="-")
        self.lbl_severity.configure(text="⚪ Unknown", text_color=UITheme.UNKNOWN, fg_color=UITheme.UNKNOWN_BG)
        self.lbl_health_score.configure(text="-")
        self.lbl_disease_score.configure(text="-")

        self.show_treatment_headers()
        self.lbl_selected_type.configure(text="-")
        self.lbl_treatment_title.configure(text="")

        self.txt_treatment.config(state=tk.NORMAL)
        self.txt_treatment.delete("1.0", tk.END)
        self.txt_treatment.insert(
            tk.END,
            "Upload a leaf image to run leaf segmentation, classification, prototype-based severity estimation, and treatment recommendation."
        )
        self.txt_treatment.config(state=tk.DISABLED)

        for child in self.rank_buttons_frame.winfo_children():
            child.destroy()

    def select_image(self):
        file_path = filedialog.askopenfilename(
            filetypes=[("Image Files", "*.jpg *.jpeg *.png *.JPG *.PNG")]
        )
        if not file_path:
            return

        self._set_status(f"Processing: {os.path.basename(file_path)}", UITheme.WARNING)
        self.root.update_idletasks()

        try:
            self.process_image(file_path)
            self._set_status("Analysis Complete", UITheme.SUCCESS)
        except Exception as e:
            self._set_status("Error", UITheme.DANGER)
            messagebox.showerror("Processing Error", str(e))

    def process_image(self, image_path):
        if self.cls_model is None:
            raise RuntimeError("Classification model is not loaded.")

        if self.leaf_model is None:
            raise RuntimeError("Leaf segmentation model is not loaded.")

        self.canvas_original.delete("all")
        self.canvas_result.delete("all")

        original_img = Image.open(image_path).convert("RGB")
        self.display_image(self.canvas_original, original_img, "Original")

        # =========================
        # 1) SEGMENTATION FIRST
        # =========================
        segmented_leaf_img, leaf_vis_img = self.segment_leaf_for_inference(original_img)

        # عرض visualization فقط
        self.display_image(self.canvas_result, leaf_vis_img, "Result")

        # =========================
        # 2) EXTRACT FEATURES
        # =========================
        # Requirement: Use original color image for better classification identity
        tensor_full = self.cls_preprocess(original_img).unsqueeze(0).to(DEVICE)
        
        # Requirement: Use segmented leaf for accurate severity feature extraction
        tensor_seg = self.cls_preprocess(segmented_leaf_img).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            # Get logits from full color image
            logits, _ = self.cls_model(tensor_full)
            probs = torch.softmax(logits, dim=1)
            top_prob, top_idx = torch.max(probs, 1)
            
            # Get features from segmented leaf for prototype similarity
            _, emb = self.cls_model(tensor_seg)

        pred_conf = float(top_prob.item())
        if pred_conf > 0.30:
            predicted_class = self.classifier_classes[top_idx.item()]
        else:
            predicted_class = "Ambiguous Diagnosis"

        if predicted_class == "Ambiguous Diagnosis":
            self.show_unknown_case(leaf_vis_img)
            return

        plant_name, disease_name = self.split_plant_disease(predicted_class)
        self.lbl_disease_value.configure(
            text=f"{plant_name} {disease_name}" if disease_name != "Healthy" else f"{plant_name} Healthy"
        )

        feature_vec = emb.squeeze(0).cpu().numpy().astype(np.float32)
        prototype_plant_name = self.extract_plant_name_for_prototype(predicted_class)
        prototype_path = self.get_prototype_path(prototype_plant_name)

        if not os.path.exists(prototype_path):
            raise FileNotFoundError(
                f"Prototype file not found for plant '{prototype_plant_name}'.\nExpected:\n{prototype_path}"
            )

        # إذا التصنيف قال healthy -> نوقف مباشرة
        if self.is_healthy_class(predicted_class):
            result = {
                "predicted_class": predicted_class,
                "plant_name": prototype_plant_name,
                "health_score": 100.0,
                "disease_score": 0.0,
                "severity_level": "HEALTHY",
                "treatments": []
            }
        else:
            healthy_prototype = np.load(prototype_path).astype(np.float32)
            similarity = self.cosine_similarity_np(feature_vec, healthy_prototype)
            health_score, disease_score, severity_percent = self.convert_similarity_to_scores(similarity)
            severity_level = self.convert_percent_to_severity_level(severity_percent)

            treatments = []
            if self.treatment_mgr is not None:
                treatments = self.treatment_mgr.get_treatments(predicted_class, severity_level)

            result = {
                "predicted_class": predicted_class,
                "plant_name": prototype_plant_name,
                "health_score": round(health_score, 2),
                "disease_score": round(disease_score, 2),
                "severity_level": severity_level,
                "treatments": treatments
            }

        self.current_result = result
        self.update_result_ui(result)

    def update_result_ui(self, result):
        severity_text, color, bg_color = self.severity_style(result["severity_level"])
        self.lbl_severity.configure(text=severity_text, text_color=color, fg_color=bg_color)
        self.lbl_health_score.configure(text=f"{result['health_score']}%")
        self.lbl_disease_score.configure(text=f"{result['disease_score']}%")

        if result["severity_level"] == "HEALTHY":
            self.hide_treatment_headers_for_healthy()
            self.display_healthy_message()
            return

        self.show_treatment_headers()
        self._update_rank_selector(result["treatments"])

    def show_unknown_case(self, result_img):
        self.lbl_disease_value.configure(text="Ambiguous Diagnosis")
        self.lbl_severity.configure(text="⚪ Unknown", text_color=UITheme.UNKNOWN, fg_color=UITheme.UNKNOWN_BG)
        self.lbl_health_score.configure(text="-")
        self.lbl_disease_score.configure(text="-")

        self.show_treatment_headers()

        self.display_image(self.canvas_result, result_img.copy(), "Result")

        for child in self.rank_buttons_frame.winfo_children():
            child.destroy()

        self.lbl_selected_type.configure(text="-")
        self.lbl_treatment_title.configure(text="No treatment recommendation")

        self.txt_treatment.config(state=tk.NORMAL)
        self.txt_treatment.delete("1.0", tk.END)
        self.txt_treatment.insert(
            tk.END,
            "Unable to determine a reliable class for this image.\n\nPlease try a clearer image."
        )
        self.txt_treatment.config(state=tk.DISABLED)

    def _update_rank_selector(self, recommendations):
        for child in self.rank_buttons_frame.winfo_children():
            child.destroy()

        if not recommendations:
            lbl = ctk.CTkLabel(
                self.rank_buttons_frame,
                text="No treatment options found.",
                text_color=UITheme.MUTED
            )
            lbl.pack(anchor="w")

            self.lbl_selected_type.configure(text="-")
            self.lbl_treatment_title.configure(text="No treatment recommendation")

            self.txt_treatment.config(state=tk.NORMAL)
            self.txt_treatment.delete("1.0", tk.END)
            self.txt_treatment.insert(
                tk.END,
                "No specific treatment found for this diagnosis.\n\nPlease consult an agricultural expert."
            )
            self.txt_treatment.config(state=tk.DISABLED)
            return

        ranks = sorted(list(set([
            str(r.get("rank", "1"))
            for r in recommendations
            if isinstance(r, dict)
        ])))

        btn_frame = ctk.CTkFrame(self.rank_buttons_frame, fg_color="transparent")
        btn_frame.pack(anchor="w")

        self.rank_btns = []

        for r in ranks:
            btn = ctk.CTkButton(
                btn_frame,
                text=f"Option {r}",
                command=lambda rank=r: self._switch_rank(recommendations, rank),
                fg_color="#252542",
                hover_color="#2D2D4F",
                text_color="#9da1ca",
                border_width=1,
                border_color="#3A3A6A",
                corner_radius=8,
                height=32,
                width=100
            )
            btn.pack(side="left", padx=(0, 8))
            self.rank_btns.append((r, btn))

        self._switch_rank(recommendations, ranks[0])

    def _switch_rank(self, recommendations, rank):
        self.active_rank.set(rank)

        for r, btn in self.rank_btns:
            if r == rank:
                btn.configure(fg_color="#302a5e", text_color="#ffffff", border_color="#5d4ef5")
            else:
                btn.configure(fg_color="#252542", text_color="#9da1ca", border_color="#3A3A6A")

        self.display_treatment(recommendations)

    def display_treatment(self, recommendations):
        self.show_treatment_headers()

        self.txt_treatment.config(state=tk.NORMAL)
        self.txt_treatment.delete("1.0", tk.END)

        shown_any = False

        for item in recommendations:
            if not isinstance(item, dict):
                continue

            rank = str(item.get("rank", "1"))
            if rank != self.active_rank.get():
                continue

            shown_any = True

            t_type = item.get("treatment_type", "")
            m_action = item.get("mode_of_action", "")
            description = item.get("description", "")
            precautions = item.get("precautions", "")
            title_guess = item.get("treatment_name") or item.get("name") or f"Treatment Option {rank}"

            selected_type_summary = f"{t_type} - {m_action}" if (t_type and m_action) else (t_type or m_action or "-")

            self.lbl_treatment_title.configure(text=title_guess)
            self.lbl_selected_type.configure(text=selected_type_summary)

            self.txt_treatment.insert(tk.END, f"{description}\n", "body")

            if precautions:
                self.txt_treatment.insert(tk.END, "\nPrecautions:\n", "section_title")
                self.txt_treatment.insert(tk.END, f"{precautions}\n", "body")

            break

        if not shown_any:
            self.lbl_selected_type.configure(text="-")
            self.lbl_treatment_title.configure(text="No treatment recommendation")
            self.txt_treatment.insert(
                tk.END,
                "No specific treatment found for this diagnosis.\n\nPlease consult an agricultural expert."
            )

        self.txt_treatment.tag_configure(
            "section_title",
            font=("Roboto", 14, "bold"),
            foreground="#f4a261",
            spacing1=6,
            spacing3=4
        )
        self.txt_treatment.tag_configure(
            "body",
            font=("Roboto", 13),
            foreground="#cbd5e1"
        )

        self.txt_treatment.config(state=tk.DISABLED)

    def display_healthy_message(self):
        for child in self.rank_buttons_frame.winfo_children():
            child.destroy()

        self.hide_treatment_headers_for_healthy()

        self.txt_treatment.config(state=tk.NORMAL)
        self.txt_treatment.delete("1.0", tk.END)

        self.txt_treatment.insert(tk.END, "Leaf is healthy\n\n", "healthy_header")
        self.txt_treatment.insert(
            tk.END,
            "No disease lesions or treatment actions are required.",
            "healthy_body"
        )

        self.txt_treatment.tag_configure(
            "healthy_header",
            font=("Roboto", 16, "bold"),
            foreground="#10b981"
        )
        self.txt_treatment.tag_configure(
            "healthy_body",
            font=("Roboto", 13),
            foreground="#cbd5e1"
        )

        self.txt_treatment.config(state=tk.DISABLED)

    def _set_status(self, text, color=None):
        self.lbl_status.configure(text=text)
        if color:
            self.status_dot.configure(text_color=color)

    def display_image(self, canvas, img, label):
        canvas.update_idletasks()

        canvas_w = canvas.winfo_width()
        canvas_h = canvas.winfo_height()

        if canvas_w < 50 or canvas_h < 50:
            canvas.delete("all")
            canvas.create_rectangle(0, 0, max(canvas_w, 10), max(canvas_h, 10), fill=UITheme.BG, outline="")
            return

        img_copy = img.copy()
        img_copy.thumbnail((canvas_w - 16, canvas_h - 16))

        tk_img = ImageTk.PhotoImage(img_copy)

        canvas.delete("all")
        canvas.create_rectangle(0, 0, canvas_w, canvas_h, fill=UITheme.BG, outline="")
        canvas.create_image(canvas_w // 2, canvas_h // 2, image=tk_img)

        if label == "Original":
            self.tk_img_original = tk_img
            self.current_original_pil = img.copy()
        else:
            self.tk_img_result = tk_img
            self.current_result_pil = img.copy()

    def _on_resize(self, event=None):
        if self._resize_after_id is not None:
            try:
                self.root.after_cancel(self._resize_after_id)
            except Exception:
                pass

        self._resize_after_id = self.root.after(120, self._redraw_images_after_resize)

    def _redraw_images_after_resize(self):
        try:
            if self.current_original_pil is not None:
                self.display_image(self.canvas_original, self.current_original_pil, "Original")
            if self.current_result_pil is not None:
                self.display_image(self.canvas_result, self.current_result_pil, "Result")
        except Exception:
            pass


if __name__ == "__main__":
    root = ctk.CTk()

    try:
        from ctypes import windll
        windll.shcore.SetProcessDpiAwareness(1)
    except Exception:
        pass

    app = PlantPrototypeLeafSegGUI(root)
    root.mainloop()