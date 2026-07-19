import os
import numpy as np
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from PIL import Image, ImageTk
import supervision as sv

# Torch and Model imports
import torch
import torch.nn as nn
from torchvision import models, transforms

try:
    import timm
    TIMM_AVAILABLE = True
except ImportError:
    TIMM_AVAILABLE = False

# RF-DETR import
try:
    from rfdetr import RFDETRSegNano
    RFDETR_AVAILABLE = True
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
except ImportError:
    RFDETR_AVAILABLE = False
    DEVICE = "cpu"

# Treatment Manager import
try:
    from treatment_manager import TreatmentManager
    TM_AVAILABLE = True
except ImportError:
    TM_AVAILABLE = False


# ========================= CTK WRAPPER =========================
import customtkinter as ctk

ctk.set_appearance_mode("Dark")
ctk.set_default_color_theme("blue")

class ModernLabel(ctk.CTkLabel):
    def __init__(self, master=None, text="", bg=None, fg=None, font=None, **kwargs):
        for k in ['highlightthickness', 'bd', 'relief', 'padx', 'pady']: kwargs.pop(k, None)
        if font and isinstance(font, tuple) and len(font) >= 2: font = (font[0], font[1])
        super().__init__(master, text=text, fg_color=bg or "transparent", text_color=fg, font=font, **kwargs)
    def config(self, text=None, bg=None, fg=None, **kwargs):
        if text is not None: kwargs['text'] = text
        if bg is not None: kwargs['fg_color'] = bg
        if fg is not None: kwargs['text_color'] = fg
        self.configure(**kwargs)

class ModernFrame(ctk.CTkFrame):
    def __init__(self, master=None, bg=None, **kwargs):
        for k in ['highlightthickness', 'highlightbackground', 'bd', 'relief', 'padx', 'pady']: kwargs.pop(k, None)
        super().__init__(master, fg_color=bg or "transparent", corner_radius=12, **kwargs)

class ModernButton(ctk.CTkButton):
    def __init__(self, master=None, text="", command=None, bg=None, fg=None, activebackground=None, font=None, **kwargs):
        for k in ['activeforeground', 'bd', 'relief', 'padx', 'pady', 'cursor']: kwargs.pop(k, None)
        if font and isinstance(font, tuple) and len(font) >= 2: font = (font[0], font[1])
        super().__init__(master, text=text, command=command, fg_color=bg, text_color=fg or "white", hover_color=activebackground, font=font, **kwargs)
    def config(self, text=None, bg=None, fg=None, **kwargs):
        if text is not None: kwargs['text'] = text
        if bg is not None: kwargs['fg_color'] = bg
        if fg is not None: kwargs['text_color'] = fg
        self.configure(**kwargs)

class ModernProgressbar(ctk.CTkProgressBar):
    def __init__(self, master=None, **kwargs):
        for k in ['style', 'orient', 'mode', 'maximum', 'value']: kwargs.pop(k, None)
        super().__init__(master, **kwargs)
        self.set(0)
    def __setitem__(self, key, value):
        if key == "value": self.set(value / 100.0)

class tk_mock:
    Frame = ModernFrame
    Label = ModernLabel
    Button = ModernButton
    Tk = ctk.CTk
    Canvas = tk.Canvas
    Text = tk.Text
    StringVar = tk.StringVar
    X = tk.X; Y = tk.Y; BOTH = tk.BOTH; LEFT = tk.LEFT; RIGHT = tk.RIGHT; TOP = tk.TOP; BOTTOM = tk.BOTTOM
    END = tk.END; FLAT = tk.FLAT; DISABLED = tk.DISABLED; NORMAL = tk.NORMAL; WORD = tk.WORD
tk = tk_mock

class ttk_mock:
    Progressbar = ModernProgressbar
    Style = ttk.Style
ttk = ttk_mock

# ========================= THEME =========================
class UITheme:
    BG = "#1e1e2e"
    HEADER_BG = "#181825"
    BANNER_BG = "#11111b"

    CARD = "#313244"
    CARD_SOFT = "#45475a"
    BORDER = "#585b70"
    IMAGE_BG = "#11111b"

    PRIMARY = "#89b4fa"
    PRIMARY_HOVER = "#b4befe"
    SUCCESS = "#a6e3a1"
    SUCCESS_BG = "#40504b"
    WARNING = "#f9e2af"
    WARNING_BG = "#4f4940"
    DANGER = "#f38ba8"
    DANGER_BG = "#583f47"

    TEXT = "#cdd6f4"
    TITLE = "#f5c2e7"
    MUTED = "#a6adc8"

    FONT_TITLE = ("Roboto", 24, "bold")
    FONT_SECTION = ("Roboto", 16, "bold")
    FONT_CARD_TITLE = ("Roboto", 16, "bold")
    FONT_LABEL = ("Roboto", 13, "bold")
    FONT_BODY = ("Roboto", 13)
    FONT_BODY_BIG = ("Roboto", 14)
    FONT_BUTTON = ("Roboto", 13, "bold")
    FONT_SMALL = ("Roboto", 12)
    FONT_STATUS = ("Roboto", 13)
    FONT_VALUE_BIG = ("Roboto", 20, "bold")


# ========================= MODEL =========================
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

    def forward(self, x):
        fe = self.eff_features(x)
        fe = self.eff_pool(fe)
        fe = torch.flatten(fe, 1)

        fs = self.swin(x)

        emb = torch.cat([fe, fs], dim=1)
        logits = self.classifier(emb)
        return logits, emb


# ========================= GUI =========================
class PlantSegmentationGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Plant Disease Diagnosis & Treatment System")
        self.root.geometry("1500x920")
        self.root.minsize(1320, 820)
        
        # ctk.CTk uses fg_color instead of bg
        try:
            self.root.configure(fg_color=UITheme.BG)
        except:
            self.root.configure(bg=UITheme.BG)

        self.base_path = os.path.dirname(os.path.abspath(__file__))

        self.tk_img_original = None
        self.tk_img_segment = None
        self.current_original_pil = None
        self.current_segment_pil = None
        self._resize_after_id = None

        self.current_disease_class = None
        self.current_ratio = None

        # Classification class names
        self.classifier_classes = [
            'Apple___Apple_scab',
            'Apple___Black_rot',
            'Apple___Cedar_apple_rust',
            'Apple___healthy',
            'Cherry_(including_sour)___Powdery_mildew',
            'Cherry_(including_sour)___healthy',
            'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot',
            'Corn_(maize)___Common_rust_',
            'Corn_(maize)___Northern_Leaf_Blight',
            'Corn_(maize)___healthy',
            'Grape___Black_rot',
            'Grape___Esca_(Black_Measles)',
            'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)',
            'Grape___healthy',
            'Orange___Haunglongbing_(Citrus_greening)',
            'Peach___Bacterial_spot',
            'Peach___healthy',
            'Pepper,_bell___Bacterial_spot',
            'Pepper,_bell___healthy',
            'Potato___Early_blight',
            'Potato___Late_blight',
            'Potato___healthy',
            'Squash - Healthy',
            'Squash___Powdery_mildew',
            'Strawberry___Leaf_scorch',
            'Strawberry___healthy',
            'Tomato___Bacterial_spot',
            'Tomato___Early_blight',
            'Tomato___Late_blight',
            'Tomato___Leaf_Mold',
            'Tomato___Septoria_leaf_spot',
            'Tomato___Spider_mites Two-spotted_spider_mite',
            'Tomato___Target_Spot',
            'Tomato___Tomato_Yellow_Leaf_Curl_Virus',
            'Tomato___Tomato_mosaic_virus',
            'Tomato___healthy',
            'orange___Healthy'
        ]

        # Segmentation class names
        self.segment_class_names = ['background', 'disease', 'leaf']

        # Severity categories
        # Early from 1 to 20
        # Moderate from 21 to 40
        # Severe from 41 to up
        self.TREATMENT_CONFIG = [
            {"min_ratio": 0.41, "category": "SEVERE", "color": UITheme.DANGER, "bg": UITheme.DANGER_BG},
            {"min_ratio": 0.21, "category": "MODERATE", "color": UITheme.WARNING, "bg": UITheme.WARNING_BG},
            {"min_ratio": 0.01, "category": "EARLY", "color": UITheme.SUCCESS, "bg": UITheme.SUCCESS_BG}
        ]

        self.active_rank = tk.StringVar(value="1")
        self.rank_btns = []

        self._setup_styles()
        self._load_models()
        self._setup_ui()

        self.root.bind("<Configure>", self._on_resize)

    # ========================= STYLES =========================
    def _setup_styles(self):
        self.style = ttk.Style()
        try:
            self.style.theme_use("clam")
        except Exception:
            pass

        self.style.configure(
            "Modern.Horizontal.TProgressbar",
            troughcolor="#e9eef5",
            bordercolor="#e9eef5",
            background=UITheme.SUCCESS,
            lightcolor=UITheme.SUCCESS,
            darkcolor=UITheme.SUCCESS,
            thickness=10
        )

    # ========================= LOAD MODELS =========================
    def _load_models(self):
        print("Loading models...")

        # Classification model
        self.cls_model = None
        cls_model_path = os.path.join(self.base_path, "best_weight_model.pth")

        if TIMM_AVAILABLE and os.path.exists(cls_model_path):
            try:
                num_classes = len(self.classifier_classes)
                self.cls_model = EffB0_SwinTiny_Fusion(num_classes=num_classes)
                state_dict = torch.load(cls_model_path, map_location=DEVICE)
                self.cls_model.load_state_dict(state_dict)
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

                print(f"Fusion classification model loaded on {DEVICE}")
            except Exception as e:
                print(f"Error loading Fusion model: {e}")
        else:
            print(f"Fusion model or timm missing. Path: {cls_model_path}")

        # Segmentation models
        self.leaf_model = None
        self.lesion_model = None

        leaf_path = os.path.join(self.base_path, "leave.pth")
        lesion_path = os.path.join(self.base_path, "lession.pth")

        if RFDETR_AVAILABLE:
            print(f"Using device: {DEVICE}")

            if os.path.exists(leaf_path):
                try:
                    self.leaf_model = RFDETRSegNano(pretrain_weights=leaf_path)
                    torch_mod = self.leaf_model.model.model
                    torch_mod.to(DEVICE)

                    if DEVICE == "cuda":
                        torch_mod.half()
                        self.leaf_model.optimize_for_inference(dtype=torch.float16)
                    else:
                        self.leaf_model.optimize_for_inference()

                    print("Leaf segmentation model loaded")
                except Exception as e:
                    print(f"Error loading leaf model: {e}")
            else:
                print(f"Leaf model not found at {leaf_path}")

            if os.path.exists(lesion_path):
                try:
                    self.lesion_model = RFDETRSegNano(pretrain_weights=lesion_path)
                    torch_mod = self.lesion_model.model.model
                    torch_mod.to(DEVICE)

                    if DEVICE == "cuda":
                        torch_mod.half()
                        self.lesion_model.optimize_for_inference(dtype=torch.float16)
                    else:
                        self.lesion_model.optimize_for_inference()

                    print("Lesion segmentation model loaded")
                except Exception as e:
                    print(f"Error loading lesion model: {e}")
            else:
                print(f"Lesion model not found at {lesion_path}")
        else:
            print("RF-DETR library not available")

        # Treatment manager
        self.treatment_mgr = None
        treatments_path = os.path.join(self.base_path, "treatments2.json")
        if TM_AVAILABLE and os.path.exists(treatments_path):
            try:
                self.treatment_mgr = TreatmentManager(treatments_path, top_k=5)
                print("Treatment manager initialized")
            except Exception as e:
                print(f"Error initializing TreatmentManager: {e}")
        else:
            print("TreatmentManager or treatments2.json missing")

    # ========================= HELPERS =========================
    def _create_info_row(self, parent, label_text, value_text="-", value_font=None):
        row = tk.Frame(parent, bg=UITheme.CARD)
        row.pack(fill=tk.X, pady=2)

        tk.Label(
            row,
            text=label_text,
            bg=UITheme.CARD,
            fg=UITheme.MUTED,
            font=UITheme.FONT_BODY_BIG,
            width=10,
            anchor="w"
        ).pack(side=tk.LEFT)

        value = tk.Label(
            row,
            text=value_text,
            bg=UITheme.CARD,
            fg=UITheme.TEXT,
            font=value_font or UITheme.FONT_BODY_BIG,
            anchor="w",
            justify="left",
            wraplength=280
        )
        value.pack(side=tk.LEFT, fill=tk.X, expand=True)
        return value

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

        if raw == "Squash - Healthy":
            return "Squash", "Healthy"

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

    def _severity_visual_percent(self, ratio):
        if ratio is None:
            return 0
        percent = ratio * 100
        if percent > 0 and percent < 1:
            return 1

        return max(0, min(int(round(percent)), 100))

    def _set_status(self, text, color=None):
        self.lbl_status.configure(text=text)
        if color:
            self.status_dot.configure(text_color=color)

    def _update_type_tabs(self, selected_type_summary):
        pass


    # ========================= UI =========================
    def _setup_ui(self):
        self._setup_ui_modern()
        
    def _setup_ui_modern(self):
        import customtkinter as ctk
        
        COLOR_BG = "#131320"
        COLOR_CARD = "#1C1C31"
        COLOR_CARD_SOFT = "#252542"
        COLOR_PRIMARY = "#5d4ef5"
        COLOR_TEXT = "#F4F4F9"
        COLOR_MUTED = "#9da1ca"
        COLOR_BORDER = "#2B2B4A"
        SECTION_COLOR = "#f4a261"
        
        self.root.configure(fg_color=COLOR_BG)
        self.root.grid_columnconfigure(0, weight=1)
        self.root.grid_rowconfigure(1, weight=1)
        
        # --- HEADER ---
        header = ctk.CTkFrame(self.root, fg_color="transparent")
        header.grid(row=0, column=0, sticky="ew", padx=40, pady=(35, 10))
        header.grid_columnconfigure(0, weight=1)
        header.grid_columnconfigure(1, weight=0)
        
        title_frame = ctk.CTkFrame(header, fg_color="transparent")
        title_frame.grid(row=0, column=0, sticky="w")
        
        title_lbl = ctk.CTkLabel(title_frame, text="Plant Disease Diagnosis & Treatment Recommendation", 
                                 font=("Roboto", 24, "bold"), text_color=COLOR_TEXT)
        title_lbl.pack(anchor="w")
        
        status_inner = ctk.CTkFrame(title_frame, fg_color="transparent")
        status_inner.pack(anchor="w", pady=(5, 0))
        self.status_dot = ctk.CTkLabel(status_inner, text="✔", font=("Roboto", 18, "bold"), text_color="#10b981")
        self.status_dot.pack(side="left", padx=(0, 8))
        self.lbl_status = ctk.CTkLabel(status_inner, text="Ready", font=("Roboto", 14), text_color=COLOR_MUTED)
        self.lbl_status.pack(side="left")

        # Hidden elements to preserve backend logic hooks
        self.btn_tab_bio = tk.Button(self.root) 
        self.btn_tab_chem = tk.Button(self.root)
        self.lbl_selected_summary = tk.Label(self.root) 
        self.lbl_plant_value = ctk.CTkLabel(self.root)
        self.lbl_percent = ctk.CTkLabel(self.root)
        self.lbl_analysis_note = ctk.CTkLabel(self.root)
        self.progress = ttk.Progressbar(self.root)
        
        self.btn_select = ctk.CTkButton(header, text="Upload Leaf Image", 
                                        command=self.select_image,
                                        font=("Roboto", 15, "bold"),
                                        fg_color=COLOR_PRIMARY, hover_color="#4f3fe0", text_color="white", corner_radius=8, height=42, width=160)
        self.btn_select.grid(row=0, column=1, sticky="e")
        
        # --- MAIN 3-COLUMN VIEW ---
        main_view = ctk.CTkFrame(self.root, fg_color="transparent")
        main_view.grid(row=1, column=0, sticky="nsew", padx=35, pady=(20, 35))
        main_view.grid_rowconfigure(0, weight=1)
        
        main_view.grid_columnconfigure(0, weight=4, uniform="maincols", minsize=400)
        main_view.grid_columnconfigure(1, weight=2, uniform="maincols", minsize=320)
        main_view.grid_columnconfigure(2, weight=4, uniform="maincols", minsize=400)
        

        
        def create_card(parent, title, icon=""):
            card = ctk.CTkFrame(parent, fg_color=COLOR_CARD, corner_radius=12, border_width=1, border_color=COLOR_BORDER)
            lbl = ctk.CTkLabel(card, text=f"{icon} {title}" if icon else title, font=("Roboto", 15, "bold"), text_color=COLOR_TEXT)
            #lbl.pack(anchor="w", padx=20, pady=(15, 5))
            lbl.pack(anchor="w", padx=15, pady=(6, 3))
            return card
            
        # -- COLUMN 1: Images --
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
        
        card_seg = create_card(col1, "Segmentation")
        card_seg.grid(row=1, column=0, sticky="nsew", pady=(10, 0))
        seg_border = ctk.CTkFrame(card_seg, fg_color=COLOR_CARD_SOFT, corner_radius=8)
        seg_border.pack(fill="both", expand=True, padx=20, pady=(5, 20))
        self.canvas_segment = tk.Canvas(seg_border, bg=COLOR_BG, highlightthickness=0, bd=0)
        self.canvas_segment.pack(fill="both", expand=True, padx=3, pady=3)
        
        # -- COLUMN 2: Disease Info --
        col2 = ctk.CTkFrame(main_view, fg_color="transparent")
        col2.grid(row=0, column=1, sticky="nsew", padx=10)
        col2.grid_columnconfigure(0, weight=1)
        
        card_mid = create_card(col2, "Disease Information")
        card_mid.pack(fill="both", expand=True)
        
        disease_box = ctk.CTkFrame(card_mid, fg_color=COLOR_CARD_SOFT, corner_radius=8)
        disease_box.pack(fill="x", padx=20, pady=(15, 25))
        self.lbl_disease_value = ctk.CTkLabel(disease_box, text="-", font=("Roboto", 17, "bold"), text_color=COLOR_TEXT, anchor="w")
        self.lbl_disease_value.pack(side="left", padx=15, pady=12)
        
        ctk.CTkLabel(card_mid, text="Severity", font=("Roboto", 14), text_color=COLOR_TEXT).pack(anchor="w", padx=20)
        self.lbl_severity = ctk.CTkLabel(card_mid, text="🟢 Healthy", font=("Roboto", 13, "bold"), 
                                         fg_color="#183d33", text_color="#10b981", corner_radius=6, anchor="w")
        self.lbl_severity.pack(fill="x", padx=20, pady=(8, 35), ipady=8)
        # Note: Padding tweaked for modern vertical rhythm

        ctk.CTkLabel(card_mid, text="Treatment Options", font=("Roboto", 14), text_color=COLOR_TEXT).pack(anchor="w", padx=20)
        self.rank_buttons_frame = ctk.CTkFrame(card_mid, fg_color="transparent")
        self.rank_buttons_frame.pack(fill="x", padx=20, pady=(8, 20))
        
        # -- COLUMN 3: Recommended Treatment --
        col3 = ctk.CTkFrame(main_view, fg_color="transparent")
        col3.grid(row=0, column=2, sticky="nsew", padx=(10, 0))
        col3.grid_columnconfigure(0, weight=1)
        
        card_right = create_card(col3, "Recommended Treatment", icon="🔬")
        card_right.pack(fill="both", expand=True)
        
        self.lbl_name_title = ctk.CTkLabel(card_right,text="Name:",font=("Roboto", 14, "bold"),text_color=SECTION_COLOR,anchor="w")
        self.lbl_name_title.pack(anchor="w", padx=20, pady=(12, 4))

        self.lbl_treatment_title = ctk.CTkLabel(
            card_right,
            text="",
            font=("Roboto", 14,"bold"),
            text_color="#cbd5e1",
            #text_color="#e5e7eb",
            justify="left",
            anchor="w",
            wraplength=420
        )
        self.lbl_treatment_title.pack(anchor="w", fill="x", padx=20, pady=(0, 14))

        self.lbl_type_title = ctk.CTkLabel(
            card_right,
            text="Type:",
            font=("Roboto", 14, "bold"),
            text_color=SECTION_COLOR,
            anchor="w"
        )
        self.lbl_type_title.pack(anchor="w", padx=20, pady=(0, 4))

        self.lbl_selected_type = ctk.CTkLabel(
            card_right,
            text="-",
            font=("Roboto", 14),
            text_color="#cbd5e1",
            justify="left",
            anchor="w",
            wraplength=420
        )
        self.lbl_selected_type.pack(anchor="w", fill="x", padx=20, pady=(0, 14))

        self.lbl_details_title = ctk.CTkLabel(
            card_right,
            text="Details:",
            font=("Roboto", 14, "bold"),
            text_color=SECTION_COLOR,
            anchor="w"
        )
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

    def _setup_ui_legacy_removed(self):
        # ================= HEADER =================
        header = tk.Frame(self.root, bg=UITheme.HEADER_BG, height=86)
        header.pack(fill=tk.X)
        header.pack_propagate(False)

        header_inner = tk.Frame(header, bg=UITheme.HEADER_BG)
        header_inner.pack(fill=tk.BOTH, expand=True, padx=18, pady=14)

        title_wrap = tk.Frame(header_inner, bg=UITheme.HEADER_BG)
        title_wrap.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        tk.Label(
            title_wrap,
            text="Plant Disease Diagnosis & Treatment System",
            bg=UITheme.HEADER_BG,
            fg=UITheme.TITLE,
            font=UITheme.FONT_TITLE
        ).pack(anchor="w")

        self.btn_select = tk.Button(
            header_inner,
            text="Upload Leaf Image",
            command=self.select_image,
            bg=UITheme.PRIMARY,
            fg="white",
            font=UITheme.FONT_BUTTON,
            padx=20,
            pady=10,
            bd=0,
            relief=tk.FLAT,
            activebackground=UITheme.PRIMARY_HOVER,
            activeforeground="white",
            cursor="hand2"
        )
        self.btn_select.pack(side=tk.RIGHT)

        # ================= BANNER =================
        banner_wrap = tk.Frame(self.root, bg=UITheme.BG)
        banner_wrap.pack(fill=tk.X, padx=14, pady=(10, 8))

        banner = tk.Frame(
            banner_wrap,
            bg=UITheme.BANNER_BG,
            highlightthickness=1,
            highlightbackground=UITheme.BORDER,
            bd=0,
            height=95
        )
        banner.pack(fill=tk.X)
        banner.pack_propagate(False)

        banner_inner = tk.Frame(banner, bg=UITheme.BANNER_BG)
        banner_inner.pack(fill=tk.BOTH, expand=True, padx=20, pady=16)

        tk.Label(
            banner_inner,
            text="AI Analysis Status",
            bg=UITheme.BANNER_BG,
            fg=UITheme.TITLE,
            font=UITheme.FONT_SECTION
        ).pack(anchor="w")

        status_row = tk.Frame(banner_inner, bg=UITheme.BANNER_BG)
        status_row.pack(anchor="w", pady=(12, 0))

        self.status_dot = tk.Label(
            status_row,
            text="●",
            fg=UITheme.SUCCESS,
            bg=UITheme.BANNER_BG,
            font=("Segoe UI", 12, "bold")
        )
        self.status_dot.pack(side=tk.LEFT)

        self.lbl_status = tk.Label(
            status_row,
            text="Ready",
            bg=UITheme.BANNER_BG,
            fg=UITheme.MUTED,
            font=UITheme.FONT_STATUS
        )
        self.lbl_status.pack(side=tk.LEFT, padx=(8, 0))

        # ================= MAIN WRAPPER =================
        main_wrapper = tk.Frame(self.root, bg=UITheme.BG)
        main_wrapper.pack(fill=tk.BOTH, expand=True, padx=14, pady=(6, 14))

        # ================= TOP SECTION =================
        top_section = tk.Frame(main_wrapper, bg=UITheme.BG, height=330)
        top_section.pack(fill=tk.X, pady=(0, 12))
        top_section.pack_propagate(False)

        top_section.grid_columnconfigure(0, weight=1)
        top_section.grid_columnconfigure(1, weight=1)
        top_section.grid_columnconfigure(2, weight=1)
        top_section.grid_rowconfigure(0, weight=1)

        # ---------- Original card ----------
        original_card = tk.Frame(
            top_section,
            bg=UITheme.CARD,
            highlightthickness=1,
            highlightbackground=UITheme.BORDER,
            bd=0
        )
        original_card.grid(row=0, column=0, sticky="nsew", padx=(0, 8))

        original_head = tk.Frame(original_card, bg=UITheme.CARD)
        original_head.pack(fill=tk.X, padx=14, pady=(12, 6))

        tk.Label(
            original_head,
            text="Original Image",
            bg=UITheme.CARD,
            fg=UITheme.TITLE,
            font=UITheme.FONT_CARD_TITLE
        ).pack(anchor="w")

        original_body = tk.Frame(original_card, bg=UITheme.CARD)
        original_body.pack(fill=tk.BOTH, expand=True, padx=14, pady=(4, 14))

        self.canvas_original = tk.Canvas(
            original_body,
            bg=UITheme.IMAGE_BG,
            highlightthickness=0,
            bd=0
        )
        self.canvas_original.pack(fill=tk.BOTH, expand=True)

        # ---------- Segmentation card ----------
        seg_card = tk.Frame(
            top_section,
            bg=UITheme.CARD,
            highlightthickness=1,
            highlightbackground=UITheme.BORDER,
            bd=0
        )
        seg_card.grid(row=0, column=1, sticky="nsew", padx=8)

        seg_head = tk.Frame(seg_card, bg=UITheme.CARD)
        seg_head.pack(fill=tk.X, padx=14, pady=(12, 6))

        tk.Label(
            seg_head,
            text="Segmentation Result",
            bg=UITheme.CARD,
            fg=UITheme.TITLE,
            font=UITheme.FONT_CARD_TITLE
        ).pack(anchor="w")

        seg_body = tk.Frame(seg_card, bg=UITheme.CARD)
        seg_body.pack(fill=tk.BOTH, expand=True, padx=14, pady=(4, 14))

        self.canvas_segment = tk.Canvas(
            seg_body,
            bg=UITheme.IMAGE_BG,
            highlightthickness=0,
            bd=0
        )
        self.canvas_segment.pack(fill=tk.BOTH, expand=True)

        # ---------- Disease analysis card ----------
        diag_card = tk.Frame(
            top_section,
            bg=UITheme.CARD,
            highlightthickness=1,
            highlightbackground=UITheme.BORDER,
            bd=0
        )
        diag_card.grid(row=0, column=2, sticky="nsew", padx=(8, 0))

        diag_head = tk.Frame(diag_card, bg=UITheme.CARD)
        diag_head.pack(fill=tk.X, padx=14, pady=(12, 6))

        tk.Label(
            diag_head,
            text="Disease Analysis",
            bg=UITheme.CARD,
            fg=UITheme.TITLE,
            font=UITheme.FONT_CARD_TITLE
        ).pack(anchor="w")

        diag_body = tk.Frame(diag_card, bg=UITheme.CARD)
        diag_body.pack(fill=tk.BOTH, expand=True, padx=14, pady=(4, 14))

        self.lbl_plant_value = self._create_info_row(diag_body, "Plant", "-")
        self.lbl_disease_value = self._create_info_row(diag_body, "Disease", "-", UITheme.FONT_VALUE_BIG)

        self.lbl_severity = tk.Label(
            diag_body,
            text="Severity: -",
            bg="#f3f4f6",
            fg=UITheme.MUTED,
            font=UITheme.FONT_LABEL,
            padx=12,
            pady=7
        )
        self.lbl_severity.pack(anchor="w", pady=(14, 10))

        percent_row = tk.Frame(diag_body, bg=UITheme.CARD)
        percent_row.pack(fill=tk.X)

        tk.Label(
            percent_row,
            text="Estimated severity",
            bg=UITheme.CARD,
            fg=UITheme.MUTED,
            font=UITheme.FONT_BODY
        ).pack(side=tk.LEFT)

        self.lbl_percent = tk.Label(
            percent_row,
            text="0%",
            bg=UITheme.CARD,
            fg=UITheme.TEXT,
            font=("Segoe UI Semibold", 14, "bold")
        )
        self.lbl_percent.pack(side=tk.RIGHT)

        self.progress = ttk.Progressbar(
            diag_body,
            style="Modern.Horizontal.TProgressbar",
            orient="horizontal",
            mode="determinate",
            maximum=100,
            value=0
        )
        self.progress.pack(fill=tk.X, pady=(10, 12))

        self.lbl_analysis_note = tk.Label(
            diag_body,
            text="Process an image to see classification and severity details.",
            bg=UITheme.CARD,
            fg=UITheme.MUTED,
            font=UITheme.FONT_BODY,
            justify="left",
            wraplength=300
        )
        self.lbl_analysis_note.pack(anchor="w")

        # ================= BOTTOM SECTION =================
        bottom_section = tk.Frame(
            main_wrapper,
            bg=UITheme.CARD,
            highlightthickness=1,
            highlightbackground=UITheme.BORDER,
            bd=0
        )
        bottom_section.pack(fill=tk.BOTH, expand=True)

        bottom_head = tk.Frame(bottom_section, bg=UITheme.CARD)
        bottom_head.pack(fill=tk.X, padx=14, pady=(12, 6))

        tk.Label(
            bottom_head,
            text="Recommended Treatment",
            bg=UITheme.CARD,
            fg=UITheme.TITLE,
            font=UITheme.FONT_CARD_TITLE
        ).pack(anchor="w")

        bottom_body = tk.Frame(bottom_section, bg=UITheme.CARD)
        bottom_body.pack(fill=tk.BOTH, expand=True, padx=14, pady=(4, 14))

        bottom_body.grid_columnconfigure(0, weight=1)
        bottom_body.grid_columnconfigure(1, weight=2)
        bottom_body.grid_rowconfigure(0, weight=1)

        # ---------- Left treatment panel ----------
        options_panel = tk.Frame(
            bottom_body,
            bg=UITheme.CARD_SOFT,
            highlightthickness=1,
            highlightbackground=UITheme.BORDER
        )
        options_panel.grid(row=0, column=0, sticky="nsew", padx=(0, 10))

        options_inner = tk.Frame(options_panel, bg=UITheme.CARD_SOFT)
        options_inner.pack(fill=tk.BOTH, expand=True, padx=14, pady=14)

        self.treatment_type_tabs = tk.Frame(options_inner, bg=UITheme.CARD_SOFT)
        self.treatment_type_tabs.pack(fill=tk.X)

        self.btn_tab_bio = tk.Button(
            self.treatment_type_tabs,
            text="Biological Treatment",
            relief=tk.FLAT,
            bd=0,
            bg="#a7dbc0",
            fg="white",
            font=UITheme.FONT_BUTTON,
            padx=14,
            pady=10
        )
        self.btn_tab_bio.pack(side=tk.LEFT, padx=(0, 6))

        self.btn_tab_chem = tk.Button(
            self.treatment_type_tabs,
            text="Chemical Treatment",
            relief=tk.FLAT,
            bd=0,
            bg="#eef3fb",
            fg=UITheme.TEXT,
            font=UITheme.FONT_BUTTON,
            padx=14,
            pady=10
        )
        self.btn_tab_chem.pack(side=tk.LEFT)

        tk.Label(
            options_inner,
            text="Available options",
            bg=UITheme.CARD_SOFT,
            fg=UITheme.MUTED,
            font=UITheme.FONT_SMALL
        ).pack(anchor="w", pady=(16, 6))

        self.rank_buttons_frame = tk.Frame(options_inner, bg=UITheme.CARD_SOFT)
        self.rank_buttons_frame.pack(fill=tk.X, pady=(0, 12))

        self.lbl_selected_type = tk.Label(
            options_inner,
            text="Type: -",
            bg=UITheme.CARD_SOFT,
            fg=UITheme.TEXT,
            font=UITheme.FONT_LABEL,
            anchor="w",
            justify="left",
            wraplength=320
        )
        self.lbl_selected_type.pack(fill=tk.X, pady=(8, 8))

        self.lbl_selected_summary = tk.Label(
            options_inner,
            text="Process an image to display treatment options.",
            bg=UITheme.CARD_SOFT,
            fg=UITheme.MUTED,
            font=UITheme.FONT_BODY,
            justify="left",
            anchor="nw",
            wraplength=320
        )
        self.lbl_selected_summary.pack(fill=tk.BOTH, expand=True)

        # ---------- Right treatment detail panel ----------
        details_panel = tk.Frame(
            bottom_body,
            bg=UITheme.CARD,
            highlightthickness=1,
            highlightbackground=UITheme.BORDER
        )
        details_panel.grid(row=0, column=1, sticky="nsew")

        details_inner = tk.Frame(details_panel, bg=UITheme.CARD)
        details_inner.pack(fill=tk.BOTH, expand=True, padx=18, pady=16)

        self.lbl_treatment_title = tk.Label(
            details_inner,
            text="ttttttttttttttttttttttttttttttttttttttttt",
            bg=UITheme.CARD,
            fg=UITheme.TITLE,
            font=("Segoe UI Semibold", 16, "bold"),
            anchor="w",
            justify="left"
        )
        self.lbl_treatment_title.pack(anchor="w")

        self.txt_treatment = tk.Text(
            details_inner,
            font=UITheme.FONT_BODY_BIG,
            wrap=tk.WORD,
            bg=UITheme.CARD,
            fg=UITheme.TEXT,
            bd=0,
            relief=tk.FLAT,
            state=tk.DISABLED,
            padx=0,
            pady=10,
            spacing1=2,
            spacing2=3,
            spacing3=7,
            insertbackground=UITheme.TEXT
        )
        self.txt_treatment.pack(fill=tk.BOTH, expand=True)

        self._reset_ui_placeholders()

    def _reset_ui_placeholders(self):
        self.lbl_plant_value.configure(text="-")
        self.lbl_disease_value.configure(text="-")
        self.lbl_severity.configure(text="⚪ Unknown", text_color="#9da1ca", fg_color="#252542")
        self.lbl_percent.configure(text="0%")
        self.progress["value"] = 0
        self.lbl_analysis_note.configure(
            text="Process an image to see classification and severity details."
        )
        self.lbl_selected_type.configure(text="Type: -")
        self.lbl_selected_summary.config(text="Process an image to display treatment options.")
        #self.lbl_treatment_title.configure(text="Treatment details will appear here")
        self.lbl_treatment_title.configure(text="")

        self.txt_treatment.config(state=tk.NORMAL)
        self.txt_treatment.delete("1.0", tk.END)
        self.txt_treatment.insert(
            tk.END,
            "Upload a leaf image to run classification, segmentation, severity estimation, and treatment recommendation."
        )
        self.txt_treatment.config(state=tk.DISABLED)

        for child in self.rank_buttons_frame.winfo_children():
            child.destroy()

    # ========================= IMAGE SELECT =========================
    def select_image(self):
        file_path = filedialog.askopenfilename(
            filetypes=[("Image Files", "*.jpg *.jpeg *.png *.JPG *.PNG")]
        )
        if not file_path:
            return

        self._set_status(f"Processing: {os.path.basename(file_path)}", UITheme.WARNING)
        self.root.update_idletasks()
        self.process_image(file_path)
        self._set_status("Analysis Complete", UITheme.SUCCESS)

    # ========================= MAIN PROCESS =========================
    def process_image(self, image_path):
        self.canvas_original.delete("all")
        self.canvas_segment.delete("all")

        original_img = Image.open(image_path).convert("RGB")
        self.display_image(self.canvas_original, original_img, "Original")

        disease_class = "Unknown"

        # Classification
        if self.cls_model:
            try:
                input_tensor = self.cls_preprocess(original_img).unsqueeze(0).to(DEVICE)

                with torch.no_grad():
                    logits, _ = self.cls_model(input_tensor)
                    probs = torch.nn.functional.softmax(logits, dim=1)
                    top_prob, top_idx = torch.max(probs, 1)

                    if top_prob.item() > 0.3:
                        disease_class = self.classifier_classes[top_idx.item()]
                    else:
                        disease_class = "Ambiguous Diagnosis"

            except Exception as e:
                print(f"Classification error: {e}")
                import traceback
                traceback.print_exc()

        self.current_disease_class = disease_class
        plant_name, disease_name = self.split_plant_disease(disease_class)

        self.lbl_plant_value.configure(text=plant_name)
        self.lbl_disease_value.configure(text=f"{plant_name} {disease_name}" if disease_name != "Healthy" else f"{plant_name} Healthy")

        # Segmentation
        detections_list = []

        try:
            is_healthy = "healthy" in disease_class.lower()

            with torch.no_grad():
                if self.leaf_model:
                    leaf_dets = self.leaf_model.predict(original_img, threshold=0.2, iou_threshold=0.2)
                    if len(leaf_dets) > 0:
                        leaf_dets.class_id = np.full(len(leaf_dets), 2, dtype=int)
                        detections_list.append(leaf_dets)

                if self.lesion_model and not is_healthy:
                    lesion_dets = self.lesion_model.predict(original_img, threshold=0.2, iou_threshold=0.2)
                    if len(lesion_dets) > 0:
                        lesion_dets.class_id = np.full(len(lesion_dets), 1, dtype=int)
                        detections_list.append(lesion_dets)

            if not detections_list and not (self.leaf_model or self.lesion_model):
                messagebox.showwarning("Models Missing", "No segmentation models loaded")
                return

            detections = sv.Detections.merge(detections_list) if detections_list else sv.Detections.empty()

            if is_healthy:
                disease_area, leaf_area, ratio = 0, 0, 0.0
                if len(detections_list) > 0:
                    _, leaf_area, _ = self.calculate_ratio(detections)
            else:
                disease_area, leaf_area, ratio = self.calculate_ratio(detections)

            segmented_img = self.create_segmentation_visualization(original_img, detections)
            self.display_image(self.canvas_segment, segmented_img, "Segmented")

            if not self.lbl_name_title.winfo_ismapped():
                self.lbl_name_title.pack(anchor="w", padx=20, pady=(12, 4))

            if not self.lbl_treatment_title.winfo_ismapped():
                self.lbl_treatment_title.pack(anchor="w", fill="x", padx=20, pady=(0, 14))

            if not self.lbl_type_title.winfo_ismapped():
                self.lbl_type_title.pack(anchor="w", padx=20, pady=(0, 4))

            if not self.lbl_selected_type.winfo_ismapped():
                self.lbl_selected_type.pack(anchor="w", fill="x", padx=20, pady=(0, 14))

            self.current_ratio = ratio
            percent = self._severity_visual_percent(ratio)
            self.lbl_percent.configure(text=f"{percent}%")
            self.progress["value"] = percent

            if is_healthy:
                self.lbl_severity.configure(
                    text="🟢 Healthy (0% Infection)",
                    text_color="#10b981",
                    fg_color="#183d33"
                )

                self.lbl_name_title.pack_forget()
                self.lbl_treatment_title.pack_forget()
                self.lbl_type_title.pack_forget()
                self.lbl_selected_type.pack_forget()

                self.lbl_details_title.pack(anchor="w", padx=20, pady=(0, 4))

                for child in self.rank_buttons_frame.winfo_children():
                    child.destroy()

                self.lbl_selected_summary.config(text="")
                self.lbl_analysis_note.configure(text="")

                self.display_healthy_message()
                return

            elif ratio is not None:
                self.determine_treatment(disease_class, ratio, percent)
                self.lbl_analysis_note.configure(
                    text=f"Disease-to-leaf estimated ratio: {ratio:.2f}"
                )
            else:
                self.lbl_severity.configure(
                    text="⚪ Unknown",
                    text_color="#9da1ca",
                    fg_color="#252542"
                )
                self.lbl_analysis_note.configure(
                    text="Unable to estimate severity ratio from the current segmentation."
                )
                self.display_no_treatment()
            print("Disease Area:", disease_area)
            print("Leaf Area:", leaf_area)
            print("Ratio:", ratio)
        except Exception as e:
            messagebox.showerror("Segmentation Error", f"Error during segmentation: {str(e)}")
            print(f"Segmentation error: {e}")

    # ========================= RATIO =========================
    def calculate_ratio(self, detections):
        disease_total = 0
        leaf_detections = []


        disease_total = 0
        leaf_areas = []

        if detections.mask is None:
            return 0, 0, None

        for i in range(len(detections)):
            class_id = int(detections.class_id[i])
            class_name = self.segment_class_names[class_id]

            mask = detections.mask[i]

            # تحويل الماسك إلى binary (0/1)
            mask_bin = (mask > 0.5).astype(np.uint8)

            area = int(np.sum(mask_bin))

            if class_name == "disease":
                disease_total += area
            elif class_name == "leaf":
                leaf_areas.append(area)

        leaf_area = max(leaf_areas) if len(leaf_areas) > 0 else 0
        ratio = (disease_total / leaf_area) if leaf_area > 0 else None

        return disease_total, leaf_area, ratio

    # ========================= VISUALIZATION =========================
    def create_segmentation_visualization(self, image, detections):
        if len(detections) == 0:
            return image

        annotated = np.array(image)

        color_disease = sv.Color(r=79, g=140, b=255)   # blue
        color_leaf = sv.Color(r=32, g=178, b=107)      # green

        disease_dets = detections[detections.class_id == 1]
        leaf_dets = detections[detections.class_id == 2]

        if len(leaf_dets) > 0:
            annotated = sv.MaskAnnotator(color=color_leaf, opacity=0.28).annotate(annotated, leaf_dets)
            labels = [f"Leaf {conf:.2f}" for conf in leaf_dets.confidence]
            annotated = sv.LabelAnnotator(
                color=color_leaf,
                text_color=sv.Color.WHITE
            ).annotate(annotated, leaf_dets, labels=labels)

        if len(disease_dets) > 0:
            annotated = sv.MaskAnnotator(color=color_disease, opacity=0.35).annotate(annotated, disease_dets)
            labels = [f"Disease {conf:.2f}" for conf in disease_dets.confidence]
            annotated = sv.LabelAnnotator(
                color=color_disease,
                text_color=sv.Color.WHITE
            ).annotate(annotated, disease_dets, labels=labels)

        return Image.fromarray(annotated)

    # ========================= SEVERITY / TREATMENT =========================
    def determine_treatment(self, disease_class, ratio, percent):
        target_category = None
        cat_color = "#9da1ca"
        bg_color = "#252542"
        severity_text = "Unknown"

        for item in self.TREATMENT_CONFIG:
            if ratio >= item["min_ratio"]:
                target_category = item["category"]
                cat_color = "#ef4444" if item["category"] == "SEVERE" else ("#f59e0b" if item["category"] in ["MODERATE", "MILD"] else "#10b981")
                bg_color = "#4c1d1d" if item["category"] == "SEVERE" else ("#473616" if item["category"] in ["MODERATE", "MILD"] else "#183d33")
                severity_text = target_category
                break

        icon = "🔴" if severity_text == "SEVERE" else ("🟡" if severity_text in ["MODERATE", "MILD"] else "🟢")

        self.lbl_severity.configure(
            text=f"{icon} {severity_text} ({percent}% Infection)",
            text_color=cat_color,
            fg_color=bg_color
        )

        self._update_rank_selector(disease_class, target_category, cat_color)

    def _update_rank_selector(self, disease_class, target_category, cat_color):
        for child in self.rank_buttons_frame.winfo_children():
            child.destroy()

        recommendations = []
        if self.treatment_mgr and target_category:
            recommendations = self.treatment_mgr.get_treatments(disease_class, target_category)

        ranks = sorted(list(set([
            str(r.get("rank", "1"))
            for r in recommendations
            if isinstance(r, dict)
        ])))

        if not ranks:
            import customtkinter as ctk
            ctk.CTkLabel(
                self.rank_buttons_frame,
                text="No treatment options found.",
                text_color="#9da1ca"
            ).pack(anchor="w")
            self.lbl_selected_type.configure(text="-")
            self.lbl_selected_summary.config(text="No recommendation is available for the detected case.")
            self.display_treatment(disease_class, target_category, cat_color)
            return

        import customtkinter as ctk
        btn_frame = ctk.CTkFrame(self.rank_buttons_frame, fg_color="transparent")
        btn_frame.pack(anchor="w")

        self.rank_btns = []
        for r in ranks:
            btn = ctk.CTkButton(
                btn_frame,
                text=f"Option {r}",
                command=lambda rank=r: self._switch_rank(disease_class, target_category, cat_color, rank),
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

        self._switch_rank(disease_class, target_category, cat_color, ranks[0])

    def _switch_rank(self, disease_class, target_category, cat_color, rank):
        self.active_rank.set(rank)

        for r, btn in self.rank_btns:
            if r == rank:
                btn.configure(fg_color="#302a5e", text_color="#ffffff", border_color="#5d4ef5")
            else:
                btn.configure(fg_color="#252542", text_color="#9da1ca", border_color="#3A3A6A")

        self.display_treatment(disease_class, target_category, cat_color)

    def display_treatment(self, disease_class, target_category, cat_color):
        self.txt_treatment.config(state=tk.NORMAL)
        self.txt_treatment.delete("1.0", tk.END)

        recommendations = []
        if self.treatment_mgr and target_category:
            recommendations = self.treatment_mgr.get_treatments(disease_class, target_category)

        shown_any = False
        selected_type_summary = "-"

        if recommendations:
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

                if t_type and m_action:
                    selected_type_summary = f"{t_type} - {m_action}"
                else:
                    selected_type_summary = t_type or m_action or "-"

                self.lbl_treatment_title.configure(text=title_guess)
                self.lbl_selected_type.configure(text=selected_type_summary)

                short_summary = description[:220] + ("..." if len(description) > 220 else "")
                self.lbl_selected_summary.config(text=short_summary)

                self.txt_treatment.insert(tk.END, f"{description}\n", "body")

                if precautions:
                    #self.txt_treatment.insert(tk.END, "Precautions:\n", "section_title")
                    self.txt_treatment.insert(tk.END, "\nPrecautions:\n", "section_title")
                    self.txt_treatment.insert(tk.END, f"{precautions}\n", "body")

                break

        if not shown_any:
            self.lbl_selected_type.configure(text="-")
            self.lbl_selected_summary.config(text="No specific treatment found for this diagnosis.")
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

        """
        self.txt_treatment.tag_configure(
            "sub_header",
            font=("Segoe UI Semibold", 11, "bold"),
            foreground=UITheme.PRIMARY
        )
        self.txt_treatment.tag_configure(
            "item_title",
            font=("Segoe UI Semibold", 12, "bold"),
            foreground=UITheme.TEXT
        )
        self.txt_treatment.tag_configure(
            "prec_title",
            font=("Segoe UI Semibold", 11, "bold"),
            foreground=UITheme.DANGER
        )
        self.txt_treatment.tag_configure(
            "body",
            font=UITheme.FONT_BODY_BIG,
            foreground=UITheme.TEXT
        )
        self.txt_treatment.tag_configure(
            "footnote",
            font=UITheme.FONT_SMALL,
            foreground=UITheme.MUTED
        )
        """
        self._update_type_tabs(selected_type_summary)
        self.txt_treatment.config(state=tk.DISABLED)

    def display_healthy_message(self):
        self.txt_treatment.config(state=tk.NORMAL)
        self.txt_treatment.delete("1.0", tk.END)

        self.txt_treatment.insert(tk.END, "Leaf is healthy\n\n", "healthy_header")
        self.txt_treatment.insert(
            tk.END,
            "No disease lesions were detected.",
            "healthy_body"
        )

        self.txt_treatment.tag_configure(
            "healthy_header",
            font=("Segoe UI Semibold", 16, "bold"),
            foreground="#10b981"
        )
        self.txt_treatment.tag_configure(
            "healthy_body",
            font=UITheme.FONT_BODY_BIG,
            foreground=UITheme.TEXT
        )

        self.txt_treatment.config(state=tk.DISABLED)

    def display_no_treatment(self):
        self.lbl_selected_type.configure(text="-")
        self.lbl_selected_summary.config(text="Unable to calculate severity ratio from the current image.")
        self.lbl_treatment_title.configure(text="No treatment recommendation")

        self.txt_treatment.config(state=tk.NORMAL)
        self.txt_treatment.delete("1.0", tk.END)
        self.txt_treatment.insert(
            tk.END,
            "Unable to calculate severity ratio.\n\nPlease ensure the image clearly shows the leaf."
        )
        self.txt_treatment.config(state=tk.DISABLED)

    # ========================= IMAGE DISPLAY =========================
    def display_image(self, canvas, img, label):
        canvas.update_idletasks()

        canvas_w = canvas.winfo_width()
        canvas_h = canvas.winfo_height()

        if canvas_w < 50 or canvas_h < 50:
            canvas.delete("all")
            canvas.create_rectangle(
                0, 0,
                max(canvas_w, 10),
                max(canvas_h, 10),
                fill=UITheme.IMAGE_BG,
                outline=""
            )
            canvas.create_text(
                max(canvas_w, 150) // 2,
                max(canvas_h, 80) // 2,
                text="Image area is loading...",
                fill=UITheme.MUTED,
                font=UITheme.FONT_BODY
            )
            return

        img_copy = img.copy()
        img_copy.thumbnail((canvas_w - 16, canvas_h - 16))

        tk_img = ImageTk.PhotoImage(img_copy)

        canvas.delete("all")
        canvas.create_rectangle(0, 0, canvas_w, canvas_h, fill=UITheme.IMAGE_BG, outline="")
        canvas.create_image(canvas_w // 2, canvas_h // 2, image=tk_img)

        if label == "Original":
            self.tk_img_original = tk_img
            self.current_original_pil = img.copy()
        else:
            self.tk_img_segment = tk_img
            self.current_segment_pil = img.copy()

    # ========================= RESIZE =========================
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
            if self.current_segment_pil is not None:
                self.display_image(self.canvas_segment, self.current_segment_pil, "Segmented")
        except Exception:
            pass

    # ========================= RUN =========================


if __name__ == "__main__":
    root = tk.Tk()

    try:
        from ctypes import windll
        windll.shcore.SetProcessDpiAwareness(1)
    except Exception:
        pass

    app = PlantSegmentationGUI(root)
    root.mainloop()