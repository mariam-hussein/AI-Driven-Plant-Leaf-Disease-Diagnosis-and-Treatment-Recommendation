# =================================================================================
#      EfficientNetB0 + Swin-Tiny Fusion | train_classification.py
# =================================================================================

import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
import timm
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import FuncFormatter, MaxNLocator
from sklearn.metrics import (
    accuracy_score, classification_report,
    confusion_matrix, ConfusionMatrixDisplay,
    precision_recall_fscore_support
)

# ================== 1. Config ==================
data_dir = "split_data"
train_dir = os.path.join(data_dir, "train")
val_dir   = os.path.join(data_dir, "val")
test_dir  = os.path.join(data_dir, "test")

SAVE_DIR = "/content/drive/MyDrive/master_models/classification_healthy_disease/Result_classification_healthy_disease"
os.makedirs(SAVE_DIR, exist_ok=True)

# Hardware Optimization
torch.backends.cudnn.benchmark = True

batch_size = 32      # Large batch for speed
num_workers = 8      # Safe number for Colab/local
num_epochs = 50      # Max epochs
patience = 15        # Early stopping patience
lr = 3e-4            # Learning rate

img_size = 224
seed = 42

torch.manual_seed(seed)
np.random.seed(seed)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device} | Batch Size: {batch_size} | Patience: {patience}")

# ================== 2. Transforms ==================
weights = EfficientNet_B0_Weights.DEFAULT
train_tf = transforms.Compose([
    transforms.Resize((img_size, img_size)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(degrees=25),
    transforms.RandomResizedCrop(
        size=img_size, scale=(0.85, 1.0), ratio=(0.9, 1.1)
    ),
    transforms.ToTensor(),
    transforms.Normalize(mean=weights.transforms().mean, std=weights.transforms().std),
])

eval_tf = transforms.Compose([
    transforms.Resize((img_size, img_size)),
    transforms.ToTensor(),
    transforms.Normalize(mean=weights.transforms().mean, std=weights.transforms().std),
])

# ================== 3. Loaders ==================
train_ds = datasets.ImageFolder(train_dir, transform=train_tf)
val_ds   = datasets.ImageFolder(val_dir, transform=eval_tf)
test_ds  = datasets.ImageFolder(test_dir, transform=eval_tf)

class_names = train_ds.classes
num_classes = len(class_names)
print(f"Classes: {num_classes}")

train_loader = DataLoader(
    train_ds, batch_size=batch_size, shuffle=True,
    num_workers=num_workers, pin_memory=True, persistent_workers=True, prefetch_factor=2
)
val_loader = DataLoader(
    val_ds, batch_size=batch_size, shuffle=False,
    num_workers=num_workers, pin_memory=True, persistent_workers=True, prefetch_factor=2
)
test_loader = DataLoader(
    test_ds, batch_size=batch_size, shuffle=False,
    num_workers=num_workers, pin_memory=True
)

# ================== 4. Model ==================
class EffB0_SwinTiny_Fusion(nn.Module):
    def __init__(self, num_classes: int, eff_weights):
        super().__init__()

        # EfficientNet
        eff = efficientnet_b0(weights=eff_weights)
        self.eff_features = eff.features
        self.eff_pool = eff.avgpool
        self.eff_out_dim = eff.classifier[1].in_features

        # Swin
        self.swin = timm.create_model(
            "swin_tiny_patch4_window7_224",
            pretrained=True,
            num_classes=0
        )
        self.swin_out_dim = self.swin.num_features

        # Classifier
        fusion_dim = self.eff_out_dim + self.swin_out_dim
        self.classifier = nn.Linear(fusion_dim, num_classes)



    def forward(self, x):
        fe = self.eff_features(x)
        fe = self.eff_pool(fe)
        fe = torch.flatten(fe, 1)

        fs = self.swin(x)

        # Concatenation
        emb = torch.cat([fe, fs], dim=1)

        logits = self.classifier(emb)
        return logits, emb

model = EffB0_SwinTiny_Fusion(
    num_classes=num_classes,
    eff_weights=weights
).to(device)

from collections import Counter

counts = Counter(train_ds.targets)  # عدد العينات لكل كلاس في التدريب
class_weights = torch.tensor(
    [1.0 / counts[i] for i in range(num_classes)],
    dtype=torch.float32,
    device=device
)
class_weights = class_weights / class_weights.sum() * num_classes

criterion = nn.CrossEntropyLoss(weight=class_weights)
print("Using weighted CE loss")

optimizer = torch.optim.AdamW(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=lr,
    weight_decay=1e-4
)


# Scheduler for fast convergence
scheduler = torch.optim.lr_scheduler.OneCycleLR(
    optimizer,
    max_lr=lr,
    steps_per_epoch=len(train_loader),
    epochs=num_epochs,
    pct_start=0.20,
    div_factor=50.0,
    final_div_factor=1e4,
    anneal_strategy="cos"
)

# Mixed Precision Scaler
scaler = torch.amp.GradScaler('cuda')

# ================== 5. Training Loop ==================
best_val_acc = -1.0
best_path = os.path.join(SAVE_DIR, "best_weight_model.pth")
history_path = os.path.join(SAVE_DIR, "history.csv")

# Early Stopping Counter
patience_counter = 0

# Store Loss, Accuracy, AND Precision/F1 for Validation
history = {
    "train_loss": [], "train_acc": [],
    "val_loss": [], "val_acc": [],
    "val_prec": [], "val_rec": [], "val_f1": []
}

print(f"\nStarting Training for {num_epochs} Epochs with Patience {patience}...")

for epoch in range(1, num_epochs + 1):
    # --- TRAIN ---
    model.train()
    train_loss_sum = 0.0
    train_correct = 0
    train_total = 0

    for x, y in train_loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)

        optimizer.zero_grad()

        with torch.amp.autocast('cuda'):
            logits, _ = model(x)
            loss = criterion(logits, y)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        train_loss_sum += loss.item() * x.size(0)
        preds = torch.argmax(logits, dim=1)
        train_correct += (preds == y).sum().item()
        train_total += x.size(0)

    avg_train_loss = train_loss_sum / train_total
    avg_train_acc  = train_correct / train_total

    # --- VALIDATION ---
    model.eval()
    val_loss_sum = 0.0
    val_correct = 0
    val_total = 0

    # Storage for calculating Precision/F1 at epoch end
    val_preds_all = []
    val_targets_all = []

    with torch.no_grad():
        for x, y in val_loader:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            with torch.amp.autocast('cuda'):
                logits, _ = model(x)
                loss = criterion(logits, y)

            val_loss_sum += loss.item() * x.size(0)
            preds = torch.argmax(logits, dim=1)
            val_correct += (preds == y).sum().item()
            val_total += x.size(0)

            # Collect for metrics
            val_preds_all.extend(preds.cpu().numpy())
            val_targets_all.extend(y.cpu().numpy())

    avg_val_loss = val_loss_sum / val_total
    avg_val_acc  = val_correct / val_total

    # Calculate Macro Precision & F1 for this epoch
    epoch_prec, epoch_rec, epoch_f1, _ = precision_recall_fscore_support(
        val_targets_all, val_preds_all, average='macro', zero_division=0
    )

    # Store History
    history["train_loss"].append(avg_train_loss)
    history["train_acc"].append(avg_train_acc)
    history["val_loss"].append(avg_val_loss)
    history["val_acc"].append(avg_val_acc)
    history["val_prec"].append(epoch_prec)
    history["val_rec"].append(epoch_rec)
    history["val_f1"].append(epoch_f1)

    print(f"Epoch {epoch:02d}/{num_epochs} | "
      f"T_Loss: {avg_train_loss:.4f} T_Acc: {avg_train_acc:.4f} | "
      f"V_Loss: {avg_val_loss:.4f} V_Acc: {avg_val_acc:.4f} | "
      f"V_Prec: {epoch_prec:.4f} V_Rec: {epoch_rec:.4f} V_F1: {epoch_f1:.4f}")


    # --- CHECK BEST MODEL & PATIENCE ---
    if avg_val_acc > best_val_acc:
        best_val_acc = avg_val_acc
        torch.save(model.state_dict(), best_path)
        print(f"  *> Saved Best (Acc: {best_val_acc:.4f})")
        patience_counter = 0


    else:
        patience_counter += 1
        print(f"  !! No improvement. Patience: {patience_counter}/{patience}")
        if patience_counter >= patience:
            print(f"\n>> Early Stopping triggered! No improvement for {patience} epochs.")
            break

# Save CSV History
pd.DataFrame(history).to_csv(history_path, index=False)
print(f"Training History saved to {history_path}")

# ================== 6. Final Evaluation & Metric Calculation ==================

# Load Best Weights
model.load_state_dict(torch.load(best_path, map_location=device))
model.eval()
print("\nLoading best model for Final Testing...")

y_true, y_pred = [], []
with torch.no_grad():
    for x, y in test_loader:
        x, y = x.to(device), y.to(device)
        with torch.amp.autocast('cuda'):
            logits, _ = model(x)
        preds = torch.argmax(logits, dim=1)
        y_true.extend(y.cpu().numpy())
        y_pred.extend(preds.cpu().numpy())

# --- Basic Terminal Report ---
acc = accuracy_score(y_true, y_pred)
precision, recall, f1, _ = precision_recall_fscore_support(
    y_true, y_pred, average='macro', zero_division=0
)

print("\n" + "="*30)
print("       FINAL TEST RESULTS")
print("="*30)
print(f"Accuracy : {acc:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall   : {recall:.4f}")
print(f"F1-Score : {f1:.4f}")
print("="*30)

report_txt = classification_report(y_true, y_pred, target_names=class_names, digits=4)
print("\nClassification Report:\n")
print(report_txt)

with open(os.path.join(SAVE_DIR, "final_report.txt"), "w") as f:
    f.write(report_txt)

# ================== 7. Enhanced Plotting ==================
# This section uses the specific plotting code requested

# Ensure saving directory exists (redundant but safe)
os.makedirs(SAVE_DIR, exist_ok=True)

# --- 1. Plotting Training Curves ---
# We define the x-axis range based on validation data length
epochs_range = range(1, len(history["val_loss"]) + 1)

def smooth_curve(values, weight=0.8):
    smoothed = []
    last = values[0]
    for v in values:
        last = last * weight + (1 - weight) * v
        smoothed.append(last)
    return smoothed

def plot_metric(metric_name, train_data, val_data, title, filename):
    plt.figure(figsize=(10, 6))

    # Only plot training data if it exists
    if train_data is not None and len(train_data) > 0:
      train_smooth = smooth_curve(train_data)
      plt.plot(epochs_range, train_smooth, label=f"Train {metric_name}", linestyle='--')


    val_smooth = smooth_curve(val_data)
    plt.plot(epochs_range, val_smooth, label=f"Val {metric_name}", linewidth=2)


    plt.title(title)
    plt.xlabel("Epoch")
    plt.ylabel(metric_name)
    plt.legend(loc='best')
    plt.grid(True, alpha=0.3)

    ax = plt.gca()

    if metric_name.lower() == "loss":
      ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f"{y:.3f}"))
    else:
      ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f"{y:.3f}"))

    # يقلل عدد العلامات حتى يظهر المنحنى أكثر استقرار
    ax.yaxis.set_major_locator(MaxNLocator(nbins=6))

    if metric_name.lower() != "loss":
      all_data = list(val_smooth)
      if train_data is not None and len(train_data) > 0:
        all_data += list(train_smooth)

      ymin = min(all_data)
      ymax = max(all_data)

      pad = max(0.005, (ymax - ymin) * 0.10)  # هامش 10%
      ax.set_ylim(max(0.0, ymin - pad), min(1.01, ymax + pad))

    plt.tight_layout(pad=1.2)
    plt.margins(x=0.02, y=0.08)

    # Save and Show
    save_path = os.path.join(SAVE_DIR, filename)
    plt.savefig(save_path, bbox_inches="tight", dpi=300)
    print(f"Saved Curve: {save_path}")
    plt.show()
    plt.close()


# Plot the 4 curves
# Note: We pass None for train data on Precision/F1 because we only calculated them for Validation
plot_metric("Accuracy", history.get("train_acc"), history["val_acc"], "Accuracy Curve", "curve_accuracy.png")
plot_metric("Loss", history.get("train_loss"), history["val_loss"], "Loss Curve", "curve_loss.png")
plot_metric("Precision", None, history["val_prec"], "Validation Precision (Macro)", "curve_precision.png")
plot_metric("Recall", None, history["val_rec"], "Validation Recall (Macro)", "curve_recall.png")
plot_metric("F1-Score", None, history["val_f1"], "Validation F1-Score (Macro)", "curve_f1.png")


# --- 2. Plotting Confusion Matrices ---

# Shorten class names for cleaner labels on the axis
short_names = [n[:20] + "..." if len(n) > 20 else n for n in class_names]

# Calculate Matrix ONCE
cm = confusion_matrix(y_true, y_pred)

def plot_cm(cm_data, display_labels, title, filename, fmt='d', normalize=None):
    # Use subplots to explicitly control the figure size
    fig, ax = plt.subplots(figsize=(16, 16))

    if normalize:
        cm_data = cm_data.astype('float') / cm_data.sum(axis=1)[:, np.newaxis]
        fmt = '.2f'

    disp = ConfusionMatrixDisplay(confusion_matrix=cm_data, display_labels=display_labels)

    # Pass 'ax=ax' to draw on the specific figure we created
    disp.plot(
        include_values=True,
        cmap='Blues',
        xticks_rotation=90,
        values_format=fmt,
        ax=ax,
        colorbar=False
    )

    ax.set_title(title, fontsize=14, pad=20)
    plt.tight_layout()

    # Save and Show
    save_path = os.path.join(SAVE_DIR, filename)
    plt.savefig(save_path, bbox_inches="tight", dpi=300)
    print(f"Saved Curve: {save_path}")
    plt.show()
    plt.close()



# 1. Counts Only (Standard)
plot_cm(cm, short_names, "Confusion Matrix (Counts)", "cm_counts.png", fmt='d')

# 2. Normalized (Percentages)
plot_cm(cm, short_names, "Confusion Matrix (Normalized)", "cm_normalized.png", normalize=True)

# --- Save final state for inference ---
# --- Save final state for inference ---
embedding_dim = model.classifier.in_features  # الصحيح (eff_dim + swin_dim)

torch.save({
    "model_state_dict": model.state_dict(),
    "class_names": class_names,
    "class_to_idx": train_ds.class_to_idx,   # مهم لاحقاً
    "img_size": img_size,
    "embedding_dim": embedding_dim
}, os.path.join(SAVE_DIR, "EffB0_Swin_Aug_inference.pth"))


print(f"\nAll Done. All plots and models saved to: {SAVE_DIR}")