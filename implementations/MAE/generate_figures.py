"""
Generate all figures for the presentation:
  - loss_curve.png (already exists, regenerate clean version)
  - confusion_matrix.png
  - accuracy_summary.png
  - patch_grid.png  (visual of patch masking at different mask ratios)
"""

import json
import os
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from sklearn.metrics import confusion_matrix
import itertools

from mae import MaskedAutoencoder, Config, get_dataloaders

cfg = Config()
CKPT = "checkpoints/mae_final.pt"
CIFAR_CLASSES = ["airplane","automobile","bird","cat","deer",
                 "dog","frog","horse","ship","truck"]

os.makedirs("figures", exist_ok=True)


# ── 1. Loss curve (clean) ───────────────────────────────────────────────────
with open("checkpoints/loss.json") as f:
    history = json.load(f)

epochs = [h["epoch"] for h in history]
losses = [h["loss"] for h in history]
lrs    = [h["lr"] for h in history]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

ax1.plot(epochs, losses, color="#1f77b4", linewidth=2)
ax1.set_xlabel("Epoch", fontsize=12)
ax1.set_ylabel("Reconstruction Loss (MSE)", fontsize=12)
ax1.set_title("Pre-training Loss", fontsize=14, fontweight="bold")
ax1.grid(True, alpha=0.3)
ax1.annotate(f"Final: {losses[-1]:.3f}", xy=(epochs[-1], losses[-1]),
             xytext=(-30, 10), textcoords="offset points",
             fontsize=10, color="#1f77b4",
             arrowprops=dict(arrowstyle="->", color="#1f77b4"))

ax2.plot(epochs, lrs, color="#1f77b4", linewidth=2)
ax2.set_xlabel("Epoch", fontsize=12)
ax2.set_ylabel("Learning Rate", fontsize=12)
ax2.set_title("LR Schedule (warmup + cosine decay)", fontsize=14, fontweight="bold")
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("figures/loss_curve.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved figures/loss_curve.png")


# ── 2. Confusion matrices (linear probe vs fine-tuning) ─────────────────────
print("Computing confusion matrices...")

import torch.nn as nn
from mae import LinearProbe, extract_features

FT_CKPT = CKPT.replace(".pt", "_finetune_best.pt")
train_loader, val_loader = get_dataloaders(cfg)

model = MaskedAutoencoder(cfg).to(cfg.device)

# --- Fine-tuning predictions ---
head_ft = nn.Sequential(
    nn.LayerNorm(cfg.enc_embed_dim),
    nn.Linear(cfg.enc_embed_dim, cfg.num_classes),
).to(cfg.device)

# --- Modèle pré-entraîné (pour le linear probe) ---
pretrain_ckpt = torch.load(CKPT, map_location=cfg.device)
model.load_state_dict(pretrain_ckpt["model"])
model.eval()

print("  Extracting features for linear probe...")
train_feats, train_labels = extract_features(model, train_loader, cfg)
val_feats, val_labels_lp = extract_features(model, val_loader, cfg)

# --- Modèle fine-tuné (pour la matrice de fine-tuning) ---
ft_ckpt = torch.load(FT_CKPT, map_location=cfg.device)
model.load_state_dict(ft_ckpt["model"])
head_ft.load_state_dict(ft_ckpt["head"])
model.eval()
head_ft.eval()

preds_ft, val_labels = [], []
with torch.no_grad():
    for imgs, labels in val_loader:
        imgs = imgs.to(cfg.device)
        x = model.patch_embed(imgs)
        x = x + model.enc_pos_embed
        for blk in model.enc_blocks:
            x = blk(x)
        x = model.enc_norm(x).mean(dim=1)
        preds_ft.append(head_ft(x).argmax(dim=1).cpu())
        val_labels.append(labels)

preds_ft = torch.cat(preds_ft)
val_labels = torch.cat(val_labels)

probe = LinearProbe(cfg.enc_embed_dim, cfg.num_classes).to(cfg.device)
optimizer_p = torch.optim.SGD(probe.parameters(), lr=cfg.probe_lr, momentum=0.9)
scheduler_p = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_p, cfg.probe_epochs)
train_ds = torch.utils.data.TensorDataset(train_feats, train_labels)
probe_loader = torch.utils.data.DataLoader(train_ds, batch_size=cfg.probe_batch_size, shuffle=True)

print("  Training linear probe...")
for _ in range(cfg.probe_epochs):
    probe.train()
    for feats, lbls in probe_loader:
        feats, lbls = feats.to(cfg.device), lbls.to(cfg.device)
        loss_p = torch.nn.functional.cross_entropy(probe(feats), lbls)
        optimizer_p.zero_grad(); loss_p.backward(); optimizer_p.step()
    scheduler_p.step()

probe.eval()
with torch.no_grad():
    preds_lp = probe(val_feats.to(cfg.device)).argmax(dim=1).cpu()

# --- Plot côte à côte ---
def plot_cm(ax, preds, labels, title):
    cm = confusion_matrix(labels.numpy(), preds.numpy())
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
    im = ax.imshow(cm_norm, interpolation="nearest", cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(10)); ax.set_yticks(range(10))
    ax.set_xticklabels(CIFAR_CLASSES, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(CIFAR_CLASSES, fontsize=8)
    ax.set_xlabel("Prédit", fontsize=10)
    ax.set_ylabel("Réel", fontsize=10)
    ax.set_title(title, fontsize=11, fontweight="bold")
    for i, j in itertools.product(range(10), range(10)):
        color = "white" if cm_norm[i, j] > 0.6 else "black"
        ax.text(j, i, f"{cm_norm[i,j]:.2f}", ha="center", va="center", fontsize=6, color=color)
    return im

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7))
im1 = plot_cm(ax1, preds_lp, val_labels_lp, "Matrice de confusion — Linear Probe (49.6%)")
im2 = plot_cm(ax2, preds_ft, val_labels, "Matrice de confusion — Fine-tuning (83.6%)")
plt.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)

plt.tight_layout()
plt.savefig("figures/confusion_matrix.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved figures/confusion_matrix.png")


# ── 3. Accuracy summary bar chart ───────────────────────────────────────────
methods = ["k-NN\n(k=20)", "Linear\nProbe", "Fine-\ntuning"]
our_acc  = [38.9, 49.6, 83.6]

# Reference: supervised ViT-Tiny trained from scratch (typical CIFAR-10 numbers)
# MAE paper doesn't do CIFAR-10 but we compare against:
# - Random init linear probe baseline ~ 10% (chance)
# - Supervised ViT-Tiny from scratch ~ 90-92% (literature)
ref_scratch = [None, None, 91.0]
chance      = [10.0, 10.0, 10.0]

x = np.arange(len(methods))
width = 0.35

fig, ax = plt.subplots(figsize=(8, 5))
bars = ax.bar(x - width/2, our_acc, width, label="MAE (ours, 50 epochs)",
              color="#e63946", alpha=0.85, zorder=3)
ref_vals = [v if v else 0 for v in ref_scratch]
bars2 = ax.bar(x + width/2, ref_vals, width, label="ViT-Tiny supervised (scratch, literature)",
               color="#457b9d", alpha=0.85, zorder=3)

# hide the None bars
for i, v in enumerate(ref_scratch):
    if v is None:
        bars2[i].set_visible(False)

ax.axhline(10, color="gray", linestyle="--", linewidth=1, label="Random chance (10%)", zorder=2)

for bar, val in zip(bars, our_acc):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
            f"{val:.1f}%", ha="center", va="bottom", fontweight="bold", fontsize=11)
for bar, val in zip(bars2, ref_scratch):
    if val:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f"{val:.1f}%", ha="center", va="bottom", fontweight="bold", fontsize=11)

ax.set_ylabel("Accuracy (%)", fontsize=12)
ax.set_title("CIFAR-10 Classification Results", fontsize=14, fontweight="bold")
ax.set_xticks(x)
ax.set_xticklabels(methods, fontsize=11)
ax.set_ylim(0, 100)
ax.legend(fontsize=10)
ax.grid(axis="y", alpha=0.3, zorder=0)
plt.tight_layout()
plt.savefig("figures/accuracy_summary.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved figures/accuracy_summary.png")


# ── 4. Masking visualization (multiple mask ratios on same image) ────────────
print("Generating masking visualization...")

transform_val = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
])
val_set = datasets.CIFAR10(root="./data", train=False, download=False, transform=transform_val)
imgs_raw, _ = next(iter(DataLoader(val_set, batch_size=16, shuffle=False)))

mean = torch.tensor([0.4914, 0.4822, 0.4465]).view(1,3,1,1)
std  = torch.tensor([0.2470, 0.2435, 0.2616]).view(1,3,1,1)

def apply_mask(img_norm, mask_ratio, patch_size=4):
    B, C, H, W = img_norm.shape
    N = (H // patch_size) ** 2
    N_mask = int(N * mask_ratio)
    noise = torch.rand(B, N)
    ids = torch.argsort(noise, dim=1)
    mask = torch.zeros(B, N)
    mask.scatter_(1, ids[:, :N_mask], 1)
    ph = pw = H // patch_size
    mask_2d = mask.view(B, ph, pw)
    mask_img = mask_2d.repeat_interleave(patch_size, dim=1).repeat_interleave(patch_size, dim=2)
    mask_img = mask_img.unsqueeze(1)
    img_denorm = img_norm * std + mean
    masked = img_denorm * (1 - mask_img) + 0.5 * mask_img
    return masked.clamp(0, 1)

ratios = [0.0, 0.5, 0.75, 0.9]
n_imgs = 4
fig, axes = plt.subplots(n_imgs, len(ratios), figsize=(len(ratios)*2, n_imgs*2))

for row in range(n_imgs):
    for col, r in enumerate(ratios):
        img = imgs_raw[row:row+1]
        if r == 0.0:
            disp = (img * std + mean).clamp(0,1)[0]
        else:
            disp = apply_mask(img, r)[0]
        axes[row, col].imshow(disp.permute(1,2,0).numpy())
        axes[row, col].axis("off")
        if row == 0:
            label = "Original" if r == 0 else f"Mask {int(r*100)}%"
            axes[row, col].set_title(label, fontsize=11, fontweight="bold")

plt.suptitle("Effect of Mask Ratio on Input", fontsize=13, fontweight="bold", y=1.01)
plt.tight_layout()
plt.savefig("figures/masking_ratios.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved figures/masking_ratios.png")

print("\nAll figures saved to figures/")
