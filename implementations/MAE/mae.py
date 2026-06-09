import math
import argparse
import json
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm


class Config:
    """Hyperparamètres du MAE sur CIFAR-10 (encodeur ViT-Tiny, décodeur léger)."""

    img_size = 32
    patch_size = 4
    num_patches = (img_size // patch_size) ** 2  # 64
    in_chans = 3
    pixels_per_patch = patch_size * patch_size * in_chans  # 48

    enc_embed_dim = 192
    enc_depth = 12
    enc_num_heads = 3

    dec_embed_dim = 96
    dec_depth = 4
    dec_num_heads = 3

    mask_ratio = 0.75

    epochs = 50
    batch_size = 256
    base_lr = 1.5e-4
    weight_decay = 0.05
    warmup_epochs = 20

    probe_epochs = 90
    probe_lr = 0.1
    probe_batch_size = 256

    ft_epochs = 30
    ft_lr = 1e-3
    ft_weight_decay = 0.05
    ft_warmup_epochs = 5

    num_classes = 10
    device = "cuda" if torch.cuda.is_available() else "cpu"
    num_workers = 4
    checkpoint_dir = "checkpoints"
    use_amp = True


cfg = Config()


def get_sincos_pos_embed(embed_dim, num_patches):
    """Embeddings positionnels sinusoïdaux 2D (non appris).

    embed_dim est divisé en deux moitiés : la première encode la position en ligne,
    la seconde encode la position en colonne. Chaque moitié entrelace sin/cos à des
    fréquences géométriquement espacées, selon la formulation 1D de "Attention Is All You Need".

    Retourne un tenseur (1, num_patches, embed_dim) sans gradient.
    """
    d = embed_dim // 2                                           # 96  (moitié lignes, moitié colonnes)
    k = torch.arange(d / 2)                                      # (48,)  indices de fréquence

    grid_size = int(num_patches ** 0.5)                          # 8
    x = torch.arange(0, grid_size)                               # (8,)
    y = torch.arange(0, grid_size)                               # (8,)

    grid_x, grid_y = torch.meshgrid(x, y, indexing='ij')        # (8, 8), (8, 8)
    grid_x_flat = grid_x.flatten()                               # (64,)
    grid_y_flat = grid_y.flatten()                               # (64,)

    omega_k = 1 / (10000 ** (2 * k / d))                        # (48,)

    sin_x = torch.sin(grid_x_flat[:, None] * omega_k[None, :])  # (64, 48)
    cos_x = torch.cos(grid_x_flat[:, None] * omega_k[None, :])  # (64, 48)
    sin_y = torch.sin(grid_y_flat[:, None] * omega_k[None, :])  # (64, 48)
    cos_y = torch.cos(grid_y_flat[:, None] * omega_k[None, :])  # (64, 48)

    emb_x = torch.stack([sin_x, cos_x], dim=2).reshape(num_patches, d)  # (64, 96)
    emb_y = torch.stack([sin_y, cos_y], dim=2).reshape(num_patches, d)  # (64, 96)

    return torch.cat([emb_x, emb_y], dim=1).unsqueeze(0)        # (1, 64, 192)


class PatchEmbed(nn.Module):
    def __init__(self, img_size=32, patch_size=4, in_chans=3, embed_dim=192):
        super().__init__()
        self.num_patches = (img_size // patch_size) ** 2
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x = self.proj(x)                    # (B, 3, 32, 32) -> (B, 192, 8, 8)
        x = x.flatten(2).transpose(1, 2)    # (B, 192, 64) -> (B, 64, 192)
        return x


class Attention(nn.Module):
    def __init__(self, dim, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.qkv = nn.Linear(dim, 3 * dim)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x):
        B, N, D = x.shape                                              # (B, N, 192)
        qkv = self.qkv(x)                                             # (B, N, 576)
        qkv = qkv.reshape(B, N, 3, self.num_heads, self.head_dim)     # (B, N, 3, 3, 64)
        qkv = qkv.permute(2, 0, 3, 1, 4)                             # (3, B, 3, N, 64)
        Q, K, V = qkv[0], qkv[1], qkv[2]                             # chacun (B, 3, N, 64)
        scores = Q @ K.transpose(-2, -1) * self.scale                 # (B, 3, N, N)
        attn = F.softmax(scores, dim=-1)                               # (B, 3, N, N)
        out = attn @ V                                                 # (B, 3, N, 64)
        out = out.transpose(1, 2).reshape(B, N, D)                    # (B, N, 192)
        out = self.proj(out)                                           # (B, N, 192)
        return out


class MlpBlock(nn.Module):
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, dim)

    def forward(self, x):
        return self.fc2(self.act(self.fc1(x)))


class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = Attention(dim, num_heads)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = MlpBlock(dim, int(dim * mlp_ratio))

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class MaskedAutoencoder(nn.Module):
    """MAE avec une architecture encodeur-décodeur asymétrique (He et al., 2022).

    L'encodeur (ViT) ne traite que les patches visibles (25% de l'image),
    ce qui le rend rapide. Le décodeur est volontairement léger : il doit seulement
    produire un signal de reconstruction, pas des représentations transférables.
    Les embeddings positionnels sont des buffers sinusoïdaux fixes (non appris)
    dans l'encodeur et le décodeur.
    """

    def __init__(self, cfg):
        super().__init__()

        self.patch_embed = PatchEmbed(cfg.img_size, cfg.patch_size, cfg.in_chans, cfg.enc_embed_dim)

        self.register_buffer(
            "enc_pos_embed",
            get_sincos_pos_embed(cfg.enc_embed_dim, cfg.num_patches)
        )

        self.enc_blocks = nn.ModuleList([
            TransformerBlock(cfg.enc_embed_dim, cfg.enc_num_heads)
            for _ in range(cfg.enc_depth)
        ])
        self.enc_norm = nn.LayerNorm(cfg.enc_embed_dim)

        self.enc_to_dec = nn.Linear(cfg.enc_embed_dim, cfg.dec_embed_dim)

        self.mask_token = nn.Parameter(torch.zeros(1, 1, cfg.dec_embed_dim))
        nn.init.normal_(self.mask_token, std=0.02)

        self.register_buffer(
            "dec_pos_embed",
            get_sincos_pos_embed(cfg.dec_embed_dim, cfg.num_patches)
        )

        self.dec_blocks = nn.ModuleList([
            TransformerBlock(cfg.dec_embed_dim, cfg.dec_num_heads)
            for _ in range(cfg.dec_depth)
        ])
        self.dec_norm = nn.LayerNorm(cfg.dec_embed_dim)
        self.dec_pred = nn.Linear(cfg.dec_embed_dim, cfg.pixels_per_patch)

        self.cfg = cfg
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def random_masking(self, x, mask_ratio):
        """Conserve aléatoirement (1 - mask_ratio) tokens par échantillon.

        Utilise un trick shuffle/unshuffle : assigner un score aléatoire à chaque token,
        argsort pour obtenir une permutation aléatoire, garder les N_visible premiers tokens,
        et sauvegarder la permutation inverse (ids_restore) pour que le décodeur puisse
        remettre les tokens dans l'ordre spatial sans bookkeeping supplémentaire.

        Retourne :
            x_visible:   (B, N_visible, D) — entrée de l'encodeur, sans tokens masqués
            mask:        (B, N) — binaire, 0=visible 1=masqué, dans l'ordre spatial d'origine
            ids_restore: (B, N) — permutation inverse pour le unshuffle du décodeur
        """
        B, N, D = x.shape                                                        # (B, 64, 192)
        N_visible = int(N * (1 - mask_ratio))                                    # 16

        noise = torch.rand(B, N, device=x.device)                               # (B, 64)
        ids_shuffle = torch.argsort(noise)                                       # (B, 64)  permutation aléatoire
        ids_restore = torch.argsort(ids_shuffle)                                 # (B, 64)  permutation inverse

        index_to_keep = ids_shuffle[:, :N_visible]                               # (B, 16)
        x_visible = torch.gather(x, dim=1, index=index_to_keep.unsqueeze(-1).expand(-1, -1, D))  # (B, 16, 192)

        mask = torch.ones(B, N, device=x.device)                                # (B, 64)  tout masqué
        mask[:, :N_visible] = 0                                                  # les 16 premiers = visibles
        mask = torch.gather(mask, dim=1, index=ids_restore)                     # (B, 64)  remis dans l'ordre spatial

        return x_visible, mask, ids_restore

    def forward_encoder(self, x):
        x = self.patch_embed(x)                                      # (B, 64, 192)
        x = x + self.enc_pos_embed                                   # (B, 64, 192)  diffusé sur B
        x, mask, ids_restore = self.random_masking(x, self.cfg.mask_ratio)  # x: (B, 16, 192)
        for block in self.enc_blocks:
            x = block(x)                                             # (B, 16, 192)
        x = self.enc_norm(x)                                         # (B, 16, 192)
        return x, mask, ids_restore

    def forward_decoder(self, latent, ids_restore):
        """Reconstruit les N patches à partir des N_visible représentations de l'encodeur.

        Les mask tokens (un unique vecteur appris) sont ajoutés pour combler les positions
        manquantes, puis ids_restore remet toute la séquence dans l'ordre spatial avant
        d'ajouter les embeddings positionnels. Le décodeur voit toutes les positions
        mais seules les représentations de l'encodeur portent de l'information réelle.
        """
        x = self.enc_to_dec(latent)                                              # (B, 16, 96)
        B, N_visible, D_dec = x.shape                                            # 16, 96
        N = ids_restore.shape[1]                                                 # 64
        N_masked = N - N_visible                                                 # 48

        mask_tokens = self.mask_token.repeat(B, N_masked, 1)                    # (B, 48, 96)
        x = torch.cat([x, mask_tokens], dim=1)                                  # (B, 64, 96)  ordre mélangé
        x = torch.gather(x, dim=1, index=ids_restore.unsqueeze(-1).expand(-1, -1, D_dec))  # (B, 64, 96)  ordre spatial
        x = x + self.dec_pos_embed                                              # (B, 64, 96)

        for block in self.dec_blocks:
            x = block(x)                                                         # (B, 64, 96)
        x = self.dec_norm(x)                                                     # (B, 64, 96)
        x = self.dec_pred(x)                                                     # (B, 64, 48)
        return x

    def patchify(self, imgs):
        p = self.cfg.patch_size                      # 4
        B, C, H, _ = imgs.shape                     # (B, 3, 32, 32)
        h = w = H // p                               # 8
        x = imgs.reshape(B, C, h, p, w, p)          # (B, 3, 8, 4, 8, 4)
        x = x.permute(0, 2, 4, 3, 5, 1)             # (B, 8, 8, 4, 4, 3)
        x = x.reshape(B, h * w, p * p * C)          # (B, 64, 48)
        return x

    def unpatchify(self, patches):
        p = self.cfg.patch_size                      # 4
        h = w = self.cfg.img_size // p              # 8
        C = self.cfg.in_chans                        # 3
        x = patches.reshape(-1, h, w, p, p, C)      # (B, 8, 8, 4, 4, 3)
        x = x.permute(0, 5, 1, 3, 2, 4)             # (B, 3, 8, 4, 8, 4)
        x = x.reshape(-1, C, h * p, w * p)          # (B, 3, 32, 32)
        return x

    def forward_loss(self, imgs, pred, mask):
        """MSE sur les patches masqués uniquement, avec normalisation par patch.

        Normaliser chaque patch indépendamment supprime le biais basse-fréquence
        (luminosité/contraste globaux) et force le modèle à prédire la texture fine
        plutôt que la simple couleur moyenne du patch. La loss est moyennée sur les
        positions masquées uniquement, pas sur l'image entière.
        """
        target = self.patchify(imgs)                              # (B, 64, 48)

        mean = target.mean(dim=-1, keepdim=True)                 # (B, 64, 1)
        var = target.var(dim=-1, keepdim=True)                   # (B, 64, 1)
        target = (target - mean) / (var + 1e-6).sqrt()           # (B, 64, 48)  normalisé par patch

        loss = (pred - target) ** 2                              # (B, 64, 48)
        loss = loss.mean(dim=-1)                                  # (B, 64)      MSE par patch
        loss = (loss * mask).sum() / mask.sum()                  # scalaire      patches masqués uniquement
        return loss

    def forward(self, imgs):
        latent, mask, ids_restore = self.forward_encoder(imgs)
        pred = self.forward_decoder(latent, ids_restore)
        loss = self.forward_loss(imgs, pred, mask)
        return loss, pred, mask


def get_dataloaders(cfg):
    transform_train = transforms.Compose([
        transforms.RandomResizedCrop(cfg.img_size, scale=(0.2, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
    ])
    transform_val = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
    ])

    train_set = datasets.CIFAR10(root="./data", train=True, download=True, transform=transform_train)
    val_set = datasets.CIFAR10(root="./data", train=False, download=True, transform=transform_val)

    train_loader = DataLoader(train_set, batch_size=cfg.batch_size, shuffle=True,
                              num_workers=cfg.num_workers, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_set, batch_size=cfg.batch_size, shuffle=False,
                            num_workers=cfg.num_workers, pin_memory=True)
    return train_loader, val_loader


def get_lr(epoch, cfg):
    """Décroissance cosinus avec échauffement linéaire."""
    if epoch < cfg.warmup_epochs:
        return cfg.base_lr * (cfg.batch_size / 256) * (epoch + 1) / cfg.warmup_epochs
    else:
        progress = (epoch - cfg.warmup_epochs) / (cfg.epochs - cfg.warmup_epochs)
        return cfg.base_lr * (cfg.batch_size / 256) * 0.5 * (1 + math.cos(math.pi * progress))


def pretrain(cfg, tag=""):
    suffix = f"_{tag}" if tag else ""
    final_path = os.path.join(cfg.checkpoint_dir, f"mae{suffix}_final.pt")
    if os.path.exists(final_path):
        print(f"Skip pretrain: {final_path} already exists")
        return
    os.makedirs(cfg.checkpoint_dir, exist_ok=True)
    train_loader, _ = get_dataloaders(cfg)

    model = MaskedAutoencoder(cfg).to(cfg.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.base_lr,
                                   weight_decay=cfg.weight_decay, betas=(0.9, 0.95))
    scaler = torch.cuda.amp.GradScaler(enabled=cfg.use_amp)

    print(f"Params: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M")
    print(f"Device: {cfg.device}")
    print(f"Epochs: {cfg.epochs}, Batch size: {cfg.batch_size}, Mask ratio: {cfg.mask_ratio}")

    loss_history = []

    for epoch in range(cfg.epochs):
        model.train()
        lr = get_lr(epoch, cfg)
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        total_loss = 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{cfg.epochs}", leave=False)
        for imgs, _ in pbar:
            imgs = imgs.to(cfg.device)

            with torch.cuda.amp.autocast(enabled=cfg.use_amp):
                loss, _, _ = model(imgs)

            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item()
            pbar.set_postfix(loss=f"{loss.item():.4f}", lr=f"{lr:.6f}")

        avg_loss = total_loss / len(train_loader)
        loss_history.append({"epoch": epoch + 1, "loss": avg_loss, "lr": lr})

        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"Epoch {epoch+1}/{cfg.epochs}  loss={avg_loss:.4f}  lr={lr:.6f}")

        if (epoch + 1) % 20 == 0:
            torch.save({
                "epoch": epoch + 1,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
            }, os.path.join(cfg.checkpoint_dir, f"mae{suffix}_epoch{epoch+1}.pt"))

    torch.save({
        "epoch": cfg.epochs,
        "model": model.state_dict(),
    }, os.path.join(cfg.checkpoint_dir, f"mae{suffix}_final.pt"))

    log_path = os.path.join(cfg.checkpoint_dir, f"loss{suffix}.json")
    with open(log_path, "w") as f:
        json.dump(loss_history, f)
    print(f"Pre-training done. Loss log saved to {log_path}")


class LinearProbe(nn.Module):
    """BN (sans affine) + classifieur linéaire, comme dans l'Annexe A.1 du papier MAE."""
    def __init__(self, embed_dim, num_classes):
        super().__init__()
        self.bn = nn.BatchNorm1d(embed_dim, affine=False)
        self.fc = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        return self.fc(self.bn(x))


@torch.no_grad()
def extract_features(model, dataloader, cfg):
    model.eval()
    all_features = []
    all_labels = []

    for imgs, labels in tqdm(dataloader, desc="Extracting features", leave=False):
        imgs = imgs.to(cfg.device)
        x = model.patch_embed(imgs)
        x = x + model.enc_pos_embed
        for blk in model.enc_blocks:
            x = blk(x)
        x = model.enc_norm(x)
        x = x.mean(dim=1)
        all_features.append(x.cpu())
        all_labels.append(labels)

    return torch.cat(all_features), torch.cat(all_labels)


def linear_probe(cfg, checkpoint_path):
    result_path = checkpoint_path.replace(".pt", "_linprobe.json")
    if os.path.exists(result_path):
        with open(result_path) as f:
            res = json.load(f)
        print(f"Skip linear_probe: already done — accuracy: {res['accuracy']*100:.1f}%")
        return res["accuracy"]

    train_loader, val_loader = get_dataloaders(cfg)

    model = MaskedAutoencoder(cfg).to(cfg.device)
    ckpt = torch.load(checkpoint_path, map_location=cfg.device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    print("Extracting features...")
    train_feats, train_labels = extract_features(model, train_loader, cfg)
    val_feats, val_labels = extract_features(model, val_loader, cfg)

    probe = LinearProbe(cfg.enc_embed_dim, cfg.num_classes).to(cfg.device)
    optimizer = torch.optim.SGD(probe.parameters(), lr=cfg.probe_lr, momentum=0.9)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, cfg.probe_epochs)

    train_dataset = torch.utils.data.TensorDataset(train_feats, train_labels)
    train_probe_loader = DataLoader(train_dataset, batch_size=cfg.probe_batch_size, shuffle=True)

    for epoch in tqdm(range(cfg.probe_epochs), desc="Linear probe"):
        probe.train()
        for feats, labels in train_probe_loader:
            feats, labels = feats.to(cfg.device), labels.to(cfg.device)
            loss = F.cross_entropy(probe(feats), labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        scheduler.step()

    probe.eval()
    val_feats_dev = val_feats.to(cfg.device)
    val_labels_dev = val_labels.to(cfg.device)
    with torch.no_grad():
        preds = probe(val_feats_dev).argmax(dim=1)
        acc = (preds == val_labels_dev).float().mean().item()
    with open(result_path, "w") as f:
        json.dump({"accuracy": acc}, f)
    print(f"Linear probe accuracy: {acc*100:.1f}%")
    return acc


@torch.no_grad()
def knn_evaluate(cfg, checkpoint_path, k=20):
    result_path = checkpoint_path.replace(".pt", "_knn.json")
    if os.path.exists(result_path):
        with open(result_path) as f:
            res = json.load(f)
        print(f"Skip knn: already done — accuracy: {res['accuracy']*100:.1f}%")
        return res["accuracy"]

    train_loader, val_loader = get_dataloaders(cfg)

    model = MaskedAutoencoder(cfg).to(cfg.device)
    ckpt = torch.load(checkpoint_path, map_location=cfg.device)
    model.load_state_dict(ckpt["model"])

    train_feats, train_labels = extract_features(model, train_loader, cfg)
    val_feats, val_labels = extract_features(model, val_loader, cfg)

    train_feats = F.normalize(train_feats, dim=1)
    val_feats = F.normalize(val_feats, dim=1)

    correct = 0
    total = 0
    batch_size = 256
    for i in range(0, len(val_feats), batch_size):
        batch = val_feats[i:i+batch_size]
        sim = batch @ train_feats.T
        _, topk_idx = sim.topk(k, dim=1)
        topk_labels = train_labels[topk_idx]
        pred = topk_labels.mode(dim=1).values
        correct += (pred == val_labels[i:i+batch_size]).sum().item()
        total += batch.shape[0]

    acc = correct / total
    with open(result_path, "w") as f:
        json.dump({"accuracy": acc}, f)
    print(f"k-NN accuracy (k={k}): {acc*100:.1f}%")
    return acc


@torch.no_grad()
def visualize(cfg, checkpoint_path, num_images=8):
    """Sauvegarde une grille : image masquée | reconstruction | original."""
    if os.path.exists("mae_reconstruction.png"):
        print("Skip visualize: mae_reconstruction.png already exists")
        return

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    _, val_loader = get_dataloaders(cfg)
    model = MaskedAutoencoder(cfg).to(cfg.device)
    ckpt = torch.load(checkpoint_path, map_location=cfg.device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    imgs, _ = next(iter(val_loader))
    imgs = imgs[:num_images].to(cfg.device)

    loss, pred, mask = model(imgs)

    mean = torch.tensor([0.4914, 0.4822, 0.4465]).view(1, 3, 1, 1).to(cfg.device)
    std = torch.tensor([0.2470, 0.2435, 0.2616]).view(1, 3, 1, 1).to(cfg.device)
    imgs_denorm = imgs * std + mean

    patches_orig = model.patchify(imgs)
    p_mean = patches_orig.mean(dim=-1, keepdim=True)
    p_var = patches_orig.var(dim=-1, keepdim=True)
    pred_pixels = pred * (p_var + 1e-6).sqrt() + p_mean
    mask_tokens = mask.unsqueeze(-1)
    combined = patches_orig * (1 - mask_tokens) + pred_pixels * mask_tokens
    recon = model.unpatchify(combined)
    recon = recon * std + mean
    recon = recon.clamp(0, 1)

    mask_img = mask.unsqueeze(-1).repeat(1, 1, cfg.pixels_per_patch)
    mask_img = model.unpatchify(mask_img)
    masked_imgs = imgs_denorm * (1 - mask_img) + 0.5 * mask_img

    fig, axes = plt.subplots(num_images, 3, figsize=(6, 2 * num_images))
    for i in range(num_images):
        for j, (img, title) in enumerate([
            (masked_imgs[i], "masked"),
            (recon[i], "reconstruction"),
            (imgs_denorm[i], "original"),
        ]):
            ax = axes[i, j] if num_images > 1 else axes[j]
            ax.imshow(img.cpu().permute(1, 2, 0).clamp(0, 1).numpy())
            ax.set_title(title, fontsize=8)
            ax.axis("off")

    plt.tight_layout()
    plt.savefig("mae_reconstruction.png", dpi=150)
    print("Saved mae_reconstruction.png")


def finetune(cfg, checkpoint_path):
    result_path = checkpoint_path.replace(".pt", "_finetune.json")
    best_ckpt_path = checkpoint_path.replace(".pt", "_finetune_best.pt")
    if os.path.exists(result_path) and os.path.exists(best_ckpt_path):
        with open(result_path) as f:
            res = json.load(f)
        print(f"Skip finetune: already done — accuracy: {res['accuracy']*100:.1f}%")
        return res["accuracy"]

    train_loader, val_loader = get_dataloaders(cfg)

    model = MaskedAutoencoder(cfg).to(cfg.device)
    ckpt = torch.load(checkpoint_path, map_location=cfg.device)
    model.load_state_dict(ckpt["model"])

    head = nn.Sequential(
        nn.LayerNorm(cfg.enc_embed_dim),
        nn.Linear(cfg.enc_embed_dim, cfg.num_classes),
    ).to(cfg.device)

    params = list(model.patch_embed.parameters()) + \
             list(model.enc_blocks.parameters()) + \
             list(model.enc_norm.parameters()) + \
             list(head.parameters())
    optimizer = torch.optim.AdamW(params, lr=cfg.ft_lr,
                                  weight_decay=cfg.ft_weight_decay, betas=(0.9, 0.999))
    scaler = torch.cuda.amp.GradScaler(enabled=cfg.use_amp)

    best_acc = 0.0
    for ft_epoch in tqdm(range(cfg.ft_epochs), desc="Fine-tuning"):
        model.train()
        head.train()
        ft_lr = cfg.ft_lr
        if ft_epoch < cfg.ft_warmup_epochs:
            ft_lr = cfg.ft_lr * (ft_epoch + 1) / cfg.ft_warmup_epochs
        else:
            progress = (ft_epoch - cfg.ft_warmup_epochs) / (cfg.ft_epochs - cfg.ft_warmup_epochs)
            ft_lr = cfg.ft_lr * 0.5 * (1 + math.cos(math.pi * progress))
        for pg in optimizer.param_groups:
            pg["lr"] = ft_lr

        for imgs, labels in train_loader:
            imgs, labels = imgs.to(cfg.device), labels.to(cfg.device)
            with torch.cuda.amp.autocast(enabled=cfg.use_amp):
                x = model.patch_embed(imgs)
                x = x + model.enc_pos_embed
                for blk in model.enc_blocks:
                    x = blk(x)
                x = model.enc_norm(x)
                x = x.mean(dim=1)
                logits = head(x)
                loss = F.cross_entropy(logits, labels)
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

        model.eval()
        head.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(cfg.device), labels.to(cfg.device)
                x = model.patch_embed(imgs)
                x = x + model.enc_pos_embed
                for blk in model.enc_blocks:
                    x = blk(x)
                x = model.enc_norm(x)
                x = x.mean(dim=1)
                preds = head(x).argmax(dim=1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)
        acc = correct / total
        if acc > best_acc:
            best_acc = acc
            torch.save({
                "model": model.state_dict(),
                "head": head.state_dict(),
            }, checkpoint_path.replace(".pt", "_finetune_best.pt"))

    with open(result_path, "w") as f:
        json.dump({"accuracy": best_acc}, f)
    print(f"Fine-tuning accuracy: {best_acc*100:.1f}%")
    return best_acc


def ablation(cfg):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if os.path.exists("ablation_mask_ratio.png") and os.path.exists("ablation_results.json"):
        print("Skip ablation: ablation_mask_ratio.png and ablation_results.json already exist")
        return

    ratios = [0.5, 0.75, 0.9]
    results = {}

    for r in ratios:
        print(f"\n{'='*50}\nMask ratio = {r}\n{'='*50}")
        cfg.mask_ratio = r
        tag = f"mr{int(r*100)}"
        pretrain(cfg, tag=tag)
        ckpt_path = os.path.join(cfg.checkpoint_dir, f"mae_{tag}_final.pt")
        lp_acc = linear_probe(cfg, ckpt_path)
        knn_acc = knn_evaluate(cfg, ckpt_path)
        results[r] = {"linear_probe": lp_acc, "knn": knn_acc}

    with open("ablation_results.json", "w") as f:
        json.dump({str(k): v for k, v in results.items()}, f, indent=2)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(ratios, [results[r]["linear_probe"]*100 for r in ratios], "o-", label="Linear probe")
    ax.plot(ratios, [results[r]["knn"]*100 for r in ratios], "s--", label="k-NN")
    ax.set_xlabel("Mask ratio")
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Mask ratio ablation (CIFAR-10)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xticks(ratios)
    plt.tight_layout()
    plt.savefig("ablation_mask_ratio.png", dpi=150)
    print("Saved ablation_mask_ratio.png and ablation_results.json")


def plot_loss(cfg):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if os.path.exists("loss_curve.png"):
        print("Skip plot_loss: loss_curve.png already exists")
        return

    loss_files = sorted([f for f in os.listdir(cfg.checkpoint_dir) if f.startswith("loss") and f.endswith(".json")])
    if not loss_files:
        print("No loss logs found in", cfg.checkpoint_dir)
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    for lf in loss_files:
        with open(os.path.join(cfg.checkpoint_dir, lf)) as f:
            history = json.load(f)
        epochs = [h["epoch"] for h in history]
        losses = [h["loss"] for h in history]
        lrs = [h["lr"] for h in history]
        label = lf.replace("loss_", "").replace("loss", "default").replace(".json", "")
        ax1.plot(epochs, losses, label=label)
        ax2.plot(epochs, lrs, label=label)

    ax1.set_xlabel("Epoch"); ax1.set_ylabel("Loss"); ax1.set_title("Training loss")
    ax1.legend(); ax1.grid(True, alpha=0.3)
    ax2.set_xlabel("Epoch"); ax2.set_ylabel("Learning rate"); ax2.set_title("LR schedule")
    ax2.legend(); ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("loss_curve.png", dpi=150)
    print("Saved loss_curve.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["pretrain", "linear_probe", "knn", "visualize",
                                         "finetune", "ablation", "plot_loss"])
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--mask_ratio", type=float, default=None)
    args = parser.parse_args()

    if args.mask_ratio is not None:
        cfg.mask_ratio = args.mask_ratio

    if args.mode == "pretrain":
        pretrain(cfg)
    elif args.mode == "linear_probe":
        assert args.checkpoint, "Fournir --checkpoint"
        linear_probe(cfg, args.checkpoint)
    elif args.mode == "knn":
        assert args.checkpoint, "Fournir --checkpoint"
        knn_evaluate(cfg, args.checkpoint)
    elif args.mode == "visualize":
        assert args.checkpoint, "Fournir --checkpoint"
        visualize(cfg, args.checkpoint)
    elif args.mode == "finetune":
        assert args.checkpoint, "Fournir --checkpoint"
        finetune(cfg, args.checkpoint)
    elif args.mode == "ablation":
        ablation(cfg)
    elif args.mode == "plot_loss":
        plot_loss(cfg)
