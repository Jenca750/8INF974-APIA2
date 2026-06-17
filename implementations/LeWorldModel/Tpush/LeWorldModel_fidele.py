

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from datasets import load_dataset
import timm
import numpy as np
import matplotlib.pyplot as plt


class Projector(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.fc = nn.Linear(in_dim, out_dim)
        self.bn = nn.BatchNorm1d(out_dim)

    def forward(self, x):
        return self.bn(self.fc(x))


class LeWMEncoder(nn.Module):
    def __init__(self, embed_dim=192):
        super().__init__()
        self.vit = timm.create_model(
            'vit_tiny_patch16_224', pretrained=False,
            num_classes=0, global_pool=''
        )
        self.projector = Projector(self.vit.num_features, embed_dim)

    def forward(self, obs):
        B, T, C, H, W = obs.shape
        x = obs.view(B * T, C, H, W)
        features = self.vit.forward_features(x)
        cls_token = features[:, 0]
        z = self.projector(cls_token)
        return z.view(B, T, -1)


class AdaLNBlock(nn.Module):
    def __init__(self, dim, nhead=16, mlp_ratio=4, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.attn = nn.MultiheadAttention(dim, nhead, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * mlp_ratio),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * mlp_ratio, dim),
            nn.Dropout(dropout),
        )
        self.adaLN = nn.Sequential(nn.SiLU(), nn.Linear(dim, 6 * dim))
        nn.init.zeros_(self.adaLN[-1].weight)
        nn.init.zeros_(self.adaLN[-1].bias)

    def forward(self, x, cond, attn_mask):
        s1, sc1, g1, s2, sc2, g2 = self.adaLN(cond).chunk(6, dim=-1)
        h = self.norm1(x) * (1 + sc1) + s1
        attn_out, _ = self.attn(h, h, h, attn_mask=attn_mask, need_weights=False)
        x = x + g1 * attn_out
        h = self.norm2(x) * (1 + sc2) + s2
        x = x + g2 * self.mlp(h)
        return x


class LeWMPredictor(nn.Module):
    """
    Prédicteur causal. Papier : 6 couches, 16 têtes, dropout=0.1 (Section 3.1).
    Les actions sont injectées via AdaLN à chaque couche.
    """
    def __init__(self, embed_dim=192, action_dim=10, num_layers=6,
                 nhead=16, dropout=0.1, max_len=64):
        super().__init__()
        self.action_proj = nn.Linear(action_dim, embed_dim)
        self.pos_embed = nn.Parameter(torch.zeros(1, max_len, embed_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        self.blocks = nn.ModuleList([
            AdaLNBlock(embed_dim, nhead=nhead, dropout=dropout)
            for _ in range(num_layers)
        ])
        self.projector = Projector(embed_dim, embed_dim)

    def forward(self, z, actions):
        B, T, D = z.shape
        cond = self.action_proj(actions)
        x = z + self.pos_embed[:, :T]
        mask = torch.triu(torch.ones(T, T, dtype=torch.bool, device=z.device), diagonal=1)
        for blk in self.blocks:
            x = blk(x, cond, attn_mask=mask)
        out = self.projector(x.reshape(B * T, D))
        return out.view(B, T, D)




def epps_pulley_stat(h, t_nodes):
    """h : (T, B, M) ; t_nodes : (K,)."""
    w = torch.exp(-t_nodes ** 2 / 2.0)
    phi_0 = torch.exp(-t_nodes ** 2 / 2.0)
    arg = h.unsqueeze(-1) * t_nodes
    phi_real = torch.cos(arg).mean(dim=1)
    phi_imag = torch.sin(arg).mean(dim=1)
    diff_sq = (phi_real - phi_0) ** 2 + phi_imag ** 2
    stats = torch.trapz(w * diff_sq, t_nodes, dim=-1)
    return stats.mean()


def sigreg(z, num_projections=256, num_knots=17):
    """
    SIGReg step-wise : un test de normalité par pas de temps, sur la dimension batch.
    Forcé en fp32 : les différences de fonctions caractéristiques sont trop
    petites pour bf16 (bruit d'arrondi → gradient inutilisable).
    """
    z = z.float()
    B, T, D = z.shape
    directions = torch.randn(D, num_projections, device=z.device, dtype=torch.float32)
    directions = F.normalize(directions, p=2, dim=0)
    h = torch.einsum('btd,dm->tbm', z, directions)
    t_nodes = torch.linspace(0.2, 4.0, steps=num_knots, device=z.device, dtype=torch.float32)
    return epps_pulley_stat(h, t_nodes)


class LeWorldModel(nn.Module):
    def __init__(self, embed_dim=192, action_dim=10):
        super().__init__()
        self.encoder = LeWMEncoder(embed_dim)
        self.predictor = LeWMPredictor(embed_dim, action_dim)

    def forward(self, obs, actions):
        z = self.encoder(obs)
        next_z_pred = self.predictor(z, actions)
        return z, next_z_pred

    def loss(self, obs, actions, lambd=0.1):
        z, next_z_pred = self(obs, actions)
        pred_loss = F.mse_loss(next_z_pred[:, :-1].float(), z[:, 1:].float())
        sigreg_loss = sigreg(z)
        total_loss = pred_loss + lambd * sigreg_loss
        return total_loss, pred_loss, sigreg_loss, z


class LeWMDecoder(nn.Module):
    def __init__(self, embed_dim=192, hidden_dim=192, patch_size=16,
                 img_size=224, num_layers=4):
        super().__init__()
        self.patch_size = patch_size
        self.num_patches_1d = img_size // patch_size
        self.num_patches = self.num_patches_1d ** 2

        self.kv_proj = nn.Linear(embed_dim, hidden_dim)
        self.query_tokens = nn.Parameter(torch.zeros(1, self.num_patches, hidden_dim))
        nn.init.trunc_normal_(self.query_tokens, std=0.02)
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, hidden_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=hidden_dim, nhead=4, dim_feedforward=hidden_dim * 4,
            dropout=0.1, batch_first=True)
        self.transformer = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, patch_size * patch_size * 3)

    def forward(self, z):
        B = z.shape[0]
        memory = self.kv_proj(z).unsqueeze(1)
        queries = (self.query_tokens + self.pos_embed).expand(B, -1, -1)
        x = self.transformer(tgt=queries, memory=memory)
        x = self.norm(x)
        patches = self.out_proj(x)
        P, N = self.patch_size, self.num_patches_1d
        patches = patches.view(B, N, N, 3, P, P)
        return patches.permute(0, 3, 1, 4, 2, 5).contiguous().view(B, 3, N * P, N * P)




class PushTImageDataset(Dataset):
    def __init__(self, hf_dataset_split, seq_len=4, frame_skip=5, image_size=224):
        self.dataset = hf_dataset_split
        self.seq_len = seq_len
        self.frame_skip = frame_skip
        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
        episodes = np.array(self.dataset['episode_index'])
        self.valid_indices = self._compute_valid_indices(episodes)

    def _compute_valid_indices(self, episodes):
        span = self.seq_len * self.frame_skip
        max_idx = len(episodes) - span
        return [i for i in range(max_idx)
                if episodes[i] == episodes[i + span - 1]]

    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, idx):
        start_idx = self.valid_indices[idx]
        indices = [start_idx + i * self.frame_skip for i in range(self.seq_len)]
        images, actions = [], []
        for i in indices:
            images.append(self.transform(self.dataset[i]['observation.image']))
            action_block = [self.dataset[i + j]['action'] for j in range(self.frame_skip)]
            actions.append(torch.tensor(action_block, dtype=torch.float32).flatten())
        return torch.stack(images), torch.stack(actions)




@torch.no_grad()
def plot_predictions(model, decoder, test_loader, device, horizon=5, n_context=3):
    model.eval()
    decoder.eval()

    obs, actions = next(iter(test_loader))
    obs, actions = obs.to(device), actions.to(device)
    B, T, C, H, W = obs.shape

    n_context = min(n_context, T)

    z = model.encoder(obs)
    z_history = z[:, :n_context]

    
    last_action = actions[:, -1:]                          
    padding = last_action.expand(-1, horizon, -1)          
    actions_ext = torch.cat([actions, padding], dim=1)     

    for t in range(horizon):
        current_actions = actions_ext[:, :n_context + t]
        next_z_pred = model.predictor(z_history, current_actions)
        z_history = torch.cat([z_history, next_z_pred[:, -1:]], dim=1)

    n_show = n_context + horizon
    flat_latents = z_history[0].view(-1, z_history.size(-1))
    pred_images = decoder(flat_latents).view(n_show, C, H, W)

    fig, axes = plt.subplots(2, n_show, figsize=(3 * n_show, 6))
    for t in range(n_show):
        if t < T:
            gt = (obs[0, t].cpu().permute(1, 2, 0).numpy() * 0.5 + 0.5).clip(0, 1)
            axes[0, t].imshow(gt)
        else:
            axes[0, t].imshow(np.ones((H, W, 3)) * 0.9)
        axes[0, t].axis('off')
        label = f"Context (t={t})" if t < n_context else f"Truth (t={t})"
        axes[0, t].set_title(label, fontsize=8)

        pr = (pred_images[t].cpu().permute(1, 2, 0).numpy() * 0.5 + 0.5).clip(0, 1)
        axes[1, t].imshow(pr)
        axes[1, t].axis('off')
        axes[1, t].set_title("Decoded" if t < n_context else f"Pred (t={t})", fontsize=8)

    plt.suptitle("Rollout autorégressif — contexte N=3 + horizon=5", fontsize=11)
    plt.tight_layout()
    plt.show()




TRAIN_WM = False
TRAIN_DECODER = True

ds = load_dataset("lerobot/pusht_image")

train_dataset = PushTImageDataset(ds["train"])
train_loader = DataLoader(
    train_dataset, batch_size=64, shuffle=True,
    drop_last=True, num_workers=4, pin_memory=True
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device : {device}")
print(f"Fenêtres train : {len(train_dataset)}")

action_dim = 10
model = LeWorldModel(embed_dim=192, action_dim=action_dim).to(device)
decoder = LeWMDecoder(embed_dim=192, img_size=224).to(device)


if TRAIN_WM:
    optimizer_wm = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    warmup_steps = 1000
    scheduler_wm = optim.lr_scheduler.LambdaLR(
        optimizer_wm, lambda step: min(1.0, (step + 1) / warmup_steps))

    epochs_wm = 10
    lambd = 0.1

    print("\nPhase 1")
    for epoch in range(epochs_wm):
        model.train()
        total_wm = total_pred = total_reg = 0.0

        for obs, actions in train_loader:
            obs = obs.to(device, non_blocking=True)
            actions = actions.to(device, non_blocking=True)

            optimizer_wm.zero_grad()
            loss_wm, pred_loss, reg_loss, _ = model.loss(obs, actions, lambd)
            loss_wm.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer_wm.step()
            scheduler_wm.step()

            total_wm += loss_wm.item()
            total_pred += pred_loss.item()
            total_reg += reg_loss.item()

        n = len(train_loader)
        print(f"WM Epoch {epoch+1}/{epochs_wm} | "
                f"Loss: {total_wm/n:.4f} | "
                f"Pred: {total_pred/n:.4f} | "
                f"SIGReg: {total_reg/n:.4f}")

        torch.save(model.state_dict(), 'lewm_model_weights.pth')
        print("Poids WM sauvegardés.")

model.load_state_dict(torch.load('lewm_model_weights (8).pth',
                                    map_location=device, weights_only=True))


if TRAIN_DECODER:
    model.eval()
    optimizer_dec = optim.AdamW(decoder.parameters(), lr=1e-4, weight_decay=1e-4)
    epochs_dec = 20
    best_dec_loss = float('inf')

    print("\nPhase 2")
    for epoch in range(epochs_dec):
        decoder.train()
        total_dec = 0.0

        for obs, _ in train_loader:
            obs = obs.to(device, non_blocking=True)

            with torch.no_grad():
                z = model.encoder(obs)

            B, T, D = z.shape
            z_flat = z.reshape(B * T, D)
            obs_flat = obs.reshape(B * T, 3, 224, 224)

            optimizer_dec.zero_grad()
            reconstructed = decoder(z_flat)
            loss_dec = F.mse_loss(reconstructed, obs_flat)
            loss_dec.backward()
            torch.nn.utils.clip_grad_norm_(decoder.parameters(), 1.0)
            optimizer_dec.step()

            total_dec += loss_dec.item()

        avg_dec = total_dec / len(train_loader)
        print(f"Decoder Epoch {epoch+1}/{epochs_dec} | Loss: {avg_dec:.4f}")

        if avg_dec <= best_dec_loss:
            best_dec_loss = avg_dec
            torch.save(decoder.state_dict(), 'lewm_decoder_weights.pth')

decoder.load_state_dict(torch.load('lewm_decoder_weights.pth',
                                    map_location=device, weights_only=True))

test_split = ds["test"] if "test" in ds else ds["train"]
test_dataset = PushTImageDataset(test_split)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=True, drop_last=True)

print("\n Rollout  (contexte N=3, horizon=5)")
plot_predictions(model, decoder, test_loader, device, horizon=5, n_context=3)