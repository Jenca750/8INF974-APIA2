import torch
from mae import MaskedAutoencoder, Config


def test_random_masking():
    print("=== Test random_masking ===")
    torch.manual_seed(42)

    B, N, D = 4, 64, 192
    mask_ratio = 0.75
    N_visible = int(N * (1 - mask_ratio))

    x = torch.randn(B, N, D)
    cfg = Config()
    model = MaskedAutoencoder(cfg)

    x_visible, mask, ids_restore = model.random_masking(x, mask_ratio)

    assert x_visible.shape == (B, N_visible, D), \
        f"x_visible shape attendue ({B}, {N_visible}, {D}), obtenue {x_visible.shape}"
    print("✓ x_visible shape correcte")

    assert mask.shape == (B, N), f"mask shape attendue ({B}, {N}), obtenue {mask.shape}"
    print("✓ mask shape correcte")

    assert ids_restore.shape == (B, N), f"ids_restore shape attendue ({B}, {N}), obtenue {ids_restore.shape}"
    print("✓ ids_restore shape correcte")

    num_visible = (mask == 0).sum(dim=1)
    num_masked = (mask == 1).sum(dim=1)
    assert (num_visible == N_visible).all()
    assert (num_masked == N - N_visible).all()
    print(f"✓ Masque : {N_visible} visibles, {N - N_visible} masqués par sample")

    for b in range(B):
        for i in range(N_visible):
            token = x_visible[b, i]
            found = any(torch.allclose(token, x[b, j], atol=1e-6) for j in range(N))
            assert found, f"Token visible [{b},{i}] ne correspond à aucun token de x"
    print("✓ Les tokens visibles proviennent bien de x")

    assert not torch.equal(mask[0], mask[1]), "Le masque devrait être différent entre les samples"
    print("✓ Masques différents entre samples")

    for b in range(B):
        sorted_ids = ids_restore[b].sort()[0]
        assert torch.equal(sorted_ids, torch.arange(N)), \
            f"ids_restore[{b}] n'est pas une permutation valide de 0..{N-1}"
    print("✓ ids_restore est une permutation valide")

    print("=== random_masking OK ===\n")


def test_forward_encoder():
    print("=== Test forward_encoder ===")
    torch.manual_seed(42)

    cfg = Config()
    model = MaskedAutoencoder(cfg)
    model.eval()

    B = 4
    imgs = torch.randn(B, 3, 32, 32)
    latent, mask, ids_restore = model.forward_encoder(imgs)
    N_visible = int(cfg.num_patches * (1 - cfg.mask_ratio))

    assert latent.shape == (B, N_visible, cfg.enc_embed_dim), \
        f"latent shape attendue ({B}, {N_visible}, {cfg.enc_embed_dim}), obtenue {latent.shape}"
    print(f"✓ latent shape correcte: {latent.shape}")

    assert mask.shape == (B, cfg.num_patches)
    print(f"✓ mask shape correcte: {mask.shape}")

    assert ids_restore.shape == (B, cfg.num_patches)
    print(f"✓ ids_restore shape correcte: {ids_restore.shape}")

    assert latent.abs().mean() > 0.01, "Le latent semble nul"
    print("✓ latent non nul")

    model.train()
    imgs.requires_grad_(True)
    latent, mask, ids_restore = model.forward_encoder(imgs)
    latent.sum().backward()
    assert imgs.grad is not None, "Pas de gradient sur les images"
    print("✓ Gradient flow OK")

    print("=== forward_encoder OK ===\n")


def test_forward_decoder():
    print("=== Test forward_decoder ===")
    torch.manual_seed(42)

    cfg = Config()
    model = MaskedAutoencoder(cfg)
    model.eval()

    B = 4
    imgs = torch.randn(B, 3, 32, 32)

    latent, mask, ids_restore = model.forward_encoder(imgs)
    pred = model.forward_decoder(latent, ids_restore)

    assert pred.shape == (B, cfg.num_patches, cfg.pixels_per_patch), \
        f"pred shape attendue ({B}, {cfg.num_patches}, {cfg.pixels_per_patch}), obtenue {pred.shape}"
    print(f"✓ pred shape correcte: {pred.shape}")

    assert pred.abs().mean() > 0.001, "Les prédictions semblent nulles"
    print("✓ Prédictions non nulles")

    model.train()
    loss, pred, mask = model(imgs)

    assert loss.ndim == 0, f"La loss devrait être un scalaire, shape={loss.shape}"
    print(f"✓ Loss est un scalaire: {loss.item():.4f}")

    assert loss.item() > 0, "La loss devrait être positive"
    assert not torch.isnan(loss), "La loss est NaN!"
    print("✓ Loss positive et non NaN")

    loss.backward()
    assert all(p.grad is not None for p in model.enc_blocks.parameters()), "Pas de gradient dans l'encodeur"
    assert all(p.grad is not None for p in model.dec_blocks.parameters()), "Pas de gradient dans le décodeur"
    print("✓ Gradient flow complet (encodeur + décodeur)")

    model.eval()
    torch.manual_seed(1)
    loss1, _, _ = model(imgs)
    torch.manual_seed(2)
    loss2, _, _ = model(imgs)
    assert not torch.allclose(loss1, loss2), "Deux masques différents devraient donner des loss différentes"
    print("✓ Loss varie avec le masque aléatoire")

    recon = model.unpatchify(pred)
    assert recon.shape == (B, 3, 32, 32), f"Reconstruction shape attendue ({B}, 3, 32, 32), obtenue {recon.shape}"
    print("✓ Reconstruction donne une image valide")

    print("=== forward_decoder OK ===\n")


if __name__ == "__main__":
    test_random_masking()
    test_forward_encoder()
    test_forward_decoder()
    print("=== TOUS LES TESTS PASSENT ===")
