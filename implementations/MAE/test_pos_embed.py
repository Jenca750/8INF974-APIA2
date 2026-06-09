import torch
from mae import get_sincos_pos_embed


def test_pos_embed():
    embed_dim = 192
    num_patches = 64

    pos = get_sincos_pos_embed(embed_dim, num_patches)

    assert pos.shape == (1, 64, 192), f"Shape attendue (1, 64, 192), obtenue {pos.shape}"
    print("✓ Shape correcte")

    assert not pos.requires_grad, "Ne doit pas avoir de gradient"
    print("✓ Pas de gradient")

    assert pos.min() >= -1.0 and pos.max() <= 1.0, f"Valeurs hors [-1,1]: min={pos.min():.3f}, max={pos.max():.3f}"
    print("✓ Valeurs dans [-1, 1]")

    pos_flat = pos.squeeze(0)
    dists = torch.cdist(pos_flat.unsqueeze(0), pos_flat.unsqueeze(0)).squeeze(0)
    dists.fill_diagonal_(float('inf'))
    min_dist = dists.min().item()
    assert min_dist > 1e-6, f"Deux patches ont le même embedding (dist min = {min_dist})"
    print(f"✓ Tous les embeddings sont uniques (dist min = {min_dist:.4f})")

    half = embed_dim // 2
    row_part = pos_flat[:, :half]
    col_part = pos_flat[:, half:]

    same_row_diff = (row_part[0] - row_part[1]).abs().max().item()
    assert same_row_diff < 1e-6, f"Patches (0,0) et (0,1) devraient avoir la même partie row, diff={same_row_diff}"
    print("✓ Même ligne -> même row embedding")

    same_col_diff = (col_part[0] - col_part[8]).abs().max().item()
    assert same_col_diff < 1e-6, f"Patches (0,0) et (1,0) devraient avoir la même partie col, diff={same_col_diff}"
    print("✓ Même colonne -> même col embedding")

    diff_row = (row_part[0] - row_part[8]).abs().max().item()
    assert diff_row > 1e-3, "Patches sur lignes différentes devraient avoir des row embeddings différents"
    print("✓ Lignes différentes -> row embeddings différents")

    pos_dec = get_sincos_pos_embed(96, 64)
    assert pos_dec.shape == (1, 64, 96), f"Shape décodeur attendue (1, 64, 96), obtenue {pos_dec.shape}"
    print("✓ Fonctionne aussi pour embed_dim=96")

    first = pos_flat[0]
    assert abs(first[0].item() - 0.0) < 1e-6, f"sin(0) devrait être 0, obtenu {first[0].item()}"
    assert abs(first[1].item() - 1.0) < 1e-6, f"cos(0) devrait être 1, obtenu {first[1].item()}"
    print("✓ Position (0,0) commence par sin(0)=0, cos(0)=1")

    print("\n=== Tous les tests passent ===")


if __name__ == "__main__":
    test_pos_embed()
