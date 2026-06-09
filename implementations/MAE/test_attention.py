import torch
from mae import Attention


def test_attention():
    torch.manual_seed(42)

    dim = 192
    num_heads = 3
    B = 4
    N = 16

    attn = Attention(dim, num_heads)
    x = torch.randn(B, N, dim)

    out = attn(x)
    assert out.shape == (B, N, dim), f"Shape attendue ({B}, {N}, {dim}), obtenue {out.shape}"
    print("✓ Shape correcte (B, N, D) -> (B, N, D)")

    loss = out.sum()
    loss.backward()
    has_grad = all(p.grad is not None for p in attn.parameters())
    assert has_grad, "Tous les paramètres doivent recevoir un gradient"
    print("✓ Gradient flow OK")

    attn.zero_grad()
    x2 = torch.randn(B, N, dim)
    out2 = attn(x2)
    assert not torch.allclose(out, out2), "Deux entrées différentes donnent la même sortie"
    print("✓ Sorties différentes pour entrées différentes")

    x_single = torch.randn(1, 1, dim)
    out_single = attn(x_single)
    assert out_single.shape == (1, 1, dim), f"Shape avec 1 token: {out_single.shape}"
    print("✓ Fonctionne avec un seul token")

    num_params = sum(p.numel() for p in attn.parameters())
    expected_min = 4 * dim * dim
    expected_max = 4 * dim * dim + 4 * dim
    assert expected_min <= num_params <= expected_max, \
        f"Nombre de params suspect: {num_params} (attendu entre {expected_min} et {expected_max})"
    print(f"✓ Nombre de paramètres cohérent: {num_params}")

    perm = torch.randperm(N)
    x_perm = x[:, perm, :]
    out_perm = attn(x_perm)
    out_orig_perm = out[:, perm, :]
    assert torch.allclose(out_perm, out_orig_perm, atol=1e-5), \
        "L'attention doit être équivariante aux permutations des tokens"
    print("✓ Équivariance aux permutations")

    attn_dec = Attention(96, 3)
    x_dec = torch.randn(2, 64, 96)
    out_dec = attn_dec(x_dec)
    assert out_dec.shape == (2, 64, 96)
    print("✓ Fonctionne avec dim=96 (décodeur)")

    print("\n=== Tous les tests passent ===")


if __name__ == "__main__":
    test_attention()
