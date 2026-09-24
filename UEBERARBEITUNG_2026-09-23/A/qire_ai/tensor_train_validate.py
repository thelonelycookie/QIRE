"""Checks TTLinear's tensor-contraction bookkeeping against an independent,
unambiguous reference (explicit einsum) before trusting it. Run:
python -m qire_ai.tensor_train_validate
"""
from __future__ import annotations

import torch

from .tensor_train import TTLinear, factorize_balanced


def _assert_close(actual, expected, msg, tol=1e-5):
    if not torch.allclose(actual, expected, atol=tol):
        diff = (actual - expected).abs().max().item()
        raise AssertionError(f"FAILED: {msg} (max abs diff={diff})")
    print(f"  ok: {msg}")


def reference_reconstruct_2core(cores) -> torch.Tensor:
    """Independent reference for exactly 2 cores, written with explicit
    einsum indices instead of the generic tensordot+permute loop in
    TTLinear.reconstruct_weight -- if both agree, the generic version's axis
    bookkeeping is very unlikely to be coincidentally wrong."""
    c0, c1 = cores[0], cores[1]  # c0: (1,m1,n1,r1), c1: (r1,m2,n2,1)
    # W[m1,m2,n1,n2] = sum_r c0[0,m1,n1,r] * c1[r,m2,n2,0]
    w = torch.einsum("amnr,rpqb->mpnq", c0, c1)  # a=b=1 (boundary ranks)
    m1, m2, n1, n2 = c0.shape[1], c1.shape[1], c0.shape[2], c1.shape[2]
    return w.reshape(m1 * m2, n1 * n2)


def test_factorize_balanced():
    print("factorize_balanced: product must cover the requested dimension")
    for dim, k in [(13, 2), (13, 3), (32, 2), (30, 3), (64, 3), (7, 1)]:
        factors = factorize_balanced(dim, k)
        assert len(factors) == k, f"expected {k} factors, got {factors}"
        product = 1
        for f in factors:
            product *= f
        assert product >= dim, f"factorize_balanced({dim},{k})={factors} covers only {product} < {dim}"
    print("  ok: all factorizations cover their target dimension")


def test_two_core_matches_reference():
    print("2-core TT reconstruction vs. independent einsum reference")
    torch.manual_seed(0)
    layer = TTLinear(in_features=13, out_features=17, num_cores=2, tt_rank=5, bias=False)
    own = layer.reconstruct_weight()
    ref_full = reference_reconstruct_2core(list(layer.cores))
    ref_cropped = ref_full[: layer.out_features, : layer.in_features]
    _assert_close(own, ref_cropped, "TTLinear.reconstruct_weight matches independent einsum reference")


def test_single_core_is_plain_dense():
    print("1-core TT (rank forced to 1) should behave like an ordinary dense matrix")
    torch.manual_seed(1)
    layer = TTLinear(in_features=8, out_features=8, num_cores=1, tt_rank=4, bias=False)
    w = layer.reconstruct_weight()
    expected = layer.cores[0][0, :, :, 0]
    _assert_close(w, expected, "single-core reconstruction equals the raw core matrix")


def test_gradient_flows():
    print("gradients must flow into every TT core during backprop")
    torch.manual_seed(2)
    layer = TTLinear(in_features=13, out_features=17, num_cores=3, tt_rank=4)
    x = torch.randn(5, 13)
    out = layer(x)
    loss = out.pow(2).sum()
    loss.backward()
    for i, core in enumerate(layer.cores):
        assert core.grad is not None and core.grad.abs().sum() > 0, f"core {i} got no gradient"
    print("  ok: all cores received nonzero gradients")


def test_can_fit_a_random_target_matrix():
    print("with generous rank, TT-Linear should be able to fit an arbitrary target linear map")
    torch.manual_seed(3)
    in_f, out_f = 8, 8
    target_w = torch.randn(out_f, in_f)
    layer = TTLinear(in_features=in_f, out_features=out_f, num_cores=2, tt_rank=8, bias=False)
    x = torch.randn(200, in_f)
    y_target = x @ target_w.t()
    opt = torch.optim.Adam(layer.parameters(), lr=0.05)
    for _ in range(500):
        opt.zero_grad()
        loss = (layer(x) - y_target).pow(2).mean()
        loss.backward()
        opt.step()
    final_loss = (layer(x) - y_target).pow(2).mean().item()
    assert final_loss < 1e-3, f"TT-Linear failed to fit a random target matrix (final MSE={final_loss})"
    print(f"  ok: fit random target matrix, final MSE={final_loss:.6f}")


def test_param_count_reduction_at_scale():
    print("parameter reduction should become real once matrices are large enough")
    small = TTLinear(in_features=13, out_features=32, num_cores=3, tt_rank=4)
    large = TTLinear(in_features=1024, out_features=1024, num_cores=3, tt_rank=4)
    small_ratio = small.num_tt_parameters() / small.num_dense_equivalent_parameters()
    large_ratio = large.num_tt_parameters() / large.num_dense_equivalent_parameters()
    print(f"  small (13x32): TT params={small.num_tt_parameters()}, dense={small.num_dense_equivalent_parameters()}, ratio={small_ratio:.3f}")
    print(f"  large (1024x1024): TT params={large.num_tt_parameters()}, dense={large.num_dense_equivalent_parameters()}, ratio={large_ratio:.5f}")
    assert large_ratio < small_ratio, "compression ratio should improve (shrink) as matrices grow"
    print("  ok: compression ratio improves with scale (as tensor-train theory predicts)")


def run_all():
    tests = [
        test_factorize_balanced,
        test_two_core_matches_reference,
        test_single_core_is_plain_dense,
        test_gradient_flows,
        test_can_fit_a_random_target_matrix,
        test_param_count_reduction_at_scale,
    ]
    for t in tests:
        t()
    print(f"\nAll {len(tests)} validation checks passed.")


if __name__ == "__main__":
    run_all()
