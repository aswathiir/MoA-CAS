"""
Unit tests for moa_cas.adapters.router, on a fake encoder that calls its
layers the way NeMo's ConformerEncoder does (keyword args incl. pad_mask).
"""

import pytest
import torch

from moa_cas.adapters.bottleneck import BottleneckAdapter
from moa_cas.adapters.router import (
    FrameRouter, MixtureOfAdapters, attach, build_experts, identity_expert, load_balancing_loss,
)

DIM, LAYERS = 8, 3


class FakeConformerLayer(torch.nn.Module):
    def forward(self, x, att_mask=None, pos_emb=None, pad_mask=None):
        return x + 1


class FakeEncoder(torch.nn.Module):
    d_model = DIM

    def __init__(self):
        super().__init__()
        self.layers = torch.nn.ModuleList(FakeConformerLayer() for _ in range(LAYERS))
        self.proj = torch.nn.Linear(DIM, DIM)

    def forward(self, x, pad_mask=None):
        for layer in self.layers:
            x = layer(x=x, pad_mask=pad_mask)
        return x


def trained_expert(seed):
    torch.manual_seed(seed)
    stack = torch.nn.ModuleList(BottleneckAdapter(DIM, 4) for _ in range(LAYERS))
    for a in stack:
        torch.nn.init.normal_(a.up.weight)  # non-identity, like a trained Stage 1 adapter
    return stack


def test_router_is_uniform_at_init():
    probs = FrameRouter(DIM, 3)(torch.randn(2, 7, DIM))
    assert torch.allclose(probs, torch.full_like(probs, 1 / 3))


def test_load_balancing_loss_bounds_and_padding():
    k = 2
    balanced = torch.tensor([[[0.9, 0.1], [0.1, 0.9]]])
    collapsed = torch.tensor([[[1.0, 0.0], [1.0, 0.0]]])
    assert load_balancing_loss(balanced).item() == pytest.approx(1.0)
    assert load_balancing_loss(collapsed).item() == pytest.approx(k)
    # The padded second frame must not count: only the first frame is real.
    valid = torch.tensor([[True, False]])
    assert load_balancing_loss(balanced, valid).item() == pytest.approx(k * 0.9)


def test_mixture_equals_weighted_sum_of_experts():
    encoder = FakeEncoder()
    a, b = trained_expert(0), trained_expert(1)
    moa = attach(encoder, MixtureOfAdapters([a, b], DIM))  # untrained router: w = (0.5, 0.5)

    x = torch.randn(1, 4, DIM)
    expected = x
    for layer in range(LAYERS):
        h = expected + 1
        expected = 0.5 * a[layer](h) + 0.5 * b[layer](h)
    assert torch.allclose(encoder(x), expected, atol=1e-5)


def test_identity_expert_with_full_weight_recovers_frozen_backbone():
    encoder = FakeEncoder()
    moa = attach(encoder, MixtureOfAdapters([trained_expert(0), identity_expert(LAYERS)], DIM))
    for router in moa.routers:
        router.mlp[-1].bias.data = torch.tensor([-50.0, 50.0])  # all weight on identity
    x = torch.randn(1, 4, DIM)
    assert torch.allclose(encoder(x), x + LAYERS, atol=1e-4)


def test_only_routers_train():
    encoder = FakeEncoder()
    moa = attach(encoder, MixtureOfAdapters([trained_expert(0), trained_expert(1)], DIM))
    assert all(not p.requires_grad for p in encoder.parameters())
    assert all(not p.requires_grad for p in moa.experts.parameters())

    out = encoder(torch.randn(2, 5, DIM))
    (out.pow(2).mean() + moa.load_balancing_loss()).backward()
    assert all(p.grad is not None for p in moa.routers.parameters())
    assert all(p.grad is None for p in moa.experts.parameters())


def test_pad_mask_is_seen_and_weights_recorded():
    encoder = FakeEncoder()
    moa = attach(encoder, MixtureOfAdapters([trained_expert(0), identity_expert(LAYERS)], DIM))
    pad = torch.tensor([[False, False, True, True]])
    encoder(torch.randn(1, 4, DIM), pad_mask=pad)
    assert torch.equal(moa._valid[0], ~pad)
    assert moa.routing_weights().shape == (1, 4, 2)


def test_hard_routing_selects_one_expert_per_frame():
    encoder = FakeEncoder()
    a, b = trained_expert(0), trained_expert(1)
    moa = attach(encoder, MixtureOfAdapters([a, b], DIM, routing="hard"))
    for router in moa.routers:
        router.mlp[-1].bias.data = torch.tensor([1.0, 0.0])  # expert 0 wins, softly
    x = torch.randn(1, 3, DIM)
    expected = x
    for layer in range(LAYERS):
        expected = a[layer](expected + 1)
    assert torch.allclose(encoder(x), expected, atol=1e-5)


def test_build_experts_loads_stage1_checkpoint_and_checks_backbone(tmp_path):
    stack = trained_expert(0)
    path = tmp_path / "hi_adapter.pt"
    torch.save({"adapters": stack.state_dict(), "lang": "hi", "bottleneck": 4}, path)

    stacks, names = build_experts([f"hi={path}", "identity"], LAYERS, DIM, backbone_lang="hi")
    assert names == ["hi", "identity"]
    x = torch.randn(1, 2, DIM)
    assert torch.allclose(stacks[0][1](x), stack[1](x))

    with pytest.raises(ValueError, match="backbone"):
        build_experts([f"hi={path}", "identity"], LAYERS, DIM, backbone_lang="bn")


def test_rejects_single_expert_and_layer_mismatch():
    with pytest.raises(ValueError):
        MixtureOfAdapters([trained_expert(0)], DIM)
    moa = MixtureOfAdapters([identity_expert(LAYERS + 1), identity_expert(LAYERS + 1)], DIM)
    with pytest.raises(ValueError, match="layers"):
        attach(FakeEncoder(), moa)


def test_fixed_weights_select_backbone_or_single_expert():
    encoder = FakeEncoder()
    a = trained_expert(0)
    moa = attach(encoder, MixtureOfAdapters([a, identity_expert(LAYERS)], DIM))
    x = torch.randn(1, 3, DIM)

    moa.fixed_weights = [0.0, 1.0]
    assert torch.allclose(encoder(x), x + LAYERS, atol=1e-6)  # untouched backbone

    moa.fixed_weights = [1.0, 0.0]
    expected = x
    for layer in range(LAYERS):
        expected = a[layer](expected + 1)
    assert torch.allclose(encoder(x), expected, atol=1e-5)  # adapter alone
    assert torch.allclose(moa.routing_weights()[0, :, 0], torch.ones(3))
