"""
Unit test for moa_cas.adapters.bottleneck — the one piece testable
without the real (499MB, gated) NeMo checkpoint. nemo_backbone.py and
language_mask.py were verified manually against the actual checkpoint
(see docs/ADAPTER_TRAINING.md) rather than covered here.
"""

import torch

from moa_cas.adapters.bottleneck import BottleneckAdapter, insert_into_encoder


def test_adapter_is_identity_at_init():
    adapter = BottleneckAdapter(dim=8, bottleneck=4)
    x = torch.randn(2, 5, 8)
    assert torch.allclose(adapter(x), x)


def test_insert_into_encoder_freezes_backbone_and_returns_adapters():
    class FakeConformerLayer(torch.nn.Module):
        def forward(self, x):
            return x + 1

    class FakeEncoder(torch.nn.Module):
        d_model = 8

        def __init__(self, n_layers=3):
            super().__init__()
            self.layers = torch.nn.ModuleList(FakeConformerLayer() for _ in range(n_layers))
            self.proj = torch.nn.Linear(8, 8)

        def forward(self, x):
            for layer in self.layers:
                x = layer(x)
            return x

    encoder = FakeEncoder()
    adapters = insert_into_encoder(encoder, bottleneck=4)

    assert len(adapters) == 3
    assert all(not p.requires_grad for p in encoder.parameters())
    assert all(p.requires_grad for a in adapters for p in a.parameters())

    x = torch.randn(1, 2, 8)
    out = encoder(x)
    assert torch.allclose(out, x + 3)  # adapters are identity at init, so untouched pass-through
