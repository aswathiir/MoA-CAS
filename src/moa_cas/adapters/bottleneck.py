"""
moa_cas/adapters/bottleneck.py
---------------------------------
Bottleneck adapters for Profile Learning: small trainable modules
inserted into a frozen encoder via forward hooks, so the pretrained
backbone's own module tree and state dict stay untouched.
"""

import torch.nn as nn


class BottleneckAdapter(nn.Module):
    """Down-project -> nonlinearity -> up-project -> residual add.

    Zero-initialized so it starts as an identity passthrough — the
    backbone's original behavior is unchanged until training moves it.
    """

    def __init__(self, dim: int, bottleneck: int = 64):
        super().__init__()
        self.down = nn.Linear(dim, bottleneck)
        self.act = nn.ReLU()
        self.up = nn.Linear(bottleneck, dim)
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)

    def forward(self, x):
        return x + self.up(self.act(self.down(x)))


def insert_into_encoder(encoder, bottleneck: int = 64) -> nn.ModuleList:
    """
    Freezes every encoder parameter and attaches one BottleneckAdapter per
    layer via a forward hook. Returns the adapters — the only trainable
    parameters this produces; the caller owns their optimizer.
    """
    for p in encoder.parameters():
        p.requires_grad = False

    hidden_dim = getattr(encoder, "d_model", 512)
    adapters = nn.ModuleList()

    def make_hook(adapter):
        def hook(module, inputs, output):
            if isinstance(output, tuple):
                return (adapter(output[0]),) + output[1:]
            return adapter(output)
        return hook

    for layer in encoder.layers:
        adapter = BottleneckAdapter(hidden_dim, bottleneck)
        adapters.append(adapter)
        layer.register_forward_hook(make_hook(adapter))

    return adapters
