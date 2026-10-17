"""
moa_cas/adapters/router.py
-----------------------------
Stage 2 "Fluidity Learning": a frame-level router that mixes several
frozen Stage 1 bottleneck adapters (the "specialists") inside a frozen
Conformer encoder.

At every encoder layer l and every frame t:

    w_t   = softmax(Router_l(h_t) / tau)                 (K weights)
    out_t = sum_k w_t,k * A_k(h_t)
          = h_t + sum_k w_t,k * (A_k(h_t) - h_t)          (since sum_k w = 1)

which is the deck's "Dynamic Gating: sum w_t . A_k(h_t)". It generalises
the two-adapter frame gate of Kulkarni et al. 2023 ("Adapting the
Adapters for Code-Switching in Multilingual ASR", arXiv:2310.07423),
out = (1 - g_t) A_1 + g_t A_2 applied at every transformer block, from a
binary gate over 2 experts to a softmax over K.

Only the routers train. The backbone and every expert stay frozen, so
the router can only *choose between* behaviours Stage 1 already learned
(the deck's Stage 2 column: adapters "Locked - Already Trained").

Load balancing follows the Switch Transformer auxiliary loss (Fedus et
al. 2021): K * sum_k f_k * P_k, where f_k is the fraction of real frames
whose top-1 choice is expert k and P_k the mean router probability for k.
It equals 1.0 when routing is perfectly balanced and K when everything
collapses onto one expert; it stops the router from ignoring an expert
before it has learned what that expert is good for.

An expert can also be IdentityExpert — "no adaptation", i.e. the raw
frozen backbone. That keeps the base model's original behaviour
reachable per frame, which is the catastrophic-forgetting guard, and it
is the only second expert available on the Hindi backbone until an
English adapter exists (see docs/05_ROUTER.md).
"""

from pathlib import Path
from typing import List, Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from moa_cas.adapters.bottleneck import BottleneckAdapter


class IdentityExpert(nn.Module):
    """An expert that changes nothing: selects the frozen backbone itself."""

    def forward(self, x):
        return x


class FrameRouter(nn.Module):
    """
    LayerNorm -> depthwise temporal conv -> MLP -> K logits per frame.

    The depthwise conv gives each frame a little acoustic context (default
    5 frames = 200 ms at 25 frames/sec): a language switch is a property of
    a stretch of speech, not of one 40 ms frame. Padded frames are zeroed
    before the conv so they never leak into real ones.

    The output layer is zero-initialised, so an untrained router gives
    exactly uniform weights. Same idea as the zero-init adapter: training
    starts from a known, measurable state.
    """

    def __init__(self, dim: int, n_experts: int, hidden: int = 128,
                 context: int = 5, temperature: float = 1.0):
        super().__init__()
        if context % 2 == 0:
            raise ValueError("context must be odd so the conv window is centred on the frame")
        self.norm = nn.LayerNorm(dim)
        self.context = nn.Conv1d(dim, dim, kernel_size=context, padding=context // 2, groups=dim)
        self.mlp = nn.Sequential(nn.Linear(dim, hidden), nn.ReLU(), nn.Linear(hidden, n_experts))
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)
        self.temperature = temperature

    def forward(self, h: torch.Tensor, pad_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """h: (B, T, D); pad_mask: (B, T), True on padding. Returns (B, T, K) probabilities."""
        x = self.norm(h)
        if pad_mask is not None:
            x = x.masked_fill(pad_mask.unsqueeze(-1), 0.0)
        x = x + self.context(x.transpose(1, 2)).transpose(1, 2)
        return torch.softmax(self.mlp(x) / self.temperature, dim=-1)


def load_balancing_loss(probs: torch.Tensor, valid: Optional[torch.Tensor] = None) -> torch.Tensor:
    """
    Switch Transformer load-balancing loss over the real (non-padded) frames.

    probs: (B, T, K) router probabilities; valid: (B, T) bool, True on real frames.
    """
    k = probs.shape[-1]
    flat = probs.reshape(-1, k)
    if valid is not None:
        flat = flat[valid.reshape(-1)]
    if flat.shape[0] == 0:
        return probs.new_zeros(())
    f = F.one_hot(flat.argmax(dim=-1), k).float().mean(dim=0)  # hard dispatch fraction (no grad)
    p = flat.mean(dim=0)                                        # soft probability mass (grad flows)
    return k * (f * p).sum()


class MixtureOfAdapters(nn.Module):
    """
    K frozen expert stacks (one adapter per encoder layer each) plus one
    trainable FrameRouter per layer.

    routing="soft" mixes by the full weight vector (used for training).
    routing="hard" uses only the top-1 expert per frame, with a
    straight-through estimator if gradients are needed — the thresholded
    gate of Kulkarni et al.
    """

    def __init__(self, experts: Sequence[nn.ModuleList], dim: int, expert_names: Sequence[str] = None,
                 router_hidden: int = 128, router_context: int = 5,
                 temperature: float = 1.0, routing: str = "soft"):
        super().__init__()
        if len(experts) < 2:
            raise ValueError("A router needs at least two experts to choose between")
        n_layers = len(experts[0])
        if any(len(e) != n_layers for e in experts):
            raise ValueError("Every expert must provide one module per encoder layer")

        self.experts = nn.ModuleList(experts)
        self.expert_names = list(expert_names or [f"expert{i}" for i in range(len(experts))])
        self.routers = nn.ModuleList(
            FrameRouter(dim, len(experts), router_hidden, router_context, temperature)
            for _ in range(n_layers)
        )
        self.routing = routing
        self._probs: List[Optional[torch.Tensor]] = [None] * n_layers
        self._valid: List[Optional[torch.Tensor]] = [None] * n_layers

        for p in self.experts.parameters():
            p.requires_grad = False

    @property
    def n_experts(self) -> int:
        return len(self.experts)

    @property
    def n_layers(self) -> int:
        return len(self.routers)

    def train(self, mode: bool = True):
        # Experts are frozen Stage 1 artefacts: keep them in eval mode
        # whatever the caller does with the routers.
        super().train(mode)
        self.experts.eval()
        return self

    def mix(self, layer: int, h: torch.Tensor, pad_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        probs = self.routers[layer](h, pad_mask)
        self._probs[layer] = probs
        self._valid[layer] = None if pad_mask is None else ~pad_mask

        if self.routing == "hard":
            hard = F.one_hot(probs.argmax(dim=-1), self.n_experts).to(probs.dtype)
            weights = hard + probs - probs.detach()  # straight-through
        else:
            weights = probs

        out = h
        for k, expert in enumerate(self.experts):
            if isinstance(expert[layer], IdentityExpert):
                continue  # contributes h - h = 0
            out = out + weights[..., k:k + 1] * (expert[layer](h) - h)
        return out

    def load_balancing_loss(self) -> torch.Tensor:
        losses = [load_balancing_loss(p, v) for p, v in zip(self._probs, self._valid) if p is not None]
        if not losses:
            raise RuntimeError("No routing recorded yet — run a forward pass through the encoder first")
        return torch.stack(losses).mean()

    def routing_weights(self) -> torch.Tensor:
        """(B, T, K) router weights of the last forward pass, averaged over layers."""
        recorded = [p for p in self._probs if p is not None]
        return torch.stack(recorded).mean(dim=0)

    def router_state_dict(self) -> dict:
        return {k: v.detach().cpu() for k, v in self.routers.state_dict().items()}


def attach(encoder: nn.Module, moa: MixtureOfAdapters) -> MixtureOfAdapters:
    """
    Freezes the encoder and routes every layer's output through `moa`.
    NeMo's ConformerLayer receives pad_mask (True on padding) as a keyword
    argument; the hook reads it from there so routing statistics and the
    load-balancing loss only ever see real frames.
    """
    if len(encoder.layers) != moa.n_layers:
        raise ValueError(f"Encoder has {len(encoder.layers)} layers but the experts cover {moa.n_layers}")
    for p in encoder.parameters():
        p.requires_grad = False

    def make_hook(idx):
        def hook(module, args, kwargs, output):
            pad_mask = kwargs.get("pad_mask")
            if isinstance(output, tuple):
                return (moa.mix(idx, output[0], pad_mask),) + output[1:]
            return moa.mix(idx, output, pad_mask)
        return hook

    for idx, layer in enumerate(encoder.layers):
        layer.register_forward_hook(make_hook(idx), with_kwargs=True)
    return moa


def identity_expert(n_layers: int) -> nn.ModuleList:
    return nn.ModuleList(IdentityExpert() for _ in range(n_layers))


def load_stage1_expert(checkpoint: str, n_layers: int, dim: int, backbone_lang: str = None) -> nn.ModuleList:
    """
    Rebuilds a Stage 1 adapter stack from scripts/train_profile_adapter.py
    output. Refuses a checkpoint trained on a different backbone: an
    adapter only means anything on the encoder whose hidden states it was
    trained to reshape.
    """
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if backbone_lang and state.get("lang") not in (None, backbone_lang):
        raise ValueError(
            f"{Path(checkpoint).name} was trained on the '{state['lang']}' backbone, "
            f"not '{backbone_lang}'; experts must share one frozen backbone."
        )
    stack = nn.ModuleList(BottleneckAdapter(dim, state.get("bottleneck", 64)) for _ in range(n_layers))
    stack.load_state_dict(state["adapters"] if "adapters" in state else state)
    for p in stack.parameters():
        p.requires_grad = False
    return stack.eval()


def build_experts(specs: Sequence[str], n_layers: int, dim: int, backbone_lang: str = None):
    """
    specs like ["hi=checkpoints/hi_adapter.pt", "identity"] ->
    (list of expert stacks, list of names).
    """
    stacks, names = [], []
    for spec in specs:
        if spec == "identity":
            stacks.append(identity_expert(n_layers))
            names.append("identity")
        else:
            name, sep, path = spec.partition("=")
            if not sep:
                raise ValueError(f"Expert spec '{spec}' must be 'identity' or 'name=checkpoint.pt'")
            stacks.append(load_stage1_expert(path, n_layers, dim, backbone_lang))
            names.append(name)
    return stacks, names
