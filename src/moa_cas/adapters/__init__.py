"""
moa_cas.adapters
------------------
No eager submodule imports — bottleneck.py is plain torch, but
nemo_backbone.py pulls in nemo_toolkit (heavy, and not everyone touching
this package needs it). Import what you need directly:

    from moa_cas.adapters.bottleneck import BottleneckAdapter, insert_into_encoder
    from moa_cas.adapters.nemo_backbone import load_backbone
    from moa_cas.adapters.language_mask import build as build_language_mask
"""
