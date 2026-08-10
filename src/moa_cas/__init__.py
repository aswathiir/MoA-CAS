"""
moa_cas
--------
Deliberately no eager submodule imports here. `preprocessing`, `pipeline`,
and `datasets.mucs` are pure-Python and have no business pulling in
torch/transformers just because someone imported the top-level package —
only scripts/benchmark.py and scripts/download_model.py need the
model/audio_io/metrics/reporting modules, and they import those directly.
"""

__version__ = "0.1.0"
