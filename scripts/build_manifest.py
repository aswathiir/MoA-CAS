"""
scripts/build_manifest.py
-----------------------------
CLI entry point for the preprocessing pipeline: turns a raw corpus into a
ManifestRow-schema JSONL manifest, driven by configs/datasets.yaml.

Usage
-----
    poetry run python scripts/build_manifest.py --list
    poetry run python scripts/build_manifest.py --config mucs_hi_en_test
"""

import argparse
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from moa_cas.pipeline.build import build_mucs_manifest, build_indicvoices_manifest
from moa_cas.pipeline.manifest import write_jsonl
from moa_cas.stats import summarize

CONFIG_PATH = Path(__file__).resolve().parent.parent / "configs" / "datasets.yaml"


def load_registry() -> dict:
    with open(CONFIG_PATH, encoding="utf-8") as f:
        return yaml.safe_load(f)


def run(name: str):
    registry = load_registry()
    if name not in registry:
        raise SystemExit(f"Unknown config '{name}'. Available: {sorted(registry)}")

    cfg = registry[name]
    adapter = cfg["adapter"]

    if adapter == "mucs":
        raw_dir = Path(cfg["raw_dir"])
        if not raw_dir.exists():
            raise SystemExit(
                f"\nRaw data not found: {raw_dir}\n"
                f"Download the MUCS {cfg['lang_pair']} package from https://www.openslr.org/104/ "
                f"and extract it there (Kaldi layout: transcripts/ + *.wav) before running this target.\n"
            )
        rows = build_mucs_manifest(
            raw_dir=raw_dir,
            out_wav_dir=Path(cfg["out_wav_dir"]),
            lang_pair=cfg["lang_pair"],
            split=cfg["split"],
        )
    elif adapter == "indicvoices":
        rows = build_indicvoices_manifest(
            lang=cfg["lang"],
            split=cfg["split"],
            limit=cfg.get("limit"),
        )
    else:
        raise SystemExit(f"Unknown adapter '{adapter}' in config '{name}'")

    if not rows:
        raise SystemExit(f"No rows produced for '{name}' — check the raw data and filters.")

    summarize(rows, label=name)
    write_jsonl(rows, Path(cfg["out_manifest"]))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build a preprocessing-pipeline manifest from configs/datasets.yaml")
    parser.add_argument("--config", help="Registry entry name (see configs/datasets.yaml)")
    parser.add_argument("--list", action="store_true", help="List available config names and exit")
    args = parser.parse_args()

    if args.list:
        for name in load_registry():
            print(name)
    elif args.config:
        run(args.config)
    else:
        parser.error("Pass --config <name> or --list")
