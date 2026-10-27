"""
scripts/export_static_demo.py
--------------------------------
Exports the demo as a static site, for hosts that can't run the model
(e.g. Vercel). Every sample is transcribed once here by the same Demo
class the live server uses, and the results are written as JSON next to
the page and its audio:

    demo_static/
      index.html               demo/index.html with window.MOA_STATIC = true
      data/info.json           model metadata and the 100-utterance benchmark
      data/samples.json        the sample list
      data/results/<id>.json   base / adapter / router output for one sample
      audio/<id>.wav

The hosted copy can only show these precomputed samples. Uploading or
recording your own audio needs the live server (scripts/demo_server.py).

Usage
-----
    python scripts/export_static_demo.py --out demo_static
    cd demo_static && npx vercel deploy --prod
"""

import argparse
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

from demo_server import Demo, build_info, build_samples, transcribe_sample


def main(args):
    out = ROOT / args.out
    # Rebuild the site files but keep the Vercel project link (.vercel/,
    # .env.local, .gitignore) so redeploys go to the same project.
    keep = {".vercel", ".env.local", ".gitignore"}
    if out.exists():
        for child in out.iterdir():
            if child.name not in keep:
                shutil.rmtree(child) if child.is_dir() else child.unlink()
    (out / "data" / "results").mkdir(parents=True)
    (out / "audio").mkdir()

    demo = Demo(str(ROOT / args.router_checkpoint), args.manifest, args.n_samples, args.device)

    def dump(path, obj):
        path.write_text(json.dumps(obj, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")

    dump(out / "data" / "info.json", build_info(demo))
    dump(out / "data" / "samples.json", build_samples(demo))
    for i, utt_id in enumerate(demo.samples, 1):
        dump(out / "data" / "results" / f"{utt_id}.json", transcribe_sample(demo, utt_id))
        shutil.copyfile(ROOT / demo.samples[utt_id]["audio_filepath"], out / "audio" / f"{utt_id}.wav")
        if i % 20 == 0:
            print(f"  {i}/{len(demo.samples)} exported", flush=True)

    page = (ROOT / "demo" / "index.html").read_text(encoding="utf-8")
    marker = "<script>\nconst $ ="
    if marker not in page:
        raise RuntimeError("demo/index.html changed shape; can't inject the static flag")
    page = page.replace(marker, "<script>window.MOA_STATIC = true;</script>\n" + marker, 1)
    (out / "index.html").write_text(page, encoding="utf-8")

    size = sum(f.stat().st_size for f in out.rglob("*") if f.is_file())
    print(f"Static demo written to {out} ({len(demo.samples)} samples, {size / 1e6:.1f} MB)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export the MoA-CAS demo as a static site")
    parser.add_argument("--out", default="demo_static")
    parser.add_argument("--router-checkpoint", default="checkpoints/router_hi_id.pt")
    parser.add_argument("--manifest", default="data/manifests/mucs_hi-en_test.jsonl")
    parser.add_argument("--n-samples", type=int, default=100)
    parser.add_argument("--device", default="cpu")
    main(parser.parse_args())
