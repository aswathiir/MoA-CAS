"""
prepare_mucs.py
---------------
Converts the MUCS 2021 Kaldi-format test folder into a JSONL manifest
that benchmark.py can read.

MUCS Kaldi structure:
  test/
  ├── transcripts/
  │   ├── text        → utterance_id  transcription text
  │   ├── wav.scp     → recording_id  /path/to/audio.wav
  │   └── segments    → utterance_id  recording_id  start_sec  end_sec
  └── *.wav           → actual audio files

What this script does:
  1. Reads text, wav.scp, and segments
  2. Extracts each segment from its parent WAV file → saves to segments_wav/
  3. Writes mucs_test_manifest.json (one JSON line per utterance)

Usage
-----
    poetry run python prepare_mucs.py --test-dir path/to/test

Output
------
    mucs_test_manifest.json   ← pass this to --mucs-manifest in benchmark.py
    test/segments_wav/        ← extracted segment WAV files (16 kHz mono)
"""

import argparse
import json
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly
from math import gcd

TARGET_SR = 16_000


# ── Kaldi file parsers ───────────────────────────────────────────────────────

def read_text(path: Path) -> dict:
    """text file: each line is  <utt_id> <transcription>"""
    result = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(maxsplit=1)
            if len(parts) == 2:
                result[parts[0]] = parts[1]
    print(f"  text      : {len(result)} utterances")
    return result


def read_wav_scp(path: Path, test_dir: Path) -> dict:
    """
    wav.scp file: each line is  <recording_id> <wav_path_or_pipe>
    Resolves relative paths relative to test_dir.
    """
    result = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(maxsplit=1)
            if len(parts) == 2:
                rec_id, wav_path = parts
                # Resolve relative paths
                resolved = Path(wav_path)
                if not resolved.is_absolute():
                    resolved = test_dir / wav_path
                result[rec_id] = str(resolved)
    print(f"  wav.scp   : {len(result)} recordings")
    return result


def read_segments(path: Path) -> dict:
    """
    segments file: each line is  <utt_id> <recording_id> <start_sec> <end_sec>
    Returns {utt_id: (recording_id, start_sec, end_sec)}
    """
    result = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) == 4:
                utt_id, rec_id, start, end = parts
                result[utt_id] = (rec_id, float(start), float(end))
    print(f"  segments  : {len(result)} segments")
    return result


# ── Segment extraction ───────────────────────────────────────────────────────

def extract_segment(wav_path: str, start: float, end: float, out_path: Path):
    """
    Loads a slice [start, end] seconds from wav_path using soundfile,
    resamples to 16 kHz mono, and saves to out_path.
    soundfile bypasses torchaudio's TorchCodec dependency entirely.
    """
    wav_np, sr = sf.read(wav_path, always_2d=True)   # shape: (samples, channels)

    start_frame = int(start * sr)
    end_frame   = int(end   * sr)
    segment_np  = wav_np[start_frame:end_frame, :]    # slice rows (time)

    # Mono: average channels
    if segment_np.shape[1] > 1:
        segment_np = segment_np.mean(axis=1, keepdims=True)

    segment = segment_np.squeeze().astype(np.float32)   # (T,) numpy array

    if sr != TARGET_SR:
        g = gcd(TARGET_SR, sr)
        segment = resample_poly(segment, TARGET_SR // g, sr // g).astype(np.float32)

    sf.write(str(out_path), segment, TARGET_SR)


# ── Main ─────────────────────────────────────────────────────────────────────

def prepare(test_dir: Path):
    transcripts_dir = test_dir / "transcripts"
    segments_wav_dir = test_dir / "segments_wav"
    segments_wav_dir.mkdir(exist_ok=True)

    text_file    = transcripts_dir / "text"
    wav_scp_file = transcripts_dir / "wav.scp"
    segments_file = transcripts_dir / "segments"

    # Validate required files
    for f in [text_file, wav_scp_file]:
        if not f.exists():
            raise FileNotFoundError(f"Required file not found: {f}")

    print("Reading Kaldi files...")
    texts    = read_text(text_file)
    wav_scps = read_wav_scp(wav_scp_file, test_dir)
    segments = read_segments(segments_file) if segments_file.exists() else {}

    manifest = []
    skipped  = 0

    print(f"\nExtracting segments to {segments_wav_dir} ...")

    for utt_id, transcription in texts.items():

        if segments:
            # Has segments file — extract slice from parent recording
            if utt_id not in segments:
                skipped += 1
                continue
            rec_id, start, end = segments[utt_id]

            if rec_id not in wav_scps:
                skipped += 1
                continue

            out_wav = segments_wav_dir / f"{utt_id}.wav"

            if not out_wav.exists():
                try:
                    extract_segment(wav_scps[rec_id], start, end, out_wav)
                except Exception as e:
                    print(f"  [skip] {utt_id} — {e}")
                    skipped += 1
                    continue

            manifest.append({
                "audio_filepath": str(out_wav),
                "text":           transcription,
                "duration":       round(end - start, 3),
            })

        else:
            # No segments file — each utterance is a full WAV
            if utt_id not in wav_scps:
                skipped += 1
                continue
            manifest.append({
                "audio_filepath": wav_scps[utt_id],
                "text":           transcription,
                "duration":       0,
            })

    # Write manifest
    out_manifest = Path("mucs_test_manifest.json")
    with open(out_manifest, "w", encoding="utf-8") as f:
        for entry in manifest:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    print(f"\nDone.")
    print(f"  Written  : {len(manifest)} entries → {out_manifest}")
    if skipped:
        print(f"  Skipped  : {skipped} (missing audio or segment info)")
    print(f"\nNext step:")
    print(f"  poetry run python benchmark.py --dataset mucs --mucs-manifest {out_manifest} --limit 100")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert MUCS Kaldi test folder to JSONL manifest")
    parser.add_argument(
        "--test-dir",
        required=True,
        help="Path to the MUCS test folder (contains transcripts/ and *.wav files)"
    )
    args = parser.parse_args()
    prepare(Path(args.test_dir))