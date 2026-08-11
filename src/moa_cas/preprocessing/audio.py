"""
moa_cas/preprocessing/audio.py
---------------------------------
Offline audio preprocessing for Kaldi-format corpora (MUCS): parses the
Kaldi triple (text, wav.scp, segments), slices each utterance out of its
parent recording, and resamples to 16 kHz mono.

Kaldi structure expected under `raw_dir`:
  raw_dir/
  ├── transcripts/
  │   ├── text        -> utterance_id  transcription text
  │   ├── wav.scp      -> recording_id  /path/to/audio.wav
  │   ├── segments      -> utterance_id  recording_id  start_sec  end_sec
  │   └── utt2spk        -> utterance_id  speaker_id  (optional)
  └── *.wav             -> recordings referenced by wav.scp

Used by pipeline/build.py to turn a raw MUCS Kaldi split (any language pair —
just point raw_dir at the right download) into per-utterance 16 kHz WAV
files ready for manifest building. This generalizes the logic that used to
live in the standalone prepare_mucs.py script.
"""

import io
import tarfile
from math import gcd
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

TARGET_SR = 16_000


class RawUtterance:
    """One parsed, audio-extracted utterance ahead of language tagging."""

    __slots__ = ("utt_id", "audio_filepath", "text", "duration", "speaker_id")

    def __init__(self, utt_id, audio_filepath, text, duration, speaker_id=None):
        self.utt_id = utt_id
        self.audio_filepath = audio_filepath
        self.text = text
        self.duration = duration
        self.speaker_id = speaker_id


# ── Kaldi file parsers ────────────────────────────────────────────────────

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
    return result


def read_wav_scp(path: Path, raw_dir: Path) -> dict:
    """wav.scp: each line is  <recording_id> <wav_path>. Relative paths resolve against raw_dir."""
    result = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(maxsplit=1)
            if len(parts) == 2:
                rec_id, wav_path = parts
                resolved = Path(wav_path)
                if not resolved.is_absolute():
                    resolved = raw_dir / wav_path
                result[rec_id] = str(resolved)
    return result


def read_segments(path: Path) -> dict:
    """segments: each line is  <utt_id> <recording_id> <start_sec> <end_sec>"""
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
    return result


def read_utt2spk(path: Path) -> dict:
    """utt2spk: each line is  <utt_id> <speaker_id>. Returns {} if the file is absent."""
    result = {}
    if not path.exists():
        return result
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(maxsplit=1)
            if len(parts) == 2:
                result[parts[0]] = parts[1]
    return result


# ── Segment extraction ───────────────────────────────────────────────────

def _slice_and_resample(wav_np, sr: int, start: float, end: float, out_path: Path) -> float:
    """Shared core: slice [start, end], downmix to mono, resample to 16 kHz, write out_path."""
    start_frame = int(start * sr)
    end_frame   = int(end * sr)
    segment_np  = wav_np[start_frame:end_frame, :]

    if segment_np.shape[1] > 1:
        segment_np = segment_np.mean(axis=1, keepdims=True)

    segment = segment_np.squeeze().astype(np.float32)

    if sr != TARGET_SR:
        g = gcd(TARGET_SR, sr)
        segment = resample_poly(segment, TARGET_SR // g, sr // g).astype(np.float32)

    sf.write(str(out_path), segment, TARGET_SR)
    return len(segment) / TARGET_SR


def extract_segment(wav_path: str, start: float, end: float, out_path: Path) -> float:
    """
    Slices [start, end] seconds out of wav_path, downmixes to mono, resamples
    to 16 kHz, and writes out_path. Returns the segment duration in seconds.
    soundfile bypasses torchaudio's TorchCodec dependency entirely.
    """
    wav_np, sr = sf.read(wav_path, always_2d=True)   # (samples, channels)
    return _slice_and_resample(wav_np, sr, start, end, out_path)


# ── Split extraction ─────────────────────────────────────────────────────

def extract_kaldi_split(raw_dir: Path, out_wav_dir: Path) -> list:
    """
    Parses a Kaldi-format split under raw_dir and extracts every utterance
    to a 16 kHz mono WAV file under out_wav_dir.

    Returns a list of RawUtterance, one per successfully extracted utterance.
    Utterances already extracted (out_path exists) are reused, not re-cut.
    """
    transcripts_dir = raw_dir / "transcripts"
    out_wav_dir.mkdir(parents=True, exist_ok=True)

    text_file     = transcripts_dir / "text"
    wav_scp_file  = transcripts_dir / "wav.scp"
    segments_file = transcripts_dir / "segments"
    utt2spk_file  = transcripts_dir / "utt2spk"

    for f in (text_file, wav_scp_file):
        if not f.exists():
            raise FileNotFoundError(f"Required Kaldi file not found: {f}")

    texts    = read_text(text_file)
    wav_scps = read_wav_scp(wav_scp_file, raw_dir)
    segments = read_segments(segments_file) if segments_file.exists() else {}
    utt2spk  = read_utt2spk(utt2spk_file)

    print(f"  text      : {len(texts)} utterances")
    print(f"  wav.scp   : {len(wav_scps)} recordings")
    if segments:
        print(f"  segments  : {len(segments)} segments")

    utterances = []
    skipped = 0

    for utt_id, transcription in texts.items():
        if segments:
            if utt_id not in segments:
                skipped += 1
                continue
            rec_id, start, end = segments[utt_id]
            if rec_id not in wav_scps:
                skipped += 1
                continue

            out_wav = out_wav_dir / f"{utt_id}.wav"
            if out_wav.exists():
                duration = round(end - start, 3)
            else:
                try:
                    duration = round(extract_segment(wav_scps[rec_id], start, end, out_wav), 3)
                except Exception as e:
                    print(f"  [skip] {utt_id} — {e}")
                    skipped += 1
                    continue

            utterances.append(RawUtterance(
                utt_id=utt_id,
                audio_filepath=str(out_wav),
                text=transcription,
                duration=duration,
                speaker_id=utt2spk.get(utt_id),
            ))
        else:
            if utt_id not in wav_scps:
                skipped += 1
                continue
            utterances.append(RawUtterance(
                utt_id=utt_id,
                audio_filepath=wav_scps[utt_id],
                text=transcription,
                duration=0.0,
                speaker_id=utt2spk.get(utt_id),
            ))

    print(f"  Extracted : {len(utterances)} utterances ({skipped} skipped)")
    return utterances


def extract_kaldi_split_from_tar(tar_path: Path, member_root: str, raw_dir: Path, out_wav_dir: Path) -> list:
    """
    Like extract_kaldi_split, but for a raw split too large to extract to
    disk in full (e.g. a 90-hour train split where the extracted audio alone
    would be ~18GB). Reads each recording's audio directly out of the .tar.gz
    into memory, slices every segment belonging to it, writes just those
    small 16kHz mono segments to out_wav_dir, then discards the recording —
    the full raw recording set is never held on disk at once.

    A single sequential pass through the archive is used deliberately:
    tarfile can't efficiently random-access a gzip-compressed stream, so
    looking up 500+ members individually would re-scan from the start each
    time. One forward pass costs the same as a plain `tar xzf` — it's the
    disk footprint that's different, not the read cost.

    `raw_dir` must already contain `transcripts/` (extract just that small
    subfolder ahead of time — it's tiny compared to the audio).
    """
    transcripts_dir = raw_dir / "transcripts"
    out_wav_dir.mkdir(parents=True, exist_ok=True)

    texts   = read_text(transcripts_dir / "text")
    segments = read_segments(transcripts_dir / "segments")
    utt2spk = read_utt2spk(transcripts_dir / "utt2spk")

    # Raw (unresolved) recording_id -> filename, to match tar member names —
    # read_wav_scp() resolves against a local raw_dir we're deliberately not
    # populating with audio, so we parse wav.scp directly here instead.
    filename_by_rec = {}
    with open(transcripts_dir / "wav.scp", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(maxsplit=1)
            if len(parts) == 2:
                filename_by_rec[parts[0]] = parts[1]

    by_recording = {}
    for utt_id, (rec_id, start, end) in segments.items():
        by_recording.setdefault(rec_id, []).append((utt_id, start, end))

    rec_id_by_filename = {fname: rec_id for rec_id, fname in filename_by_rec.items()}

    print(f"  text      : {len(texts)} utterances")
    print(f"  segments  : {len(segments)} segments across {len(by_recording)} recordings")

    utterances = []
    skipped = 0
    seen_recordings = 0

    with tarfile.open(tar_path, "r:gz") as tf:
        for member in tf:
            if not member.isfile():
                continue
            filename = member.name.rsplit("/", 1)[-1]
            rec_id = rec_id_by_filename.get(filename)
            if rec_id is None or rec_id not in by_recording:
                continue

            seen_recordings += 1
            fobj = tf.extractfile(member)
            wav_np, sr = sf.read(io.BytesIO(fobj.read()), always_2d=True)

            for utt_id, start, end in by_recording[rec_id]:
                if utt_id not in texts:
                    skipped += 1
                    continue

                out_wav = out_wav_dir / f"{utt_id}.wav"
                if out_wav.exists():
                    duration = round(end - start, 3)
                else:
                    try:
                        duration = round(_slice_and_resample(wav_np, sr, start, end, out_wav), 3)
                    except Exception as e:
                        print(f"  [skip] {utt_id} — {e}")
                        skipped += 1
                        continue

                utterances.append(RawUtterance(
                    utt_id=utt_id,
                    audio_filepath=str(out_wav),
                    text=texts[utt_id],
                    duration=duration,
                    speaker_id=utt2spk.get(utt_id),
                ))

            if seen_recordings % 50 == 0:
                print(f"  ...{seen_recordings}/{len(by_recording)} recordings processed, "
                      f"{len(utterances)} utterances so far")

    missing_recordings = len(by_recording) - seen_recordings
    if missing_recordings:
        print(f"  [warn] {missing_recordings} recordings referenced in segments were never found in the tar")

    print(f"  Extracted : {len(utterances)} utterances from {seen_recordings} recordings ({skipped} skipped)")
    return utterances
