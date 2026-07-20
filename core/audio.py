"""
core/audio.py
-------------
Audio preprocessing: decode → resample to 16 kHz → mono (1, T) tensor.

Handles two sources:
  hf_bytes   : Audio(decode=False) dict from HuggingFace {"bytes": ..., "path": ...}
  local_path : string path to a local WAV file (MUCS)

soundfile is used for decoding in both cases — no torchcodec needed.
"""

import io
import torch
import torchaudio
import soundfile as sf

TARGET_SR = 16_000


def prepare_audio(sample: dict, device: str) -> torch.Tensor:
    """
    Dispatches to the right decoder based on sample["audio_type"].
    Returns a mono float32 tensor at 16 kHz on the target device.
    """
    if sample["audio_type"] == "hf_bytes":
        return _from_hf(sample["audio_data"], device)
    else:
        return _from_path(sample["audio_data"], device)


def _from_hf(audio_dict: dict, device: str) -> torch.Tensor:
    """Decode from HuggingFace Audio(decode=False) — bytes in memory."""
    if audio_dict.get("bytes"):
        wav_np, sr = sf.read(io.BytesIO(audio_dict["bytes"]))
    else:
        wav_np, sr = sf.read(audio_dict["path"])
    return _to_tensor(wav_np, sr, device)


def _from_path(path: str, device: str) -> torch.Tensor:
    """Decode from a local WAV file path (MUCS)."""
    wav_np, sr = sf.read(path)
    return _to_tensor(wav_np, sr, device)


def _to_tensor(wav_np, sr: int, device: str) -> torch.Tensor:
    """Shared postprocessing: numpy → tensor → resample → mono → device."""
    wav = torch.tensor(wav_np, dtype=torch.float32)

    if wav.ndim == 1:
        wav = wav.unsqueeze(0)                        # (T,) → (1, T)
    elif wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)            # stereo → mono

    if sr != TARGET_SR:
        wav = torchaudio.functional.resample(wav, sr, TARGET_SR)

    return wav.to(device)