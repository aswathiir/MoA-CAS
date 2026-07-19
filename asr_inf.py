"""
IndicConformer ASR — single-file WebSocket server + test client

Run server : python asr_ws.py
Run client : python asr_ws.py --client --file audio.wav
             python asr_ws.py --client --file audio.wav --lang kn

Dependencies:
    pip install nemo_toolkit[asr] fastapi uvicorn[standard] websockets soundfile numpy
"""

import argparse
import asyncio
import json
import os
import sys
import tempfile

import numpy as np
import soundfile as sf
import torch

# ── Config ────────────────────────────────────────────────────────────────────
LANG        = "hi"                 # change or pass --lang
SAMPLE_RATE = 16000
HOST        = "0.0.0.0"
PORT        = 8000
WS_URI      = f"ws://localhost:{PORT}/ws"


# ══════════════════════════════════════════════════════════════════════════════
# SERVER
# ══════════════════════════════════════════════════════════════════════════════

def run_server(lang: str):
    import uvicorn
    import nemo.collections.asr as nemo_asr
    from fastapi import FastAPI, WebSocket

    model_name = f"ai4bharat/indicconformer_stt_{lang}_hybrid_rnnt_large"
    app        = FastAPI()
    asr        = None

    @app.on_event("startup")
    async def load():
        nonlocal asr
        print(f"Loading {model_name} …")
        asr = nemo_asr.models.ASRModel.from_pretrained(model_name=model_name)
        asr = asr.to("cuda" if torch.cuda.is_available() else "cpu")
        asr.eval()
        # CTC decoding is faster; swap to 'rnnt' for better accuracy
        if hasattr(asr, "change_decoding_strategy"):
            asr.change_decoding_strategy(decoder_type="ctc")
        print("Model ready — listening on", WS_URI)

    def _transcribe(audio: np.ndarray) -> str:
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            sf.write(f.name, audio.astype(np.float32), SAMPLE_RATE)
            path = f.name
        try:
            with torch.no_grad():
                out = asr.transcribe([path], batch_size=1)
            r = out[0]
            return r if isinstance(r, str) else getattr(r, "text", str(r))
        finally:
            os.unlink(path)

    @app.websocket("/ws")
    async def ws_endpoint(websocket: WebSocket):
        await websocket.accept()
        print("Client connected")
        try:
            while True:
                # Expect raw int16 PCM bytes at 16 kHz mono
                raw   = await websocket.receive_bytes()
                audio = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
                loop  = asyncio.get_event_loop()
                text  = await loop.run_in_executor(None, _transcribe, audio)
                await websocket.send_text(json.dumps({"text": text}))
        except Exception:
            print("Client disconnected")

    uvicorn.run(app, host=HOST, port=PORT)


# ══════════════════════════════════════════════════════════════════════════════
# CLIENT
# ══════════════════════════════════════════════════════════════════════════════

async def run_client(filepath: str, uri: str):
    import websockets

    # Load and validate audio
    audio, sr = sf.read(filepath, dtype="int16", always_2d=False)
    if audio.ndim > 1:
        audio = audio[:, 0]
    if sr != SAMPLE_RATE:
        sys.exit(f"Need {SAMPLE_RATE} Hz mono. Got {sr} Hz.\n"
                 f"Convert: ffmpeg -i {filepath} -ar 16000 -ac 1 out.wav")

    print(f"Sending {filepath} ({len(audio)/SAMPLE_RATE:.1f}s) → {uri}")

    async with websockets.connect(uri) as ws:
        await ws.send(audio.tobytes())
        reply = json.loads(await ws.recv())
        print(f"\nTranscription: {reply['text']}")


# ══════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--client", action="store_true", help="Run WebSocket test client")
    parser.add_argument("--file",   default=None,        help="WAV file to transcribe (client mode)")
    parser.add_argument("--lang",   default=LANG,        help="Language code, e.g. hi kn te ta")
    parser.add_argument("--uri",    default=WS_URI,      help="WebSocket URI (client mode)")
    args = parser.parse_args()

    if args.client:
        if not args.file:
            sys.exit("--file is required in client mode")
        asyncio.run(run_client(args.file, args.uri))
    else:
        run_server(args.lang)