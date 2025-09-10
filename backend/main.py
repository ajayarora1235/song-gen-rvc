# app/routers/audio_pipeline.py
import asyncio
import os
import math
from contextlib import asynccontextmanager
from typing import AsyncIterator, Optional, Literal

import uvicorn
from fastapi import APIRouter, FastAPI, Request, HTTPException, Query
from fastapi.responses import StreamingResponse

import numpy as np
import aiofiles
import httpx
import subprocess
import sys
import tempfile

# Optional: ElevenLabs Music streaming (server-pulled source)
# pip install elevenlabs python-dotenv
try:
    from elevenlabs.client import ElevenLabs
except Exception:
    ElevenLabs = None

router = APIRouter(prefix="/audio", tags=["audio"])

# ---------------------------
# Audio helpers (ffmpeg IPC)
# ---------------------------

class FFmpegDecoder:
    """
    Decode arbitrary compressed input (mp3/aac/wav/etc) to PCM float32 mono/stereo via ffmpeg pipe.
    Produces fixed-size frames in samples per channel (frame_hop).
    """
    def __init__(self, sample_rate=44100, channels=2, frame_hop=2048):
        self.sr = sample_rate
        self.channels = channels
        self.frame_hop = frame_hop
        self.proc = None
        self.buf = b""

    def start(self):
        # -i pipe: reads compressed bytes from stdin
        # -f f32le -ac <ch> -ar <sr> : output raw PCM float32
        args = [
            "ffmpeg", "-hide_banner", "-loglevel", "error",
            "-i", "pipe:0",
            "-f", "f32le", "-acodec", "pcm_f32le",
            "-ac", str(self.channels), "-ar", str(self.sr),
            "pipe:1"
        ]
        self.proc = subprocess.Popen(args, stdin=subprocess.PIPE, stdout=subprocess.PIPE)

    async def feed(self, chunk: bytes):
        if self.proc is None or self.proc.stdin is None:
            raise RuntimeError("Decoder not started")
        # Write compressed chunk
        self.proc.stdin.write(chunk)
        self.proc.stdin.flush()

    def close_stdin(self):
        if self.proc and self.proc.stdin:
            try:
                self.proc.stdin.close()
            except Exception:
                pass

    async def read_frames(self) -> AsyncIterator[np.ndarray]:
        """
        Async generator yielding float32 numpy frames of shape (samples, channels)
        """
        if self.proc is None or self.proc.stdout is None:
            raise RuntimeError("Decoder not started")

        bytes_per_sample = 4  # float32
        frame_bytes = self.frame_hop * self.channels * bytes_per_sample

        loop = asyncio.get_running_loop()
        while True:
            # readexactly-like behavior on blocking pipe without trio
            chunk = await loop.run_in_executor(None, self.proc.stdout.read, frame_bytes)
            if not chunk:
                break
            if len(chunk) < frame_bytes:
                # tail frame; pad
                chunk = chunk + b"\x00" * (frame_bytes - len(chunk))
            arr = np.frombuffer(chunk, dtype=np.float32).reshape(-1, self.channels)
            yield arr

    def terminate(self):
        if self.proc:
            try:
                self.proc.terminate()
            except Exception:
                pass


class FFmpegEncoder:
    """
    Encode PCM float32 frames to compressed output (opus/mp3) via ffmpeg pipe, yielding bytes.
    """
    def __init__(self, codec: Literal["opus","mp3"]="opus", sample_rate=44100, channels=2, bitrate="128k"):
        self.codec = codec
        self.sr = sample_rate
        self.channels = channels
        self.bitrate = bitrate
        self.proc = None

    def start(self):
        if self.codec == "opus":
            acodec = "libopus"
            container = "ogg"
        elif self.codec == "mp3":
            acodec = "libmp3lame"
            container = "mp3"
        else:
            raise ValueError("Unsupported codec")

        self.container = container
        args = [
            "ffmpeg", "-hide_banner", "-loglevel", "error",
            "-f", "f32le", "-ac", str(self.channels), "-ar", str(self.sr),
            "-i", "pipe:0",
            "-acodec", acodec, "-b:a", self.bitrate,
            "-f", container, "pipe:1"
        ]
        self.proc = subprocess.Popen(args, stdin=subprocess.PIPE, stdout=subprocess.PIPE)

    async def write_pcm(self, frames: np.ndarray):
        if self.proc is None or self.proc.stdin is None:
            raise RuntimeError("Encoder not started")
        # frames: (samples, channels), float32
        self.proc.stdin.write(frames.astype(np.float32).tobytes())
        self.proc.stdin.flush()

    def close_stdin(self):
        if self.proc and self.proc.stdin:
            try:
                self.proc.stdin.close()
            except Exception:
                pass

    async def stream_encoded(self) -> AsyncIterator[bytes]:
        if self.proc is None or self.proc.stdout is None:
            raise RuntimeError("Encoder not started")
        loop = asyncio.get_running_loop()
        while True:
            out = await loop.run_in_executor(None, self.proc.stdout.read, 4096)
            if not out:
                break
            yield out

    def terminate(self):
        if self.proc:
            try:
                self.proc.terminate()
            except Exception:
                pass


# ----------------------------------
# Stubs: your separator + RVC engine
# ----------------------------------

class StreamingSeparator:
    """
    Replace implementation with your streaming Demucs/MDX (overlap-add).
    For each input frame, return (vocals, instrumental) aligned to same shape.
    """
    def __init__(self, sr=44100, channels=2):
        self.sr = sr
        self.channels = channels
        # init model weights / warm-up here if needed

    async def stream(self, frame: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        # TODO: swap with real separator; placeholder = naive L/R fake split
        # Return zeros to illustrate the contract
        zeros = np.zeros_like(frame)
        return zeros, frame  # (vocals=0, instrumental=original)


class RVCProcessor:
    """
    Bridge to your existing RVC. Implement process_chunk() to run VC on short windows.
    """
    def __init__(self, sr=44100):
        self.sr = sr
        # init your RVC model/session

    async def process_chunk(self, vocals_chunk: np.ndarray) -> np.ndarray:
        # TODO: call into your pipeline (sync or async)
        # For demo, return passthrough
        return vocals_chunk


# ---------------------------
# Mixing utility
# ---------------------------
def mix(vocals: np.ndarray, instrumental: np.ndarray, gain_v=1.0, gain_i=1.0) -> np.ndarray:
    y = gain_v * vocals + gain_i * instrumental
    # soft clip
    return np.tanh(y)


# ============================================================
# Route A: Client streams audio to us -> we process -> we stream back
# ============================================================
@router.post("/process/stream")
async def process_streaming_audio(
    request: Request,
    input_codec: Optional[str] = Query(default=None, description="Hint: mp3, wav, opus. If None, ffmpeg will auto-detect."),
    out_codec: Literal["opus","mp3"] = Query(default="opus"),
    sr: int = Query(default=44100),
    ch: int = Query(default=2),
    hop: int = Query(default=2048),
):
    """
    Client sends compressed audio via chunked transfer (POST body).
    We decode -> separate -> RVC on vocals -> mix -> encode -> stream back.
    Content-Type of response matches selected 'out_codec' container (ogg for opus, audio/mpeg for mp3).
    """

    decoder = FFmpegDecoder(sample_rate=sr, channels=ch, frame_hop=hop)
    encoder = FFmpegEncoder(codec=out_codec, sample_rate=sr, channels=ch, bitrate="160k")
    separator = StreamingSeparator(sr=sr, channels=ch)
    rvc = RVCProcessor(sr=sr)

    decoder.start()
    encoder.start()

    async def reader_pump():
        async for chunk in request.stream():
            await decoder.feed(chunk)
        decoder.close_stdin()

    async def pipeline() -> AsyncIterator[bytes]:
        # Fan-out: run decode->separate->RVC->mix->encode in streaming fashion
        pump = asyncio.create_task(reader_pump())
        try:
            async for frame in decoder.read_frames():
                vocals, instrumental = await separator.stream(frame)
                processed_vocals = await rvc.process_chunk(vocals)
                mixed = mix(processed_vocals, instrumental)
                await encoder.write_pcm(mixed)
                # drain some encoded bytes early to minimize latency
                async for enc_chunk in encoder.stream_encoded():
                    if enc_chunk:
                        yield enc_chunk
            # signal encoder end
            encoder.close_stdin()
            async for tail in encoder.stream_encoded():
                if tail:
                    yield tail
        finally:
            try:
                pump.cancel()
            except Exception:
                pass
            decoder.terminate()
            encoder.terminate()

    media_type = "audio/ogg" if out_codec == "opus" else "audio/mpeg"
    return StreamingResponse(pipeline(), media_type=media_type)


# ==================================================================================
# Route B: Server pulls ElevenLabs Music stream -> we process -> we stream back
# ==================================================================================
@router.get("/process/eleven-music")
async def process_eleven_music(
    prompt: str,
    music_length_ms: int = 10000,
    out_codec: Literal["opus","mp3"] = Query(default="opus"),
    sr: int = 44100,
    ch: int = 2,
    hop: int = 2048,
):
    """
    Uses ElevenLabs Music streaming to obtain audio chunks, then runs the same pipeline.
    Requires ELEVENLABS_API_KEY and the `elevenlabs` Python SDK.
    """

    if ElevenLabs is None:
        raise HTTPException(500, "elevenlabs SDK not installed. pip install elevenlabs")

    eleven = ElevenLabs(api_key=os.getenv("ELEVENLABS_API_KEY"))
    # Per docs: elevenlabs.music.stream(...) yields audio bytes in chunks.  [oai_citation:5‡ElevenLabs](https://elevenlabs.io/docs/cookbooks/music/streaming)
    music_stream = eleven.music.stream(prompt=prompt, music_length_ms=music_length_ms)

    decoder = FFmpegDecoder(sample_rate=sr, channels=ch, frame_hop=hop)
    encoder = FFmpegEncoder(codec=out_codec, sample_rate=sr, channels=ch, bitrate="160k")
    separator = StreamingSeparator(sr=sr, channels=ch)
    rvc = RVCProcessor(sr=sr)

    decoder.start()
    encoder.start()

    async def reader_pump():
        loop = asyncio.get_running_loop()
        for chunk in music_stream:
            if chunk:
                await loop.run_in_executor(None, decoder.feed, chunk)
        decoder.close_stdin()

    async def pipeline() -> AsyncIterator[bytes]:
        pump = asyncio.create_task(reader_pump())
        try:
            async for frame in decoder.read_frames():
                vocals, instrumental = await separator.stream(frame)
                processed_vocals = await rvc.process_chunk(vocals)
                mixed = mix(processed_vocals, instrumental)
                await encoder.write_pcm(mixed)
                async for enc_chunk in encoder.stream_encoded():
                    if enc_chunk:
                        yield enc_chunk
            encoder.close_stdin()
            async for tail in encoder.stream_encoded():
                if tail:
                    yield tail
        finally:
            try:
                pump.cancel()
            except Exception:
                pass
            decoder.terminate()
            encoder.terminate()

    media_type = "audio/ogg" if out_codec == "opus" else "audio/mpeg"
    return StreamingResponse(pipeline(), media_type=media_type)