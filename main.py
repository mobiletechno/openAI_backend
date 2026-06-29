"""
SalesIQ Backend Server — WebSocket Real-Time Edition v3.0
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Supports three binary audio input modes for minimum latency:

  MODE 1 — Raw PCM (FASTEST ⚡)
    • No encoding/decoding overhead
    • Client sends raw int16 bytes (16kHz, mono)
    • Server reads directly into numpy array
    • Set "format": "pcm" in START_SESSION

  MODE 2 — OGG/Opus or WebM (SMALLEST SIZE ✅)
    • Best for mobile networks (13x smaller than WAV)
    • Designed for real-time voice (WhatsApp/Discord codec)
    • Auto-detected via magic bytes OR set "format": "ogg"
    • Server converts via pydub (fast, ~5ms)

  MODE 3 — Any File Format (WAV, MP3, M4A, FLAC)
    • Auto-detected via magic bytes
    • Server converts via pydub
    • Slowest due to larger payload size

Auto-Detection (magic bytes):
  Raw PCM  → no header → detected by format hint or fallback
  RIFF     → WAV
  OggS     → OGG/Opus
  0x1A45   → WebM/Opus
  ID3/FF   → MP3
  ftyp     → M4A/AAC
  fLaC     → FLAC

WebSocket Protocol:
  Client → Server:
    { "type": "START_SESSION", "language": "en", "format": "pcm" }
    <binary audio bytes>
    { "type": "PING" }
    { "type": "END_SESSION" }

  Server → Client:
    { "type": "SESSION_STARTED", "session_id": "...", "format_mode": "pcm" }
    { "type": "PROCESSING", "chunk_id": "...", "format": "pcm", "size": 160000 }
    { "type": "SPEECH", "speaker": "You", "text": "...", "start": 0.0, "end": 2.5 }
    { "type": "SILENCE", "chunk_id": "..." }
    { "type": "PONG" }
    { "type": "SESSION_ENDED", "stats": {...} }
    { "type": "ERROR", "message": "..." }

REST Endpoints (Postman testing):
  GET  /           → Health check
  GET  /status     → Server model status
  POST /transcribe → Audio file → plain transcript
  POST /diarize    → Audio file → speaker-labeled transcript
"""

import os
import io
import uuid
import struct
import asyncio
import logging
import tempfile
import json as _json
from typing import Optional, Literal

import numpy as np
import torch
import torchaudio
import soundfile as sf
from fastapi import FastAPI, File, UploadFile, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydub import AudioSegment
from openai import AsyncOpenAI, OpenAI
from pyannote.audio import Pipeline
from dotenv import load_dotenv

# ── Environment ───────────────────────────────────────────────────────────────
load_dotenv()
OPENAI_API_KEY    = os.getenv("OPENAI_API_KEY", "")
HUGGINGFACE_TOKEN = os.getenv("HUGGINGFACE_TOKEN", "")

# ── Constants ─────────────────────────────────────────────────────────────────
PCM_SAMPLE_RATE   = 16000    # Hz — must match mobile recorder
PCM_CHANNELS      = 1        # mono
PCM_DTYPE         = np.int16 # 16-bit signed

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s │ %(levelname)s │ %(message)s",
)
log = logging.getLogger("salesiq")

def get_val(obj, key, default=None):
    if obj is None:
        return default
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


# ── FastAPI ───────────────────────────────────────────────────────────────────
app = FastAPI(
    title="SalesIQ Backend API",
    description="Real-time WebSocket speech transcription + speaker diarization. Supports PCM, OGG, WAV, MP3.",
    version="3.0.0",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Create static directory if it doesn't exist
os.makedirs("static", exist_ok=True)

# Mount static directory to serve files at /static
app.mount("/static", StaticFiles(directory="static"), name="static")

# ── Global model instances ────────────────────────────────────────────────────
sync_openai:      Optional[OpenAI]      = None
async_openai:     Optional[AsyncOpenAI] = None
diarize_pipeline: Optional[Pipeline]   = None


@app.on_event("startup")
async def startup_event():
    global sync_openai, async_openai, diarize_pipeline

    if OPENAI_API_KEY:
        sync_openai  = OpenAI(api_key=OPENAI_API_KEY)
        async_openai = AsyncOpenAI(api_key=OPENAI_API_KEY)
        log.info("✅ OpenAI clients initialized (sync + async)")
    else:
        log.warning("⚠️  OPENAI_API_KEY not set")

    if HUGGINGFACE_TOKEN:
        try:
            log.info("⏳ Loading pyannote model (first run ~2–5 min)...")
            diarize_pipeline = Pipeline.from_pretrained(
                "pyannote/speaker-diarization-3.1",
                use_auth_token=HUGGINGFACE_TOKEN,
            )
            if diarize_pipeline is None:
                raise ValueError(
                    "Pipeline returned None. Please verify your HUGGINGFACE_TOKEN is correct and "
                    "that you accepted terms for both 'pyannote/speaker-diarization-3.1' and "
                    "'pyannote/segmentation-3.0' on Hugging Face."
                )
            device = "cuda" if torch.cuda.is_available() else "cpu"
            diarize_pipeline.to(torch.device(device))
            log.info(f"✅ pyannote loaded on {device.upper()}")
        except Exception as e:
            log.error(f"❌ pyannote load failed: {e}")
    else:
        log.warning("⚠️  HUGGINGFACE_TOKEN not set — diarization disabled")


# =============================================================================
# AUDIO FORMAT DETECTION
# =============================================================================
AudioFormat = Literal["pcm", "wav", "ogg", "webm", "mp3", "m4a", "flac", "unknown"]

MAGIC_BYTES: list[tuple[bytes, AudioFormat]] = [
    (b"RIFF",       "wav"),    # WAV
    (b"OggS",       "ogg"),    # OGG/Opus
    (b"\x1a\x45",  "webm"),   # WebM/Opus (EBML)
    (b"ID3",        "mp3"),    # MP3 with ID3 tag
    (b"\xff\xfb",  "mp3"),    # MP3 without ID3
    (b"\xff\xf3",  "mp3"),    # MP3 variant
    (b"fLaC",      "flac"),   # FLAC
]
M4A_MAGIC_OFFSET = 4  # 'ftyp' sits at byte offset 4 in M4A


def detect_format(data: bytes, hint: str = "") -> AudioFormat:
    """
    Detect audio format from magic bytes.
    Falls back to hint string if magic bytes are inconclusive (raw PCM has no header).
    """
    if hint in ("pcm", "raw"):
        return "pcm"

    for magic, fmt in MAGIC_BYTES:
        if data[:len(magic)] == magic:
            return fmt

    # M4A: 'ftyp' at byte 4
    if len(data) >= 8 and data[4:8] == b"ftyp":
        return "m4a"

    # No magic bytes matched → likely raw PCM
    if hint in ("ogg", "wav", "mp3", "m4a", "flac", "webm"):
        return hint  # trust explicit hint

    return "pcm"   # default fallback — treat as raw PCM


# =============================================================================
# AUDIO CONVERSION
# =============================================================================
def pcm_bytes_to_wav_file(pcm_bytes: bytes, sample_rate: int = PCM_SAMPLE_RATE) -> str:
    """
    Convert raw PCM int16 bytes → temp WAV file (fastest path, no pydub).
    Uses soundfile directly — ~0.5ms overhead.
    """
    audio_np  = np.frombuffer(pcm_bytes, dtype=PCM_DTYPE).astype(np.float32) / 32768.0
    tmp_path  = tempfile.mktemp(suffix=".wav")
    sf.write(tmp_path, audio_np, sample_rate)
    return tmp_path


def pcm_bytes_to_wav_bytes(pcm_bytes: bytes, sample_rate: int = PCM_SAMPLE_RATE) -> bytes:
    """
    Convert raw PCM int16 bytes → in-memory WAV bytes for Whisper API upload.
    No disk I/O — pure memory operation (~0.3ms).
    """
    audio_np = np.frombuffer(pcm_bytes, dtype=PCM_DTYPE)
    buf      = io.BytesIO()
    sf.write(buf, audio_np.astype(np.float32) / 32768.0, sample_rate, format="WAV")
    buf.seek(0)
    return buf.read()


def file_bytes_to_wav_file(audio_bytes: bytes) -> str:
    """
    Convert any file format (OGG/WAV/MP3/M4A/FLAC) → temp WAV 16kHz mono via pydub.
    Slower than PCM path (~5–20ms) but handles all formats.
    """
    tmp_path = tempfile.mktemp(suffix=".wav")
    audio    = AudioSegment.from_file(io.BytesIO(audio_bytes))
    audio    = audio.set_channels(1).set_frame_rate(16000)
    audio.export(tmp_path, format="wav")
    return tmp_path


def file_bytes_to_wav_bytes(audio_bytes: bytes) -> bytes:
    """
    Convert any file format → in-memory WAV bytes for Whisper API.
    No disk I/O — pure memory (~5–20ms).
    """
    audio = AudioSegment.from_file(io.BytesIO(audio_bytes))
    audio = audio.set_channels(1).set_frame_rate(16000)
    buf   = io.BytesIO()
    audio.export(buf, format="wav")
    buf.seek(0)
    return buf.read()


def any_audio_to_wav_bytes(raw: bytes, fmt: AudioFormat) -> bytes:
    """
    Universal converter: routes to fast PCM path or pydub path based on format.
    Returns WAV bytes ready for Whisper API or pyannote.
    """
    if fmt == "pcm":
        return pcm_bytes_to_wav_bytes(raw)
    else:
        return file_bytes_to_wav_bytes(raw)


def any_audio_to_wav_file(raw: bytes, fmt: AudioFormat) -> str:
    """
    Universal converter: routes to fast PCM path or pydub path based on format.
    Returns path to temp WAV file for pyannote.
    """
    if fmt == "pcm":
        return pcm_bytes_to_wav_file(raw)
    else:
        return file_bytes_to_wav_file(raw)


# =============================================================================
# SESSION MANAGER
# =============================================================================
class Session:
    """Holds per-connection state. Speaker map persists across all chunks."""

    def __init__(self, session_id: str, language: str = "en", audio_format: str = "pcm"):
        self.session_id   = session_id
        self.language     = language
        self.audio_format: AudioFormat = audio_format  # user-declared hint
        self.speaker_map: dict[str, str] = {}
        self.chunk_count  = 0
        self.total_words  = 0

    def resolve_speaker(self, raw_label: str) -> str:
        """Map pyannote SPEAKER_XX → persistent friendly name for this session."""
        if raw_label not in self.speaker_map:
            idx = len(self.speaker_map)
            name = "You" if idx == 0 else "Client" if idx == 1 else f"Speaker_{idx + 1}"
            self.speaker_map[raw_label] = name
        return self.speaker_map[raw_label]


# =============================================================================
# DIARIZATION HELPER
# =============================================================================
def run_diarization(wav_path: str) -> list[dict]:
    """Run pyannote speaker diarization. Returns [{speaker, start, end}]."""
    if not diarize_pipeline:
        return []
    waveform, sample_rate = torchaudio.load(wav_path)
    result = diarize_pipeline({"waveform": waveform, "sample_rate": sample_rate})
    return [
        {"speaker": spk, "start": round(turn.start, 3), "end": round(turn.end, 3)}
        for turn, _, spk in result.itertracks(yield_label=True)
    ]


# =============================================================================
# SEGMENT MERGE
# =============================================================================
def merge_segments(
    whisper_segs: list[dict],
    diarize_segs: list[dict],
    session: Session,
    full_text: str,
) -> list[dict]:
    """Match Whisper text segments to pyannote speaker windows. Merge same-speaker lines."""
    merged: list[dict] = []

    for ws in whisper_segs:
        ws_start = ws.get("start", 0.0)
        ws_end   = ws.get("end", 0.0)
        ws_text  = ws.get("text", "").strip()
        if not ws_text:
            continue

        # Overlap-based speaker matching
        best_raw   = None
        best_score = 0.0
        for ds in diarize_segs:
            overlap = max(0.0, min(ws_end, ds["end"]) - max(ws_start, ds["start"]))
            if overlap > best_score:
                best_score = overlap
                best_raw   = ds["speaker"]

        speaker = (
            session.resolve_speaker(best_raw) if best_raw
            else (list(session.speaker_map.values())[0] if session.speaker_map else "You")
        )

        # Merge consecutive same-speaker lines
        if merged and merged[-1]["speaker"] == speaker:
            merged[-1]["text"] += " " + ws_text
            merged[-1]["end"]   = round(ws_end, 3)
        else:
            merged.append({
                "speaker": speaker,
                "text":    ws_text,
                "start":   round(ws_start, 3),
                "end":     round(ws_end, 3),
            })

    if not merged:
        merged.append({
            "speaker": list(session.speaker_map.values())[0] if session.speaker_map else "You",
            "text":    full_text.strip(),
            "start":   0.0,
            "end":     0.0,
        })

    return merged


# =============================================================================
# CORE ASYNC PROCESSOR
# =============================================================================
async def process_chunk(raw_bytes: bytes, session: Session) -> list[dict]:
    """
    Full async pipeline per audio chunk:
      1. Detect format (magic bytes + session hint)
      2. Convert to WAV bytes (PCM fast path OR pydub)
      3. Transcribe via OpenAI Whisper async
      4. Diarize via pyannote in thread pool (non-blocking)
      5. Merge and return speaker-labeled segments

    Latency breakdown (typical):
      PCM → WAV conversion : ~0.3 ms  (in-memory, no disk)
      File → WAV conversion : ~5–20 ms (pydub, in-memory, no disk)
      Whisper API call      : ~1–2 s
      pyannote (CPU)        : ~1–3 s
    """
    loop = asyncio.get_running_loop()

    # ── Step 1: Detect format ──────────────────────────────────────────────────
    fmt = detect_format(raw_bytes, hint=session.audio_format)
    log.info(f"🔍 [{session.session_id}] Detected format: {fmt} | {len(raw_bytes)} bytes")

    # ── Step 2: Convert to WAV bytes (in-memory, no disk I/O for Whisper) ──────
    # Run in thread pool to avoid blocking the async event loop
    wav_bytes: bytes = await loop.run_in_executor(
        None, any_audio_to_wav_bytes, raw_bytes, fmt
    )

    if len(wav_bytes) < 4000:
        log.warning(f"⚠️ [{session.session_id}] Audio chunk too short ({len(wav_bytes)} bytes). Skipping Whisper.")
        return []

    # ── Step 3: Transcribe with Whisper (async API call) ──────────────────────
    audio_io      = io.BytesIO(wav_bytes)
    audio_io.name = "audio.wav"

    whisper_resp = await async_openai.audio.transcriptions.create(
        model="whisper-1",
        file=audio_io,
        language=session.language,
        response_format="verbose_json",
        timestamp_granularities=["segment"],
    )

    whisper_text = get_val(whisper_resp, "text")
    full_text = (whisper_text or "").strip()
    raw_segs = get_val(whisper_resp, "segments") or []
    whisper_segs = []
    for s in raw_segs:
        s_text = get_val(s, "text")
        if s_text and s_text.strip():
            whisper_segs.append({
                "text": s_text.strip(),
                "start": get_val(s, "start", 0.0),
                "end": get_val(s, "end", 0.0)
            })

    if not full_text:
        return []   # silence chunk

    log.info(f"🎙️  [{session.session_id}] Whisper: \"{full_text[:70]}\"")

    # ── Step 4: Diarize with pyannote (thread pool, non-blocking) ─────────────
    diarize_segs: list[dict] = []
    if diarize_pipeline:
        # Write WAV to temp file for pyannote (torchaudio.load requires a file)
        wav_path: str = await loop.run_in_executor(
            None, _write_wav_temp, wav_bytes
        )
        try:
            diarize_segs = await loop.run_in_executor(None, run_diarization, wav_path)
            log.info(
                f"🧑‍🤝‍🧑 [{session.session_id}] Speakers: "
                f"{list(set(s['speaker'] for s in diarize_segs))}"
            )
        finally:
            if os.path.exists(wav_path):
                os.remove(wav_path)

    # ── Step 5: Merge ──────────────────────────────────────────────────────────
    return merge_segments(whisper_segs, diarize_segs, session, full_text)


def _write_wav_temp(wav_bytes: bytes) -> str:
    """Write WAV bytes to a temp file (needed by torchaudio.load for pyannote)."""
    path = tempfile.mktemp(suffix=".wav")
    with open(path, "wb") as f:
        f.write(wav_bytes)
    return path


# =============================================================================
# WEBSOCKET — PRIMARY REAL-TIME ENDPOINT
# =============================================================================
@app.websocket("/ws/audio")
async def websocket_audio(websocket: WebSocket):
    """
    ┌─────────────────────────────────────────────────────────────────────────┐
    │  SalesIQ Real-Time Audio WebSocket                                      │
    │                                                                         │
    │  Connect: ws://host/ws/audio                                            │
    │                                                                         │
    │  STEP 1 — Start session (text frame):                                   │
    │    { "type": "START_SESSION", "language": "en", "format": "pcm" }      │
    │                                                                         │
    │    format options:                                                      │
    │      "pcm"  → Raw int16 bytes, 16kHz mono  (FASTEST ⚡)               │
    │      "ogg"  → OGG/Opus compressed          (SMALLEST ✅)               │
    │      "wav"  → WAV file bytes               (COMPATIBLE)                │
    │      "mp3"  → MP3 file bytes               (COMPATIBLE)                │
    │      "auto" → Auto-detect from magic bytes (DEFAULT)                   │
    │                                                                         │
    │  STEP 2 — Send audio chunk (binary frame):                              │
    │    <raw PCM bytes>  OR  <OGG/WAV/MP3/M4A file bytes>                   │
    │    Recommended chunk duration: 3–5 seconds                              │
    │                                                                         │
    │  STEP 3 — Repeat step 2 for each audio chunk                            │
    │                                                                         │
    │  STEP 4 — End session (text frame):                                     │
    │    { "type": "END_SESSION" }                                            │
    └─────────────────────────────────────────────────────────────────────────┘
    """
    await websocket.accept()
    client_addr = websocket.client.host if websocket.client else "unknown"
    log.info(f"🔌 Connected: {client_addr}")

    session: Optional[Session] = None

    try:
        while True:
            msg = await websocket.receive()

            # ── TEXT FRAME (control) ──────────────────────────────────────────
            if "text" in msg:
                try:
                    ctrl     = _json.loads(msg["text"])
                    msg_type = ctrl.get("type", "").upper()
                except Exception:
                    await websocket.send_json({"type": "ERROR", "message": "Invalid JSON"})
                    continue

                # START_SESSION
                if msg_type == "START_SESSION":
                    audio_fmt = ctrl.get("format", "auto").lower()
                    session   = Session(
                        session_id   = str(uuid.uuid4())[:8],
                        language     = ctrl.get("language", "en"),
                        audio_format = audio_fmt,
                    )
                    log.info(
                        f"▶️  Session {session.session_id} | "
                        f"lang={session.language} | fmt={audio_fmt}"
                    )
                    await websocket.send_json({
                        "type":        "SESSION_STARTED",
                        "session_id":  session.session_id,
                        "format_mode": audio_fmt,
                        "tip": (
                            "Send raw int16 PCM bytes for fastest processing"
                            if audio_fmt == "pcm"
                            else "Send audio chunks as binary frames"
                        ),
                    })

                # END_SESSION
                elif msg_type == "END_SESSION":
                    stats = {}
                    if session:
                        stats = {
                            "session_id":  session.session_id,
                            "chunks":      session.chunk_count,
                            "total_words": session.total_words,
                            "speakers":    list(session.speaker_map.values()),
                        }
                        log.info(f"⏹️  Session {session.session_id} ended | {stats}")
                    await websocket.send_json({"type": "SESSION_ENDED", "stats": stats})
                    session = None

                # PING
                elif msg_type == "PING":
                    await websocket.send_json({"type": "PONG"})

                else:
                    await websocket.send_json({
                        "type":    "ERROR",
                        "message": f"Unknown control type: {msg_type}",
                    })

            # ── BINARY FRAME (audio chunk) ────────────────────────────────────
            elif "bytes" in msg:
                raw_bytes = msg["bytes"]
                if not raw_bytes:
                    continue

                # Auto-create session if none exists
                if not session:
                    session = Session(session_id=str(uuid.uuid4())[:8])
                    log.info(f"⚡ Auto-session: {session.session_id}")
                    await websocket.send_json({
                        "type":        "SESSION_STARTED",
                        "session_id":  session.session_id,
                        "format_mode": "auto",
                        "message":     "Auto session. Send START_SESSION to set format explicitly.",
                    })

                if not async_openai:
                    await websocket.send_json({
                        "type":    "ERROR",
                        "message": "OPENAI_API_KEY not configured on server.",
                    })
                    continue

                session.chunk_count += 1
                chunk_id = f"{session.session_id}-c{session.chunk_count}"
                detected = detect_format(raw_bytes, hint=session.audio_format)

                log.info(
                    f"📥 [{session.session_id}] Chunk #{session.chunk_count} "
                    f"| {len(raw_bytes):,} bytes | fmt={detected}"
                )

                # Immediate ACK with format info
                await websocket.send_json({
                    "type":     "PROCESSING",
                    "chunk_id": chunk_id,
                    "size":     len(raw_bytes),
                    "format":   detected,
                })

                # Process chunk
                try:
                    segments = await process_chunk(raw_bytes, session)

                    if not segments:
                        await websocket.send_json({
                            "type":     "SILENCE",
                            "chunk_id": chunk_id,
                        })
                    else:
                        for seg in segments:
                            session.total_words += len(seg["text"].split())
                            await websocket.send_json({
                                "type":       "SPEECH",
                                "speaker":    seg["speaker"],
                                "text":       seg["text"],
                                "start":      seg["start"],
                                "end":        seg["end"],
                                "chunk_id":   chunk_id,
                                "session_id": session.session_id,
                            })
                        log.info(
                            f"✅ [{session.session_id}] Chunk #{session.chunk_count} "
                            f"→ {len(segments)} segment(s) sent"
                        )

                except Exception as e:
                    log.error(f"❌ [{session.session_id}] Error: {e}")
                    await websocket.send_json({
                        "type":     "ERROR",
                        "message":  str(e),
                        "chunk_id": chunk_id,
                    })

            elif msg.get("type") == "websocket.disconnect":
                break

    except WebSocketDisconnect:
        sid = session.session_id if session else "unknown"
        log.info(f"🔌 Disconnected: {sid}")
    except Exception as e:
        log.error(f"❌ Fatal WebSocket error: {e}")
        try:
            await websocket.send_json({"type": "ERROR", "message": str(e)})
        except Exception:
            pass


# =============================================================================
# REST ENDPOINTS (Postman testing & fallback)
# =============================================================================
@app.get("/", tags=["Health"])
async def root():
    return {
        "status":    "ok",
        "service":   "SalesIQ Backend API",
        "version":   "3.0.0",
        "websocket": "ws://host/ws/audio",
        "formats_supported": ["pcm (fastest)", "ogg/opus", "wav", "mp3", "m4a", "flac", "webm"],
    }


@app.get("/status", tags=["Health"])
async def status():
    return {
        "status":               "ok",
        "openai_whisper":       "ready" if async_openai else "not configured",
        "pyannote_diarization": "ready" if diarize_pipeline else "not configured",
        "gpu_available":        torch.cuda.is_available(),
        "device":               "cuda" if torch.cuda.is_available() else "cpu",
        "pcm_config": {
            "sample_rate": PCM_SAMPLE_RATE,
            "channels":    PCM_CHANNELS,
            "dtype":       "int16",
        },
    }


@app.post("/transcribe", tags=["REST"])
async def transcribe(
    file: UploadFile = File(...),
    language: str = "en",
    format: str = "auto",
):
    """Upload any audio file → plain transcript. Supports PCM, WAV, OGG, MP3, M4A."""
    if not sync_openai:
        raise HTTPException(503, "OPENAI_API_KEY not configured.")
    raw  = await file.read()
    if not raw:
        raise HTTPException(400, "Empty file.")
    fmt  = detect_format(raw, hint=format)
    wav  = any_audio_to_wav_bytes(raw, fmt)
    try:
        audio_io      = io.BytesIO(wav)
        audio_io.name = "audio.wav"
        resp = sync_openai.audio.transcriptions.create(
            model="whisper-1", file=audio_io, language=language,
            response_format="verbose_json", timestamp_granularities=["segment"],
        )
        resp_text = get_val(resp, "text")
        raw_segs = get_val(resp, "segments") or []
        segments = []
        for s in raw_segs:
            s_text = get_val(s, "text")
            if s_text and s_text.strip():
                segments.append({
                    "text": s_text.strip(),
                    "start": round(get_val(s, "start", 0.0), 2),
                    "end": round(get_val(s, "end", 0.0), 2)
                })
        duration = get_val(resp, "duration")
        lang = get_val(resp, "language")
        return JSONResponse({
            "transcript":      resp_text,
            "language":        lang,
            "duration":        round(duration, 2) if duration else None,
            "detected_format": fmt,
            "segments":        segments,
        })
    except Exception as e:
        raise HTTPException(500, str(e))


@app.post("/diarize", tags=["REST"])
async def diarize(
    file: UploadFile = File(...),
    language: str = "en",
    format: str = "auto",
):
    """Upload any audio file → speaker-labeled transcript. Supports PCM, WAV, OGG, MP3."""
    if not sync_openai:
        raise HTTPException(503, "OPENAI_API_KEY not configured.")
    raw = await file.read()
    if not raw:
        raise HTTPException(400, "Empty file.")
    fmt  = detect_format(raw, hint=format)
    wav  = any_audio_to_wav_bytes(raw, fmt)
    loop = asyncio.get_running_loop()
    session = Session(session_id="rest-test", language=language, audio_format=fmt)
    wav_path = None
    try:
        audio_io      = io.BytesIO(wav)
        audio_io.name = "audio.wav"
        resp = sync_openai.audio.transcriptions.create(
            model="whisper-1", file=audio_io, language=language,
            response_format="verbose_json", timestamp_granularities=["segment"],
        )
        raw_segs = get_val(resp, "segments") or []
        whisper_segs = []
        for s in raw_segs:
            s_text = get_val(s, "text")
            if s_text and s_text.strip():
                whisper_segs.append({
                    "text": s_text.strip(),
                    "start": get_val(s, "start", 0.0),
                    "end": get_val(s, "end", 0.0)
                })
        diarize_segs: list[dict] = []
        if diarize_pipeline:
            wav_path     = await loop.run_in_executor(None, _write_wav_temp, wav)
            diarize_segs = await loop.run_in_executor(None, run_diarization, wav_path)
        resp_text = get_val(resp, "text")
        merged = merge_segments(whisper_segs, diarize_segs, session, resp_text)
        return JSONResponse({
            "speakers_detected": len(session.speaker_map),
            "speaker_map":       session.speaker_map,
            "detected_format":   fmt,
            "transcript":        merged,
        })
    except Exception as e:
        raise HTTPException(500, str(e))
    finally:
        if wav_path and os.path.exists(wav_path):
            os.remove(wav_path)
