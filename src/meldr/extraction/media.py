"""Media pipeline: ffprobe/ffmpeg, keyframes, vision captions, time-windowed segments (P2).

Subprocesses always use argument lists (not a shell), timeouts, and temp directories.
"""

from __future__ import annotations

import asyncio
import base64
import json
import re
import shutil
import subprocess
import tempfile
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from meldr.config import MediaConfig
from meldr.core.errors import ExternalToolError
from meldr.core.segments import ContentSegment, Locator, segment_id
from meldr.core.types import Modality, RawRecord
from meldr.extraction.documents import record_bytes
from meldr.extraction.transcribe import Transcriber, TranscriptSegment
from meldr.llm.base import Capability, ContentPart, Message
from meldr.llm.gateway import LLMGateway
from meldr.llm.policy import DataClass, SourcePolicy
from meldr.llm.prompts import load as load_prompt

CAPTION_PROMPT_VERSION = "captions-v1"
_PTS = re.compile(r"pts_time:([0-9.]+)")


class FrameCaption(BaseModel):
    frame_ms: int
    caption: str
    visible_text: str = ""


class FrameCaptions(BaseModel):
    captions: list[FrameCaption]


@dataclass
class Frame:
    ms: int
    jpeg: bytes


def _binary(name: str) -> str:
    path = shutil.which(name)
    if path is None:
        raise ExternalToolError(
            f"'{name}' is not on PATH. To process media, install ffmpeg (for example, "
            "`brew install ffmpeg` or `apt-get install ffmpeg`)."
        )
    return path


def _run(args: list[str], timeout: float) -> subprocess.CompletedProcess[bytes]:
    try:
        return subprocess.run(args, capture_output=True, timeout=timeout, check=True)  # noqa: S603
    except subprocess.TimeoutExpired as e:
        raise ExternalToolError(f"{Path(args[0]).name} timed out after {timeout}s.") from e
    except subprocess.CalledProcessError as e:
        tail = e.stderr.decode(errors="replace").strip().splitlines()[-1:] if e.stderr else []
        raise ExternalToolError(f"{Path(args[0]).name} failed: {' '.join(tail)}") from e


def probe(path: Path, *, timeout: float = 60.0) -> dict[str, Any]:
    """Duration and stream kinds from ffprobe."""
    out = _run(
        [
            _binary("ffprobe"),
            "-v",
            "error",
            "-show_entries",
            "format=duration:stream=codec_type",
            "-of",
            "json",
            str(path),
        ],
        timeout,
    )
    data = json.loads(out.stdout or b"{}")
    kinds = {s.get("codec_type") for s in data.get("streams", [])}
    return {
        "duration_s": float(data.get("format", {}).get("duration") or 0.0),
        "has_audio": "audio" in kinds,
        "has_video": "video" in kinds,
    }


def extract_audio(path: Path, out_wav: Path, *, timeout: float = 600.0) -> Path:
    _run(
        [
            _binary("ffmpeg"),
            "-y",
            "-v",
            "error",
            "-i",
            str(path),
            "-vn",
            "-ac",
            "1",
            "-ar",
            "16000",
            "-f",
            "wav",
            str(out_wav),
        ],
        timeout,
    )
    return out_wav


def keyframes(path: Path, workdir: Path, cfg: MediaConfig, duration_s: float) -> list[Frame]:
    """Scene-change frames (gt(scene,0.3)), capped and spaced, with fixed intervals as fallback."""
    scale = "scale='min(768,iw)':-2"
    scene_dir = workdir / "scene"
    scene_dir.mkdir()
    res = _run(
        [
            _binary("ffmpeg"),
            "-v",
            "info",
            "-i",
            str(path),
            "-vf",
            f"select='gt(scene,0.3)',showinfo,{scale}",
            "-vsync",
            "vfr",
            "-q:v",
            "4",
            str(scene_dir / "f%05d.jpg"),
        ],
        cfg.timeout_s,
    )
    times = [float(t) for t in _PTS.findall(res.stderr.decode(errors="replace"))]
    files = sorted(scene_dir.glob("f*.jpg"))
    frames: list[Frame] = []
    last = -1e9
    for t, f in zip(times, files, strict=False):
        if t - last >= cfg.min_frame_gap_s and len(frames) < cfg.max_frames:
            frames.append(Frame(ms=int(t * 1000), jpeg=f.read_bytes()))
            last = t
    wanted = max(1, min(cfg.max_frames, int(duration_s // cfg.frame_interval_s) or 1))
    if len(frames) < wanted:
        fixed_dir = workdir / "fixed"
        fixed_dir.mkdir()
        _run(
            [
                _binary("ffmpeg"),
                "-v",
                "error",
                "-i",
                str(path),
                "-vf",
                f"fps=1/{cfg.frame_interval_s},{scale}",
                "-q:v",
                "4",
                str(fixed_dir / "f%05d.jpg"),
            ],
            cfg.timeout_s,
        )
        frames = [
            Frame(ms=int(i * cfg.frame_interval_s * 1000), jpeg=f.read_bytes())
            for i, f in enumerate(sorted(fixed_dir.glob("f*.jpg"))[: cfg.max_frames])
        ]
    return frames


async def caption_frames(
    gateway: LLMGateway | None,
    frames: Sequence[Frame],
    *,
    per_call: int = 8,
    source_policy: SourcePolicy | None = None,
    warnings: list[str] | None = None,
) -> list[FrameCaption]:
    """Vision captions through the gateway (none, with a warning, when vision is unavailable)."""
    if gateway is None or Capability.VISION not in gateway.provider.capabilities:
        if warnings is not None and frames:
            warnings.append("captions_skipped:no_vision_provider")
        return []
    out: list[FrameCaption] = []
    for i in range(0, len(frames), per_call):
        group = frames[i : i + per_call]
        parts = [
            ContentPart(
                type="text",
                text=json.dumps({"task": "captions", "frame_ms": [f.ms for f in group]}),
            )
        ]
        for f in group:
            parts.append(
                ContentPart(
                    type="image",
                    image_b64=base64.b64encode(f.jpeg).decode(),
                    media_type="image/jpeg",
                )
            )
        res = await gateway.generate(
            [Message(role="user", content=parts)],
            response_model=FrameCaptions,
            contains={DataClass.CONTENT},
            purpose="captions",
            prompt_version=CAPTION_PROMPT_VERSION,
            system=load_prompt("captions_v1"),
            source_policy=source_policy,
        )
        out.extend(res.parsed.captions)
    return out


def windows(
    transcript: Sequence[TranscriptSegment],
    captions: Sequence[FrameCaption],
    *,
    record_id: str,
    source_id: str,
    duration_ms: int,
    window_s: float = 60.0,
    title: str | None = None,
) -> list[ContentSegment]:
    """Merge transcript and captions into fixed windows with start/end locators."""
    width = int(window_s * 1000)
    end = max(duration_ms, *(t.end_ms for t in transcript), *(c.frame_ms for c in captions), 1)
    segments: list[ContentSegment] = []
    for n, start in enumerate(range(0, end, width)):
        stop = min(start + width, end)
        spoken = [t.text for t in transcript if t.start_ms < stop and t.end_ms > start]
        seen = [
            c.caption + (f" [text: {c.visible_text}]" if c.visible_text else "")
            for c in captions
            if start <= c.frame_ms < stop
        ]
        if not spoken and not seen:
            continue
        lines = []
        if spoken:
            lines.append("Transcript: " + " ".join(spoken))
        if seen:
            lines.append("Visual: " + " | ".join(seen))
        segments.append(
            ContentSegment(
                segment_id=segment_id(source_id, record_id, n),
                source_id=source_id,
                record_id=record_id,
                modality=Modality.VIDEO,
                text="\n".join(lines),
                title=title,
                locator=Locator(start_ms=start, end_ms=stop),
            )
        )
    return segments


async def segment_image(
    record: RawRecord,
    vision: LLMGateway | None,
    *,
    source_policy: SourcePolicy | None = None,
) -> tuple[list[ContentSegment], list[str]]:
    """One captioned segment for an image record. Returns (segments, warnings)."""
    warnings: list[str] = []
    captions = await caption_frames(
        vision,
        [Frame(ms=0, jpeg=record_bytes(record))],
        source_policy=source_policy,
        warnings=warnings,
    )
    segments = [
        ContentSegment(
            segment_id=segment_id(record.source_id, record.record_id, i),
            source_id=record.source_id,
            record_id=record.record_id,
            modality=Modality.IMAGE,
            text=c.caption + (f"\nVisible text: {c.visible_text}" if c.visible_text else ""),
            locator=Locator(),
            title=record.record_id,
        )
        for i, c in enumerate(captions)
    ]
    return segments, warnings


async def segment_media(
    path: Path,
    *,
    record_id: str,
    source_id: str,
    cfg: MediaConfig,
    transcriber: Transcriber | None,
    vision: LLMGateway | None,
    source_policy: SourcePolicy | None = None,
) -> tuple[list[ContentSegment], list[str]]:
    """Full media pipeline for one file. Returns (segments, warnings)."""
    warnings: list[str] = []
    info = await asyncio.to_thread(probe, path, timeout=cfg.timeout_s)
    with tempfile.TemporaryDirectory(prefix="meldr-media-") as tmp:
        work = Path(tmp)
        transcript: list[TranscriptSegment] = []
        if info["has_audio"]:
            if transcriber is None:
                warnings.append("transcript_skipped:no_transcriber")
            else:
                wav = await asyncio.to_thread(
                    extract_audio, path, work / "audio.wav", timeout=cfg.timeout_s
                )
                transcript = await transcriber.transcribe(wav)
        frames = (
            await asyncio.to_thread(keyframes, path, work, cfg, info["duration_s"])
            if info["has_video"]
            else []
        )
        captions = await caption_frames(
            vision,
            frames,
            per_call=cfg.frames_per_call,
            source_policy=source_policy,
            warnings=warnings,
        )
    segs = windows(
        transcript,
        captions,
        record_id=record_id,
        source_id=source_id,
        duration_ms=int(info["duration_s"] * 1000),
        window_s=cfg.window_s,
        title=path.stem,
    )
    return segs, warnings
