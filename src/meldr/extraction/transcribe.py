"""Transcribers: local faster-whisper or an OpenAI-compatible transcription endpoint."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any, Protocol

from pydantic import BaseModel

from meldr.core.errors import MissingExtraError
from meldr.llm.policy import DataClass, DataPolicy, check


class TranscriptSegment(BaseModel):
    start_ms: int
    end_ms: int
    text: str


class Transcriber(Protocol):
    name: str
    is_local: bool

    async def transcribe(self, wav_path: Path) -> list[TranscriptSegment]: ...


class FasterWhisperTranscriber:
    """Runs faster-whisper locally in a worker thread (extra `media`)."""

    name = "faster_whisper"
    is_local = True

    def __init__(self, model: str = "small", *, whisper_model: Any = None) -> None:
        if whisper_model is None:
            try:
                from faster_whisper import WhisperModel
            except ImportError as e:
                raise MissingExtraError("media", "Local transcription") from e
            whisper_model = WhisperModel(model, device="auto", compute_type="int8")
        self._model = whisper_model

    async def transcribe(self, wav_path: Path) -> list[TranscriptSegment]:
        def run() -> list[TranscriptSegment]:
            segments, _info = self._model.transcribe(str(wav_path))
            return [
                TranscriptSegment(
                    start_ms=int(s.start * 1000), end_ms=int(s.end * 1000), text=s.text.strip()
                )
                for s in segments
            ]

        return await asyncio.to_thread(run)


class OpenAITranscriber:
    """`audio.transcriptions.create` with segment timestamps. Vendor unless `is_local`."""

    name = "openai_transcription"

    def __init__(
        self,
        model: str = "whisper-1",
        *,
        client: Any,
        is_local: bool = False,
        policy: DataPolicy | None = None,
    ) -> None:
        self.model = model
        self.is_local = is_local
        self._client = client
        self.policy = policy or DataPolicy.default_for(is_local=is_local)

    async def transcribe(self, wav_path: Path) -> list[TranscriptSegment]:
        check(self.policy, {DataClass.CONTENT}, provider=self.name)
        with wav_path.open("rb") as fh:
            resp = await self._client.audio.transcriptions.create(
                model=self.model,
                file=fh,
                response_format="verbose_json",
                timestamp_granularities=["segment"],
            )
        return [
            TranscriptSegment(
                start_ms=int(s.start * 1000), end_ms=int(s.end * 1000), text=str(s.text).strip()
            )
            for s in (getattr(resp, "segments", None) or [])
        ]
