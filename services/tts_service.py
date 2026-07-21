"""Local Edge-TTS generation with SQLite-backed timing metadata."""
import asyncio
import logging
import os
from typing import Callable, Optional

from core import database, workspace

LOGGER = logging.getLogger(__name__)
NARRATOR_VOICE = "en-US-AriaNeural"
CHARACTER_VOICE = "en-US-GuyNeural"


def _duration(path: str) -> float:
    try:
        from moviepy import AudioFileClip
        clip = AudioFileClip(path)
        try: return float(clip.duration)
        finally: clip.close()
    except Exception as exc:
        raise RuntimeError(f"Could not read generated audio duration: {exc}") from exc


async def _save(text: str, voice: str, path: str) -> None:
    try:
        import edge_tts
    except ImportError as exc:
        raise RuntimeError("edge-tts is not installed. Install project requirements.") from exc
    await edge_tts.Communicate(text=text, voice=voice).save(path)


def generate_audio(project_name: str, narrator_voice: str = NARRATOR_VOICE, character_voice: str = CHARACTER_VOICE, progress_callback: Optional[Callable[[int, int], None]] = None) -> dict:
    """Generate a timed MP3 for every non-empty narration/dialogue sequence."""
    sequences = database.get_story_sequences(project_name)
    if not sequences: raise ValueError("Build the story before generating audio.")
    audio_dir = os.path.join(workspace.get_project_path(project_name), "audio"); os.makedirs(audio_dir, exist_ok=True)
    produced = skipped = 0; errors: list[str] = []
    for index, sequence in enumerate(sequences, 1):
        text = " ".join(part for part in (sequence.get("narration"), sequence.get("text")) if part).strip()
        if not text:
            skipped += 1
        else:
            voice = narrator_voice if not sequence.get("text") else character_voice
            path = os.path.join(audio_dir, f"sequence_{sequence['sequence_index']:04d}.mp3")
            try:
                asyncio.run(_save(text, voice, path))
                database.upsert_audio_asset(project_name, sequence["id"], path, voice, _duration(path), text)
                produced += 1
            except Exception as exc:
                LOGGER.exception("TTS failed for sequence %s", sequence["id"]); errors.append(f"Sequence {sequence['sequence_index']}: {exc}")
        if progress_callback: progress_callback(index, len(sequences))
    if not errors: database.update_project_state(project_name, "AUDIO_GENERATED")
    return {"total": len(sequences), "generated": produced, "skipped": skipped, "errors": errors}
