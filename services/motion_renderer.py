"""Per-panel MoviePy motion clips with audio and subtitle synchronization."""
import os
from typing import Callable, Optional
from core import database, workspace

OUTPUT_SIZE = (1280, 720); FPS = 24

def _moviepy():
    try:
        from moviepy import AudioFileClip, ImageClip, TextClip, CompositeVideoClip
        return AudioFileClip, ImageClip, TextClip, CompositeVideoClip
    except ImportError as exc: raise RuntimeError("moviepy is not installed. Install project requirements.") from exc

def render_motion_clips(project_name: str, progress_callback: Optional[Callable[[int, int], None]] = None) -> dict:
    """Render resumable motion clips from story sequences."""
    AudioFileClip, ImageClip, TextClip, CompositeVideoClip = _moviepy()
    sequences = database.get_story_sequences(project_name)
    if not sequences: raise ValueError("Build the story before rendering clips.")
    target = os.path.join(workspace.get_project_path(project_name), "video_clips"); os.makedirs(target, exist_ok=True)
    clips = database.get_render_clips(project_name); existing = {clip['sequence_id']: clip for clip in clips if os.path.exists(clip['clip_path'])}
    done = skipped = 0; errors: list[str] = []
    for number, item in enumerate(sequences, 1):
        if item['id'] in existing: skipped += 1; continue
        video = audio = None
        try:
            duration = float(item.get('audio_duration') or item['duration_seconds'])
            image = ImageClip(item['panel_path']).with_duration(duration).resized(height=OUTPUT_SIZE[1])
            if image.w < OUTPUT_SIZE[0]: image = image.resized(width=OUTPUT_SIZE[0])
            motion = item.get('camera_suggestion', 'static')
            if motion in ('zoom_in', 'zoom_out'):
                start, end = (1.0, 1.12) if motion == 'zoom_in' else (1.12, 1.0)
                image = image.resized(lambda t: start + (end - start) * (t / duration))
            image = image.with_position('center')
            layers = [image]
            subtitle = ' '.join(x for x in (item.get('narration'), item.get('text')) if x).strip()
            if subtitle:
                layers.append(TextClip(text=subtitle, font_size=34, color='white', stroke_color='black', stroke_width=2, size=(1160, None), method='caption').with_duration(duration).with_position(('center', 620)))
            video = CompositeVideoClip(layers, size=OUTPUT_SIZE).with_duration(duration)
            if item.get('audio_path') and os.path.exists(item['audio_path']): audio = AudioFileClip(item['audio_path']); video = video.with_audio(audio)
            output = os.path.join(target, f"clip_{item['sequence_index']:04d}.mp4")
            video.write_videofile(output, fps=FPS, codec='libx264', audio_codec='aac', logger=None)
            database.upsert_render_clip(project_name, item['id'], output, duration); done += 1
        except Exception as exc: database.upsert_render_clip(project_name, item['id'], '', 0, 'ERROR', str(exc)); errors.append(f"Sequence {item['sequence_index']}: {exc}")
        finally:
            if video: video.close()
            if audio: audio.close()
        if progress_callback: progress_callback(number, len(sequences))
    if not errors: database.update_project_state(project_name, 'RENDERED')
    return {'total': len(sequences), 'rendered': done, 'skipped': skipped, 'errors': errors}
