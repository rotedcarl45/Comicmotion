"""Resumable final MP4 assembly from per-panel clips."""
import os
from typing import Callable, Optional
from core import database, workspace

def assemble_video(project_name: str, filename: str = 'comicmotion_final.mp4', progress_callback: Optional[Callable[[int, int], None]] = None) -> str:
    """Concatenate validated clips, record a job, and return the MP4 path."""
    try: from moviepy import VideoFileClip, concatenate_videoclips
    except ImportError as exc: raise RuntimeError('moviepy is not installed. Install project requirements.') from exc
    clips = database.get_render_clips(project_name)
    if not clips or any(not item['clip_path'] or not os.path.exists(item['clip_path']) for item in clips): raise ValueError('All motion clips must render successfully before assembly.')
    output = os.path.join(workspace.get_project_path(project_name), 'output', filename)
    job = database.create_render_job(project_name, output); opened = []; final = None
    try:
        for index, item in enumerate(clips, 1):
            opened.append(VideoFileClip(item['clip_path']))
            if progress_callback: progress_callback(index, len(clips))
        final = concatenate_videoclips(opened, method='compose')
        final.write_videofile(output, fps=24, codec='libx264', audio_codec='aac', logger=None)
        database.update_render_job(project_name, job, 'COMPLETED', 100); database.update_project_state(project_name, 'COMPLETED'); return output
    except Exception as exc:
        database.update_render_job(project_name, job, 'ERROR', 0, str(exc)); raise RuntimeError(f'Final assembly failed: {exc}') from exc
    finally:
        if final: final.close()
        for clip in opened: clip.close()
