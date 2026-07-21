"""Build a deterministic, editable playback story from panel analysis."""
import json
from core import database


def _speech(analysis: dict) -> tuple[str, str]:
    dialogue = analysis.get("dialogue") or []
    lines: list[str] = []; speaker = "Narrator"
    for item in dialogue:
        if isinstance(item, dict) and item.get("text"):
            speaker = str(item.get("speaker") or speaker); lines.append(str(item["text"]))
    return speaker, " ".join(lines)


def build_story(project_name: str) -> dict:
    """Merge ordered panel analyses into one sequence per panel.

    The source order is page then panel index; character names are normalized
    case-insensitively to maintain continuity without inventing content.
    """
    panels = database.get_all_panels(project_name)
    sequences: list[dict] = []; canonical: dict[str, str] = {}
    for index, panel in enumerate(panels, 1):
        if not panel.get("analysis_json"): raise ValueError(f"Panel {panel['id']} has not been analyzed.")
        analysis = json.loads(panel["analysis_json"])
        characters = [canonical.setdefault(str(name).casefold(), str(name)) for name in analysis.get("characters", []) if str(name).strip()]
        speaker, text = _speech(analysis)
        narration = str(analysis.get("narration") or "")
        words = len((text + " " + narration).split())
        sequences.append({"sequence_index": index, "panel_id": panel["id"], "speaker": speaker, "text": text, "narration": narration, "emotion": str(analysis.get("emotion") or "neutral"), "camera_suggestion": str(analysis.get("camera_suggestion") or "static"), "duration_seconds": max(2.5, min(12.0, words / 2.6 if words else 3.0)), "metadata_json": json.dumps({"scene": analysis.get("scene", ""), "characters": characters, "actions": analysis.get("actions", [])}, ensure_ascii=False)})
    if not sequences: raise ValueError("No panels are available to build a story.")
    database.replace_story_sequences(project_name, sequences)
    database.update_project_state(project_name, "STORY_READY")
    return {"sequences": len(sequences), "characters": sorted(canonical.values())}
