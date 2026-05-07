#!/usr/bin/env python3
"""
Generate and save rhyme-only output (step 1 of prompt pipeline).
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
from datetime import datetime
from pathlib import Path
from typing import Any

from google import genai
from google.genai import types

from config import Config


DEFAULT_PIPELINE_JSON = "outputs/pipeline_output_20260505_103714.json"
DEFAULT_OUTPUT_ROOT = "outputs"
STRICT_RHYME_LINE_COUNT = 9


SYSTEM_PROMPT = """You are an expert children's rhyme writer for premium 3D cartoon nursery videos.

Write one original, catchy, simple kids rhyme (age 3-7) that is educational and positive.

Rules:
- Keep language simple and playful.
- Avoid scary, unsafe, violent, or sad content.
- Rhyme must be exactly 9 lines.
- Keep total runtime around 1 minute when spoken/sung.
- Output is for a "3d animation cartoon" style video.

Return ONLY valid JSON with this structure:
{
  "title": "short title for this rhyme",
  "rhyme": "multi-line rhyme text",
  "rhyme_style": "brief description of literary style: meter, tone, age band (e.g. playful, ages 3-5)",
  "rhyme_music_style_prompt": "single detailed prompt for generating only 1-minute kids-safe background music matching the rhyme mood and rhythm, with a gentle start (no heavy opening)"
}
"""


def _sanitize_for_dir(name: str) -> str:
    cleaned = re.sub(r"[<>:\"/\\|?*\x00-\x1F]", "", name)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    # Windows does not allow folder names ending with space or dot.
    cleaned = cleaned.rstrip(" .")
    cleaned = cleaned[:80].rstrip(" .")
    return cleaned if cleaned else f"untitled_{datetime.now().strftime('%Y%m%d_%H%M%S')}"


def _load_pipeline_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _extract_candidates(payload: dict[str, Any]) -> list[dict[str, Any]]:
    ranked = payload.get("ranked", {}).get("top_shorts", [])
    processed = payload.get("processed_shorts", [])
    merged: list[dict[str, Any]] = []
    seen: set[str] = set()
    for item in ranked + processed:
        vid = str(item.get("video_id", "")).strip()
        if vid and vid not in seen:
            seen.add(vid)
            merged.append(item)
    return merged


def _find_video(candidates: list[dict[str, Any]], video_id: str) -> dict[str, Any] | None:
    for item in candidates:
        if str(item.get("video_id", "")).strip() == video_id.strip():
            return item
    return None


def _build_rhyme_request(reference: str) -> str:
    return f"""Create a kids rhyme from this reference:
{reference}

Important constraints:
- Make the rhyme educational and memorable.
- Keep total runtime around 1 minute.
- Music prompt must target about 1 minute (around 60 seconds).
- Ensure the music direction says the beginning should be gentle and not too heavy.
- Rhyme length must be exactly 9 lines.
- Output is for a "3d animation cartoon" video.
"""


def _call_gemini_for_rhyme(reference: str) -> dict[str, Any]:
    api_key = (os.getenv("GEMINI_API_KEY", "").strip() or os.getenv("GOOGLE_API_KEY", "").strip())
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY is missing in environment/.env.")
    model = os.getenv("GEMINI_TEXT_MODEL", "").strip() or "gemini-3.1-flash-lite-preview"
    client = genai.Client(api_key=api_key)
    response = client.models.generate_content(
        model=model,
        contents=[
            types.Content(
                role="user",
                parts=[
                    types.Part.from_text(
                        text=f"{SYSTEM_PROMPT}\n\n{_build_rhyme_request(reference)}"
                    )
                ],
            )
        ],
        config=types.GenerateContentConfig(
            temperature=0.8,
            response_mime_type="application/json",
        ),
    )
    raw = (response.text or "").strip()
    if not raw:
        raise RuntimeError("Gemini returned an empty response for rhyme generation.")
    raw = re.sub(r"^```json\s*|```$", "", raw, flags=re.MULTILINE).strip()
    return json.loads(raw)

def _enforce_exact_rhyme_line_count(payload: dict[str, Any]) -> dict[str, Any]:
    rhyme_text = str(payload.get("rhyme", "")).strip()
    rhyme_lines = [line.strip() for line in rhyme_text.splitlines() if line.strip()]
    if len(rhyme_lines) > STRICT_RHYME_LINE_COUNT:
        rhyme_lines = rhyme_lines[:STRICT_RHYME_LINE_COUNT]
    elif len(rhyme_lines) < STRICT_RHYME_LINE_COUNT:
        while len(rhyme_lines) < STRICT_RHYME_LINE_COUNT:
            rhyme_lines.append("Clap and count, then smile and play!")
    out = dict(payload)
    out["rhyme"] = "\n".join(rhyme_lines)
    return out


def _ensure_rhyme_fields(payload: dict[str, Any]) -> dict[str, Any]:
    out = dict(payload)
    if not str(out.get("rhyme_style", "")).strip():
        raise ValueError("Model output missing required field: rhyme_style")
    if not str(out.get("rhyme_music_style_prompt", "")).strip():
        raise ValueError("Model output missing required field: rhyme_music_style_prompt")
    return out


def build_reference_from_pipeline_video(
    payload: dict[str, Any], video_id: str, theme: str | None = None
) -> tuple[str, str]:
    candidates = _extract_candidates(payload)
    selected = _find_video(candidates, video_id)
    if not selected:
        available_ids = ", ".join(str(item.get("video_id", "")) for item in candidates[:30])
        raise ValueError(
            f"video_id '{video_id}' not found in pipeline output."
            + (f" Available IDs (sample): {available_ids}" if available_ids else "")
        )
    title = str(selected.get("title", "")).strip() or f"video_{video_id}"
    reference = (
        f"Video ID: {video_id}\n"
        f"Title: {title}\n"
        f"Channel: {selected.get('channel', '')}\n"
        f"Description: {selected.get('description', '')}\n"
        f"Scene Description: {selected.get('scene_description', '')}\n"
        f"Transcript: {selected.get('transcript', '')}\n"
        f"Story Arc: {selected.get('detailed_video_analysis', {}).get('story_arc', '')}\n"
        f"Visual Timeline: {selected.get('detailed_video_analysis', {}).get('visual_timeline', [])}\n"
    )
    if theme:
        reference += f"\nAdditional theme/mood requested: {theme}\n"
    return title, reference


def generate_rhyme_for_studio(
    *,
    video_id: str | None = None,
    custom_prompt: str | None = None,
    theme: str | None = None,
    output_root: Path | None = None,
) -> dict[str, Any]:
    """
    Optional video_id runs the main pipeline (saves pipeline JSON), then generates a rhyme from that video's row.
    Without video_id, uses custom_prompt and/or theme only.
    Returns rhyme payload plus paths for downstream steps.
    """
    root = output_root or Path(DEFAULT_OUTPUT_ROOT)
    vid = (video_id or "").strip()
    prompt = (custom_prompt or "").strip()
    mood = (theme or "").strip()

    if not vid and not prompt and not mood:
        raise ValueError("Provide a YouTube video ID and/or a custom prompt or theme.")

    pipeline_json_path: str | None = None
    reference = ""
    folder_title = ""

    if vid:
        from main import run_pipeline

        pipeline_json_path = asyncio.run(run_pipeline(vid))
        if not pipeline_json_path:
            raise RuntimeError("Pipeline did not produce an output JSON path.")
        loaded = _load_pipeline_json(Path(pipeline_json_path))
        folder_title, reference = build_reference_from_pipeline_video(loaded, vid, mood or None)
        if prompt:
            reference += f"\nUser creative direction (blend with the reference): {prompt}\n"
    else:
        folder_title = f"custom_rhyme_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        chunks: list[str] = []
        if prompt:
            chunks.append(f"User Prompt: {prompt}")
        if mood:
            chunks.append(f"Theme/Mood: {mood}")
        reference = "\n".join(chunks)

    rhyme_payload = _call_gemini_for_rhyme(reference)
    rhyme_payload = _enforce_exact_rhyme_line_count(rhyme_payload)
    rhyme_payload = _ensure_rhyme_fields(rhyme_payload)

    out_dir = _save_rhyme(root, folder_title, rhyme_payload, reference)
    safe_title = _sanitize_for_dir(folder_title)

    return {
        **rhyme_payload,
        "folder_title": safe_title,
        "output_prompts_dir": str(out_dir),
        "pipeline_json_path": pipeline_json_path,
    }


def _save_rhyme(output_root: Path, folder_title: str, payload: dict[str, Any], reference: str) -> Path:
    target_dir = output_root / _sanitize_for_dir(folder_title) / "prompts"
    target_dir.mkdir(parents=True, exist_ok=True)

    json_path = target_dir / "rhyme_only.json"
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    txt_path = target_dir / "rhyme_only.txt"
    with txt_path.open("w", encoding="utf-8") as f:
        f.write(f"Title: {payload.get('title', '')}\n\n")
        f.write("=== RHYME ===\n")
        f.write(payload.get("rhyme", ""))
        f.write("\n\n=== RHYME STYLE ===\n")
        f.write(payload.get("rhyme_style", ""))
        f.write("\n\n=== RHYME MUSIC STYLE PROMPT ===\n")
        f.write(payload.get("rhyme_music_style_prompt", ""))

    (target_dir / "reference_input.txt").write_text(reference, encoding="utf-8")
    return target_dir


def _resolve_reference(args: argparse.Namespace) -> tuple[str, str]:
    video_id = (args.video_id or "").strip()
    if not video_id:
        video_id = input("Enter video_id (leave blank for manual prompt mode): ").strip()

    if video_id:
        json_path = Path(args.json_path)
        if not json_path.exists():
            raise FileNotFoundError(f"Pipeline JSON not found: {json_path}")

        payload = _load_pipeline_json(json_path)
        candidates = _extract_candidates(payload)
        selected = _find_video(candidates, video_id)
        if not selected:
            available_ids = ", ".join(item.get("video_id", "") for item in candidates[:20])
            raise ValueError(
                f"video_id '{video_id}' not found in {json_path}. Available IDs: {available_ids}"
            )

        title = str(selected.get("title", "")).strip() or f"video_{video_id}"
        reference = (
            f"Video ID: {video_id}\n"
            f"Title: {title}\n"
            f"Channel: {selected.get('channel', '')}\n"
            f"Description: {selected.get('description', '')}\n"
            f"Scene Description: {selected.get('scene_description', '')}\n"
            f"Transcript: {selected.get('transcript', '')}\n"
            f"Story Arc: {selected.get('detailed_video_analysis', {}).get('story_arc', '')}\n"
            f"Visual Timeline: {selected.get('detailed_video_analysis', {}).get('visual_timeline', [])}\n"
        )
        return title, reference

    custom_prompt = input("Enter your reference prompt/theme for the kids rhyme: ").strip()
    if not custom_prompt:
        raise ValueError("You must enter a prompt when video_id is not provided.")
    custom_title = input("Optional title for output folder (press Enter to auto-generate): ").strip()
    folder_title = custom_title or f"custom_rhyme_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    return folder_title, f"User Prompt: {custom_prompt}"


def main() -> None:
    parser = argparse.ArgumentParser(description="Step 1: kids rhyme writer")
    parser.add_argument("--video-id", type=str, default=None, help="Video ID from pipeline JSON.")
    parser.add_argument("--json-path", type=str, default=DEFAULT_PIPELINE_JSON, help="Pipeline JSON path.")
    parser.add_argument("--output-root", type=str, default=DEFAULT_OUTPUT_ROOT, help="Root output directory.")
    args = parser.parse_args()

    folder_title, reference = _resolve_reference(args)
    rhyme_payload = _call_gemini_for_rhyme(reference)
    rhyme_payload = _enforce_exact_rhyme_line_count(rhyme_payload)
    rhyme_payload = _ensure_rhyme_fields(rhyme_payload)

    out_dir = _save_rhyme(Path(args.output_root), folder_title, rhyme_payload, reference)
    print(f"Saved rhyme-only prompts in: {out_dir}")


if __name__ == "__main__":
    main()
