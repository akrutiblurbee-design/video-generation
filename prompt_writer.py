#!/usr/bin/env python3
"""
Generate a kids rhyme + scene-by-scene visual prompts.

Flow:
1) Ask for video_id.
2) If empty, ask for custom user prompt.
3) If video_id exists, fetch reference info from pipeline JSON.
4) Generate:
   - A rhyme for kids
   - Detailed scene-by-scene generation prompts aligned to the rhyme
5) Save result in a directory named from the video title.
"""

from __future__ import annotations

import argparse
import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any

import openai

from config import Config


DEFAULT_PIPELINE_JSON = "outputs/pipeline_output_20260505_103714.json"
DEFAULT_OUTPUT_ROOT = "outputs"
STRICT_SCENE_COUNT = 7


SYSTEM_PROMPT = """You are an expert children's rhyme writer and storyboard prompt designer for premium 3D cartoon animation videos.

Your job:
1) Write one original, catchy, simple kids rhyme (age 3-7), 11-14 lines.
2) Create scene-by-scene prompts to generate visuals that perfectly match each rhyme beat.
3) Create one dedicated prompt to generate background music for the rhyme.

Rules:
- Keep language simple, playful, and positive.
- Avoid scary, unsafe, violent, or sad content.
- Ensure visual continuity across scenes (same characters, costumes, props, and color style).
- Scenes must directly reflect rhyme lines.
- Prefer bright, high-saturation, kid-friendly visuals.
- The video concept and every scene prompt must be for high-quality 3D cartoon animation in a Cocomelon-like nursery-rhyme style.
- Each scene prompt must be highly detailed, cinematic, and production-ready (not generic).
- Each scene prompt should include: environment details, character look + outfit, expressions, action beats, camera framing, lens feel, lighting, color mood, depth cues, kid-friendly texture/style details, and explicit foreground/midground/background composition.
- Keep the look consistent with Cocomelon-like traits: soft rounded forms, clean textures, bright pastel-heavy palette, friendly oversized expressions, gentle daytime lighting, and playful preschool environments.
- Mention only child-safe visual elements.
- Generate exactly 7 scenes (not fewer, not more).
- Rhyme must be at least 11 lines; do not generate fewer.
- Each visual_prompt must be highly descriptive and production-usable, around 150-220 words with specific visual details (avoid vague one-liners).
- Make each visual_prompt internally structured as a single paragraph that naturally covers: (1) setting and atmosphere, (2) character blocking and expressions, (3) camera shot + movement + lens feel, (4) lighting + palette + render texture cues.
- Avoid generic phrasing like "colorful background" or "kids are happy"; replace with concrete visual details (specific props, spatial layout, micro-actions, and facial cues).

Return ONLY valid JSON with this structure:
{
  "title": "short title for this rhyme",
  "rhyme": "multi-line rhyme text",
  "rhyme_music_style_prompt": "single detailed prompt for generating about 1-minute kids-safe background music matching the rhyme mood and rhythm, with a gentle start (no heavy opening)",
  "visual_style_guide": {
    "style": "Cocomelon-like 3D cartoon nursery-rhyme style details",
    "characters": ["character 1 detailed profile", "character 2 detailed profile"],
    "palette": "dominant color palette with mood guidance",
    "camera_language": "camera guidance for playful kid content",
    "render_notes": "materials, shading, and quality notes"
  },
  "scene_prompts": [
    {
      "scene_number": 1,
      "rhyme_line_reference": "line(s) covered",
      "duration_seconds": 4,
      "visual_prompt": "very detailed 3D cartoon generation prompt with environment, character, camera, and lighting",
      "motion_and_action": "how characters move",
      "continuity_notes": "what must stay consistent"
    }
  ]
}
"""


def _sanitize_for_dir(name: str) -> str:
    cleaned = re.sub(r"[<>:\"/\\|?*\x00-\x1F]", "", name)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    cleaned = cleaned.rstrip(".")
    return cleaned[:80] if cleaned else f"untitled_{datetime.now().strftime('%Y%m%d_%H%M%S')}"


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


def _build_user_request(reference: str) -> str:
    return f"""Create output based on this reference:
{reference}

Important constraints:
- Make the rhyme educational and memorable.
- Keep total runtime around 1 minute.
- Music prompt must target about 1 minute (around 60 seconds).
- Ensure the music direction says the beginning should be gentle and not too heavy.
- Generate exactly 7 scenes with clear progression.
- Rhyme length must be 11 to 14 lines (minimum 11 lines).
- Every scene must be visually tied to the exact rhyme idea it represents.
- Output is for a 3D cartoon animation video.
- Prefer a Cocomelon-like visual direction (friendly preschool world, soft rounded characters, bright pastel colors, cheerful lighting).
- Include rich visual details in each scene prompt: setting, background layers, props, character design details, facial expression, pose, action, camera angle, shot type, lighting setup, color mood, and texture/material cues.
- Make each scene visual prompt detailed enough for direct text-to-video generation, with cinematic framing, lens feel, depth layering, foreground-midground-background composition, and clear subject choreography.
- Keep style consistency across all scenes so they feel like one continuous 3D cartoon short film.
- For every scene visual_prompt, target 150-220 words and pack it with concrete, non-generic details.
- In each scene visual_prompt, explicitly specify:
  - precise setting layout (foreground, midground, background)
  - character blocking (who stands/sits where), pose, and expression beats
  - camera plan (shot size, angle, movement, and lens feel)
  - lighting design (key/fill/rim feel), palette accents, and material/texture cues
  - child-safe environmental props and motion choreography tied to rhyme meaning
"""


def _call_openai_for_rhyme(reference: str) -> dict[str, Any]:
    if not Config.OPENAI_API_KEY:
        raise RuntimeError("OPENAI_API_KEY is missing in environment/.env.")

    client = openai.OpenAI(api_key=Config.OPENAI_API_KEY, timeout=90)
    response = client.chat.completions.create(
        model=Config.OPENAI_MODEL,
        temperature=0.8,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": _build_user_request(reference)},
        ],
    )
    raw = (response.choices[0].message.content or "").strip()
    raw = re.sub(r"^```json\s*|```$", "", raw, flags=re.MULTILINE).strip()
    return json.loads(raw)


def _fallback_output(seed_title: str, seed_text: str) -> dict[str, Any]:
    rhyme = (
        "Clap, clap, little hands, clap in a row,\n"
        "Tap, tap, tiny toes, off we go!\n"
        "Share your smile, bright and wide,\n"
        "Friends together, side by side.\n"
        "Zoom goes the toy car, round and round,\n"
        "Happy giggles are the best sound.\n"
        "Take a turn, then let it be,\n"
        "Sharing makes it fun for three!\n"
        "Build a tower, one, two, three,\n"
        "Cheer for friends so happily,\n"
        "Play, learn, laugh, in harmony!\n"
    )
    return {
        "title": seed_title,
        "rhyme": rhyme,
        "rhyme_music_style_prompt": (
            f"{seed_text}. Create cheerful, child-safe instrumental nursery music for 60 seconds, "
            "with a soft, gentle intro (do not make the beginning too heavy), gentle percussion, "
            "playful marimba and ukulele textures, bright major-key harmony, steady 120 BPM groove, "
            "no vocals, smooth loop-friendly ending."
        ),
        "visual_style_guide": {
            "style": "Cocomelon-like 3D cartoon animation, rounded forms, soft global illumination, bright pastel nursery palette",
            "characters": ["two preschool kids", "a smiling toy car"],
            "palette": "sky blue, sunny yellow, candy pink, leaf green",
            "camera_language": "gentle dolly moves, medium and close shots, playful pacing",
            "render_notes": "clean toy-like materials, smooth shading, soft shadows, kid-friendly proportions",
        },
        "scene_prompts": [
            {
                "scene_number": 1,
                "rhyme_line_reference": "Clap, clap, little hands, clap in a row",
                "duration_seconds": 5,
                "visual_prompt": (
                    f"{seed_text}\n"
                    "Inside a bright playroom, two kids clap rhythmically in sync, "
                    "sunlight through windows, colorful toys in background, kid-safe props only."
                ),
                "motion_and_action": "Hands clap to beat, slight bounce dance.",
                "continuity_notes": "Keep same kids, clothing colors, and room layout.",
            },
            {
                "scene_number": 2,
                "rhyme_line_reference": "Tap, tap, tiny toes, off we go",
                "duration_seconds": 5,
                "visual_prompt": "Kids tap toes and march in a circle around a toy car on a soft rug.",
                "motion_and_action": "Playful marching and toe taps.",
                "continuity_notes": "Same playroom and same toy car.",
            },
            {
                "scene_number": 3,
                "rhyme_line_reference": "Share your smile, bright and wide",
                "duration_seconds": 5,
                "visual_prompt": "Close-up of both kids smiling and nodding kindly to each other.",
                "motion_and_action": "Friendly eye contact and hand gesture to share.",
                "continuity_notes": "Maintain facial style and color palette.",
            },
            {
                "scene_number": 4,
                "rhyme_line_reference": "Friends together, side by side",
                "duration_seconds": 5,
                "visual_prompt": "Both kids sit side by side and roll the toy car gently between them.",
                "motion_and_action": "Pass-and-receive toy action with cheerful expressions.",
                "continuity_notes": "Same costumes, toy car, and bright playroom background.",
            },
            {
                "scene_number": 5,
                "rhyme_line_reference": "Zoom goes the toy car, round and round",
                "duration_seconds": 5,
                "visual_prompt": "Wide shot of the toy car looping around colorful floor markers in 3D cartoon style.",
                "motion_and_action": "Fast toy-car movement, kids clap in rhythm.",
                "continuity_notes": "Preserve color palette and playful lighting.",
            },
            {
                "scene_number": 6,
                "rhyme_line_reference": "Happy giggles are the best sound",
                "duration_seconds": 5,
                "visual_prompt": "Medium close-up of both kids laughing while the toy car stops near them.",
                "motion_and_action": "Soft bouncing laugh motion and joyful hand gestures.",
                "continuity_notes": "Keep facial design and lighting continuity.",
            },
            {
                "scene_number": 7,
                "rhyme_line_reference": "Take a turn, then let it be, Sharing makes it fun for three",
                "duration_seconds": 6,
                "visual_prompt": "A third friend joins, and all three kids share turns with the toy car in a final happy group shot.",
                "motion_and_action": "Turn-taking sequence ending with group wave to camera.",
                "continuity_notes": "Same 3D style; add one new kid while preserving main scene setup.",
            },
        ],
    }


def _enforce_exact_scene_count(payload: dict[str, Any], required_count: int = STRICT_SCENE_COUNT) -> dict[str, Any]:
    scenes = payload.get("scene_prompts")
    if not isinstance(scenes, list):
        raise ValueError("Model output missing 'scene_prompts' list.")
    if len(scenes) != required_count:
        raise ValueError(
            f"Model output has {len(scenes)} scene(s); expected exactly {required_count} scenes."
        )

    normalized_scenes: list[dict[str, Any]] = []
    for idx, scene in enumerate(scenes, start=1):
        if not isinstance(scene, dict):
            raise ValueError(f"Scene {idx} is not a JSON object.")
        normalized_scene = dict(scene)
        normalized_scene["scene_number"] = idx
        normalized_scenes.append(normalized_scene)

    normalized_payload = dict(payload)
    normalized_payload["scene_prompts"] = normalized_scenes
    return normalized_payload


def _save_result(output_root: Path, folder_title: str, payload: dict[str, Any], reference: str) -> Path:
    target_dir = output_root / _sanitize_for_dir(folder_title) / "prompts"
    target_dir.mkdir(parents=True, exist_ok=True)

    json_path = target_dir / "rhyme_and_scene_prompts.json"
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    txt_path = target_dir / "rhyme_and_scene_prompts.txt"
    with txt_path.open("w", encoding="utf-8") as f:
        f.write(f"Title: {payload.get('title', '')}\n\n")
        f.write("=== RHYME ===\n")
        f.write(payload.get("rhyme", ""))
        f.write("\n\n=== RHYME MUSIC STYLE PROMPT ===\n")
        f.write(payload.get("rhyme_music_style_prompt", ""))
        f.write("\n\n=== VISUAL STYLE GUIDE ===\n")
        style = payload.get("visual_style_guide", {})
        f.write(json.dumps(style, indent=2, ensure_ascii=False))
        f.write("\n\n=== SCENE PROMPTS ===\n")
        for scene in payload.get("scene_prompts", []):
            f.write(f"\nScene {scene.get('scene_number', '')}\n")
            f.write(f"Rhyme Reference: {scene.get('rhyme_line_reference', '')}\n")
            f.write(f"Duration: {scene.get('duration_seconds', '')}s\n")
            f.write(f"Visual Prompt: {scene.get('visual_prompt', '')}\n")
            f.write(f"Motion/Action: {scene.get('motion_and_action', '')}\n")
            f.write(f"Continuity: {scene.get('continuity_notes', '')}\n")

    reference_path = target_dir / "reference_input.txt"
    reference_path.write_text(reference, encoding="utf-8")
    return target_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="Kids rhyme + scene prompt writer")
    parser.add_argument(
        "--video-id",
        type=str,
        default=None,
        help="Video ID to pull reference from pipeline JSON.",
    )
    parser.add_argument(
        "--json-path",
        type=str,
        default=DEFAULT_PIPELINE_JSON,
        help="Path to pipeline output JSON file.",
    )
    parser.add_argument(
        "--output-root",
        type=str,
        default=DEFAULT_OUTPUT_ROOT,
        help="Root directory where title-based folders are created.",
    )
    args = parser.parse_args()

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
        folder_title = title
    else:
        custom_prompt = input("Enter your reference prompt/theme for the kids rhyme: ").strip()
        if not custom_prompt:
            raise ValueError("You must enter a prompt when video_id is not provided.")
        custom_title = input("Optional title for output folder (press Enter to auto-generate): ").strip()
        folder_title = custom_title or f"custom_rhyme_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        reference = f"User Prompt: {custom_prompt}"

    try:
        generated = _call_openai_for_rhyme(reference)
    except Exception:
        # Graceful fallback so user still gets usable output.
        generated = _fallback_output(seed_title=_sanitize_for_dir(folder_title), seed_text=reference)
    generated = _enforce_exact_scene_count(generated)

    output_dir = _save_result(Path(args.output_root), folder_title, generated, reference)
    print(f"Saved rhyme + scene prompts in: {output_dir}")


if __name__ == "__main__":
    main()
