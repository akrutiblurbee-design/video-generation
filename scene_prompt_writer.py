#!/usr/bin/env python3
"""
Generate scene prompts from a saved rhyme file (step 2 of prompt pipeline).
"""

from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path
from typing import Any

from google import genai
from google.genai import types


DEFAULT_OUTPUT_ROOT = "outputs"
DEFAULT_RHYME_PATH = str(Path("outputs") / "learn numbers" / "prompts" / "rhyme_only.json")
STRICT_SCENE_COUNT = 8


SYSTEM_PROMPT = """You are an expert storyboard prompt designer for premium 3D cartoon animation videos.

Your job:
1) Analyze the provided rhyme.
2) Create scene-by-scene prompts to generate visuals that perfectly match each rhyme beat.

Rules:
- Keep language simple, playful, and positive.
- Avoid scary, unsafe, violent, or sad content.
- Ensure visual continuity across scenes (same characters, costumes, props, and color style).
- Scenes must directly reflect rhyme lines.
- Prefer bright, high-saturation, kid-friendly visuals.
- The output must be explicitly for "3d animation cartoon" in a Cocomelon-like nursery-rhyme style.
- Never output live-action or photorealistic directions.
- Generate exactly 8 scenes (not fewer, not more).
- Each visual_prompt must be highly descriptive and production-ready, around 150-220 words.

Return ONLY valid JSON with this structure:
{
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


STYLE_ANCHOR = (
    'Style anchor: "3d animation cartoon", Cocomelon-like nursery-rhyme look; '
    "rounded preschool characters, bright pastel palette, soft daylight GI, clean toy-like textures."
)
DETAIL_APPENDIX = (
    "Expand this scene with concrete production details: include explicit foreground/midground/background "
    "layout, exact character placement and facial expression beats, shot size + angle + camera movement + "
    "lens feel, key/fill/rim lighting cues, material textures, and child-safe props tied directly to the rhyme moment."
)
ANTI_REALISM_GUARDRAIL = (
    "Hard visual guardrails: fully stylized 3d animation cartoon render only; "
    "not live-action, not photorealistic, not cinematic real-camera footage, "
    "not real humans, not real kids, not documentary look."
)


def _sanitize_for_dir(name: str) -> str:
    cleaned = re.sub(r"[<>:\"/\\|?*\x00-\x1F]", "", name)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    cleaned = cleaned.rstrip(".")
    return cleaned[:80] if cleaned else "untitled_video"


def _stylize_child_terms(text: str) -> str:
    replacements = {
        "two kids": "two stylized cartoon toddlers",
        "both kids": "both stylized cartoon toddlers",
        "three kids": "three stylized cartoon toddlers",
        "kids": "stylized cartoon children",
        "child": "stylized cartoon child",
        "children": "stylized cartoon children",
        "boy": "stylized cartoon toddler boy",
        "girl": "stylized cartoon toddler girl",
    }
    styled = text
    for source, target in replacements.items():
        styled = re.sub(rf"\b{re.escape(source)}\b", target, styled, flags=re.IGNORECASE)
    return styled


def _load_rhyme_payload(rhyme_file: Path) -> dict[str, Any]:
    if not rhyme_file.exists():
        raise FileNotFoundError(f"Rhyme file not found: {rhyme_file}")
    if rhyme_file.suffix.lower() == ".json":
        return json.loads(rhyme_file.read_text(encoding="utf-8"))

    text = rhyme_file.read_text(encoding="utf-8")
    title_match = re.search(r"^\s*Title:\s*(.+)$", text, flags=re.MULTILINE)
    rhyme_match = re.search(r"===\s*RHYME\s*===\s*(.*?)\s*(?:===|$)", text, flags=re.DOTALL | re.IGNORECASE)
    music_match = re.search(
        r"===\s*RHYME MUSIC STYLE PROMPT\s*===\s*(.*?)\s*(?:===|$)",
        text,
        flags=re.DOTALL | re.IGNORECASE,
    )
    return {
        "title": title_match.group(1).strip() if title_match else "",
        "rhyme": rhyme_match.group(1).strip() if rhyme_match else "",
        "rhyme_music_style_prompt": music_match.group(1).strip() if music_match else "",
    }


def _build_scene_request(reference: str, rhyme_payload: dict[str, Any]) -> str:
    return f"""Create scene prompts using this source:

Reference:
{reference}

Rhyme Title:
{rhyme_payload.get("title", "")}

Rhyme Text:
{rhyme_payload.get("rhyme", "")}

Constraints:
- Generate exactly 8 scenes.
- Every scene must map to the rhyme progression.
- Output is for "3d animation cartoon" in a Cocomelon-like style.
- Include rich scene details: environment layout, character poses/expressions, camera, lighting, palette, texture cues.
"""


def _parse_model_json_response(raw_text: str) -> dict[str, Any]:
    cleaned = re.sub(r"^```json\s*|```$", "", raw_text.strip(), flags=re.MULTILINE).strip()
    if not cleaned:
        raise RuntimeError("Gemini returned an empty response for scene generation.")

    try:
        parsed = json.loads(cleaned)
        if isinstance(parsed, dict):
            return parsed
        raise ValueError("Expected top-level JSON object from model response.")
    except json.JSONDecodeError:
        pass

    decoder = json.JSONDecoder()
    for match in re.finditer(r"[\{\[]", cleaned):
        candidate = cleaned[match.start() :]
        try:
            parsed, _ = decoder.raw_decode(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            return parsed
        if isinstance(parsed, list):
            return {"scene_prompts": parsed}

    raise ValueError("Unable to parse valid JSON payload from Gemini response.")


def _call_gemini_for_scenes(reference: str, rhyme_payload: dict[str, Any]) -> dict[str, Any]:
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
                        text=f"{SYSTEM_PROMPT}\n\n{_build_scene_request(reference, rhyme_payload)}"
                    )
                ],
            )
        ],
        config=types.GenerateContentConfig(
            temperature=0.3,
            response_mime_type="application/json",
        ),
    )
    raw = (response.text or "").strip()
    return _parse_model_json_response(raw)


def _enforce_exact_scene_count(payload: dict[str, Any]) -> dict[str, Any]:
    return _normalize_scene_count(payload, STRICT_SCENE_COUNT)


def _normalize_scene_count(payload: dict[str, Any], expected: int) -> dict[str, Any]:
    scenes = payload.get("scene_prompts")
    if not isinstance(scenes, list):
        raise ValueError("Model output missing 'scene_prompts' list.")
    expected = max(1, int(expected))
    if len(scenes) > expected:
        trimmed = scenes[:expected]
    else:
        trimmed = list(scenes)
        last = trimmed[-1] if trimmed else {}
        while len(trimmed) < expected:
            trimmed.append(dict(last) if isinstance(last, dict) else {})
    out = dict(payload)
    normalized_scenes: list[dict[str, Any]] = []
    for idx, scene in enumerate(trimmed, start=1):
        if isinstance(scene, dict):
            normalized_scenes.append({**scene, "scene_number": idx})
        else:
            normalized_scenes.append(
                {
                    "scene_number": idx,
                    "rhyme_line_reference": "",
                    "duration_seconds": 6,
                    "visual_prompt": str(scene),
                    "motion_and_action": "",
                    "continuity_notes": "",
                }
            )
    out["scene_prompts"] = normalized_scenes
    return out


def _enforce_detailed_scene_prompts(payload: dict[str, Any]) -> dict[str, Any]:
    scenes = payload.get("scene_prompts")
    if not isinstance(scenes, list):
        return payload
    out = dict(payload)
    normalized_scenes: list[dict[str, Any]] = []
    for scene in scenes:
        if not isinstance(scene, dict):
            normalized_scenes.append(scene)
            continue
        scene_copy = dict(scene)
        visual_prompt = _stylize_child_terms(str(scene_copy.get("visual_prompt", "")).strip())
        composed = visual_prompt
        if STYLE_ANCHOR.lower() not in composed.lower():
            composed = f"{STYLE_ANCHOR} {composed}".strip()
        if ANTI_REALISM_GUARDRAIL.lower() not in composed.lower():
            composed = f"{composed} {ANTI_REALISM_GUARDRAIL}".strip()
        if len(composed.split()) < 120:
            composed = f"{composed} {DETAIL_APPENDIX}".strip()
        scene_copy["visual_prompt"] = composed
        continuity_notes = str(scene_copy.get("continuity_notes", "")).strip()
        if continuity_notes:
            if "3d animation cartoon" not in continuity_notes.lower():
                continuity_notes = f"{continuity_notes} Keep the same 3d animation cartoon Cocomelon-like look across scenes."
            if "not photorealistic" not in continuity_notes.lower():
                continuity_notes = f"{continuity_notes} Keep characters stylized and not photorealistic."
            scene_copy["continuity_notes"] = continuity_notes
        else:
            scene_copy["continuity_notes"] = (
                "Keep consistent character designs, outfits, props, palette, and 3d animation cartoon "
                "Cocomelon-like rendering style; keep all characters stylized and not photorealistic."
            )
        normalized_scenes.append(scene_copy)
    out["scene_prompts"] = normalized_scenes
    return out


def _save_combined_prompts(
    output_root: Path,
    folder_title: str,
    rhyme_payload: dict[str, Any],
    scene_payload: dict[str, Any],
    reference: str,
) -> Path:
    target_dir = output_root / _sanitize_for_dir(folder_title) / "prompts"
    target_dir.mkdir(parents=True, exist_ok=True)

    final_payload = {
        "title": rhyme_payload.get("title", folder_title),
        "rhyme": rhyme_payload.get("rhyme", ""),
        "rhyme_music_style_prompt": rhyme_payload.get("rhyme_music_style_prompt", ""),
        "visual_style_guide": scene_payload.get("visual_style_guide", {}),
        "scene_prompts": scene_payload.get("scene_prompts", []),
    }

    json_path = target_dir / "rhyme_and_scene_prompts.json"
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(final_payload, f, indent=2, ensure_ascii=False)

    txt_path = target_dir / "rhyme_and_scene_prompts.txt"
    with txt_path.open("w", encoding="utf-8") as f:
        f.write(f"Title: {final_payload.get('title', '')}\n\n")
        f.write("=== RHYME ===\n")
        f.write(final_payload.get("rhyme", ""))
        f.write("\n\n=== RHYME MUSIC STYLE PROMPT ===\n")
        f.write(final_payload.get("rhyme_music_style_prompt", ""))
        f.write("\n\n=== VISUAL STYLE GUIDE ===\n")
        f.write(json.dumps(final_payload.get("visual_style_guide", {}), indent=2, ensure_ascii=False))
        f.write("\n\n=== SCENE PROMPTS ===\n")
        for scene in final_payload.get("scene_prompts", []):
            f.write(f"\nScene {scene.get('scene_number', '')}\n")
            f.write(f"Rhyme Reference: {scene.get('rhyme_line_reference', '')}\n")
            f.write(f"Duration: {scene.get('duration_seconds', '')}s\n")
            f.write(f"Visual Prompt: {scene.get('visual_prompt', '')}\n")
            f.write(f"Motion/Action: {scene.get('motion_and_action', '')}\n")
            f.write(f"Continuity: {scene.get('continuity_notes', '')}\n")

    (target_dir / "reference_input.txt").write_text(reference, encoding="utf-8")
    return target_dir


def _system_prompt_for_count(scene_count: int) -> str:
    n = max(1, int(scene_count))
    return SYSTEM_PROMPT.replace(
        "exactly 8 scenes (not fewer, not more).",
        f"exactly {n} scenes (not fewer, not more).",
    )


def _build_scene_user_for_studio(
    reference: str, rhyme_payload: dict[str, Any], scene_count: int, video_style: str
) -> str:
    vs = (video_style or "").strip() or (
        "3D cartoon nursery animation, bright, child-safe, high-saturation"
    )
    return f"""Create scene prompts using this source:

Reference:
{reference}

Rhyme Title:
{rhyme_payload.get("title", "")}

Rhyme Text:
{rhyme_payload.get("rhyme", "")}

Constraints:
- Generate exactly {scene_count} scenes.
- The user-chosen video style is: {vs!r}
- Every visual_prompt must BEGIN with an explicit style clause that includes those video-style words, then continue with the detailed shot description (do not rely on a separate style section alone).
- Scenes must map to the rhyme progression.
- Output is for "3d animation cartoon" in a Cocomelon-like style.
- Include rich scene details: environment layout, character poses/expressions, camera, lighting, palette, texture cues.
"""


def _call_gemini_scenes_studio(
    reference: str, rhyme_payload: dict[str, Any], scene_count: int, video_style: str
) -> dict[str, Any]:
    api_key = (os.getenv("GEMINI_API_KEY", "").strip() or os.getenv("GOOGLE_API_KEY", "").strip())
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY is missing in environment/.env.")
    model = os.getenv("GEMINI_TEXT_MODEL", "").strip() or "gemini-3.1-flash-lite-preview"
    client = genai.Client(api_key=api_key)
    system = _system_prompt_for_count(scene_count)
    user = _build_scene_user_for_studio(reference, rhyme_payload, scene_count, video_style)
    response = client.models.generate_content(
        model=model,
        contents=[
            types.Content(
                role="user",
                parts=[types.Part.from_text(text=f"{system}\n\n{user}")],
            )
        ],
        config=types.GenerateContentConfig(
            temperature=0.7,
            response_mime_type="application/json",
        ),
    )
    raw = (response.text or "").strip()
    return _parse_model_json_response(raw)


def _force_user_style_prefix(payload: dict[str, Any], video_style: str) -> dict[str, Any]:
    vs = (video_style or "").strip() or "Bright 3D cartoon nursery animation"
    out = dict(payload)
    lead = f"Visual style: {vs}. "
    new_scenes: list[Any] = []
    for sc in out.get("scene_prompts", []):
        if not isinstance(sc, dict):
            new_scenes.append(sc)
            continue
        c = dict(sc)
        vp = str(c.get("visual_prompt", "")).strip()
        low = vp.lower()
        if low.startswith(lead.strip().lower()[:40]):
            c["visual_prompt"] = vp
        elif vs.lower() in low[:220]:
            c["visual_prompt"] = vp
        else:
            c["visual_prompt"] = lead + vp
        new_scenes.append(c)
    out["scene_prompts"] = new_scenes
    return out


def generate_scene_prompts_for_api(
    *,
    rhyme_payload: dict[str, Any],
    scene_count: int,
    video_style: str,
    reference: str = "",
    output_root: Path | None = None,
    folder_title: str | None = None,
) -> dict[str, Any]:
    """Used by the Studio HTTP API: variable scene count and user video style baked into each scene."""
    root = output_root or Path(DEFAULT_OUTPUT_ROOT)
    n = max(1, min(int(scene_count), 24))
    ref = (reference or "").strip()
    if not ref:
        ref = (
            f"Rhyme title: {rhyme_payload.get('title', '')}\n"
            f"Rhyme:\n{rhyme_payload.get('rhyme', '')}\n"
        )

    scene_payload = _call_gemini_scenes_studio(ref, rhyme_payload, n, video_style)

    scene_payload = _normalize_scene_count(scene_payload, n)
    scene_payload = _enforce_detailed_scene_prompts(scene_payload)
    scene_payload = _force_user_style_prefix(scene_payload, video_style)

    title = folder_title or rhyme_payload.get("title") or "studio_session"
    safe = _sanitize_for_dir(str(title))
    out_dir = _save_combined_prompts(root, safe, rhyme_payload, scene_payload, ref)

    scenes_for_response: list[dict[str, Any]] = []
    for sc in scene_payload.get("scene_prompts", []):
        if not isinstance(sc, dict):
            continue
        num = sc.get("scene_number")
        scenes_for_response.append(
            {
                "scene_number": num,
                "title": f"Scene {num}",
                "rhyme_line_reference": sc.get("rhyme_line_reference", ""),
                "duration_seconds": sc.get("duration_seconds"),
                "description": sc.get("visual_prompt", ""),
                "motion_and_action": sc.get("motion_and_action", ""),
                "continuity_notes": sc.get("continuity_notes", ""),
            }
        )

    return {
        "scenes": scenes_for_response,
        "output_prompts_dir": str(out_dir),
        "scene_count": len(scenes_for_response),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Step 2: generate scene prompts from saved rhyme")
    parser.add_argument("--rhyme-file", type=str, default=DEFAULT_RHYME_PATH, help="Path to rhyme_only.json or rhyme_only.txt")
    parser.add_argument("--output-root", type=str, default=DEFAULT_OUTPUT_ROOT, help="Root output directory")
    parser.add_argument("--reference-file", type=str, default=None, help="Optional reference_input.txt path")
    args = parser.parse_args()

    rhyme_file = Path(args.rhyme_file)
    rhyme_payload = _load_rhyme_payload(rhyme_file)
    folder_title = rhyme_payload.get("title") or rhyme_file.parent.parent.name

    reference = ""
    if args.reference_file:
        reference = Path(args.reference_file).read_text(encoding="utf-8").strip()
    else:
        default_ref = rhyme_file.parent / "reference_input.txt"
        if default_ref.exists():
            reference = default_ref.read_text(encoding="utf-8").strip()

    scene_payload = _call_gemini_for_scenes(reference, rhyme_payload)

    scene_payload = _enforce_exact_scene_count(scene_payload)
    scene_payload = _enforce_detailed_scene_prompts(scene_payload)
    output_dir = _save_combined_prompts(
        output_root=Path(args.output_root),
        folder_title=str(folder_title),
        rhyme_payload=rhyme_payload,
        scene_payload=scene_payload,
        reference=reference,
    )
    print(f"Saved rhyme + scene prompts in: {output_dir}")


if __name__ == "__main__":
    main()
