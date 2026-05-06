#!/usr/bin/env python3
"""
Generate 50-60s background music from rhyme prompt text using Gemini generate_content.

Lyria 3 Clip (lyria-3-clip-preview) always outputs ~30s; use Lyria 3 Pro for 50–60s.
"""

from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path

from dotenv import load_dotenv
from google import genai
from google.genai import errors as genai_errors
from google.genai import types


DEFAULT_PROMPTS_PATH = str(
    Path("outputs") / "wheels on the bus" / "prompts" / "rhyme_and_scene_prompts.txt"
)
DEFAULT_OUTPUT_ROOT = "outputs"
DEFAULT_MODEL = "lyria-3-pro-preview"
DEFAULT_API_VERSION = "v1"
FALLBACK_API_VERSIONS = ["v1beta", "v1alpha", "v1"]
MODEL_CANDIDATES = [
    "lyria-3-pro-preview",
    "lyria-3-clip-preview",
    "lyria-002",
]


def _sanitize_for_dir(name: str) -> str:
    cleaned = re.sub(r"[<>:\"/\\|?*\x00-\x1F]", "", name)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    cleaned = cleaned.rstrip(".")
    return cleaned[:80] if cleaned else "untitled_video"


def _get_api_key_from_env() -> str:
    candidates = [
        os.getenv("GEMINI_API_KEY", "").strip(),
        os.getenv("GOOGLE_API_KEY", "").strip(),
    ]
    api_key = next((value for value in candidates if value), "")
    if not api_key:
        raise RuntimeError("Missing API key. Set GEMINI_API_KEY (or GOOGLE_API_KEY) in .env.")
    return api_key


def _extract_section(text: str, section_name: str) -> str:
    section_match = re.search(
        rf"===\s*{re.escape(section_name)}\s*===\s*(.*?)\s*(?:===|$)",
        text,
        flags=re.DOTALL | re.IGNORECASE,
    )
    return section_match.group(1).strip() if section_match and section_match.group(1).strip() else ""


def _extract_music_inputs(prompts_file: Path) -> tuple[str, str]:
    if not prompts_file.exists():
        raise FileNotFoundError(f"Prompt file not found: {prompts_file}")

    text = prompts_file.read_text(encoding="utf-8")
    rhyme_text = _extract_section(text, "RHYME")
    music_style_prompt = _extract_section(text, "RHYME MUSIC STYLE PROMPT")

    if not music_style_prompt:
        line_match = re.search(r"^\s*Rhyme Music Style Prompt:\s*(.+)$", text, flags=re.MULTILINE)
        if line_match:
            music_style_prompt = line_match.group(1).strip()

    if not music_style_prompt:
        raise ValueError(
            "Music style prompt not found. Ensure prompt writer output contains "
            "'=== RHYME MUSIC STYLE PROMPT ==='."
        )
    if not rhyme_text:
        raise ValueError("Rhyme lyrics not found. Ensure prompt file contains '=== RHYME ==='.")
    return rhyme_text, music_style_prompt


def _collect_audio_parts(response: types.GenerateContentResponse) -> tuple[bytes, str]:
    collected_bytes: list[bytes] = []
    mime_type = "audio/mp3"
    checked_parts = 0

    def _collect_from_parts(parts: object) -> None:
        nonlocal mime_type, checked_parts
        for part in parts or []:
            checked_parts += 1
            inline_data = getattr(part, "inline_data", None)
            if inline_data and getattr(inline_data, "data", None):
                collected_bytes.append(inline_data.data)
                if getattr(inline_data, "mime_type", None):
                    mime_type = inline_data.mime_type

    # Some SDK versions expose top-level convenience parts.
    _collect_from_parts(getattr(response, "parts", None))

    # In many responses, audio arrives via candidates[*].content.parts.
    for candidate in getattr(response, "candidates", None) or []:
        content = getattr(candidate, "content", None)
        _collect_from_parts(getattr(content, "parts", None))

    if not collected_bytes:
        finish_reasons = [
            str(getattr(candidate, "finish_reason", "unknown"))
            for candidate in (getattr(response, "candidates", None) or [])
        ]
        response_text = (getattr(response, "text", "") or "").strip()
        text_hint = f" Model text response: {response_text[:250]!r}" if response_text else ""
        raise RuntimeError(
            "Lyria response did not include audio inline_data "
            f"(checked {checked_parts} parts; finish_reasons={finish_reasons})."
            f"{text_hint} Check model access/quota or try another prompt."
        )
    return b"".join(collected_bytes), mime_type


def _write_debug_response_snapshot(
    *, response: types.GenerateContentResponse | None, output_dir: Path, reason: str
) -> Path | None:
    output_dir.mkdir(parents=True, exist_ok=True)
    debug_file = output_dir / "music_generation_debug.json"
    payload: dict[str, object] = {"reason": reason}

    if response is None:
        payload["response"] = None
    else:
        try:
            # Pydantic model dump when available.
            payload["response"] = response.model_dump(exclude_none=True)
        except Exception:
            payload["response"] = str(response)

        prompt_feedback = getattr(response, "prompt_feedback", None)
        if prompt_feedback is not None:
            try:
                prompt_feedback = prompt_feedback.model_dump(exclude_none=True)
            except Exception:
                prompt_feedback = str(prompt_feedback)

        payload["summary"] = {
            "has_top_level_parts": bool(getattr(response, "parts", None)),
            "candidate_count": len(getattr(response, "candidates", None) or []),
            "prompt_feedback": prompt_feedback,
            "text_preview": (getattr(response, "text", "") or "")[:300],
        }

    debug_file.write_text(
        json.dumps(payload, indent=2, ensure_ascii=True, default=str),
        encoding="utf-8",
    )
    return debug_file


def _extension_for_mime(mime_type: str) -> str:
    mime = (mime_type or "").lower()
    if "mpeg" in mime or "mp3" in mime:
        return "mp3"
    if "wav" in mime or "x-wav" in mime or "wave" in mime:
        return "wav"
    if "ogg" in mime:
        return "ogg"
    if "aac" in mime:
        return "aac"
    return "bin"


def _generate_music_bytes(
    *,
    api_key: str,
    api_version: str,
    model: str,
    lyrics: str,
    style_prompt: str,
    target_seconds: int,
    output_dir: Path,
) -> tuple[bytes, str]:
    client = genai.Client(
        api_key=api_key,
        http_options=types.HttpOptions(api_version=api_version),
    )
    final_prompt = (
        "Create a complete children's song using the exact lyrics provided below.\n\n"
        f"Style guidance:\n{style_prompt.strip()}\n\n"
        f"Lyrics to sing (keep wording exactly as written):\n{lyrics.strip()}\n\n"
        f"Requirements: generate exactly {target_seconds} seconds of audio (do not exceed), "
        "STRICT intro rule: instrumental intro must be ONLY 3-4 seconds, then vocals must begin immediately "
        "at or before second 4. Do not create a longer intro. "
        "Arrangement timeline: 0-4s soft intro only, 4s onward full song with vocals and rhythm. "
        "No long instrumental opening, clear vocal melody, playful arrangement, and a clear musical ending "
        f"right at {target_seconds} seconds."
    )

    response: types.GenerateContentResponse | None = None
    try:
        response = client.models.generate_content(
            model=model,
            contents=final_prompt,
            config=types.GenerateContentConfig(
                response_modalities=["AUDIO", "TEXT"],
            ),
        )
    except genai_errors.ClientError as exc:
        message = str(exc)
        if "429" in message and "spending cap" in message.lower():
            raise RuntimeError(
                "Gemini API rejected music generation because your project monthly spending cap "
                "is reached. Increase/reset project cap at https://ai.studio/spend, or switch to "
                "another API key/project with available quota."
            ) from exc
        if "not found" in message.lower() or "not supported" in message.lower():
            raise RuntimeError(
                f"Model '{model}' is not available for this API key/project/version. "
                "Use a key/project with Lyria access or change the model."
            ) from exc
        raise

    prompt_feedback = getattr(response, "prompt_feedback", None)
    block_reason = getattr(prompt_feedback, "block_reason", None)
    if block_reason:
        debug_file = _write_debug_response_snapshot(
            response=response,
            output_dir=output_dir,
            reason=f"prompt_blocked_{block_reason}",
        )
        raise RuntimeError(
            "Music generation was blocked by Gemini prompt feedback "
            f"(block_reason={block_reason}). Debug details saved to: {debug_file}. "
            "Try adjusting prompt wording, switching to another API key/project, or checking quota/billing."
        )

    try:
        return _collect_audio_parts(response)
    except RuntimeError as exc:
        debug_file = _write_debug_response_snapshot(
            response=response,
            output_dir=output_dir,
            reason="no_audio_inline_data",
        )
        raise RuntimeError(f"{exc} Debug details saved to: {debug_file}") from exc


def _model_supports_generate_content(model_obj: object) -> bool:
    methods = getattr(model_obj, "supported_actions", None) or getattr(
        model_obj, "supported_generation_methods", None
    )
    if not methods:
        return True
    methods_lower = [str(method).lower() for method in methods]
    return any("generatecontent" in method for method in methods_lower)


def _normalize_model_name(name: str) -> str:
    return name.split("/", 1)[1] if name.startswith("models/") else name


def _discover_music_models(client: genai.Client) -> list[str]:
    discovered: list[str] = []
    for model_obj in client.models.list():
        raw_name = getattr(model_obj, "name", None)
        if not raw_name:
            continue
        name = _normalize_model_name(str(raw_name))
        lowered = name.lower()
        if "lyria" not in lowered and "music" not in lowered:
            continue
        if not _model_supports_generate_content(model_obj):
            continue
        discovered.append(name)
    return discovered


def _resolve_model(client: genai.Client, requested_model: str) -> tuple[str, list[str], bool]:
    available = _discover_music_models(client)
    if not available:
        return requested_model, available, False

    requested_normalized = _normalize_model_name(requested_model)
    available_set = {name.lower() for name in available}
    if requested_normalized.lower() in available_set:
        return requested_normalized, available, False

    for candidate in MODEL_CANDIDATES:
        if candidate.lower() in available_set:
            return candidate, available, True

    return available[0], available, True


def _resolve_client_model_and_version(
    *,
    api_key: str,
    requested_api_version: str,
    requested_model: str,
) -> tuple[genai.Client, str, str, list[str], bool, bool]:
    checked_versions: list[str] = [requested_api_version]
    checked_versions.extend(v for v in FALLBACK_API_VERSIONS if v != requested_api_version)

    first_client = genai.Client(
        api_key=api_key,
        http_options=types.HttpOptions(api_version=checked_versions[0]),
    )
    first_model, first_available, first_used_fallback = _resolve_model(first_client, requested_model)
    if first_available:
        return (
            first_client,
            checked_versions[0],
            first_model,
            first_available,
            first_used_fallback,
            False,
        )

    for api_version in checked_versions[1:]:
        client = genai.Client(
            api_key=api_key,
            http_options=types.HttpOptions(api_version=api_version),
        )
        model, available, used_fallback = _resolve_model(client, requested_model)
        if available:
            return client, api_version, model, available, used_fallback, True

    return (
        first_client,
        checked_versions[0],
        first_model,
        first_available,
        first_used_fallback,
        False,
    )


def _infer_video_title_from_prompts_file(prompts_file: Path) -> str:
    prompts_dir = prompts_file.parent
    if prompts_dir.name == "prompts":
        return prompts_dir.parent.name
    return prompts_dir.name


def _build_output_dir(root: Path, video_title: str) -> Path:
    target = root / video_title / "generated_music"
    target.mkdir(parents=True, exist_ok=True)
    return target


def main() -> None:
    load_dotenv()

    parser = argparse.ArgumentParser(description="Generate music from rhyme music style prompt")
    parser.add_argument("--prompts-file", type=str, default=DEFAULT_PROMPTS_PATH)
    parser.add_argument("--output-root", type=str, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--model", type=str, default=os.getenv("GEMINI_MUSIC_MODEL", DEFAULT_MODEL))
    parser.add_argument(
        "--api-version",
        type=str,
        default=os.getenv("GEMINI_API_VERSION", DEFAULT_API_VERSION),
        help="Gemini API version for generate_content endpoint (recommended: v1).",
    )
    parser.add_argument(
        "--target-seconds",
        type=int,
        default=60,
        help="Target audio duration in seconds (strictly 60).",
    )
    args = parser.parse_args()

    if args.target_seconds != 60:
        raise ValueError("Only 60-second output is supported. Use --target-seconds 60.")

    prompts_file = Path(args.prompts_file)
    rhyme_lyrics, music_prompt = _extract_music_inputs(prompts_file)
    api_key = _get_api_key_from_env()
    video_title = _sanitize_for_dir(_infer_video_title_from_prompts_file(prompts_file))
    output_dir = _build_output_dir(Path(args.output_root), video_title)
    (
        _client,
        selected_api_version,
        selected_model,
        available_models,
        used_model_fallback,
        used_version_fallback,
    ) = _resolve_client_model_and_version(
        api_key=api_key,
        requested_api_version=args.api_version,
        requested_model=args.model,
    )

    if available_models:
        preview = ", ".join(available_models[:8])
        suffix = " ..." if len(available_models) > 8 else ""
        print(f"Detected music-capable models: {preview}{suffix}")
    else:
        print("Could not discover any music-capable model from ListModels for this key/version.")

    if used_version_fallback:
        print(
            f"Requested API version '{args.api_version}' has no music-capable models. "
            f"Falling back to '{selected_api_version}'."
        )

    if used_model_fallback:
        print(
            f"Requested model '{args.model}' is unavailable. "
            f"Falling back to '{selected_model}'."
        )

    print(f"Using model: {selected_model}")
    if "lyria-3-clip" in selected_model.lower() and args.target_seconds > 30:
        print(
            "Note: Lyria 3 Clip always returns ~30s audio. For ~50–60s tracks, use "
            "--model lyria-3-pro-preview (or set GEMINI_MUSIC_MODEL)."
        )
    print(f"Using API version: {selected_api_version}")
    print(f"Reading prompt from: {prompts_file}")
    print(f"Lyrics:\n{rhyme_lyrics}")
    print(f"Music prompt: {music_prompt}")

    audio_bytes, mime_type = _generate_music_bytes(
        api_key=api_key,
        api_version=selected_api_version,
        model=selected_model,
        lyrics=rhyme_lyrics,
        style_prompt=music_prompt,
        target_seconds=args.target_seconds,
        output_dir=output_dir,
    )

    ext = _extension_for_mime(mime_type)
    output_audio = output_dir / f"music_{args.target_seconds}s.{ext}"
    output_audio.write_bytes(audio_bytes)
    full_prompt = (
        f"Style guidance:\n{music_prompt}\n\n"
        f"Lyrics:\n{rhyme_lyrics}\n\n"
        f"Target duration: {args.target_seconds}s"
    )
    (output_dir / "music_prompt.txt").write_text(full_prompt, encoding="utf-8")
    (output_dir / "music_mime_type.txt").write_text(mime_type, encoding="utf-8")
    print(f"Saved generated music: {output_audio}")


if __name__ == "__main__":
    main()
