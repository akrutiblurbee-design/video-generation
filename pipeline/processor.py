"""
Steps 5–14 — Process each short:
  5.  Loop each short
  6.  Download MP4 + extract SRT captions
  7.  Parse download result
  8.  Needs transcription?
  9a. Whisper transcription
  9b. Use existing captions
  10. (optional) FFmpeg frame extraction
  11. Build Gemini request with video URL
  12. Gemini JSON analysis
  13. Parse AI analysis + compute viral score
  14. (removed) Persist output via JSON in main pipeline
"""

import asyncio
import base64
import json
import logging
import os
import re
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from google import genai

from config import Config

log = logging.getLogger("processor")

# Ensure download directory exists
os.makedirs(Config.DOWNLOAD_DIR, exist_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
# Public entry point
# ─────────────────────────────────────────────────────────────────────────────

async def process_shorts(shorts: list[dict]) -> list[dict]:
    """Process all shorts sequentially (respects rate limits, avoids OOM)."""
    processed = []
    for i, short in enumerate(shorts, 1):
        log.info("[%d/%d] Processing: %s", i, len(shorts), short.get("title", short.get("video_id")))
        try:
            result = await _process_one(short)
            processed.append(result)
        except asyncio.CancelledError:
            # Do not convert cancellation into successful-looking rows.
            raise
        except Exception as e:
            log.exception("Failed to process %s", short.get("video_id"))
            failed_short = dict(short)
            failed_short.update(_empty_analysis(short=failed_short))
            failed_short["transcript"] = ""
            failed_short["transcript_source"] = "none"
            failed_short["transcript_available"] = False
            failed_short["transcript_char_count"] = 0
            failed_short["views_per_hour"] = _compute_views_per_hour(failed_short)
            failed_short["viral_score"] = _compute_viral_score(failed_short)
            failed_short["engagement_rate"] = _compute_engagement_rate(failed_short)
            failed_short["error"] = f"{e.__class__.__name__}: {str(e).strip()}" if str(e).strip() else e.__class__.__name__
            processed.append(failed_short)
    return processed


# ─────────────────────────────────────────────────────────────────────────────
# Step 5-6 — Download + captions
# ─────────────────────────────────────────────────────────────────────────────

async def _process_one(short: dict) -> dict:
    video_id = short["video_id"]
    output_template = os.path.join(Config.DOWNLOAD_DIR, f"{video_id}.%(ext)s")

    # Step 6 — Download MP4 + auto-subtitles
    download_result = await _download_video(short["url"], output_template, video_id)

    # Step 7 — Parse download result
    mp4_path = download_result.get("mp4_path")
    srt_path = download_result.get("srt_path")
    has_captions = bool(srt_path and os.path.exists(srt_path))

    # Step 8/9 — Transcription
    transcript_source = "none"
    if has_captions:
        log.info("[%s] Step 9b: Using existing captions.", video_id)
        transcript = _parse_srt(srt_path)
        transcript_source = "captions"
    elif mp4_path and os.path.exists(mp4_path):
        log.info("[%s] Step 9a: Running Whisper transcription.", video_id)
        transcript = await _whisper_transcribe(mp4_path)
        transcript_source = "whisper"
    else:
        log.warning("[%s] No video or captions — skipping transcription.", video_id)
        transcript = ""

    # Step 11-12 — Gemini video analysis (URL-based)
    log.info("[%s] Step 11-12: Running Gemini video analysis.", video_id)
    ai_analysis = await _gemini_video_analysis(short, transcript)

    # Step 13 — Parse AI analysis + compute viral score
    short.update(ai_analysis)
    short["transcript"] = transcript
    short["transcript_source"] = transcript_source
    short["transcript_available"] = bool(transcript.strip())
    short["transcript_char_count"] = len(transcript)
    short["views_per_hour"] = _compute_views_per_hour(short)
    short["viral_score"] = _compute_viral_score(short)
    short["engagement_rate"] = _compute_engagement_rate(short)

    # Cleanup downloaded files
    if not Config.KEEP_DOWNLOADED_MEDIA:
        _cleanup(mp4_path, srt_path)

    return short


# ─────────────────────────────────────────────────────────────────────────────
# Step 6 — yt-dlp download
# ─────────────────────────────────────────────────────────────────────────────

async def _download_video(url: str, output_template: str, video_id: str) -> dict:
    cmd = [
        "yt-dlp",
        "--extractor-args", "youtube:player_client=android,web,ios",
        "--format", "bestvideo[ext=mp4][height<=720]+bestaudio[ext=m4a]/best[ext=mp4]/best",
        "--merge-output-format", "mp4",
        "--write-auto-sub",
        "--sub-lang", "en",
        "--sub-format", "srt",
        "--convert-subs", "srt",
        "--output", output_template,
        "--no-playlist",
        url,
    ]

    try:
        returncode, stdout, stderr = await _run_subprocess_capture(cmd)
    except FileNotFoundError:
        log.warning(
            "yt-dlp not found. Install yt-dlp and ensure it is available in PATH."
        )
        return {"mp4_path": None, "srt_path": None, "raw_output": "yt-dlp command not found"}
    output_text = stdout.decode(errors="replace") + stderr.decode(errors="replace")

    if returncode != 0:
        log.warning(
            "[%s] yt-dlp failed (exit=%d): %s",
            video_id,
            returncode,
            output_text[-500:],
        )

    # Scan for actual output files
    base = os.path.join(Config.DOWNLOAD_DIR, video_id)
    mp4_path = f"{base}.mp4" if os.path.exists(f"{base}.mp4") else None
    srt_path = next(
        (p for p in Path(Config.DOWNLOAD_DIR).glob(f"{video_id}*.srt")), None
    )
    srt_path = str(srt_path) if srt_path else None

    if not mp4_path and not srt_path:
        log.warning(
            "[%s] Download produced no MP4/SRT. yt-dlp output tail: %s",
            video_id,
            output_text[-500:],
        )

    return {
        "mp4_path": mp4_path,
        "srt_path": srt_path,
        "raw_output": output_text[-500:],
    }


# ─────────────────────────────────────────────────────────────────────────────
# Step 9a — Whisper
# ─────────────────────────────────────────────────────────────────────────────

async def _whisper_transcribe(mp4_path: str) -> str:
    """Run openai-whisper CLI in a subprocess."""
    out_dir = os.path.dirname(mp4_path)
    cmd = [
        sys.executable,
        "-m",
        "whisper",
        mp4_path,
        "--model", Config.WHISPER_MODEL,
        "--output_format", "txt",
        "--output_dir", out_dir,
        "--language", "en",
        "--fp16", "False",
    ]

    try:
        returncode, _, stderr = await _run_subprocess_capture(cmd)
    except FileNotFoundError:
        log.warning(
            "Whisper CLI not found. Install with `pip install openai-whisper` "
            "or ensure `whisper` is available in PATH."
        )
        return ""

    if returncode != 0:
        log.warning("Whisper error: %s", stderr.decode(errors="replace")[-300:])
        return ""

    txt_path = mp4_path.replace(".mp4", ".txt")
    if os.path.exists(txt_path):
        with open(txt_path) as f:
            return f.read().strip()
    return ""


# ─────────────────────────────────────────────────────────────────────────────
# Step 9b — Parse SRT
# ─────────────────────────────────────────────────────────────────────────────

def _parse_srt(srt_path: str) -> str:
    """Strip SRT timestamps and indices, return clean text."""
    with open(srt_path, encoding="utf-8", errors="replace") as f:
        content = f.read()

    # Remove sequence numbers and timestamps
    content = re.sub(r"^\d+\s*$", "", content, flags=re.MULTILINE)
    content = re.sub(r"\d{2}:\d{2}:\d{2},\d{3} --> \d{2}:\d{2}:\d{2},\d{3}", "", content)
    content = re.sub(r"<[^>]+>", "", content)  # Remove HTML tags in SRT
    lines = [ln.strip() for ln in content.splitlines() if ln.strip()]
    return " ".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# Step 10 — FFmpeg frame extraction
# ─────────────────────────────────────────────────────────────────────────────

async def _extract_frames(mp4_path: str, video_id: str) -> list[str]:
    """Extract up to MAX_FRAMES frames at FRAMES_PER_SECOND rate, return base64 list."""
    frames_dir = os.path.join(Config.DOWNLOAD_DIR, f"frames_{video_id}")
    os.makedirs(frames_dir, exist_ok=True)
    ffmpeg_cmd = _resolve_ffmpeg_cmd()

    cmd = [
        ffmpeg_cmd,
        "-i", mp4_path,
        "-vf", f"fps={Config.FRAMES_PER_SECOND},scale={Config.FRAME_WIDTH}:-1",
        "-frames:v", str(Config.MAX_FRAMES),
        "-q:v", "2",
        os.path.join(frames_dir, "frame_%03d.jpg"),
        "-y",
        "-loglevel", "error",
    ]

    try:
        returncode, _, stderr = await _run_subprocess_capture(cmd)
    except FileNotFoundError:
        log.warning(
            "FFmpeg CLI not found. Install ffmpeg and ensure it is available in PATH."
        )
        return []

    if returncode != 0:
        log.warning("FFmpeg error for %s: %s", video_id, stderr.decode(errors="replace")[-200:])
        return []

    frame_files = sorted(Path(frames_dir).glob("frame_*.jpg"))[: Config.MAX_FRAMES]
    frames_b64 = []
    for fp in frame_files:
        with open(fp, "rb") as f:
            frames_b64.append(base64.b64encode(f.read()).decode())

    # Cleanup frame dir
    import shutil
    shutil.rmtree(frames_dir, ignore_errors=True)

    return frames_b64


def _resolve_ffmpeg_cmd() -> str:
    """
    Resolve ffmpeg executable across environments.
    Prioritises PATH, then common winget install location on Windows.
    """
    in_path = shutil.which("ffmpeg")
    if in_path:
        return in_path

    local_app_data = os.getenv("LOCALAPPDATA")
    if local_app_data:
        winget_pkg_dir = Path(local_app_data) / "Microsoft" / "WinGet" / "Packages"
        if winget_pkg_dir.exists():
            matches = sorted(
                winget_pkg_dir.glob(
                    "Gyan.FFmpeg_*/ffmpeg-*/bin/ffmpeg.exe"
                )
            )
            if matches:
                return str(matches[-1])

    return "ffmpeg"


async def _run_subprocess_capture(cmd: list[str]) -> tuple[int, bytes, bytes]:
    """
    Run subprocess commands across platforms/event loops.
    On some Windows event loops, asyncio subprocess transport is unavailable.
    """
    try:
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await proc.communicate()
        return proc.returncode, stdout or b"", stderr or b""
    except NotImplementedError:
        log.warning(
            "Active event loop does not support asyncio subprocesses; "
            "falling back to threaded subprocess execution."
        )
        completed = await asyncio.to_thread(
            subprocess.run,
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
        return completed.returncode, completed.stdout or b"", completed.stderr or b""


# ─────────────────────────────────────────────────────────────────────────────
# Steps 11-12 — Gemini Video Analysis
# ─────────────────────────────────────────────────────────────────────────────

ANALYSIS_PROMPT = """You are a senior viral-content strategist and visual storyteller.
Analyse this YouTube Short deeply using the transcript and metadata. The video URL is provided.
Be specific and concrete. Avoid generic filler language.
Return ONLY valid JSON.

Video metadata:
- Title: {title}
- Channel: {channel}
- Views: {views}
- Likes: {likes}
- Comments: {comments}
- Duration: {duration}s
- Upload date: {upload_date}
- Video URL: {video_url}

Transcript:
{transcript}

Quality requirements:
- Infer a clear timeline from opening hook to final payoff.
- Mention specific visual elements (characters, props, setting shifts, on-screen text, camera behavior).
- Explain why the edit rhythm and story beats help retention.
- If something is uncertain, state the most likely interpretation from available evidence.

Return ONLY this JSON structure (no markdown, no explanation):
{{
  "scene_description": "6-10 sentence visual summary with subject, action, setting, color/style, and progression",
  "detailed_video_analysis": {{
    "story_arc": "Describe beginning, middle, and end of this short in detail (6-10 sentences)",
    "visual_timeline": [
      "Beat 1 [0-3s]: what appears on screen, what changes, and why it matters",
      "Beat 2 [3-7s]: key visual/audio shift and retention effect",
      "Beat 3 [7-12s]: escalation and audience expectation",
      "Beat 4 [12-20s]: payoff/reveal and emotional response"
    ],
    "shot_composition": "Camera angle/style or animation framing pattern throughout (include shot types and movement where possible)",
    "editing_and_pacing": "Cut rhythm, transitions, movement speed, and retention pattern with concrete pacing notes",
    "on_screen_text_and_graphics": "Any text overlays, captions style, emojis, graphics, and where they appear",
    "audio_and_voice_pattern": "Music style, vocals/voiceover cadence, and sync with visuals",
    "recreation_blueprint": "Step-by-step production blueprint to recreate a similar short (at least 6 actionable steps)"
  }},
  "hook": "The opening hook line or concept in ≤15 words",
  "hook_style": "one of: question | shock | story | tutorial | challenge | transformation | comedy | controversy",
  "niche": "one of: tech | fitness | finance | comedy | food | travel | education | gaming | beauty | motivation | lifestyle | other",
  "format": "one of: talking-head | voiceover | text-on-screen | POV | tutorial | reaction | montage | vlog | animation",
  "primary_emotion": "one of: awe | curiosity | humor | inspiration | anger | shock | nostalgia | fear | joy | disgust",
  "pacing": "one of: fast | medium | slow",
  "has_trending_audio": true or false,
  "cta_present": true or false,
  "estimated_retention": "a percentage estimate like 78%",
  "strengths": ["up to 3 short bullet points"],
  "weaknesses": ["up to 2 short bullet points"],
  "replication_tips": "1-2 actionable tips to replicate virality"
}}"""


def _get_gemini_api_key() -> str:
    return (Config.GEMINI_API_KEY or "").strip()


async def _gemini_video_analysis(short: dict, transcript: str) -> dict:
    gemini_api_key = _get_gemini_api_key()
    if not gemini_api_key:
        log.warning("No Gemini API key — skipping analysis.")
        return _empty_analysis(short=short)

    client = genai.Client(api_key=gemini_api_key)
    model = (
        (Config.GEMINI_VIDEO_ANALYSIS_MODEL or "").strip()
        or (Config.GEMINI_TEXT_MODEL or "").strip()
        or "gemini-3.1-flash-lite-preview"
    )
    prompt = ANALYSIS_PROMPT.format(
        title=short.get("title", ""),
        channel=short.get("channel", ""),
        views=short.get("views", 0),
        likes=short.get("likes", 0),
        comments=short.get("comments", 0),
        duration=short.get("duration", 0),
        upload_date=short.get("upload_date", ""),
        video_url=short.get("url", ""),
        transcript=transcript[:3000] if transcript else "No transcript available.",
    )

    last_error = None
    for attempt in range(1, 4):
        try:
            response = await asyncio.to_thread(
                client.models.generate_content,
                model=model,
                contents=prompt,
            )
            raw = (response.text or "").strip()
            # Strip markdown fences if present
            raw = re.sub(r"^```json\s*|```$", "", raw, flags=re.MULTILINE).strip()
            return json.loads(raw)
        except asyncio.CancelledError:
            # Respect cooperative cancellation (for Ctrl+C or task cancellation).
            raise
        except json.JSONDecodeError as e:
            log.error("Gemini analysis returned non-JSON output: %s", e)
            return _empty_analysis(short=short)
        except Exception as e:
            last_error = e
            if attempt < 3:
                backoff_seconds = attempt * 2
                log.warning(
                    "Gemini request failed on attempt %d/3 (%s). Retrying in %ds.",
                    attempt,
                    e.__class__.__name__,
                    backoff_seconds,
                )
                await asyncio.sleep(backoff_seconds)
                continue
            break

    log.error("Gemini analysis failed after retries: %s", last_error)
    return _empty_analysis(short=short)


def _empty_analysis(short: Optional[dict] = None) -> dict:
    fallback_scene_description = _build_fallback_scene_description(short or {})
    return {
        "scene_description": fallback_scene_description,
        "detailed_video_analysis": {
            "story_arc": "Unable to generate full scene analysis for this video in this run.",
            "visual_timeline": ["Beat 1: Metadata-only fallback, frame timeline unavailable."],
            "shot_composition": "",
            "editing_and_pacing": "",
            "on_screen_text_and_graphics": "",
            "audio_and_voice_pattern": "",
            "recreation_blueprint": "Retry processing when video/captions/Gemini are available.",
        },
        "hook": "Watch this short's key moment.",
        "hook_style": "unknown",
        "niche": "other",
        "format": "unknown",
        "primary_emotion": "unknown",
        "pacing": "medium",
        "has_trending_audio": False,
        "cta_present": False,
        "estimated_retention": "0%",
        "strengths": [],
        "weaknesses": [],
        "replication_tips": "",
    }


def _build_fallback_scene_description(short: dict) -> str:
    title = str(short.get("title", "")).strip()
    source_description = str(short.get("description", "")).strip()
    if source_description:
        snippet = source_description.replace("\n", " ").strip()
        if len(snippet) > 320:
            snippet = snippet[:317].rstrip() + "..."
        return snippet
    if title:
        return (
            "Animated short likely centered around: "
            f"{title}. Full visual scene description was unavailable in this run."
        )
    return "Scene description unavailable for this video in the current run."


# ─────────────────────────────────────────────────────────────────────────────
# Step 13 — Viral score + engagement rate
# ─────────────────────────────────────────────────────────────────────────────

def _compute_viral_score(short: dict) -> float:
    """
    viral_score = (views / hours_since_posted) + (likes * LIKE_WEIGHT) + (comments * COMMENT_WEIGHT)
    """
    views_per_hour = short.get("views_per_hour")
    if views_per_hour is None:
        views_per_hour = _compute_views_per_hour(short)
    likes = short.get("likes", 0)
    comments = short.get("comments", 0)

    return round(
        views_per_hour
        + (likes * Config.LIKE_WEIGHT)
        + (comments * Config.COMMENT_WEIGHT),
        2,
    )


def _compute_views_per_hour(short: dict) -> float:
    """Estimate growth speed from upload timestamp and current views."""
    views = short.get("views", 0) or 0
    upload_dt_str = short.get("upload_datetime")
    hours_since = 72.0  # default fallback when upload time is unknown

    if upload_dt_str:
        try:
            upload_dt = datetime.fromisoformat(upload_dt_str)
            if upload_dt.tzinfo is None:
                upload_dt = upload_dt.replace(tzinfo=timezone.utc)
            now = datetime.now(timezone.utc)
            delta_seconds = max((now - upload_dt).total_seconds(), 3600)
            hours_since = delta_seconds / 3600
        except ValueError:
            pass

    return round(views / hours_since, 4)


def _compute_engagement_rate(short: dict) -> float:
    views = short.get("views", 1)
    likes = short.get("likes", 0)
    comments = short.get("comments", 0)
    return round((likes + comments) / max(views, 1), 6)


# ─────────────────────────────────────────────────────────────────────────────
# Cleanup helpers
# ─────────────────────────────────────────────────────────────────────────────

def _cleanup(*paths):
    for p in paths:
        if p and os.path.exists(p):
            try:
                os.remove(p)
            except OSError:
                pass