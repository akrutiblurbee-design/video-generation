"""
Step 2 — Fetch Shorts metadata via yt-dlp.
Scrapes views, likes, comments, upload date from each channel.
"""

import asyncio
import json
import logging
import shutil
import subprocess
import sys
from datetime import datetime, timedelta, timezone

log = logging.getLogger("fetcher")


def _yt_dlp_argv() -> list[str]:
    """Invoke yt-dlp: PATH binary if present, else same-interpreter ``python -m yt_dlp``."""
    for name in ("yt-dlp", "yt-dlp.exe"):
        path = shutil.which(name)
        if path:
            return [path]
    return [sys.executable, "-m", "yt_dlp"]


async def fetch_shorts_metadata(
    channels: list[str],
    days_back: int = 3,
    max_duration: int = 60,
) -> list[dict]:
    """
    Runs yt-dlp for each channel concurrently and returns merged metadata list.
    """
    tasks = [_fetch_channel(ch, days_back, max_duration) for ch in channels]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    all_shorts = []
    for ch, result in zip(channels, results):
        if isinstance(result, Exception):
            log.error(
                "Failed to fetch channel %s (%s): %r",
                ch,
                type(result).__name__,
                result,
            )
        else:
            log.info("Channel %s — got %d shorts", ch, len(result))
            all_shorts.extend(result)

    return all_shorts


async def fetch_single_short_metadata(video_id: str) -> list[dict]:
    """
    Fetch metadata for one specific YouTube Short video ID.
    """
    if not video_id:
        return []

    url = f"https://www.youtube.com/shorts/{video_id}"
    cmd = [
        *_yt_dlp_argv(),
        "--dump-single-json",
        "--no-playlist",
        "--extractor-args",
        "youtube:player_client=android,web,ios",
        url,
    ]

    log.info("Running yt-dlp for video: %s", video_id)
    proc = await _run_subprocess(cmd)
    stdout, stderr = proc.stdout, proc.stderr

    if proc.returncode != 0:
        err = stderr.decode(errors="replace")
        raise RuntimeError(f"yt-dlp error for {video_id}: {err[:500]}")

    try:
        entry = json.loads(stdout.decode())
    except json.JSONDecodeError as e:
        raise RuntimeError(f"JSON parse error for {video_id}: {e}")

    duration = entry.get("duration") or 0
    upload_date_str = entry.get("upload_date")
    upload_dt = None
    if upload_date_str:
        try:
            upload_dt = datetime.strptime(upload_date_str, "%Y%m%d").replace(
                tzinfo=timezone.utc
            )
        except ValueError:
            pass

    short = {
        "video_id": entry.get("id") or video_id,
        "title": entry.get("title"),
        "url": f"https://www.youtube.com/watch?v={entry.get('id') or video_id}",
        "channel": entry.get("uploader_id") or entry.get("uploader") or "",
        "views": entry.get("view_count") or 0,
        "likes": entry.get("like_count") or 0,
        "comments": entry.get("comment_count") or 0,
        "duration": duration,
        "upload_date": upload_date_str,
        "upload_datetime": upload_dt.isoformat() if upload_dt else None,
        "thumbnail": entry.get("thumbnail"),
        "description": (entry.get("description") or "")[:500],
    }

    return [short]


async def _fetch_channel(channel: str, days_back: int, max_duration: int) -> list[dict]:
    """
    Calls yt-dlp in a subprocess to extract JSON metadata for a channel's Shorts.
    """
    cutoff = datetime.now(timezone.utc) - timedelta(days=days_back)
    cutoff_str = cutoff.strftime("%Y%m%d")

    # yt-dlp flat playlist for /shorts — dumps JSON per video, no download
    cmd = [
        *_yt_dlp_argv(),
        "--flat-playlist",
        "--dump-single-json",
        "--extractor-args", "youtubetab:approximate_date",
        f"https://www.youtube.com/{channel}/shorts",
    ]

    log.info("Running yt-dlp for channel: %s", channel)
    proc = await _run_subprocess(cmd)
    stdout, stderr = proc.stdout, proc.stderr

    if proc.returncode != 0:
        err = stderr.decode(errors="replace")
        raise RuntimeError(f"yt-dlp error for {channel}: {err[:500]}")

    try:
        playlist = json.loads(stdout.decode())
    except json.JSONDecodeError as e:
        raise RuntimeError(f"JSON parse error for {channel}: {e}")

    entries = playlist.get("entries", [])
    shorts = []

    for entry in entries:
        duration = entry.get("duration") or 0
        if duration > max_duration:
            continue

        upload_date_str = entry.get("upload_date")  # YYYYMMDD
        if upload_date_str and upload_date_str < cutoff_str:
            continue

        upload_dt = None
        if upload_date_str:
            try:
                upload_dt = datetime.strptime(upload_date_str, "%Y%m%d").replace(
                    tzinfo=timezone.utc
                )
            except ValueError:
                pass

        shorts.append({
            "video_id": entry.get("id"),
            "title": entry.get("title"),
            # Flat-playlist "url" is often an ID; force a canonical URL for downloads.
            "url": f"https://www.youtube.com/watch?v={entry.get('id')}",
            "channel": channel,
            "views": entry.get("view_count") or 0,
            "likes": entry.get("like_count") or 0,
            "comments": entry.get("comment_count") or 0,
            "duration": duration,
            "upload_date": upload_date_str,
            "upload_datetime": upload_dt.isoformat() if upload_dt else None,
            "thumbnail": entry.get("thumbnail"),
            "description": entry.get("description", "")[:500],
        })

    return shorts


async def _run_subprocess(cmd: list[str]) -> subprocess.CompletedProcess:
    """Run command in a thread for cross-platform asyncio compatibility."""
    return await asyncio.to_thread(
        subprocess.run,
        cmd,
        capture_output=True,
        check=False,
    )