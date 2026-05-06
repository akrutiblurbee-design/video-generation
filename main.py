#!/usr/bin/env python3
"""
YouTube Shorts Viral Analysis Pipeline
Equivalent to the n8n flow - runs on a configurable schedule
"""

import asyncio
import argparse
import json
import logging
from pathlib import Path
import time
from datetime import datetime, timezone
import sys

import schedule

from config import Config
from pipeline.aggregator import aggregate_and_rank
from pipeline.fetcher import fetch_shorts_metadata, fetch_single_short_metadata
from pipeline.processor import process_shorts

# Avoid Windows cp1252 logging crashes when titles contain emoji.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("pipeline.log", encoding="utf-8"),
    ],
)
log = logging.getLogger("main")


async def run_pipeline(video_id: str | None = None):
    log.info("=" * 60)
    log.info("Pipeline started at %s", datetime.now().isoformat())
    log.info("=" * 60)

    # Step 2 - Fetch metadata from YouTube channels
    log.info("Step 2: Fetching Shorts metadata...")
    if video_id:
        raw_shorts = await fetch_single_short_metadata(video_id=video_id)
    else:
        raw_shorts = await fetch_shorts_metadata(
            channels=Config.YOUTUBE_CHANNELS,
            days_back=Config.DAYS_BACK,
            max_duration=Config.MAX_DURATION_SECONDS,
        )

    # Step 3 - Deduplicate, filter, sort, top 30
    log.info("Step 3: Filtering and sorting %d raw shorts...", len(raw_shorts))
    shorts = _parse_and_filter(raw_shorts)

    # Step 4 - Has videos?
    if not shorts:
        summary = {
            "total_shorts": 0,
            "total_views": 0,
            "total_likes": 0,
            "avg_engagement_rate": 0,
            "empty": True,
        }
        output_path = _save_pipeline_output(
            raw_shorts=raw_shorts,
            filtered_shorts=shorts,
            processed_shorts=[],
            ranked={
                "top_shorts": [],
                "niche_groups": {},
                "top_formats": [],
                "top_hook_styles": [],
                "top_emotions": [],
            },
            summary=summary,
        )
        log.info("Structured JSON saved to %s", output_path)
        log.warning("No shorts found - pipeline finished with empty report.")
        return

    log.info("Processing %d shorts...", len(shorts))

    # Steps 5-14 - Process each short (download, transcribe, analyse, store)
    processed = await process_shorts(shorts)

    # Step 16 - Aggregate and rank
    log.info("Step 16: Aggregating and ranking results...")
    ranked = aggregate_and_rank(processed)
    summary = _build_summary(processed)

    # Step 16.5 - Save structured JSON output
    output_path = _save_pipeline_output(
        raw_shorts=raw_shorts,
        filtered_shorts=shorts,
        processed_shorts=processed,
        ranked=ranked,
        summary=summary,
    )
    log.info("Structured JSON saved to %s", output_path)

    log.info("Pipeline complete. Processed %d shorts.", len(processed))


def _parse_and_filter(raw_shorts: list) -> list:
    """Deduplicate, filter by duration, sort by views/hour, take top N."""
    seen = set()
    unique = []
    for short in raw_shorts:
        vid_id = short.get("video_id") or short.get("id")
        if vid_id and vid_id not in seen:
            seen.add(vid_id)
            if short.get("duration", 9999) <= Config.MAX_DURATION_SECONDS:
                short["views_per_hour"] = _compute_views_per_hour(short)
                unique.append(short)

    unique.sort(key=lambda item: item.get("views_per_hour", 0), reverse=True)
    return unique[: Config.TOP_N]


def _compute_views_per_hour(short: dict) -> float:
    """Estimate growth speed from upload timestamp and current views."""
    views = short.get("views", 0) or 0
    upload_dt_str = short.get("upload_datetime")
    hours_since = 72.0  # fallback if timestamp is missing

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


def _build_summary(processed: list) -> dict:
    total_views = sum(item.get("views", 0) for item in processed)
    total_likes = sum(item.get("likes", 0) for item in processed)
    avg_engagement = (
        sum(item.get("engagement_rate", 0) for item in processed) / len(processed)
        if processed
        else 0
    )
    return {
        "total_shorts": len(processed),
        "total_views": total_views,
        "total_likes": total_likes,
        "avg_engagement_rate": round(avg_engagement, 4),
        "empty": False,
    }


def _save_pipeline_output(
    raw_shorts: list,
    filtered_shorts: list,
    processed_shorts: list,
    ranked: dict,
    summary: dict,
) -> str:
    output_dir = Path(Config.JSON_OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)

    generated_at = datetime.now().isoformat()
    filename = f"pipeline_output_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    output_path = output_dir / filename

    payload = {
        "generated_at": generated_at,
        "config": {
            "channels": Config.YOUTUBE_CHANNELS,
            "days_back": Config.DAYS_BACK,
            "max_duration_seconds": Config.MAX_DURATION_SECONDS,
            "top_n": Config.TOP_N,
            "openai_model": Config.OPENAI_MODEL,
            "openai_vision_model": Config.OPENAI_VISION_MODEL,
        },
        "counts": {
            "raw_shorts": len(raw_shorts),
            "filtered_shorts": len(filtered_shorts),
            "processed_shorts": len(processed_shorts),
            "top_ranked_shorts": len(ranked.get("top_shorts", [])),
        },
        "summary": summary,
        "ranked": ranked,
        "processed_shorts": processed_shorts,
    }

    with output_path.open("w", encoding="utf-8") as file_obj:
        json.dump(payload, file_obj, indent=2, ensure_ascii=False)

    return str(output_path)


def schedule_pipeline():
    if not Config.SCHEDULER_ENABLED:
        log.info(
            "Scheduler is disabled via SCHEDULER_ENABLED=false. "
            "Use --now to run the pipeline manually."
        )
        return

    log.info(
        "Scheduler started - will run every %d day(s) at %s",
        Config.SCHEDULE_EVERY_DAYS,
        Config.CRON_TIME,
    )
    schedule.every(Config.SCHEDULE_EVERY_DAYS).days.at(Config.CRON_TIME).do(
        lambda: asyncio.run(run_pipeline())
    )

    while True:
        schedule.run_pending()
        time.sleep(30)


if __name__ == "__main__":
    try:
        parser = argparse.ArgumentParser(description="YouTube Shorts Viral Analysis Pipeline")
        parser.add_argument(
            "--now",
            action="store_true",
            help="Run immediately once instead of scheduler mode.",
        )
        parser.add_argument(
            "--schedule",
            action="store_true",
            help="Run in scheduler mode.",
        )
        parser.add_argument(
            "--video-id",
            type=str,
            default=None,
            help="Process exactly one YouTube video ID once.",
        )
        args = parser.parse_args()

        if args.video_id:
            asyncio.run(run_pipeline(video_id=args.video_id.strip()))
        elif args.schedule:
            schedule_pipeline()
        elif args.now:
            asyncio.run(run_pipeline())
        else:
            video_id = input("Enter YouTube video ID: ").strip()
            if not video_id:
                raise ValueError("videoID is required.")
            asyncio.run(run_pipeline(video_id=video_id))
    except KeyboardInterrupt:
        log.warning("Pipeline interrupted by user. Exiting cleanly.")
