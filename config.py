"""
Central configuration — edit this file or set environment variables.
"""

import os
from dotenv import load_dotenv

load_dotenv()


class Config:
    # ── Runtime / scheduling ─────────────────────────────────────────────────
    SCHEDULER_ENABLED: bool = os.getenv("SCHEDULER_ENABLED", "false").lower() == "true"
    SCHEDULE_EVERY_DAYS: int = int(os.getenv("SCHEDULE_EVERY_DAYS", "1"))
    JSON_OUTPUT_DIR: str = os.getenv("JSON_OUTPUT_DIR", "outputs")

    # ── Scheduling ──────────────────────────────────────────────────────────
    CRON_TIME: str = os.getenv("CRON_TIME", "08:00")  # 24h HH:MM

    # ── YouTube Channels to monitor ─────────────────────────────────────────
    YOUTUBE_CHANNELS: list[str] = [
        c.strip()
        for c in os.getenv(
            "YOUTUBE_CHANNELS",
            "@MrBeast,@mkbhd,@veritasium",  # ← replace with your targets
        ).split(",")
    ]

    # ── Fetch settings ──────────────────────────────────────────────────────
    DAYS_BACK: int = int(os.getenv("DAYS_BACK", "3"))
    MAX_DURATION_SECONDS: int = int(os.getenv("MAX_DURATION", "60"))
    TOP_N: int = int(os.getenv("TOP_N", "30"))

    # ── Download / transcription ────────────────────────────────────────────
    DOWNLOAD_DIR: str = os.getenv("DOWNLOAD_DIR", "tmp/yt_shorts")
    KEEP_DOWNLOADED_MEDIA: bool = os.getenv("KEEP_DOWNLOADED_MEDIA", "false").lower() == "true"
    WHISPER_MODEL: str = os.getenv("WHISPER_MODEL", "base")  # tiny|base|small|medium|large
    FRAMES_PER_SECOND: float = float(os.getenv("FRAMES_PER_SECOND", "0.5"))   # 1 frame / 2s
    MAX_FRAMES: int = int(os.getenv("MAX_FRAMES", "8"))
    FRAME_WIDTH: int = int(os.getenv("FRAME_WIDTH", "512"))

    # ── OpenAI ──────────────────────────────────────────────────────────────
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
    OPENAI_MODEL: str = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    OPENAI_VISION_MODEL: str = os.getenv("OPENAI_VISION_MODEL", OPENAI_MODEL)

    # ── Musicful (kids rhyme background music generation) ───────────────────
    MUSICFUL_ENABLED: bool = os.getenv("MUSICFUL_ENABLED", "true").lower() == "true"
    MUSICFUL_API_KEY: str = os.getenv("MUSICFUL_API_KEY", "")
    MUSICFUL_BASE_URL: str = os.getenv("MUSICFUL_BASE_URL", "https://api.musicful.ai")
    MUSICFUL_GENERATE_PATH: str = os.getenv("MUSICFUL_GENERATE_PATH", "/v1/music/generate")
    MUSICFUL_STATUS_PATH: str = os.getenv(
        "MUSICFUL_STATUS_PATH", "/v1/music/generations/{job_id}"
    )
    MUSICFUL_TIMEOUT_SECONDS: int = int(os.getenv("MUSICFUL_TIMEOUT_SECONDS", "180"))
    MUSICFUL_POLL_SECONDS: float = float(os.getenv("MUSICFUL_POLL_SECONDS", "4"))
    MUSICFUL_VOLUME: float = float(os.getenv("MUSICFUL_VOLUME", "1.0"))

    # ── Airtable ────────────────────────────────────────────────────────────
    AIRTABLE_API_KEY: str = os.getenv("AIRTABLE_API_KEY", "")
    AIRTABLE_BASE_ID: str = os.getenv("AIRTABLE_BASE_ID", "")
    AIRTABLE_TABLE_NAME: str = os.getenv("AIRTABLE_TABLE_NAME", "YT Shorts Analysis")

    # ── Viral score weights ──────────────────────────────────────────────────
    # viral_score = (views / hours_since_posted) + (likes * LIKE_WEIGHT) + (comments * COMMENT_WEIGHT)
    LIKE_WEIGHT: int = int(os.getenv("LIKE_WEIGHT", "2"))
    COMMENT_WEIGHT: int = int(os.getenv("COMMENT_WEIGHT", "3"))