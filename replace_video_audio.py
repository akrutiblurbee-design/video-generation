#!/usr/bin/env python3
"""
Replace a video's original audio with generated music.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
from pathlib import Path


DEFAULT_VIDEO_PATH = "Scene_7_Rhyme_Reference__The_202605061201.mp4"
DEFAULT_AUDIO_PATH = "outputs\wheels on the bus\generated_music\music_60s.mp3"
DEFAULT_OUTPUT_PATH = "outputs/final_videos/final_with_generated_music.mp4"


def _ensure_ffmpeg() -> str:
    ffmpeg_path = shutil.which("ffmpeg")
    if ffmpeg_path:
        return ffmpeg_path
    try:
        import imageio_ffmpeg

        return imageio_ffmpeg.get_ffmpeg_exe()
    except ImportError:
        pass
    raise RuntimeError(
        "FFmpeg executable not found. The PyPI package named 'ffmpeg' is not FFmpeg itself.\n"
        "Options:\n"
        "  • pip install imageio-ffmpeg  (bundled ffmpeg; then re-run)\n"
        "  • Or install FFmpeg and add it to PATH, e.g. Windows: winget install Gyan.FFmpeg"
    )


def _replace_audio(video_path: Path, audio_path: Path, output_path: Path) -> None:
    if not video_path.exists():
        raise FileNotFoundError(f"Video file not found: {video_path}")
    if not audio_path.exists():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    ffmpeg = _ensure_ffmpeg()
    command = [
        ffmpeg,
        "-y",
        "-i",
        str(video_path),
        "-i",
        str(audio_path),
        "-map",
        "0:v:0",
        "-map",
        "1:a:0",
        "-c:v",
        "copy",
        "-c:a",
        "aac",
        "-b:a",
        "192k",
        "-shortest",
        str(output_path),
    ]
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg failed.\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Replace video audio with generated music")
    parser.add_argument("--video", type=str, default=DEFAULT_VIDEO_PATH, help="Path to input video.")
    parser.add_argument("--audio", type=str, default=DEFAULT_AUDIO_PATH, help="Path to generated audio.")
    parser.add_argument("--output", type=str, default=DEFAULT_OUTPUT_PATH, help="Path to output video.")
    args = parser.parse_args()

    video_path = Path(args.video)
    audio_path = Path(args.audio)
    output_path = Path(args.output)

    _replace_audio(video_path, audio_path, output_path)
    print(f"Saved final video: {output_path}")


if __name__ == "__main__":
    main()
