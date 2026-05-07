#!/usr/bin/env python3
"""
HTTP API for Blurbee Studio — pipeline, rhyme, scene prompts, music, replace audio.

Run: uvicorn api_server:app --reload --host 0.0.0.0 --port 8000
"""

from __future__ import annotations

import logging
import tempfile
from pathlib import Path

import openai
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from config import Config
from music_generator import generate_music_for_studio
from replace_video_audio import _build_timestamped_output_path, _replace_audio
from rhyme_writer import generate_rhyme_for_studio
from scene_prompt_writer import generate_scene_prompts_for_api

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("api_server")

OUTPUT_DIR = Path(Config.JSON_OUTPUT_DIR or "outputs")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

app = FastAPI(title="Blurbee Studio API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/outputs", StaticFiles(directory=str(OUTPUT_DIR)), name="outputs")


class GenerateRhymeBody(BaseModel):
    video_id: str | None = None
    prompt: str | None = None
    theme: str | None = None


class GenerateScenesBody(BaseModel):
    rhyme_description: str = Field(..., min_length=1)
    scene_count: int = Field(6, ge=1, le=24)
    style: str | None = Field(None, description="Overall video / visual style")
    video_style: str | None = None
    title: str | None = None
    rhyme_music_style_prompt: str | None = None
    folder_title: str | None = None


class GenerateMusicBody(BaseModel):
    rhyme_description: str = Field(..., min_length=1)
    style: str | None = ""
    duration: int = Field(60, ge=60, le=60)
    rhyme_music_style_prompt: str | None = None
    folder_title: str | None = None


class EnhanceScenePromptBody(BaseModel):
    prompt: str = Field(..., min_length=1)


def _enhance_scene_prompt(prompt: str) -> str:
    if not Config.OPENAI_API_KEY:
        raise RuntimeError("OPENAI_API_KEY is missing in environment/.env.")
    client = openai.OpenAI(api_key=Config.OPENAI_API_KEY, timeout=90)
    response = client.chat.completions.create(
        model=Config.OPENAI_MODEL,
        temperature=0.4,
        messages=[
            {
                "role": "system",
                "content": (
                    "You improve scene-prompt briefs for kid-safe 3D cartoon video generation. "
                    "Keep the original meaning and constraints, but make the prompt clearer, richer, "
                    "and more production-ready with concrete visual details."
                ),
            },
            {
                "role": "user",
                "content": (
                    "Enhance the following prompt. Return only the enhanced prompt text without code fences.\n\n"
                    f"{prompt.strip()}"
                ),
            },
        ],
    )
    return (response.choices[0].message.content or "").strip()


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/generate-rhyme")
def generate_rhyme(body: GenerateRhymeBody) -> dict:
    vid = (body.video_id or "").strip()
    prompt = (body.prompt or "").strip()
    theme = (body.theme or "").strip()

    if not vid and not prompt and not theme:
        raise HTTPException(
            status_code=400,
            detail="Provide a YouTube video ID and/or custom prompt (theme can supplement either).",
        )

    try:
        payload = generate_rhyme_for_studio(
            video_id=vid or None,
            custom_prompt=prompt or None,
            theme=theme or None,
            output_root=OUTPUT_DIR,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        log.exception("generate-rhyme failed")
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return {
        "rhyme": str(payload.get("rhyme", "")).strip(),
        "rhyme_style": str(payload.get("rhyme_style", "")).strip(),
    }


@app.post("/generate-scenes")
def generate_scenes(body: GenerateScenesBody) -> dict:
    vs = (body.video_style or body.style or "").strip() or "3D Pixar-style cartoon nursery animation"
    rhyme_payload = {
        "title": body.title or "Studio rhyme",
        "rhyme": body.rhyme_description.strip(),
        "rhyme_music_style_prompt": body.rhyme_music_style_prompt or "",
    }
    try:
        return generate_scene_prompts_for_api(
            rhyme_payload=rhyme_payload,
            scene_count=body.scene_count,
            video_style=vs,
            reference="",
            output_root=OUTPUT_DIR,
            folder_title=body.folder_title,
        )
    except Exception as exc:
        log.exception("generate-scenes failed")
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/generate-music")
def generate_music(body: GenerateMusicBody) -> dict:
    folder = (body.folder_title or "").strip() or "studio_music"
    try:
        raw = generate_music_for_studio(
            rhyme_lyrics=body.rhyme_description.strip(),
            rhyme_music_style_prompt=body.rhyme_music_style_prompt or "",
            user_music_style=body.style or "",
            duration_seconds=body.duration,
            output_root=OUTPUT_DIR,
            folder_title=folder,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        log.exception("generate-music failed")
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    url = raw.get("audio_url")
    return {**raw, "url": url, "file_url": url}


@app.post("/enhance-scene-prompt")
def enhance_scene_prompt(body: EnhanceScenePromptBody) -> dict:
    prompt = body.prompt.strip()
    if not prompt:
        raise HTTPException(status_code=400, detail="Prompt cannot be empty.")
    try:
        enhanced = _enhance_scene_prompt(prompt)
        if not enhanced:
            raise ValueError("LLM returned an empty enhanced prompt.")
        return {"enhanced_prompt": enhanced}
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        log.exception("enhance-scene-prompt failed")
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/replace-audio")
async def replace_audio_endpoint(
    video: UploadFile = File(...),
    audio: UploadFile = File(...),
    offset: str = Form("0"),
):
    _ = offset  # Reserved for future ffmpeg alignment
    suffix_v = Path(video.filename or "video.mp4").suffix or ".mp4"
    suffix_a = Path(audio.filename or "audio.mp3").suffix or ".mp3"
    try:
        with tempfile.TemporaryDirectory() as tmp:
            vpath = Path(tmp) / f"in_video{suffix_v}"
            apath = Path(tmp) / f"in_audio{suffix_a}"
            vpath.write_bytes(await video.read())
            apath.write_bytes(await audio.read())
            final_dir = OUTPUT_DIR / "final_videos"
            final_dir.mkdir(parents=True, exist_ok=True)
            out_path = _build_timestamped_output_path(final_dir / "final_with_generated_music.mp4")
            _replace_audio(vpath, apath, out_path)
            return FileResponse(
                str(out_path),
                media_type="video/mp4",
                filename=out_path.name,
            )
    except Exception as exc:
        log.exception("replace-audio failed")
        raise HTTPException(status_code=500, detail=str(exc)) from exc
