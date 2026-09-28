"""Shared image and reward-output utilities."""

from __future__ import annotations

import base64
import json
import re
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, Tuple

from PIL import Image


def load_rgb_image(path: str | Path) -> Image.Image:
    with Image.open(path) as image:
        image.load()
        return image.convert("RGB")


def image_to_data_url(path: str | Path, image_format: str = "PNG") -> str:
    image_format = image_format.upper()
    mime = "image/jpeg" if image_format in {"JPG", "JPEG"} else "image/png"
    image = load_rgb_image(path)
    buffer = BytesIO()
    image.save(buffer, format=image_format)
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:{mime};base64,{encoded}"


def strip_json_fence(text: str) -> str:
    raw = (text or "").strip()
    raw = re.sub(
        r"<think\b[^>]*>.*?</think\s*>",
        "",
        raw,
        flags=re.DOTALL | re.IGNORECASE,
    ).strip()
    if raw.startswith("```"):
        raw = re.sub(r"^```(?:json)?\s*", "", raw)
        raw = re.sub(r"\s*```$", "", raw)
    return raw.strip()


def parse_reward_json(text: str) -> Tuple[float, Dict[str, Any]]:
    """Parse model output and return (final_score, parsed_json).

    The reward model is prompted to emit pure JSON, but this parser tolerates
    fenced JSON or a short preamble as a convenience for smoke tests.
    """

    raw = strip_json_fence(text)
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", raw, flags=re.DOTALL)
        if match is None:
            raise
        data = json.loads(match.group(0))

    score = float(data["final_score"])
    score = max(0.0, min(10.0, score))
    return score, data


def reward_from_score(final_score: float) -> float:
    return max(0.0, min(10.0, float(final_score))) / 10.0
