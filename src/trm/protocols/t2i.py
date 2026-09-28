"""Text-to-image reward protocol.

Input:
    generated image + text prompt

Output:
    JSON rubric.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

from trm.prompts import T2I_SYSTEM_PROMPT, build_t2i_user_text
from trm.utils import image_to_data_url

DEFAULT_MAX_NEW_TOKENS = 4096
RESULT_KEY = "trm_t2i"


def build_transformers_messages(
    image: str | Path,
    prompt: str,
) -> List[Dict[str, Any]]:
    return [
        {"role": "system", "content": T2I_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": [
                {"type": "image", "image": str(image)},
                {"type": "text", "text": build_t2i_user_text(prompt)},
            ],
        },
    ]


def build_openai_messages(
    image: str | Path,
    prompt: str,
    image_format: str = "PNG",
) -> List[Dict[str, Any]]:
    return [
        {"role": "system", "content": T2I_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {"url": image_to_data_url(image, image_format)},
                },
                {"type": "text", "text": build_t2i_user_text(prompt)},
            ],
        },
    ]
