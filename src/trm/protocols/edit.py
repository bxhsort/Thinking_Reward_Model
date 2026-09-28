"""Image-editing reward protocol.

Input:
    source image + edited image + editing instruction

Output:
    JSON rubric.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

from trm.prompts import EDIT_SYSTEM_PROMPT, build_edit_user_text
from trm.utils import image_to_data_url

DEFAULT_MAX_NEW_TOKENS = 1536
RESULT_KEY = "trm_edit"


def build_transformers_messages(
    source_image: str | Path,
    edited_image: str | Path,
    instruction: str,
) -> List[Dict[str, Any]]:
    return [
        {"role": "system", "content": EDIT_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": [
                {"type": "image", "image": str(source_image)},
                {"type": "image", "image": str(edited_image)},
                {"type": "text", "text": build_edit_user_text(instruction)},
            ],
        },
    ]


def build_openai_messages(
    source_image: str | Path,
    edited_image: str | Path,
    instruction: str,
    image_format: str = "PNG",
) -> List[Dict[str, Any]]:
    return [
        {"role": "system", "content": EDIT_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {"url": image_to_data_url(source_image, image_format)},
                },
                {
                    "type": "image_url",
                    "image_url": {"url": image_to_data_url(edited_image, image_format)},
                },
                {"type": "text", "text": build_edit_user_text(instruction)},
            ],
        },
    ]
