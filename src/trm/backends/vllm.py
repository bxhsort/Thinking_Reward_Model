"""OpenAI-compatible vLLM backend for reward inference."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Sequence

from trm.utils import parse_reward_json, reward_from_score


@dataclass
class VLLMBackend:
    base_url: str
    model: str
    api_key: str = "EMPTY"
    timeout: float = 600.0
    max_tokens: int = 1536
    temperature: float = 0.0
    seed: Optional[int] = 42
    chat_template_kwargs: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        try:
            from openai import OpenAI
        except ImportError as error:
            raise ImportError("Install the OpenAI client first: pip install openai") from error

        base_url = self.base_url.rstrip("/")
        if not base_url.endswith("/v1"):
            base_url = f"{base_url}/v1"
        self._client = OpenAI(
            base_url=base_url,
            api_key=self.api_key,
            timeout=self.timeout,
        )

    def score_messages(self, messages: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
        extra_body: Dict[str, Any] = {}
        if self.chat_template_kwargs:
            extra_body["chat_template_kwargs"] = self.chat_template_kwargs

        response = self._client.chat.completions.create(
            model=self.model,
            messages=messages,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            seed=self.seed,
            extra_body=extra_body or None,
        )
        content = response.choices[0].message.content
        if isinstance(content, list):
            output_text = "".join(str(item) for item in content)
        else:
            output_text = str(content or "")

        final_score, parsed = parse_reward_json(output_text)
        return {
            "reward": reward_from_score(final_score),
            "final_score": final_score,
            "parsed": parsed,
            "raw_output": output_text,
            "response_id": response.id,
        }

