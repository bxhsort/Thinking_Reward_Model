"""Hugging Face Transformers backend for reward inference."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Sequence

import torch
import transformers
from transformers import AutoProcessor

from trm.utils import parse_reward_json, reward_from_score


@dataclass
class TransformersBackend:
    model_name_or_path: str
    device_map: Optional[str] = "none"
    torch_dtype: str = "auto"
    trust_remote_code: bool = True
    attn_implementation: Optional[str] = None

    def __post_init__(self) -> None:
        self.processor = AutoProcessor.from_pretrained(
            self.model_name_or_path,
            trust_remote_code=self.trust_remote_code,
        )
        kwargs: Dict[str, Any] = {
            "trust_remote_code": self.trust_remote_code,
            "dtype": _resolve_torch_dtype(self.torch_dtype),
        }
        if self.device_map and self.device_map.lower() != "none":
            kwargs["device_map"] = self.device_map
        if self.attn_implementation:
            kwargs["attn_implementation"] = self.attn_implementation

        self.model = _resolve_model_class().from_pretrained(
            self.model_name_or_path,
            **kwargs,
        )
        if not self.device_map or self.device_map.lower() == "none":
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            self.model.to(device)
        self.model.eval()

    def score_messages(
        self,
        messages: Sequence[Dict[str, Any]],
        max_new_tokens: int,
        temperature: float = 0.0,
        enable_thinking: bool = False,
    ) -> Dict[str, Any]:
        try:
            from qwen_vl_utils import process_vision_info
        except ImportError as error:
            raise ImportError(
                "qwen-vl-utils is required for Transformers inference: "
                "pip install qwen-vl-utils"
            ) from error

        prompt = _apply_chat_template(
            self.processor,
            messages,
            enable_thinking=enable_thinking,
        )
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(
            text=[prompt],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to(_first_model_device(self.model))

        generate_kwargs: Dict[str, Any] = {
            "max_new_tokens": int(max_new_tokens),
            "do_sample": temperature > 0,
        }
        if temperature > 0:
            generate_kwargs["temperature"] = float(temperature)

        eos_token_id = getattr(getattr(self.processor, "tokenizer", None), "eos_token_id", None)
        if eos_token_id is not None:
            generate_kwargs["pad_token_id"] = eos_token_id

        with torch.inference_mode():
            generated_ids = self.model.generate(**inputs, **generate_kwargs)

        input_ids = inputs["input_ids"]
        generated_trimmed = [
            output_ids[len(input_ids_i) :]
            for input_ids_i, output_ids in zip(input_ids, generated_ids)
        ]
        output_text = self.processor.batch_decode(
            generated_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0].strip()

        final_score, parsed = parse_reward_json(output_text)
        return {
            "reward": reward_from_score(final_score),
            "final_score": final_score,
            "parsed": parsed,
            "raw_output": output_text,
        }


def _resolve_model_class() -> Any:
    for name in (
        "AutoModelForImageTextToText",
        "AutoModelForVision2Seq",
        "Qwen2_5_VLForConditionalGeneration",
        "AutoModelForCausalLM",
    ):
        model_cls = getattr(transformers, name, None)
        if model_cls is not None:
            return model_cls
    raise RuntimeError("No compatible Transformers model class was found.")


def _resolve_torch_dtype(dtype: str) -> Any:
    if dtype == "auto":
        return "auto"
    if not hasattr(torch, dtype):
        raise ValueError(f"Unsupported torch dtype: {dtype}")
    return getattr(torch, dtype)


def _first_model_device(model: torch.nn.Module) -> torch.device:
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _apply_chat_template(
    processor: Any,
    messages: Sequence[Dict[str, Any]],
    enable_thinking: bool,
) -> str:
    kwargs = {
        "tokenize": False,
        "add_generation_prompt": True,
        "enable_thinking": enable_thinking,
    }
    try:
        return processor.apply_chat_template(messages, **kwargs)
    except TypeError:
        kwargs.pop("enable_thinking")
        return processor.apply_chat_template(messages, **kwargs)
