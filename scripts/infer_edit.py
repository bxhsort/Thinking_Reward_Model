#!/usr/bin/env python3
"""Run image-editing reward inference with Transformers or a vLLM server."""

from __future__ import annotations

import argparse
import concurrent.futures as futures
import json
import sys
from pathlib import Path
from typing import Any, Dict, Iterable

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from trm.backends.vllm import VLLMBackend
from trm.protocols import edit as edit_protocol


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("transformers", "vllm"), default="transformers")
    parser.add_argument("--model", required=True, help="HF repo/local path or vLLM served model name.")
    parser.add_argument("--source-image", help="Source/original image path.")
    parser.add_argument("--edited-image", help="Edited image path.")
    parser.add_argument("--instruction", help="Image-editing instruction.")
    parser.add_argument("--input-jsonl", help="Batch JSONL for vLLM backend.")
    parser.add_argument("--output-jsonl", help="Batch JSONL output path for vLLM backend.")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--max-new-tokens", type=int, default=1536)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--enable-thinking", action="store_true")

    parser.add_argument("--device-map", default="none")
    parser.add_argument("--torch-dtype", default="auto")
    parser.add_argument("--attn-implementation", default=None)
    parser.add_argument("--no-trust-remote-code", action="store_true")

    parser.add_argument("--base-url", default="http://localhost:8000")
    parser.add_argument("--api-key", default="EMPTY")
    parser.add_argument("--timeout", type=float, default=600)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def iter_jsonl(path: str | Path) -> Iterable[Dict[str, Any]]:
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            record["_line_number"] = line_number
            yield record


def resolve_record(record: Dict[str, Any]) -> tuple[str, str, str]:
    source = record.get("source_image") or record.get("source")
    edited = record.get("edited_image") or record.get("edited") or record.get("output_image")
    instruction = record.get("instruction") or record.get("prompt")
    if not source or not edited or not instruction:
        raise ValueError(
            "Edit JSONL records require source_image, edited_image, and instruction "
            "(aliases: source, edited/output_image, prompt)."
        )
    return str(source), str(edited), str(instruction)


def build_vllm_backend(args: argparse.Namespace) -> VLLMBackend:
    return VLLMBackend(
        base_url=args.base_url,
        model=args.model,
        api_key=args.api_key,
        timeout=args.timeout,
        max_tokens=args.max_new_tokens,
        temperature=args.temperature,
        seed=args.seed,
        chat_template_kwargs={"enable_thinking": bool(args.enable_thinking)},
    )


def run_transformers(args: argparse.Namespace) -> None:
    from trm.backends.transformers import TransformersBackend

    if args.input_jsonl:
        raise SystemExit("Batch JSONL mode is only implemented for the vLLM backend.")
    if not (args.source_image and args.edited_image and args.instruction):
        raise SystemExit("Edit inference requires --source-image, --edited-image, and --instruction.")

    backend = TransformersBackend(
        model_name_or_path=args.model,
        device_map=args.device_map,
        torch_dtype=args.torch_dtype,
        trust_remote_code=not args.no_trust_remote_code,
        attn_implementation=args.attn_implementation,
    )
    messages = edit_protocol.build_transformers_messages(
        source_image=args.source_image,
        edited_image=args.edited_image,
        instruction=args.instruction,
    )
    result = backend.score_messages(
        messages,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        enable_thinking=args.enable_thinking,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


def run_vllm_single(args: argparse.Namespace) -> None:
    if not (args.source_image and args.edited_image and args.instruction):
        raise SystemExit("Edit inference requires --source-image, --edited-image, and --instruction.")
    backend = build_vllm_backend(args)
    messages = edit_protocol.build_openai_messages(
        args.source_image,
        args.edited_image,
        args.instruction,
    )
    result = backend.score_messages(messages)
    print(json.dumps(result, ensure_ascii=False, indent=2))


def run_vllm_jsonl(args: argparse.Namespace) -> None:
    if not args.output_jsonl:
        raise SystemExit("Batch mode requires --output-jsonl.")

    records = list(iter_jsonl(args.input_jsonl))
    backend = build_vllm_backend(args)
    with Path(args.output_jsonl).open("w", encoding="utf-8") as handle:
        with futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
            pending = {}
            for record in records:
                source, edited, instruction = resolve_record(record)
                messages = edit_protocol.build_openai_messages(source, edited, instruction)
                future = executor.submit(backend.score_messages, messages)
                pending[future] = record
            for future in futures.as_completed(pending):
                record = pending[future]
                try:
                    output = {**record, edit_protocol.RESULT_KEY: future.result()}
                except Exception as exc:
                    output = {**record, "ok": False, "error": str(exc)}
                handle.write(json.dumps(output, ensure_ascii=False) + "\n")
                handle.flush()


def main() -> None:
    args = parse_args()
    if args.backend == "transformers":
        run_transformers(args)
    elif args.input_jsonl:
        run_vllm_jsonl(args)
    else:
        run_vllm_single(args)


if __name__ == "__main__":
    main()
