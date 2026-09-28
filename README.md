# Think Before You Score

Official inference code for **Think Before You Score: Thinking Reward Model for
Visual Generation**.

This repository is referred to as **TRM** in code and model names.

TRM provides two visual reward models:

- **TRM-Edit**: scores an edited image given a source image and an editing instruction.
- **TRM-T2I**: scores a generated image given a text-to-image prompt.

## Installation

We recommend Python 3.12 with the tested dependency versions below.

```bash
git clone https://github.com/your-org/TRM.git
cd TRM

conda create -n trm python=3.12 -y
conda activate trm
pip install -U pip
```

Install the tested inference stack:

```bash
pip install \
  torch==2.11.0 \
  torchvision==0.26.0 \
  transformers==5.8.1 \
  tokenizers==0.22.2 \
  safetensors==0.8.0 \
  qwen-vl-utils==0.0.14 \
  pillow==12.3.0 \
  huggingface_hub==1.25.1 \
  openai==2.50.0 \
  vllm==0.21.0
```

Install this repository in editable mode:

```bash
pip install -e .
```

## Model Download

Download the released checkpoints from Hugging Face. Replace the repo ids below
with the official model ids from the paper release.

```bash
pip install -U "huggingface_hub[cli]"
# hf auth login  # only needed for gated/private checkpoints

mkdir -p checkpoints

hf download your-org/TRM-Edit \
  --local-dir checkpoints/trm-edit

hf download your-org/TRM-T2I \
  --local-dir checkpoints/trm-t2i
```

Then set:

```bash
export TRM_EDIT_MODEL=checkpoints/trm-edit
export TRM_T2I_MODEL=checkpoints/trm-t2i
```

## Transformers Inference

### Image Editing Reward

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/infer_edit.py \
  --backend transformers \
  --model "$TRM_EDIT_MODEL" \
  --source-image examples/source.png \
  --edited-image examples/edited.png \
  --instruction "Replace the red car with a blue car." \
  --temperature 0
```

### Text-to-Image Reward

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/infer_t2i.py \
  --backend transformers \
  --model "$TRM_T2I_MODEL" \
  --image examples/generated.png \
  --prompt "A photo of a yellow bus parked under a rainy neon street." \
  --temperature 0
```

For Transformers inference, use a single GPU, `temperature=0`, and keep
thinking disabled. The scripts do not enable thinking unless `--enable-thinking`
is explicitly passed.

## vLLM Server Inference

Start one vLLM server per reward model.
For vLLM, choose tensor parallelism with `TENSOR_PARALLEL_SIZE`.

### Serve TRM-Edit

```bash
MODEL_PATH="$TRM_EDIT_MODEL" \
SERVED_MODEL_NAME=trm-edit-reward \
IMAGE_LIMIT=2 \
TENSOR_PARALLEL_SIZE=1 \
MAX_MODEL_LEN=32768 \
VLLM_EXTRA_ARGS="--gdn-prefill-backend triton" \
bash scripts/serve_vllm.sh
```

In another terminal:

```bash
python scripts/infer_edit.py \
  --backend vllm \
  --base-url http://localhost:8000 \
  --model trm-edit-reward \
  --source-image examples/source.png \
  --edited-image examples/edited.png \
  --instruction "Replace the red car with a blue car."
```

### Serve TRM-T2I

```bash
MODEL_PATH="$TRM_T2I_MODEL" \
SERVED_MODEL_NAME=trm-t2i-reward \
IMAGE_LIMIT=1 \
TENSOR_PARALLEL_SIZE=2 \
MAX_MODEL_LEN=16384 \
VLLM_EXTRA_ARGS="--gdn-prefill-backend triton" \
bash scripts/serve_vllm.sh
```

In another terminal:

```bash
python scripts/infer_t2i.py \
  --backend vllm \
  --base-url http://localhost:8000 \
  --model trm-t2i-reward \
  --image examples/generated.png \
  --prompt "A photo of a yellow bus parked under a rainy neon street."
```

## Batch Scoring

Edit JSONL:

```json
{"source_image": "source.png", "edited_image": "edited.png", "instruction": "Make the sky cloudy."}
```

```bash
python scripts/infer_edit.py \
  --backend vllm \
  --base-url http://localhost:8000 \
  --model trm-edit-reward \
  --input-jsonl data/edit_requests.jsonl \
  --output-jsonl outputs/edit_scores.jsonl \
  --workers 16
```

T2I JSONL:

```json
{"image": "generated.png", "prompt": "A photo of a yellow bus."}
```

```bash
python scripts/infer_t2i.py \
  --backend vllm \
  --base-url http://localhost:8000 \
  --model trm-t2i-reward \
  --input-jsonl data/t2i_requests.jsonl \
  --output-jsonl outputs/t2i_scores.jsonl \
  --workers 16
```

## Output

Each call prints a JSON object:

```json
{
  "reward": 0.83,
  "final_score": 8.3,
  "parsed": {
    "eval_points": [],
    "dimension_summary": {},
    "score_reason": "...",
    "final_score": 8.3
  },
  "raw_output": "..."
}
```

## Configs

Reference server settings are provided in:

- `configs/edit_reward.yaml`
- `configs/t2i_reward.yaml`

These files mirror the settings used by the released checkpoints and are useful
when adapting the scripts to a custom serving stack.

## Citation

If you use TRM, please cite our paper:

```bibtex
@article{trm2026,
  title={Think Before You Score: Thinking Reward Model for Visual Generation},
  author={TRM Authors},
  journal={arXiv preprint},
  year={2026}
}
```
