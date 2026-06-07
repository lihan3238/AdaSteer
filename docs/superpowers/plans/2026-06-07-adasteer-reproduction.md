# AdaSteer Reproduction Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Reproduce AdaSteer end to end on `dell`, first with the official shipped vectors and then, where source data permits, by regenerating RD/HD vectors and evaluation tables.

**Architecture:** Treat the official repository as the baseline artifact and add a thin reproducibility layer around it: environment checks, path normalization, smoke runs, full generation, deterministic result aggregation, and reproduction notes. Do not change model behavior until smoke tests prove which compatibility patches are actually required.

**Tech Stack:** Python 3.9/3.10, PyTorch, Transformers 4.46.3-compatible model APIs, Accelerate, scikit-learn, PEFT, OpenAI-compatible classification API, Hugging Face local model directories on `dell`.

---

## Context From Notes And Paper

- The `05` note is not an AdaSteer single-paper note; it is the active research-positioning note for capability-positive agent safety. Its constraint for this reproduction is: do not treat "steering for agent safety" as a novelty claim by itself; treat AdaSteer as a module/baseline to understand adaptive latent intervention and its limits.
- The `04_safety_steering_repe` map has not yet added a dedicated AdaSteer note, even though the PDF is in the paper folder. After reproduction, add a concise single-paper/reproduction entry only if requested.
- The paper's core method is dual-direction activation steering:
  - RD rejects harmful/jailbreak inputs.
  - HD offsets over-refusal and preserves benign utility.
  - Adaptive coefficients are fit from input positions along RD/HD.
- The official code embeds the adaptive coefficient formulas directly in the modified model classes, not as a configurable logistic-regression module.
- Official code already ships vectors for LLaMA-3.1, Qwen2.5, and Gemma-2. That makes result reproduction with provided vectors the first target; vector-extraction reproduction is a second target because the original direction-identification datasets are only partially present in the repo.

## Known Machine State

- Run location: `dell`, SSH host `10.77.0.102`, user `dell`.
- GPU state checked on 2026-06-07: 8 x RTX 5090, 32GB each.
- Complete local models on `dell`:
  - `/home/dell/Downloads/Llama-3.1-8B-Instruct-hf`
  - `/home/dell/Downloads/Qwen2.5-7B-Instruct`
  - `/home/dell/Downloads/Qwen2.5-7B-Instruct-hf`
  - `/home/dell/Downloads/gemma-2-9b-it`
  - `/home/dell/Downloads/gemma2-9b-it`
- `hello` was cleaned of incomplete AdaSteer models and should not be used for this reproduction unless more disk is freed.
- Current fork:
  - `origin`: `git@github.com:lihan3238/AdaSteer.git`
  - `upstream`: `git@github.com:MuyuenLP/AdaSteer.git`

## File Structure

- Modify or create `scripts/repro/` for local, non-SLURM reproduction scripts.
- Create `scripts/repro/env_check.py` to verify imports, GPU, model paths, and vector shapes.
- Create `scripts/repro/run_smoke_generation.sh` to run small subsets before full evaluation.
- Create `scripts/repro/run_full_generation.sh` to run provided-vector full generation per model.
- Create `scripts/repro/classify_outputs.sh` to run `main_classify.py` on generated files when API credentials are present.
- Create `scripts/repro/summarize_results.py` to aggregate classifier scores and generation counts into a markdown/CSV summary.
- Create `docs/reproduction/adasteer_reproduction_log.md` for decisions, exact commands, deviations, and final results.
- Keep official `scripts/{llama31,qwen25,gemma2}/*.sh` unchanged until the local repro scripts work.

## Task 1: Clone And Sync On `dell`

**Files:**
- No repo file changes.

- [ ] **Step 1: Check whether the fork is already cloned on `dell`**

Run:

```bash
ssh -o BatchMode=yes 10.77.0.102 'test -d ~/Code/AdaSteer/.git && echo exists || echo missing'
```

Expected: `exists` or `missing`.

- [ ] **Step 2: Clone or update the fork**

Run if missing:

```bash
ssh -o BatchMode=yes 10.77.0.102 'mkdir -p ~/Code && git clone git@github.com:lihan3238/AdaSteer.git ~/Code/AdaSteer'
```

Run if exists:

```bash
ssh -o BatchMode=yes 10.77.0.102 'cd ~/Code/AdaSteer && git remote set-url origin git@github.com:lihan3238/AdaSteer.git && git fetch origin && git status --short'
```

Expected: clone succeeds, or existing repo has no unexpected dirty tracked files.

- [ ] **Step 3: Add upstream remote on `dell`**

Run:

```bash
ssh -o BatchMode=yes 10.77.0.102 'cd ~/Code/AdaSteer && (git remote get-url upstream >/dev/null 2>&1 && git remote set-url upstream git@github.com:MuyuenLP/AdaSteer.git || git remote add upstream git@github.com:MuyuenLP/AdaSteer.git) && git remote -v'
```

Expected: `origin` points to `lihan3238/AdaSteer`, `upstream` points to `MuyuenLP/AdaSteer`.

## Task 2: Build A Compatible Environment

**Files:**
- Create: `scripts/repro/env_check.py`
- Create: `docs/reproduction/adasteer_reproduction_log.md`

- [ ] **Step 1: Create a fresh env**

Use a fresh env because the existing `qwen` env has `transformers 4.57.1` and no `flash_attn`; the official code targets `transformers 4.46.3`.

Run:

```bash
ssh -o BatchMode=yes 10.77.0.102 'conda create -y -n adasteer-repro python=3.10'
```

Expected: env is created at `/home/dell/.conda/envs/adasteer-repro`.

- [ ] **Step 2: Install packages**

Run:

```bash
ssh -o BatchMode=yes 10.77.0.102 'source ~/miniconda3/etc/profile.d/conda.sh && conda activate adasteer-repro && pip install --upgrade pip && pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128 && pip install transformers==4.46.3 accelerate scikit-learn peft openai tqdm numpy safetensors sentencepiece protobuf'
```

Expected: install succeeds. If `torch` for CUDA 12.8 is unavailable from that index, use the installed `torch 2.9.1+cu128` base from the `qwen` env as the reference and create a conda env with the same CUDA-compatible torch first, then pin `transformers==4.46.3`.

- [ ] **Step 3: Decide flash-attention path**

Run:

```bash
ssh -o BatchMode=yes 10.77.0.102 'source ~/miniconda3/etc/profile.d/conda.sh && conda activate adasteer-repro && python - <<'"'"'PY'"'"'
try:
    import flash_attn
    print("flash_attn", flash_attn.__version__)
except Exception as e:
    print("flash_attn_missing", e)
PY'
```

Expected: either `flash_attn <version>` or `flash_attn_missing ...`.

If missing, do not install immediately. First patch the local repro scripts to pass a `--no_flash_attention_2` option or patch model loading to use `attn_implementation="eager"` / `use_flash_attention_2=False` for smoke runs. RTX 5090 plus flash-attn build can be the longest setup risk.

- [ ] **Step 4: Write `scripts/repro/env_check.py`**

Create a script that checks:

```python
from __future__ import annotations

import json
import os
import pickle
from pathlib import Path

import torch
import transformers

ROOT = Path(__file__).resolve().parents[2]

MODEL_PATHS = {
    "llama31": Path("/home/dell/Downloads/Llama-3.1-8B-Instruct-hf"),
    "qwen25": Path("/home/dell/Downloads/Qwen2.5-7B-Instruct"),
    "gemma2": Path("/home/dell/Downloads/gemma-2-9b-it"),
}

VECTOR_PATHS = {
    "llama31": ROOT / "vectors/llama31-8b-instruct",
    "qwen25": ROOT / "vectors/qwen25-7b-instruct",
    "gemma2": ROOT / "vectors/gemma2-9b-it",
}


def load_pickle_shape(path: Path) -> tuple[int, ...]:
    with path.open("rb") as handle:
        value = pickle.load(handle)
    return tuple(value.shape)


def main() -> None:
    print("torch", torch.__version__)
    print("transformers", transformers.__version__)
    print("cuda_available", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("gpu_count", torch.cuda.device_count())
        for index in range(torch.cuda.device_count()):
            print("gpu", index, torch.cuda.get_device_name(index))

    for name, path in MODEL_PATHS.items():
        assert path.exists(), f"missing model path: {path}"
        assert (path / "config.json").exists(), f"missing config: {path}"
        assert (path / "tokenizer.json").exists(), f"missing tokenizer: {path}"
        shards = sorted(path.glob("model-*.safetensors"))
        assert len(shards) == 4, f"{name}: expected 4 safetensors shards, got {len(shards)}"
        with (path / "config.json").open() as handle:
            config = json.load(handle)
        print("model", name, config.get("model_type"), len(shards), str(path))

    for name, path in VECTOR_PATHS.items():
        assert path.exists(), f"missing vector directory: {path}"
        for subdir in ["RD", "HD"]:
            vector_dir = path / subdir
            assert (vector_dir / "class_a.pkl").exists(), f"missing {vector_dir}/class_a.pkl"
            assert (vector_dir / "class_b.pkl").exists(), f"missing {vector_dir}/class_b.pkl"
            assert (vector_dir / "mean_diff.pkl").exists(), f"missing {vector_dir}/mean_diff.pkl"
            print("vector", name, subdir, "class_a", load_pickle_shape(vector_dir / "class_a.pkl"))
            print("vector", name, subdir, "mean_diff", load_pickle_shape(vector_dir / "mean_diff.pkl"))
        assert (path / "HD/proj.pkl").exists(), f"missing {path}/HD/proj.pkl"


if __name__ == "__main__":
    main()
```

- [ ] **Step 5: Run the env check**

Run:

```bash
ssh -o BatchMode=yes 10.77.0.102 'cd ~/Code/AdaSteer && source ~/miniconda3/etc/profile.d/conda.sh && conda activate adasteer-repro && pip install -e . && python scripts/repro/env_check.py'
```

Expected: torch/transformers versions print, CUDA is available, three model paths and vector shapes validate.

## Task 3: Add Local Smoke Generation Scripts

**Files:**
- Create: `scripts/repro/run_smoke_generation.sh`
- Modify only if needed: `adasteer/src/main_generate_steering_multi_adasteer.py`
- Modify only if needed: `adasteer/src/main_generate_steering_multi.py`

- [ ] **Step 1: Create tiny input subsets without changing source data**

Run:

```bash
ssh -o BatchMode=yes 10.77.0.102 'cd ~/Code/AdaSteer && mkdir -p tmp/smoke_inputs/llama31 tmp/smoke_inputs/qwen25 tmp/smoke_inputs/gemma2 && python - <<'"'"'PY'"'"'
import json
from pathlib import Path

for model in ["llama31", "qwen25", "gemma2"]:
    src = Path("data/inputs") / model
    dst = Path("tmp/smoke_inputs") / model
    for name in ["GCG", "XSTest", "OKTest", "alpaca_eval"]:
        data = json.load((src / f"{name}.json").open())
        with (dst / f"{name}.json").open("w") as handle:
            json.dump(data[:2], handle, indent=2, ensure_ascii=False)
PY'
```

Expected: each smoke dataset file contains 2 samples.

- [ ] **Step 2: Create `scripts/repro/run_smoke_generation.sh`**

Use this content:

```bash
#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
MODEL="${1:-llama31}"

case "$MODEL" in
  llama31)
    MODEL_PATH="/home/dell/Downloads/Llama-3.1-8B-Instruct-hf"
    DATA_DIR="$ROOT_DIR/tmp/smoke_inputs/llama31"
    VECTOR="$ROOT_DIR/vectors/llama31-8b-instruct/RD/mean_diff.pkl"
    MODEL_SIGN="llama31"
    DATASETS="GCG,XSTest,OKTest,alpaca_eval"
    BS=1
    ;;
  qwen25)
    MODEL_PATH="/home/dell/Downloads/Qwen2.5-7B-Instruct"
    DATA_DIR="$ROOT_DIR/tmp/smoke_inputs/qwen25"
    VECTOR="$ROOT_DIR/vectors/qwen25-7b-instruct/RD/mean_diff.pkl"
    MODEL_SIGN="qwen25"
    DATASETS="GCG,XSTest,OKTest,alpaca_eval"
    BS=1
    ;;
  gemma2)
    MODEL_PATH="/home/dell/Downloads/gemma-2-9b-it"
    DATA_DIR="$ROOT_DIR/tmp/smoke_inputs/gemma2"
    VECTOR="$ROOT_DIR/vectors/gemma2-9b-it/RD/mean_diff.pkl"
    MODEL_SIGN="gemma2"
    DATASETS="GCG,XSTest,OKTest,alpaca_eval"
    BS=1
    ;;
  *)
    echo "unknown model: $MODEL" >&2
    exit 2
    ;;
esac

mkdir -p "$ROOT_DIR/results/smoke/$MODEL/generate"

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-6}" \
python "$ROOT_DIR/adasteer/src/main_generate_steering_multi_adasteer.py" \
  --model_name_or_path "$MODEL_PATH" \
  --data_dir "$DATA_DIR" \
  --output_dir "$ROOT_DIR/results/smoke/$MODEL" \
  --model_sign "$MODEL_SIGN" \
  --dataset_list "$DATASETS" \
  --steer_vector "$VECTOR" \
  --alpha 0 \
  --overwrite True \
  --if_all_layers True \
  --if_support_sys_prompt False \
  --bs "$BS"
```

- [ ] **Step 3: Run LLaMA smoke**

Run:

```bash
ssh -o BatchMode=yes 10.77.0.102 'cd ~/Code/AdaSteer && source ~/miniconda3/etc/profile.d/conda.sh && conda activate adasteer-repro && bash scripts/repro/run_smoke_generation.sh llama31'
```

Expected: four JSON files appear under `results/smoke/llama31/generate/`, each with 2 outputs.

- [ ] **Step 4: Fix the first actual compatibility failure**

If the smoke run fails because `flash_attn` is missing, patch model loading in both generation entrypoints from:

```python
use_flash_attention_2 = True
```

to an argument-controlled value:

```python
use_flash_attention_2 = not args.disable_flash_attention_2
```

Add this field to both `Arguments` dataclasses:

```python
disable_flash_attention_2: bool = field(default=False)
```

Then add `--disable_flash_attention_2 True` to the smoke script.

Expected: smoke rerun either passes or exposes the next compatibility issue.

- [ ] **Step 5: Run Qwen and Gemma smoke**

Run:

```bash
ssh -o BatchMode=yes 10.77.0.102 'cd ~/Code/AdaSteer && source ~/miniconda3/etc/profile.d/conda.sh && conda activate adasteer-repro && bash scripts/repro/run_smoke_generation.sh qwen25 && bash scripts/repro/run_smoke_generation.sh gemma2'
```

Expected: `results/smoke/qwen25/generate/` and `results/smoke/gemma2/generate/` contain the four smoke files.

## Task 4: Full Generation With Official Vectors

**Files:**
- Create: `scripts/repro/run_full_generation.sh`

- [ ] **Step 1: Create full generation script**

Use this content:

```bash
#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
MODEL="${1:?usage: run_full_generation.sh llama31|qwen25|gemma2}"

case "$MODEL" in
  llama31)
    MODEL_PATH="/home/dell/Downloads/Llama-3.1-8B-Instruct-hf"
    DATA_DIR="$ROOT_DIR/data/inputs/llama31"
    VECTOR="$ROOT_DIR/vectors/llama31-8b-instruct/RD/mean_diff.pkl"
    MODEL_SIGN="llama31"
    DATASETS="Autodan,aim,Cipher,ReNeLLM,GCG,Multilingual,Jailbroken,XSTest,OKTest,alpaca_eval"
    BS="${BS:-8}"
    GPU="${GPU:-6}"
    ;;
  qwen25)
    MODEL_PATH="/home/dell/Downloads/Qwen2.5-7B-Instruct"
    DATA_DIR="$ROOT_DIR/data/inputs/qwen25"
    VECTOR="$ROOT_DIR/vectors/qwen25-7b-instruct/RD/mean_diff.pkl"
    MODEL_SIGN="qwen25"
    DATASETS="Autodan,aim,Cipher,ReNeLLM,GCG,Multilingual,Jailbroken,XSTest,OKTest,alpaca_eval"
    BS="${BS:-8}"
    GPU="${GPU:-7}"
    ;;
  gemma2)
    MODEL_PATH="/home/dell/Downloads/gemma-2-9b-it"
    DATA_DIR="$ROOT_DIR/data/inputs/gemma2"
    VECTOR="$ROOT_DIR/vectors/gemma2-9b-it/RD/mean_diff.pkl"
    MODEL_SIGN="gemma2"
    DATASETS="Autodan,aim,Cipher,ReNeLLM,GCG,Multilingual,Jailbroken,XSTest,OKTest,alpaca_eval"
    BS="${BS:-4}"
    GPU="${GPU:-3}"
    ;;
  *)
    echo "unknown model: $MODEL" >&2
    exit 2
    ;;
esac

OUT="$ROOT_DIR/results/full/$MODEL/adasteer"
mkdir -p "$OUT/generate"

CUDA_VISIBLE_DEVICES="$GPU" \
python "$ROOT_DIR/adasteer/src/main_generate_steering_multi_adasteer.py" \
  --model_name_or_path "$MODEL_PATH" \
  --data_dir "$DATA_DIR" \
  --output_dir "$OUT" \
  --model_sign "$MODEL_SIGN" \
  --dataset_list "$DATASETS" \
  --steer_vector "$VECTOR" \
  --alpha 0 \
  --overwrite True \
  --if_all_layers True \
  --if_support_sys_prompt False \
  --bs "$BS"
```

- [ ] **Step 2: Run full LLaMA generation**

Run:

```bash
ssh -o BatchMode=yes 10.77.0.102 'cd ~/Code/AdaSteer && source ~/miniconda3/etc/profile.d/conda.sh && conda activate adasteer-repro && GPU=6 BS=8 bash scripts/repro/run_full_generation.sh llama31'
```

Expected: `results/full/llama31/adasteer/generate/` has 10 JSON files with counts matching source datasets.

- [ ] **Step 3: Run full Qwen generation**

Run:

```bash
ssh -o BatchMode=yes 10.77.0.102 'cd ~/Code/AdaSteer && source ~/miniconda3/etc/profile.d/conda.sh && conda activate adasteer-repro && GPU=7 BS=8 bash scripts/repro/run_full_generation.sh qwen25'
```

Expected: `results/full/qwen25/adasteer/generate/` has 10 JSON files with counts matching source datasets.

- [ ] **Step 4: Run full Gemma generation**

Run:

```bash
ssh -o BatchMode=yes 10.77.0.102 'cd ~/Code/AdaSteer && source ~/miniconda3/etc/profile.d/conda.sh && conda activate adasteer-repro && GPU=3 BS=4 bash scripts/repro/run_full_generation.sh gemma2'
```

Expected: `results/full/gemma2/adasteer/generate/` has 10 JSON files with counts matching source datasets.

## Task 5: Classification And Result Aggregation

**Files:**
- Create: `scripts/repro/classify_outputs.sh`
- Create: `scripts/repro/summarize_results.py`

- [ ] **Step 1: Confirm API credential**

Run:

```bash
ssh -o BatchMode=yes 10.77.0.102 'test -n "${OPENAI_API_KEY:-}" && echo has_key || echo missing_key'
```

Expected: `has_key`. If `missing_key`, stop classification and report that generation is ready but GPT-4o scoring cannot run.

- [ ] **Step 2: Create `scripts/repro/classify_outputs.sh`**

Use this content:

```bash
#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
MODEL="${1:?usage: classify_outputs.sh llama31|qwen25|gemma2}"
BASE="$ROOT_DIR/results/full/$MODEL/adasteer"

for file in "$BASE"/generate/*.json; do
  dataset="$(basename "$file" .json)"
  mkdir -p "$BASE/classify/$dataset"
  python "$ROOT_DIR/adasteer/src/main_classify.py" \
    --data_path "$file" \
    --output_dir "$BASE/classify/$dataset" \
    --num_threads "${NUM_THREADS:-16}"
done
```

- [ ] **Step 3: Run classification per model**

Run:

```bash
ssh -o BatchMode=yes 10.77.0.102 'cd ~/Code/AdaSteer && source ~/miniconda3/etc/profile.d/conda.sh && conda activate adasteer-repro && NUM_THREADS=16 bash scripts/repro/classify_outputs.sh llama31'
ssh -o BatchMode=yes 10.77.0.102 'cd ~/Code/AdaSteer && source ~/miniconda3/etc/profile.d/conda.sh && conda activate adasteer-repro && NUM_THREADS=16 bash scripts/repro/classify_outputs.sh qwen25'
ssh -o BatchMode=yes 10.77.0.102 'cd ~/Code/AdaSteer && source ~/miniconda3/etc/profile.d/conda.sh && conda activate adasteer-repro && NUM_THREADS=16 bash scripts/repro/classify_outputs.sh gemma2'
```

Expected: each generated dataset has `classify/<dataset>/score.txt`.

- [ ] **Step 4: Create `scripts/repro/summarize_results.py`**

Use this content:

```python
from __future__ import annotations

import ast
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MODELS = ["llama31", "qwen25", "gemma2"]


def read_score(path: Path) -> dict[str, float | int | str]:
    lines = path.read_text().splitlines()
    counts = ast.literal_eval(lines[0])
    total = sum(counts.values())
    return {
        "total": total,
        "full_compliance": counts.get("1_full_compliance", 0),
        "full_refusal": counts.get("2_full_refusal", 0),
        "partial_refusal": counts.get("3_partial_refusal", 0),
        "errors": counts.get("error", 0),
        "full_compliance_rate": float(lines[1]),
        "full_refusal_rate": float(lines[2]),
    }


def main() -> None:
    rows: list[dict[str, object]] = []
    for model in MODELS:
        base = ROOT / "results" / "full" / model / "adasteer"
        for gen_path in sorted((base / "generate").glob("*.json")):
            dataset = gen_path.stem
            generated = json.load(gen_path.open())
            score_path = base / "classify" / dataset / "score.txt"
            row: dict[str, object] = {
                "model": model,
                "dataset": dataset,
                "generated": len(generated),
                "score_path": str(score_path.relative_to(ROOT)) if score_path.exists() else "",
            }
            if score_path.exists():
                row.update(read_score(score_path))
            rows.append(row)

    out_dir = ROOT / "results" / "full"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "summary.md"
    headers = ["model", "dataset", "generated", "full_compliance_rate", "full_refusal_rate", "errors", "score_path"]
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(str(row.get(h, "")) for h in headers) + " |")
    out_path.write_text("\n".join(lines) + "\n")
    print(out_path)


if __name__ == "__main__":
    main()
```

- [ ] **Step 5: Summarize results**

Run:

```bash
ssh -o BatchMode=yes 10.77.0.102 'cd ~/Code/AdaSteer && source ~/miniconda3/etc/profile.d/conda.sh && conda activate adasteer-repro && python scripts/repro/summarize_results.py'
```

Expected: `results/full/summary.md` exists and includes every generated dataset.

## Task 6: Fixed-Steering Baseline For LLaMA

**Files:**
- Create: `scripts/repro/run_llama31_fixed_refusal.sh`

- [ ] **Step 1: Create fixed baseline script**

Use this content:

```bash
#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
MODEL_PATH="/home/dell/Downloads/Llama-3.1-8B-Instruct-hf"
DATA_DIR="$ROOT_DIR/data/inputs/llama31"
VECTOR="$ROOT_DIR/vectors/llama31-8b-instruct/RD/mean_diff.pkl"
DATASETS="${DATASETS:-ReNeLLM,GCG,XSTest}"
ALPHA="${ALPHA:--0.2}"
OUT="$ROOT_DIR/results/full/llama31/refusal/ALPHA_$ALPHA"

mkdir -p "$OUT/generate"

CUDA_VISIBLE_DEVICES="${GPU:-6}" \
python "$ROOT_DIR/adasteer/src/main_generate_steering_multi.py" \
  --model_name_or_path "$MODEL_PATH" \
  --data_dir "$DATA_DIR" \
  --output_dir "$OUT" \
  --model_sign llama31 \
  --dataset_list "$DATASETS" \
  --steer_vector "$VECTOR" \
  --alpha "$ALPHA" \
  --overwrite True \
  --if_all_layers True \
  --if_support_sys_prompt False \
  --bs "${BS:-8}"
```

- [ ] **Step 2: Run fixed baseline**

Run:

```bash
ssh -o BatchMode=yes 10.77.0.102 'cd ~/Code/AdaSteer && source ~/miniconda3/etc/profile.d/conda.sh && conda activate adasteer-repro && GPU=6 BS=8 ALPHA=-0.2 bash scripts/repro/run_llama31_fixed_refusal.sh'
```

Expected: fixed-refusal outputs exist for `ReNeLLM`, `GCG`, and `XSTest`.

## Task 7: Vector Extraction Reproduction Audit

**Files:**
- Create: `docs/reproduction/vector_extraction_audit.md`

- [ ] **Step 1: Record what source data is present**

Run:

```bash
ssh -o BatchMode=yes 10.77.0.102 'cd ~/Code/AdaSteer && find data/anchors configs adasteer/extract -maxdepth 4 -type f -printf "%p %s\n" | sort'
```

Expected: currently only `data/anchors/llama31/harmful_break_or_not/test.jsonl` is present for extraction; full paper sources are not all present.

- [ ] **Step 2: Run shipped LLaMA RD extraction only after env smoke passes**

Run:

```bash
ssh -o BatchMode=yes 10.77.0.102 'cd ~/Code/AdaSteer && source ~/miniconda3/etc/profile.d/conda.sh && conda activate adasteer-repro && CUDA_VISIBLE_DEVICES=6 python adasteer/extract/Probing/main.py --config configs/llama31_RD.ini'
```

Expected: new vectors are written under `vectors/llama31-8b-instruct/refusal/`. Do not overwrite `vectors/llama31-8b-instruct/RD/mean_diff.pkl`.

- [ ] **Step 3: Compare generated refusal vector shape**

Run:

```bash
ssh -o BatchMode=yes 10.77.0.102 'cd ~/Code/AdaSteer && python - <<'"'"'PY'"'"'
import pickle
from pathlib import Path
for p in ["vectors/llama31-8b-instruct/refusal/mean_diff.pkl", "vectors/llama31-8b-instruct/RD/mean_diff.pkl"]:
    with Path(p).open("rb") as f:
        value = pickle.load(f)
    print(p, value.shape)
PY'
```

Expected: both are `(32, 4096)`. Numerical equality is not expected because the config points to a tiny anchor file, not the full paper data pipeline.

## Task 8: Reproduction Report

**Files:**
- Modify: `docs/reproduction/adasteer_reproduction_log.md`

- [ ] **Step 1: Write final log structure**

Use this structure:

```markdown
# AdaSteer Reproduction Log

Date: 2026-06-07
Machine: dell / 10.77.0.102

## Scope

- Provided-vector generation: completed / blocked
- GPT-4o classification: completed / blocked
- Fixed-steering baseline: completed / blocked
- Vector extraction: completed / partial / blocked

## Deviations From Paper

- GPU: RTX 5090 instead of single NVIDIA Tesla A100.
- Environment: record exact torch/transformers/flash-attn versions.
- Vector extraction source data: record missing original datasets if not acquired.

## Model Paths

List exact local paths and validation output.

## Results

Link to `results/full/summary.md`.

## Failure Notes

Record exact error messages and fixes.

## Interpretation For Research Direction

Summarize how AdaSteer informs VisualDojo/action-authorization work:
- useful as latent risk/steering module;
- not enough as standalone agent-action safety method;
- hidden-state intervention needs external action/provenance checks for tool agents.
```

- [ ] **Step 2: Commit reproduction-layer files**

Run:

```bash
git add docs/superpowers/plans/2026-06-07-adasteer-reproduction.md docs/reproduction/adasteer_reproduction_log.md scripts/repro
git commit -m "docs: add AdaSteer reproduction plan"
```

Expected: commit contains only reproduction scripts/docs, not generated results or model files.

## Self-Review

- Spec coverage: the plan covers note positioning, model availability, environment setup, provided-vector generation, GPT-4o classification, fixed baseline, vector extraction audit, and final report.
- Placeholder scan: no `TBD`, `TODO`, or "fill in details" instructions are present.
- Type consistency: script names, model names, vector paths, and result paths are consistent across tasks.

