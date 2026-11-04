# GEAR supervised fine-tuning

This directory collects the cold-start SFT utilities used before GEAR reinforcement learning. The
training itself is delegated to [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory); GEAR does
not vendor a second trainer here.

## Contents

- `configs/qwen2_vl_3b_full_sft.yaml`: paper-style full-parameter Qwen2.5-VL-3B configuration.
- `data/dataset_info.example.json`: LLaMA-Factory registrations for solver and rubricator data.
- `scripts/prepare_frozenlake_sft.py`: normalize and split FrozenLake/Navigation-style samples.
- `scripts/extract_svg_solver_sft.py`: extract high-quality SVG solver traces from rollouts.
- `scripts/generate_svg_rubrics.py`: generate SVG rubricator targets with an OpenAI-compatible VLM.
- `scripts/convert_generated_rubrics.py`: convert generated SVG rubric records into SFT messages.
- `scripts/convert_sokoban_rubric_sft.py`: convert Sokoban rubric traces to `messages` JSONL.
- `scripts/sample_jsonl.py`: reproducible JSONL sampling.
- `scripts/verify_sft_data.py`: validate roles, image placeholders, files, and assistant targets.

The copied utilities originated in the adjacent LLaMA-Factory workspace and were cleaned up for
portable paths, command-line use, deterministic sampling, and environment-based credentials.

## Expected data format

Both solver and rubricator datasets use one JSON object per line:

```json
{
  "messages": [
    {"role": "system", "content": "..."},
    {"role": "user", "content": "<image>\n..."},
    {"role": "assistant", "content": "..."}
  ],
  "images": ["/absolute/path/to/image.png"]
}
```

Keep solver trajectories and rubricator traces distinguishable during data construction, then
register the final files as `gear_solver_sft` and `gear_rubricator_sft`. For a paper-style cold
start, prepare 20K high-quality reasoning/solver examples and 20K rubric evaluation examples with
reject sampling. The scripts in this directory are preparation tools; generated data and teacher
model outputs are not included in the repository.

## Training

1. Install LLaMA-Factory in a separate environment or checkout.
2. Copy `data/dataset_info.example.json` into its dataset directory as `dataset_info.json` and edit
   both `file_name` values.
3. Edit `dataset_dir`, model path, and output path in the YAML if necessary.
4. Validate each JSONL file before starting training:

```bash
python SFT/scripts/verify_sft_data.py /path/to/gear_solver_sft.jsonl
python SFT/scripts/verify_sft_data.py /path/to/gear_rubricator_sft.jsonl
```

5. Start full-parameter SFT:

```bash
bash SFT/train_llamafactory.sh
```

The YAML follows the manuscript settings where LLaMA-Factory supports them directly: sequence
length 32768, batch size 1, gradient accumulation 1, learning rate `2e-5`, cosine decay toward
`2e-6`, weight decay 0.1, warmup ratio 0.01, and 5 epochs. Confirm that the installed
LLaMA-Factory/Transformers version supports `cosine_with_min_lr` and `lr_scheduler_kwargs`.

API keys must be supplied through environment variables or command-line arguments. Do not commit
teacher-model credentials or generated private data.

## Task preparation examples

Sokoban rubricator traces:

```bash
python SFT/scripts/convert_sokoban_rubric_sft.py \
  --input /path/to/raw_sokoban_rubrics.jsonl \
  --output /path/to/sokoban_rubricator_sft.jsonl \
  --skip_invalid
```

FrozenLake solver traces:

```bash
python SFT/scripts/prepare_frozenlake_sft.py /path/to/frozenlake.jsonl \
  --train-output /path/to/frozenlake_train.jsonl \
  --val-output /path/to/frozenlake_val.jsonl
```

SVG solver and rubricator traces:

```bash
python SFT/scripts/extract_svg_solver_sft.py \
  --root_dir /path/to/svg_rollout_data \
  --output_file /path/to/svg_solver.jsonl

export OPENAI_API_KEY=...
export OPENAI_BASE_URL=https://your-teacher-endpoint/v1
export RUBRIC_TEACHER_MODEL=your-vision-language-model
python SFT/scripts/generate_svg_rubrics.py \
  --input /path/to/svg_solver.jsonl \
  --output /path/to/svg_rubrics.jsonl
python SFT/scripts/convert_generated_rubrics.py \
  /path/to/svg_rubrics.jsonl /path/to/svg_rubricator_sft.jsonl
```

Merge or register the task files according to the desired experiment mixture. The repository does
not silently concatenate them because the final solver/rubricator balance is an experimental choice.
