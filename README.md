# GEAR: Co-Evolving Grounded VLM Agents with Environment-Aligned Rubrics

This repository implements **GEAR (Grounded Environment-Aligned Rubrics)**.
GEAR trains a multi-turn vision-language agent under two coupled roles:

- **Solver** observes the environment, reasons over interaction history, and emits executable actions.
- **Rubricator** proposes task- and trajectory-specific process criteria.
- **Verifier** judges whether a fixed criterion is satisfied by complete trajectories or local transitions.
- **Environment alignment** accepts a criterion only when its satisfaction is positively correlated with grounded environment feedback.

The central idea is simple: self-generated rubrics are hypotheses, not truth. They become process
rewards only after environment-aligned validation.

## Installation

```bash
conda create -n vagen python=3.12 -y
conda activate vagen

git submodule update --init --recursive

cd verl
USE_MEGATRON=0 bash scripts/install_vllm_sglang_mcore.sh
pip install --no-deps -e .
cd ..

pip install -e .
pip install "trl==0.26.2"
```

Verify the renamed package:

```bash
PYTHONPATH=.:verl python -c \
  "from vagen.gear import compute_score; print('GEAR import OK')"
```

## Supervised fine-tuning

GEAR uses a cold-start SFT stage for solver reasoning and rubricator behavior before reinforcement
learning. The task-oriented preparation utilities for Sokoban, FrozenLake, and SVG are collected in
[`SFT/`](SFT/README.md). Model training is performed with
[LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory); update the example dataset registrations
and model paths, then run `bash SFT/train_llamafactory.sh`.

## Quick start

### Development launcher

```bash
bash examples/train/sokoban/train_ppo_qwen25vl3b_gear_policy_dual_role.sh
```

This is a development configuration, not an exact paper reproduction: it uses `rollout.n=1`,
`alpha=0.2`, a heuristic verifier, and repository-specific optimization weights.

### Multi-trajectory training

The launchers sample eight trajectories per prompt and expose grounded validation, warmup,
protected role updates, rubric auditing, and task-specific targets:

```bash
# Sokoban
CUDA_VISIBLE_DEVICES=0,1 N_GPUS=2 \
bash examples/train/sokoban/train_gear_multi_trajectory.sh

# FrozenLake
CUDA_VISIBLE_DEVICES=0,1 N_GPUS=2 \
bash examples/train/frozenlake/train_gear_multi_trajectory.sh

# SVG
CUDA_VISIBLE_DEVICES=0,1 N_GPUS=2 \
bash examples/train/svg/train_gear_multi_trajectory.sh
```

### External verifier

```bash
custom_reward_function.reward_kwargs.verifier.mode=external \
custom_reward_function.reward_kwargs.verifier.base_url=http://127.0.0.1:5009/v1 \
custom_reward_function.reward_kwargs.verifier.model=qwen-verifier
```


## Configuration

Core options live under `custom_reward_function.reward_kwargs`.

| Option | Meaning |
|---|---|
| `alpha` | Minimum accepted rubric–environment Pearson correlation |
| `lambda_cot` | Scale of validated solver process reward |
| `outcome_weight` | Scale of grounded environment reward |
| `turn_level_process_reward` | Enable local transition scoring and turn-end placement |
| `normalize_rubric_weights` | Clamp weights and normalize positive weights |
| `alignment_target` | `auto`, `traj_success`, `env_reward`, `visual_similarity`, or an extra-info key |
| `rubric_validation_enabled` | Enable environment-aligned filtering |
| `ema_rule_corr_filter` | Allow supported historical-correlation fallback |
| `corr_min_samples` | Minimum historical support |
| `corr_schema_version` | Namespace version for tracker state |
| `rubricator.mode` | `policy`, `external`, or `heuristic` |
| `verifier.mode` | `heuristic`, `grounded`, `svg_grounded`, or `external` |
| `enable_dual_role_update` | Train the rubricator role as well as the solver |

Rubric weights are clamped to [-20, 20]. Positive weights are normalized to sum to one, while
negative weights remain penalties. The selected score is normalized to [0, 1].

The scheduler can warm up criterion supervision, decay it when alignment remains weak, fall back to
pure outcome reward, freeze on plateaus, or enter recovery/warm-restart states.

## Citation

The supplied manuscript is anonymous and under review. Until public bibliographic metadata is
available, cite it by title:

> *GEAR: Co-Evolving Grounded VLM Agents with Environment-Aligned Rubrics.* Anonymous authors,
> under review, 2027.

Please also cite VAGEN and VERL when using this implementation.

This work is on top of
[VAGEN](https://github.com/mll-lab-nu/VAGEN) and
[VERL](https://github.com/volcengine/verl).
