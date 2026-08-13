# Refinement threshold — status and how to run

Training the comparative model and **tuning the refinement threshold** are two steps. Training does not apply `refine_prediction()`. Validation Dice from training is unrefined.

`train.py` now runs the sweep automatically after comparative training ends — whether that is early stopping or `max_epochs`. It loads `best_model.pt` (the best epoch), not the last-epoch in-memory weights. Job `688160` finished before this hook existed, so that run still needs the standalone command below.

## Already run

| Run | Result | Artifact |
|---|---|---|
| Baseline training | Best val Dice **0.779**, converged epoch 45 | `src/training/checkpoints_baseline/best_model.pt` |
| Comparative training (Sonic job `688160`) | Best val Dice **0.7322** at epoch 32; early stop epoch 42 | `src/training/checkpoints_comparative/best_model.pt` |

`uncertainty_loss_weight` was **0.1** for the comparative run (single value, not swept).

## Still to run

**Refinement threshold sweep** on the **validation** split only (never test).

The script loads the comparative checkpoint, runs MC Dropout once per val case, then scores `refine_prediction()` at several thresholds. It selects the threshold with the highest mean Dice on eCSF / GM / dGM.

## How to run (Sonic)

From `code/`, venv active:

```bash
python3 -u -m src.training.tune_refinement_threshold \
    --config src/training/train_config_comparative.yaml \
    --checkpoint src/training/checkpoints_comparative/best_model.pt \
    --thresholds 0.3,0.4,0.5,0.6,0.7 \
    --n-mc-samples 10 \
    --out-dir src/training/tuning_results
```

Change `--checkpoint` if `best_model.pt` lives somewhere else.

Later comparative training jobs (`run_train_comparative.sh` / `train.py`) run this sweep themselves when training ends. Set `run_refinement_sweep: false` in the config to skip it.

### What to look for

Stdout:

```
Selected threshold: 0.X (target_dice=... vs. no-refinement ...)
```

Written to disk:

- `src/training/tuning_results/refinement_threshold_sweep.json` — full table + `"selected"`
- `src/training/tuning_results/refinement_threshold_sweep.png` — sweep plot

Give Role 5 / Role 6 the selected threshold from the JSON `"selected"` field. That is the number they use for with/without-refinement tests. The code default of `0.5` is only a placeholder until this sweep finishes.
