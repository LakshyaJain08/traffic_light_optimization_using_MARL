# Model Performance Report

## Run Summary
- Date: 2026-04-06
- Training mode: MARL PPO (`torch`)
- Episodes: 10
- Reward objective: waiting time + CO2 + fuel
- CO2/fuel normalization: aggressive log scaling enabled (`co2_log_scale=100000`, `fuel_log_scale=100000`)

## Final Evaluation (Baseline vs MARL)
| Metric | Baseline | MARL | Improvement |
|---|---:|---:|---:|
| Waiting time per step | 187.8457 | 12.9447 | 93.1% reduction |
| CO2 per step | 166988.4694 | 129398.2338 | 22.5% reduction |
| Fuel per step | 54179.7351 | 41984.9426 | 22.5% reduction |

## Validation Against Targets
- Waiting time reduction target: >10% -> Achieved (93.1%)
- CO2 reduction target: >5% -> Achieved (22.5%)
- Fuel reduction target: >5% -> Achieved (22.5%)

## Why This Model Is Better
- The trained MARL policy dramatically reduces congestion compared to fixed-time control.
- Emissions and fuel consumption are also reduced substantially, showing that optimization is not only traffic-speed focused.
- The objective now reflects true multi-objective behavior (traffic + environmental impact), and all success criteria are met.

## Saved Model
- Archived checkpoint: `results/archive/good_model_20260406_000709/checkpoints`
- Active checkpoint used for evaluation: `results/checkpoints`

## Generated Artifacts
- Evaluation metrics CSV: `results/evaluation_metrics.csv`
- Comparison plot: `results/comparison_plot.png`
- Training log: `results/training_logs.csv`
