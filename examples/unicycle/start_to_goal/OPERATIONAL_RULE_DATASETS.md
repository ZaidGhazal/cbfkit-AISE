# Operational Rule Datasets (Unicycle)

This setup generates operational-rule datasets for two systems:

1. `unicycle_static_obstacle`
2. `unicycle_dynamic_obstacle`

## Safety rule

A trajectory is labeled `Pass` iff the barrier stays nonnegative for the full rollout:

`h(x(t), t) = ||p(t) - p_obs(t)||^2 - r^2 >= 0` for all simulated timesteps.

Otherwise the outcome is `Fail`.

## Input features

### Static obstacle system

- `initial_distance_to_obstacle` [m], range `[0.7, 2.6]`
- `initial_speed` [m/s], range `[0.0, 2.5]`
- `obstacle_radius` [m], range `[0.35, 1.0]`
- `initial_heading_error` [rad], range `[-pi, pi]`

### Dynamic obstacle system

- `initial_distance_to_obstacle` [m], range `[0.9, 3.0]`
- `initial_speed` [m/s], range `[0.0, 2.5]`
- `obstacle_radius` [m], range `[0.35, 0.9]`
- `initial_heading_error` [rad], range `[-pi, pi]`
- `obstacle_speed` [m/s], range `[0.0, 1.2]`
- `obstacle_heading` [rad], range `[-pi, pi]`

## Sampling strategy

- Uniform random sampling over feature bounds.
- Rejection condition to enforce initially safe starts: `initial_distance_to_obstacle > obstacle_radius + 0.05`.

## Controllers

- `D_legacy`: nominal proportional controller (no CBF)
- `D_evolved`: vanilla CBF-QP controller with zeroing CBF condition `alpha(h) = alpha * h`

`D_evolved` is filtered to include only rows labeled `Pass`.

## Simulation settings

- Time step: `dt = 0.02 s`
- Duration: `tf = 6.0 s` (`300` steps)

## Reproducibility

Random seeds are saved in each system's `metadata.json`.

When generating multiple batch folders, pass a distinct batch index or seed offset.
Otherwise each batch reuses the same sampled operational cases.

## Generate datasets

From repo root:

```bash
python3 examples/unicycle/start_to_goal/generate_operational_rule_datasets.py --n-samples 200

# Also export paired one-to-one comparison dataset
python3 examples/unicycle/start_to_goal/generate_operational_rule_datasets.py \
  --n-samples 200 \
  --paired-index-datasets

# Independent batch run; records the effective offset in metadata.json
python3 examples/unicycle/start_to_goal/generate_operational_rule_datasets.py \
  --n-samples 50 \
  --paired-index-datasets \
  --batch-index 7 \
  --out-dir examples/unicycle/start_to_goal/results/vanilla/batch7/samples_50
```

Outputs are written to:

`examples/unicycle/start_to_goal/results/operational_rule_datasets/`

With `--paired-index-datasets`, each system folder also includes:

- `D_paired_comparison.csv`: same test case per row, with both `legacy_*` and `evolved_*` labels/results.

## Visualize simulations

Use the visualization utility to replay any row from either dataset and render:

- trajectory
- obstacle(s)
- first failure location (if safety violation occurs)
- optional animated GIF

Script:

`examples/unicycle/start_to_goal/visualize_operational_rule_simulations.py`

Examples (paired mode only):

```bash
# Static obstacle, paired comparison for row 12, save PNG
python3 examples/unicycle/start_to_goal/visualize_operational_rule_simulations.py \
  --system unicycle_static_obstacle \
  --index 12

# Dynamic obstacle, paired comparison for row 3, save PNG + GIF
python3 examples/unicycle/start_to_goal/visualize_operational_rule_simulations.py \
  --system unicycle_dynamic_obstacle \
  --index 3 \
  --animate
```

Default output location:

`examples/unicycle/start_to_goal/results/operational_rule_datasets/visualizations/<system>/`
