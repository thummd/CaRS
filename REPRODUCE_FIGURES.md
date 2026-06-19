# Reproducing the two CaRS paper figures

This repo ships the code **and** a small curated data subset needed to
regenerate two figures from the paper:

1. `paper/figs/cargo_multiseed/regime_timeseries_panel.pdf` — day-ahead
   price history for the 12 markets, coloured by the CaRS-CAM discovered
   regime (daily, multi-seed consensus).
2. `paper/figs/cargo/spillover_three_panel_augmented_effective.pdf` —
   stable vs. crisis cross-border spillover network (inside the ring) with
   domestic feature drivers (outside the ring), using the effective
   (soft-prior-gated) structural weights.

The reference PDFs are committed alongside this file so you can diff your
output against them.

## Environment

```bash
# Figure 1 needs only: numpy, pandas, matplotlib
# Figure 2 (from the shipped extract) needs only: numpy, matplotlib
pip install numpy pandas matplotlib
```

`torch` is **not** required to reproduce the figures from the committed
data — the heavy PyTorch checkpoints have been replaced by a tiny
pre-extracted weight tensor (see "How the data was slimmed" below). It is
only needed if you re-run `electricity/extract_W_tensors.py` against the
original checkpoints.

## Figure 1 — regime time-series panel

```bash
python3 electricity/plot_regime_timeseries.py \
    --trajectory_dir outputs/regime_trajectories_daily \
    --output_dir paper/figs/cargo_multiseed \
    --frequency D --no_split_by_group \
    --markets SE DE CZ HU AT IT ES FR BE NL PL DK
```

Reads:
- `data/unified/unified_<MKT>_2015_2026_clean.csv` — daily price + feature
  series per market (the plot uses `Day_Ahead_Price`; the AT panel also
  reads the DE series to back-fill the pre-2018 merged-bidding-zone period).
- `outputs/regime_trajectories_daily/regime_trajectory_<MKT>.csv` — the
  multi-seed consensus regime posterior per market.

## Figure 2 — augmented three-panel spillover (effective W)

```bash
python3 electricity/visualize_spillover_three_panel_augmented.py \
    --use_effective_weights
```

Reads `outputs/experiments/12market_cargo_controls_daily/<MKT>/h1/seed<N>/`
for the 12 markets × 5 seeds, using per seed dir:
- `W_tensors.npz` — extracted structural weight tensors + soft-prior alpha
  logits (replaces the original `checkpoints/final.tar`).
- `config.json` — holds `data.feature_cols`.
- `regime_assignments.npy`, `actuals.npy` — used to price-canonicalise the
  arbitrary per-(market, seed) regime indices so the Stable/Crisis panels
  are comparable (regime 1 = higher-mean-price = Crisis).

The figure is the five-seed mean; weights are price-canonicalised per
(market, seed) before averaging.

## How the data was slimmed (Figure 2)

The spillover figure originally read the full `checkpoints/final.tar`
PyTorch checkpoints (~859 MB across 12 markets × 5 seeds). Only two
tensors per regime are actually used: the structural weight matrix
`causal_emissions.<r>.icgnn.W` and the soft-prior gate logit
`causal_emissions.<r>.icgnn.physical_prior_alpha_logit`. These are
extracted into a ~40 KB-per-seed `W_tensors.npz` by:

```bash
python3 electricity/extract_W_tensors.py     # needs torch + the final.tar files
```

`visualize_european_network._load_W_from_experiment` and
`visualize_spillover_three_panel._load_trained_alpha` transparently prefer
`final.tar` when present and fall back to `W_tensors.npz` otherwise, so the
figure reproduces bit-for-bit from the extract alone (verified: the
spillover and domestic-driver weight dictionaries are numerically
identical to the full-checkpoint render).

## Results tables

The forecast-quality table is computed entirely from the small per-seed
`results.json` metric files (no model checkpoint needed):

```bash
python3 electricity/render_forecast_quality_table.py
# CARGO rows come from
#   outputs/experiments/12market_cargo_controls_daily/<MKT>/h{1,7,14}/seed<N>/results.json
# Multi-horizon aggregation only:
python3 electricity/aggregate_multihorizon_metrics.py
```

**Baseline rows caveat.** `render_forecast_quality_table.load_baselines`
expects `--baseline_dir <dir>/<MKT>/h{h}/baseline_results_seed42.json`
(default `outputs/baselines/daily`). The shipped baseline metrics are the
single-seed (seed 42) run at `outputs/baselines/<MKT>/baseline_results_seed42.json`
— a flat, h1-only layout — so with the defaults the CARGO rows populate
and the baseline rows render as `---`. Point `--baseline_dir` at a matching
per-horizon layout, or regenerate the baselines with
`electricity/baselines/run_baselines.py`, to fill them in.

## Model checkpoints & CaRS model code

The full trained CaRS checkpoints are committed under
`outputs/experiments/<experiment>/<MKT>/h{H}/seed<N>/checkpoints/final.tar`
(106 checkpoints, ~1.2 GB total) across four experiments
(`12market_cargo_controls_daily`, `12market_cargo_controls`,
`daily_dec8_confirm`, `12market_gat_spillover`). The model/training code
that produced and can load them lives in `shared_backbone/`:

- `shared_backbone/models/ds3m_causal.py` — the DS3M-causal (CaRS) model.
- `shared_backbone/modules/{causal_emission,shared_dag,hierarchical_dag,physical_prior}.py`
  — the per-regime causal emission heads (the `causal_emissions.<r>.icgnn.W`
  weights), the shared structural DAG, and the soft physical prior.
- `shared_backbone/training/train_e2e.py`, `shared_backbone/run_shared_backbone.py`
  — end-to-end training / experiment entry points.

A checkpoint's `model_state_dict` is a plain tensor dict, e.g.:

```python
import torch
sd = torch.load(".../seed42/checkpoints/final.tar",
                map_location="cpu", weights_only=False)["model_state_dict"]
W = sd["causal_emissions.0.icgnn.W"]   # structural weights, regime 0
```

Note: the checkpoints let you re-run **inference** without retraining, but
inference and retraining both also require the full ~31 GB unified feature
data (only the 12 daily price CSVs needed for Figure 1 are shipped here).
