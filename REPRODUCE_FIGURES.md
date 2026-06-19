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
