# CaRS — Causal Regime-Switching for European electricity prices

CaRS (a.k.a. CaRS-CARGO) is a deep state-space model that jointly learns
(i) a latent **regime** sequence (e.g. stable vs. crisis) and (ii) a
per-regime **structural causal graph** over price drivers and cross-border
spillovers, trained end-to-end for day-ahead price forecasting across 12
European markets.

This README is the entry point for the repo: it maps the (many) code
files, then walks the pipeline **data → train CaRS → results tables →
figures**. Figure-specific details and the data manifest live in
[`REPRODUCE_FIGURES.md`](REPRODUCE_FIGURES.md).

> **Checkpoints use Git LFS.** Install `git-lfs` *before* cloning (or run
> `git lfs install && git lfs pull` after) to fetch the `*.tar` model
> checkpoints; otherwise you get small pointer files.

## Repository layout

| Path | What's there |
|---|---|
| `shared_backbone/` | **The CaRS model & training code** (see below). |
| `electricity/` | Data pipeline, baselines, table generators, figure scripts. |
| `experiments/` | Shell launchers for the multi-seed / multi-horizon training sweeps. |
| `outputs/experiments/<exp>/<MKT>/h<H>/seed<N>/` | Per-run outputs: `checkpoints/final.tar`, `results.json`, `config.json`, predictions/actuals. |
| `outputs/baselines/<MKT>/` | Baseline metric JSONs. |
| `outputs/regime_trajectories_daily/` | Multi-seed consensus regime posteriors (for Figure 1). |
| `data/unified/` | Unified per-market feature CSVs (only the 12 daily price files are shipped; the full set is ~31 GB). |
| `paper/figs/`, `paper/tables/` | Rendered figures and LaTeX tables. |
| `validation/` | Cross-border spillover validation (DY/BK/TE/Granger vs. CaRS edges). |
| `upstream/` | Reference implementations (DS3M, FANTOM) CaRS builds on. |

### CaRS model code (`shared_backbone/`)

| File | Role |
|---|---|
| `models/ds3m_causal.py` | The DS3M-causal (CaRS) model: regime HMM + per-regime causal emission. |
| `modules/causal_emission.py` | Per-regime causal emission head — holds the structural weights `causal_emissions.<r>.icgnn.W`. |
| `modules/shared_dag.py` | Shared structural DAG / NOTEARS acyclicity machinery. |
| `modules/hierarchical_dag.py` | Hierarchical DAG variant. |
| `modules/physical_prior.py` | Soft physical prior gating forbidden cross-border edges (`physical_prior_alpha_logit`). |
| `training/train_e2e.py` | End-to-end augmented-Lagrangian training loop. |
| `run_shared_backbone.py` | **CLI entry point** — trains one (market, horizon, seed) run. |
| `data_loader.py` | Builds train/val/test tensors from the unified dataset. |

## Setup

```bash
git lfs install            # once per machine
pip install torch numpy pandas matplotlib scikit-learn statsmodels xgboost
```

All paths derive from `CARS_ROOT` (defaults to the repo root; override with
the `CARS_ROOT` env var). See `paths.py`.

## 1. Build the unified dataset (optional — daily price CSVs are shipped)

Training reads `data/unified/unified_<MKT>_<range>{_hourly}_clean.csv`. To
(re)build it from the raw source data:

```bash
python3 electricity/create_unified_dataset.py --countries DE,FR,NL,BE,AT,CZ,PL,HU,IT,ES,DK,SE --frequency D
```

The raw inputs (ENTSO-E, weather, commodities, …) are ~31 GB and are **not**
shipped; the `electricity/download_*.py` scripts document each source.

## 2. Train CaRS

One run = one `(market, horizon, seed)`. This is the exact command used for
the daily 5-seed sweep (from `experiments/launch_cargo_controls_daily_multiseed.sh`):

```bash
python3 -m shared_backbone.run_shared_backbone \
    --market DE --horizon 1 --seed 42 --frequency D \
    --output_dir outputs/experiments/12market_cargo_controls_daily/DE/h1/seed42 \
    --h_dim 32 --z_dim 8 --d_dim 2 --lag 1 \
    --sharing_mode independent --w_init_scale 0.1 \
    --aggregation_mode cam_gat --cam_hidden_dim 32 \
    --lambda_dag 10.0 --lambda_sparse 1.0 --lambda_var_reg 0.01 \
    --lambda_target 10.0 --lambda_regime_diff 1.0 \
    --elastic_threshold 0.05 --elastic_weight 0.3 \
    --feature_groups price,load,weather,calendar,gen_forecast,demand_forecast,spillover,spgci,commodity \
    --spillover --physical_prior_mode soft --physical_prior_alpha_init 0.05 \
    --batch_size 256 --task_type prediction \
    --early_stopping_metric directional_accuracy \
    --max_auglag_steps 15 --max_inner_epochs 15 --early_stopping_patience 15 \
    --patience_dag 3 --use_amp
```

Each run writes to its `--output_dir`:
- `checkpoints/final.tar` — model weights (`d_dim=2` ⇒ regimes 0 and 1).
- `results.json` — RMSE / MAE / Spearman / directional accuracy.
- `regime_assignments.npy`, `actuals.npy`, `predictions.npy`, `config.json`.

**Full sweep** (12 markets × {h1,h7,h14} × 5 seeds), with filesystem-based
idempotent skipping:

```bash
CARS_GPU=0 bash experiments/launch_cargo_controls_daily_multiseed.sh
# Override the grid via env vars, e.g.:
HORIZONS="1" SEEDS="42" MARKETS="DE FR" CARS_GPU=0 \
    bash experiments/launch_cargo_controls_daily_multiseed.sh
```

> Re-training/inference needs the full ~31 GB unified data. To just **load**
> a shipped checkpoint, see the snippet in `REPRODUCE_FIGURES.md`.

## 3. Baselines

```bash
python3 -m electricity.baselines.run_baselines \
    --countries DE,FR,NL,BE,AT,CZ,PL,HU,IT,ES,DK,SE --baselines all
# -> outputs/baselines/<MKT>/baseline_results_seed42.json
# Models: XGBoost, Lasso-per-regime, MS-VAR, naive {zero, mean, persistence}.
# (electricity/baselines/lstm_baseline.py is a standalone LSTM, not wired into this runner.)
```

## 4. Results tables

The forecast-quality table reads only the per-seed `results.json` files (no
checkpoint needed):

```bash
python3 electricity/render_forecast_quality_table.py
# CARGO rows  <- outputs/experiments/12market_cargo_controls_daily/<MKT>/h{1,7,14}/seed<N>/results.json
# baseline rows <- --baseline_dir (see caveat)
# writes paper/tables/forecast_quality_vs_baselines.tex + CSVs

python3 electricity/aggregate_multihorizon_metrics.py   # multi-horizon metric aggregation
python3 electricity/aggregate_multiseed_W.py            # per-edge W mean/std/CI/sign-stability over seeds
```

**Baseline-rows caveat.** `render_forecast_quality_table.load_baselines`
expects `--baseline_dir <dir>/<MKT>/h<H>/baseline_results_seed42.json`
(default `outputs/baselines/daily`), but `run_baselines.py` writes a flat
`outputs/baselines/<MKT>/baseline_results_seed42.json` (single seed, h1).
With the defaults the CARGO rows populate and baseline rows render as `---`;
point `--baseline_dir` at a matching per-horizon layout to fill them in.

## 5. Figures

See [`REPRODUCE_FIGURES.md`](REPRODUCE_FIGURES.md) for the two paper figures
(regime time-series panel; augmented stable/crisis spillover network),
including how the heavy checkpoints were slimmed to a tiny `W_tensors.npz`
extract so Figure 2 reproduces without the full weights.
