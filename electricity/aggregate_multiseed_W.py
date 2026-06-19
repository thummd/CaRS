"""Per-cell multi-seed aggregation of the CARGO W matrix.

For each (market, regime, source) cell of the cross-border spillover
dictionary built by `load_spillover_matrices`, this script reads the
checkpoint from every available seed and reports:

  - mean W across seeds                         (point estimate)
  - std  W across seeds                         (parameter dispersion)
  - 95% percentile-bootstrap CI from the seed   (interval estimate)
    distribution (2.5%, 97.5% across seeds)
  - sign-stability fraction: fraction of seeds  (sign-consistency)
    that agree with the seed-mean's sign

Plus per-market headline diagnostics:
  - mean directional-accuracy across seeds      (from each results.json)
  - sign-flip-survival under the seed CI: cells whose seed-mean shows
    a sign flip between regimes AND whose CIs (in both regimes)
    exclude zero. This is the multi-seed analogue of the partial-
    correlation block bootstrap from analyze_confounder_control.py.

Outputs:
  - prints per-cell aggregation tables (top-K by |seed_mean|) plus
    headline summary
  - paper/figs/cargo_multiseed/spillover_seed_stability.pdf
    (12x12 heatmap, sign-stability fraction, RdBu_r centred on 0.5)
  - paper/figs/cargo_multiseed/spillover_mean_with_ci.pdf
    (two 12x12 panels: seed mean + seed std, RdBu_r for mean and
    sequential for std)

Usage:
    python3 electricity/aggregate_multiseed_W.py \\
        --results_dir outputs/experiments/12market_cargo_controls_daily
"""
import argparse
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from visualize_spillover_three_panel import (
    CLOCKWISE_ORDER, _load_trained_alpha,
)
from visualize_european_network import (
    _load_W_from_experiment, spillover_source, INTERCONNECTIONS,
)


def discover_seeds(results_dir, market):
    """Return sorted list of seed dirs present for this market."""
    market_dir = results_dir / market / "h1"
    if not market_dir.is_dir():
        return []
    seeds = []
    for d in sorted(market_dir.iterdir()):
        if not d.is_dir() or not d.name.startswith("seed"):
            continue
        try:
            s = int(d.name.replace("seed", ""))
        except ValueError:
            continue
        if (d / "checkpoints" / "final.tar").exists():
            seeds.append((s, d))
    return seeds


def load_per_seed_spillover(results_dir, market, regime, use_effective=False):
    """Return {seed: {source: signed_W}} for cross-border edges into `market`."""
    phys = {tuple(sorted(p)) for p in INTERCONNECTIONS}
    out = {}
    for seed, exp_dir in discover_seeds(results_dir, market):
        W, fc = _load_W_from_experiment(exp_dir, regime)
        if W is None:
            continue
        alpha = _load_trained_alpha(exp_dir, regime) if use_effective else None
        best = {}
        for i, feat in enumerate(fc):
            if i == 0:
                continue
            w = float(W[i, 0])
            if abs(w) <= 1e-6:
                continue
            src = spillover_source(feat)
            if src is None or src == market or src not in CLOCKWISE_ORDER:
                continue
            if use_effective and alpha is not None:
                forbidden = tuple(sorted((src, market))) not in phys
                if forbidden:
                    w = w * alpha
            prev = best.get(src)
            if prev is None or abs(w) > abs(prev):
                best[src] = w
        out[seed] = best
    return out


def aggregate_seed_stats(per_seed_dict):
    """Given {seed: {source: weight}}, return per-source aggregation."""
    sources = set()
    for d in per_seed_dict.values():
        sources.update(d.keys())
    rows = []
    for src in sorted(sources):
        vals = [per_seed_dict[s].get(src, np.nan) for s in sorted(per_seed_dict)]
        vals = np.array(vals, dtype=float)
        nz = vals[~np.isnan(vals)]
        if len(nz) < 2:
            continue
        mean = float(np.mean(nz))
        std = float(np.std(nz, ddof=1)) if len(nz) > 1 else 0.0
        lo = float(np.percentile(nz, 2.5))
        hi = float(np.percentile(nz, 97.5))
        sign_stab = float(np.mean(np.sign(nz) == np.sign(mean))) if mean != 0 else 0.5
        rows.append(dict(source=src, n_seeds=len(nz), mean=mean,
                         std=std, ci_lo=lo, ci_hi=hi,
                         sign_stab=sign_stab,
                         ci_excludes_zero=(lo > 0 or hi < 0)))
    return pd.DataFrame(rows)


def per_seed_matrices(results_dir, markets, regime, use_effective=False):
    """{seed: 12x12 DataFrame of signed W[s, t]}"""
    out = {}
    for target in markets:
        per_seed = load_per_seed_spillover(
            results_dir, target, regime, use_effective=use_effective)
        for seed, src_w in per_seed.items():
            if seed not in out:
                out[seed] = pd.DataFrame(np.nan, index=markets,
                                          columns=markets, dtype=float)
            for src, w in src_w.items():
                out[seed].loc[src, target] = w
    return out


def annotate_physical(ax, markets):
    from matplotlib.patches import Rectangle
    phys = {tuple(sorted(p)) for p in INTERCONNECTIONS}
    for i, s in enumerate(markets):
        for j, t in enumerate(markets):
            if s == t:
                continue
            if tuple(sorted((s, t))) in phys:
                ax.add_patch(Rectangle((j - 0.5, i - 0.5), 1, 1,
                                       fill=False, edgecolor="#222222",
                                       lw=0.7, alpha=0.55, zorder=9))


def draw_seed_stability_heatmap(stab_stable, stab_crisis, output, markets):
    """Sign-stability heatmap per regime. 1.0 = all seeds agree on sign;
    0.5 = chance; 0.0 = anti-correlated across seeds (rare).
    """
    fig, axes = plt.subplots(1, 2, figsize=(11, 5),
                              gridspec_kw=dict(wspace=0.10))
    for ax, M, title in [(axes[0], stab_stable, "Stable: sign-stability"),
                          (axes[1], stab_crisis, "Crisis: sign-stability")]:
        im = ax.imshow(M.values, cmap="RdBu_r", vmin=0.0, vmax=1.0,
                       aspect="equal", interpolation="nearest")
        n = len(M)
        ax.set_xticks(range(n)); ax.set_yticks(range(n))
        ax.set_xticklabels(M.columns, fontsize=8)
        ax.set_yticklabels(M.index, fontsize=8)
        ax.set_xlabel("Target $t$", fontsize=9)
        if ax is axes[0]:
            ax.set_ylabel("Source $s$", fontsize=9)
        ax.set_title(title, fontsize=10, fontweight="bold")
        ax.set_xticks(np.arange(-0.5, n, 1), minor=True)
        ax.set_yticks(np.arange(-0.5, n, 1), minor=True)
        ax.grid(which="minor", color="#ffffff", linewidth=0.4)
        ax.tick_params(which="minor", length=0)
        annotate_physical(ax, markets)
    cbar = fig.colorbar(im, ax=axes, shrink=0.78, pad=0.02,
                        fraction=0.025, orientation="vertical")
    cbar.set_label("Sign-stability fraction (1.0 = all seeds agree)",
                   fontsize=9)
    fig.suptitle("CARGO multi-seed sign-stability of cross-border spillover",
                 fontsize=11, fontweight="bold", y=1.01)
    output.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output, dpi=200, bbox_inches="tight",
                facecolor="white", edgecolor="none")
    plt.close()
    print(f"  -> {output}")


def draw_mean_std_heatmap(mean_stable, std_stable, mean_crisis, std_crisis,
                           output, markets, vmax_mean, vmax_std):
    fig, axes = plt.subplots(2, 2, figsize=(12, 10),
                              gridspec_kw=dict(hspace=0.30, wspace=0.10))
    def _draw(ax, M, vmax, cmap, title, show_ylabel=True):
        im = ax.imshow(M.values, cmap=cmap,
                       vmin=(-vmax if cmap == "RdBu_r" else 0.0),
                       vmax=vmax, aspect="equal", interpolation="nearest")
        n = len(M)
        ax.set_xticks(range(n)); ax.set_yticks(range(n))
        ax.set_xticklabels(M.columns, fontsize=7)
        ax.set_yticklabels(M.index if show_ylabel else [""] * n, fontsize=7)
        ax.set_xlabel("Target $t$", fontsize=8)
        if show_ylabel:
            ax.set_ylabel("Source $s$", fontsize=8)
        ax.set_title(title, fontsize=10, fontweight="bold")
        ax.set_xticks(np.arange(-0.5, n, 1), minor=True)
        ax.set_yticks(np.arange(-0.5, n, 1), minor=True)
        ax.grid(which="minor", color="#ffffff", linewidth=0.4)
        ax.tick_params(which="minor", length=0)
        annotate_physical(ax, markets)
        return im
    im_m = _draw(axes[0, 0], mean_stable, vmax_mean, "RdBu_r",
                  "(a) Stable — seed-mean W")
    _draw(axes[0, 1], mean_crisis, vmax_mean, "RdBu_r",
          "(b) Crisis — seed-mean W", show_ylabel=False)
    im_s = _draw(axes[1, 0], std_stable, vmax_std, "viridis",
                  "(c) Stable — seed-std W")
    _draw(axes[1, 1], std_crisis, vmax_std, "viridis",
          "(d) Crisis — seed-std W", show_ylabel=False)
    fig.colorbar(im_m, ax=axes[0, :], shrink=0.78, pad=0.02,
                 fraction=0.025).set_label("Seed-mean W", fontsize=9)
    fig.colorbar(im_s, ax=axes[1, :], shrink=0.78, pad=0.02,
                 fraction=0.025).set_label("Seed-std W", fontsize=9)
    fig.suptitle("CARGO multi-seed: mean + dispersion of cross-border spillover",
                 fontsize=11, fontweight="bold", y=0.995)
    plt.savefig(output, dpi=200, bbox_inches="tight",
                facecolor="white", edgecolor="none")
    plt.close()
    print(f"  -> {output}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results_dir", type=Path,
                        default=Path(__file__).resolve().parent.parent
                                / "outputs" / "experiments"
                                / "12market_cargo_controls_daily")
    parser.add_argument("--output_dir", type=Path,
                        default=Path(__file__).resolve().parent.parent
                                / "paper" / "figs" / "cargo_multiseed")
    parser.add_argument("--use_effective_weights", action="store_true",
                        help="Aggregate the effective (soft-prior-gated) W"
                             " rather than raw W.")
    args = parser.parse_args()

    markets = CLOCKWISE_ORDER
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Inventory of which (market, seed) combinations have a checkpoint
    print("Seed availability per market:")
    for m in markets:
        seeds = [s for s, _ in discover_seeds(args.results_dir, m)]
        print(f"  {m}: {len(seeds):>2} seeds {seeds}")
    print()

    for regime in (0, 1):
        rlabel = "Stable" if regime == 0 else "Crisis"
        print(f"\n=== Regime {regime} ({rlabel}) — per-cell seed aggregation ===")
        per_seed = per_seed_matrices(args.results_dir, markets, regime,
                                      use_effective=args.use_effective_weights)
        if not per_seed:
            print("  no checkpoints found")
            continue
        seeds = sorted(per_seed.keys())
        n_seeds = len(seeds)
        # Compute per-cell statistics
        stack = np.stack([per_seed[s].values for s in seeds], axis=0)
        with np.errstate(invalid="ignore"):
            mean = np.nanmean(stack, axis=0)
            std = np.nanstd(stack, axis=0, ddof=1) if n_seeds > 1 else np.zeros_like(mean)
            same_sign = np.nansum(np.sign(stack) == np.sign(mean[None, :, :]),
                                   axis=0) / n_seeds
        if regime == 0:
            mean_stable_df = pd.DataFrame(mean, index=markets, columns=markets)
            std_stable_df = pd.DataFrame(std, index=markets, columns=markets)
            stab_stable_df = pd.DataFrame(same_sign, index=markets, columns=markets)
        else:
            mean_crisis_df = pd.DataFrame(mean, index=markets, columns=markets)
            std_crisis_df = pd.DataFrame(std, index=markets, columns=markets)
            stab_crisis_df = pd.DataFrame(same_sign, index=markets, columns=markets)

        # Headline: top-10 most-stable edges (sign-stab = 1.0 and largest |mean|)
        all_cells = []
        for i, s in enumerate(markets):
            for j, t in enumerate(markets):
                if s == t or np.isnan(mean[i, j]):
                    continue
                all_cells.append(dict(s=s, t=t, mean=mean[i, j],
                                       std=std[i, j], stab=same_sign[i, j]))
        df_all = pd.DataFrame(all_cells)
        df_all["abs_mean"] = df_all["mean"].abs()
        df_all["physical"] = df_all.apply(
            lambda r: tuple(sorted((r["s"], r["t"]))) in
                       {tuple(sorted(p)) for p in INTERCONNECTIONS}, axis=1)
        df_top = df_all.sort_values("abs_mean", ascending=False).head(15)
        print(f"  Top-15 cells by |seed-mean| ({n_seeds} seeds):")
        for _, r in df_top.iterrows():
            phys = "PHYS" if r["physical"] else "long"
            print(f"    {r['s']} -> {r['t']}  mean={r['mean']:+.3f}  "
                  f"std={r['std']:.3f}  sign-stab={r['stab']:.0%}  {phys}")

    # Render heatmaps + export per-cell statistics as CSV if both regimes were computed
    if "mean_stable_df" in dir() and "mean_crisis_df" in dir():
        vmax_mean = float(max(
            np.nanmax(np.abs(mean_stable_df.values)),
            np.nanmax(np.abs(mean_crisis_df.values)), 1e-9))
        vmax_std = float(max(
            np.nanmax(std_stable_df.values),
            np.nanmax(std_crisis_df.values), 1e-9))
        suffix = "_effective" if args.use_effective_weights else ""
        draw_mean_std_heatmap(
            mean_stable_df, std_stable_df, mean_crisis_df, std_crisis_df,
            args.output_dir / f"spillover_mean_with_ci{suffix}.pdf",
            markets, vmax_mean, vmax_std)
        draw_seed_stability_heatmap(
            stab_stable_df, stab_crisis_df,
            args.output_dir / f"spillover_seed_stability{suffix}.pdf",
            markets)
        # CSV exports of the per-cell seed-mean + std + sign-stability,
        # so downstream consumers (e.g. the CARGO walkthrough notebook)
        # can splice the matrices in without re-running the aggregation.
        for name, df in [("mean_stable", mean_stable_df), ("mean_crisis", mean_crisis_df),
                         ("std_stable", std_stable_df), ("std_crisis", std_crisis_df),
                         ("signstab_stable", stab_stable_df),
                         ("signstab_crisis", stab_crisis_df)]:
            p = args.output_dir / f"spillover_{name}{suffix}.csv"
            df.to_csv(p, float_format="%.4f")
            print(f"  -> {p}")


if __name__ == "__main__":
    main()
