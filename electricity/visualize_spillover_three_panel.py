"""Three-panel inter-market spillover comparison: stable | crisis | sign flips.

For each regime, extracts the CaRS-learned cross-border spillover weight
between every ordered pair of the 12 European markets from per-country
checkpoints under `shared_backbone/results/` and renders three circular
network panels:

  (1) Stable regime spillover (all edges)
  (2) Crisis regime spillover (all edges)
  (3) Sign-flipped edges only — edges whose sign differs between regimes
      AND whose magnitude in BOTH regimes exceeds a noise threshold
      (default 10% of the shared max|W|). The threshold suppresses
      shrinkage-driven sign flips on near-zero weights, which would
      otherwise dominate the panel without representing genuine regime
      structure.

All three panels share:
  - Identical node positions on a circle (clockwise geographic ordering,
    SE at 12 o'clock).
  - Edge widths on a single normalisation: w_max = max|W| across both
    regimes, so a thick edge means the same thing in every panel.
  - Sign-coded colour: red = positive spillover, blue = negative.

The original per-regime figures in
  paper/figs/european_causal_network_gat_spillover_regime{0,1}.pdf
overlay domestic feature importance with inter-market spillover and use
per-regime normalisation, which makes the headline finding — sign
inversions across regimes — hard to read by eye. This script renders the
spillover layer in isolation so that finding becomes the visual claim.

Usage:
    python3 visualize_spillover_three_panel.py
    python3 visualize_spillover_three_panel.py --mask_frac 0.05
"""
import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch

from visualize_european_network import (
    _load_W_from_experiment, spillover_source, INTERCONNECTIONS,
)


def find_gat_spillover_experiment(country_dir, seed=42):
    """For the `outputs/experiments/12market_gat_spillover/` layout,
    each country has a fixed `h1/seed42/` subdirectory holding a single
    checkpoint. Return that path if it exists, else None.
    """
    if not country_dir.is_dir():
        return None
    exp = country_dir / "h1" / f"seed{seed}"
    if (exp / "checkpoints" / "final.tar").exists() and (exp / "config.json").exists():
        return exp
    return None


# Clockwise geographic ordering: SE at 12 o'clock, going clockwise around
# the European map. DE (central hub) is placed between DK and PL so most of
# its strong physical interconnections (DK, PL, NL, BE, CZ) sit near it on
# the circle, keeping short chords for expected neighbour effects and long
# chords for surprising long-range spillovers.
CLOCKWISE_ORDER = [
    "SE", "DK", "DE", "PL", "CZ", "HU", "AT", "IT", "ES", "FR", "BE", "NL",
]

POS_COLOR = "#E24B4A"   # positive spillover  (red)
NEG_COLOR = "#378ADD"   # negative spillover  (blue)
FLIP_DASH = (0, (5, 3))  # dashed style for the sign-flipped panel


def circle_positions(markets, radius=1.0, center=(0.0, 0.0)):
    """Return {market: (x, y)} on a circle, SE at 12 o'clock, going clockwise."""
    n = len(markets)
    pos = {}
    for i, m in enumerate(markets):
        theta = np.pi / 2 - 2 * np.pi * i / n  # clockwise from top
        pos[m] = (center[0] + radius * np.cos(theta),
                  center[1] + radius * np.sin(theta))
    return pos


def _load_trained_alpha(exp_dir, regime):
    """Return the trained `sigmoid(physical_prior_alpha_logit)` for this
    market+regime checkpoint, or None if the checkpoint was trained
    without the soft prior (`physical_prior_mode=off`).
    """
    import math
    ckpt_path = exp_dir / "checkpoints" / "final.tar"
    if ckpt_path.exists():
        import torch
        sd = torch.load(ckpt_path, map_location="cpu",
                        weights_only=False)["model_state_dict"]
        key = f"causal_emissions.{regime}.icgnn.physical_prior_alpha_logit"
        if key not in sd:
            return None
        return 1.0 / (1.0 + math.exp(-float(sd[key].item())))

    # Lightweight reproduction path: alpha logit pre-extracted next to the
    # config by electricity/extract_W_tensors.py (NaN when the checkpoint
    # was trained with the soft prior off).
    npz_path = exp_dir / "W_tensors.npz"
    if npz_path.exists():
        data = np.load(npz_path)
        key = f"alpha_logit_{regime}"
        if key not in data.files:
            return None
        val = float(data[key])
        if np.isnan(val):
            return None
        return 1.0 / (1.0 + math.exp(-val))

    return None


def load_spillover_matrices(results_dir, markets, use_effective_weights=False):
    """Return {regime: {(source, target): weight}} for inter-market spillovers.

    For each target market T we load its per-country checkpoint, then for
    every source country S != T we keep the signed weight with the largest
    |W| across S's price/flow lag features. This is the same per-source
    aggregation rule used by `visualize_european_network.load_network_data`,
    but emitted as a dense (source, target) dictionary rather than a per-
    market top-N list, so the two regimes can be compared edge-for-edge.

    If `use_effective_weights=True`, multiply each cross-border weight by
    its CARGO soft-prior pass-through factor:
      gate = 1                          if (S, T) is a physical interconnect
      gate = sigmoid(alpha)             if (S, T) is forbidden by the prior
      gate = 1                          if the checkpoint has no soft prior
    This reflects what the *operational* model emits into the prediction
    (raw W on forbidden edges is scaled up to compensate for the gate
    during training, so the raw-W matrix overstates non-physical
    contribution; the effective view is the honest one).
    """
    physical_pairs = {tuple(sorted(p)) for p in INTERCONNECTIONS}
    out = {0: {}, 1: {}}
    market_set = set(markets)
    for target in markets:
        country_dir = results_dir / target
        exp_dir = find_gat_spillover_experiment(country_dir)
        if exp_dir is None:
            print(f"  WARNING: no spillover checkpoint for {target}"
                  f" (expected {country_dir / 'h1' / 'seed42'})")
            continue
        for regime in (0, 1):
            W, feature_cols = _load_W_from_experiment(exp_dir, regime)
            if W is None or feature_cols is None:
                continue
            alpha = (_load_trained_alpha(exp_dir, regime)
                     if use_effective_weights else None)
            best_per_source = {}
            for i, feat in enumerate(feature_cols):
                if i == 0:  # price index, autoregressive
                    continue
                w = float(W[i, 0])
                if abs(w) <= 1e-6:
                    continue
                src = spillover_source(feat)
                if src is None or src == target or src not in market_set:
                    continue
                if use_effective_weights and alpha is not None:
                    is_forbidden = (tuple(sorted((src, target)))
                                    not in physical_pairs)
                    if is_forbidden:
                        w = w * alpha
                prev = best_per_source.get(src)
                if prev is None or abs(w) > abs(prev):
                    best_per_source[src] = w
            for src, w in best_per_source.items():
                out[regime][(src, target)] = w
    return out


def edge_endpoints(positions, src, tgt, node_radius=0.085, curvature=0.18):
    """Endpoints offset to the node boundary; signed curvature so reverse
    pairs sit on opposite arcs and don't overlap visually.
    """
    sx, sy = positions[src]
    tx, ty = positions[tgt]
    dx, dy = tx - sx, ty - sy
    length = float(np.hypot(dx, dy)) or 1.0
    ux, uy = dx / length, dy / length
    x1, y1 = sx + node_radius * ux, sy + node_radius * uy
    x2, y2 = tx - node_radius * ux, ty - node_radius * uy
    rad = curvature if src < tgt else -curvature
    return (x1, y1), (x2, y2), rad


def draw_panel(ax, positions, edges, w_max, title,
               node_radius=0.085, dashed=False):
    """Draw one circular network panel.

    edges: iterable of (src, tgt, signed_weight). `dashed=True` switches
    the line style to dashed (used for the sign-flipped panel).
    """
    ax.set_xlim(-1.30, 1.30)
    ax.set_ylim(-1.30, 1.30)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title(title, fontsize=11, fontweight="bold", pad=6)

    # Faint guide ring
    ax.add_patch(plt.Circle((0, 0), 1.0, fill=False,
                            ec="#dcdcdc", lw=0.5, ls=(0, (2, 3)),
                            zorder=1))

    for (src, tgt, w) in edges:
        if src not in positions or tgt not in positions:
            continue
        (x1, y1), (x2, y2), rad = edge_endpoints(positions, src, tgt,
                                                  node_radius)
        color = POS_COLOR if w > 0 else NEG_COLOR
        rel = abs(w) / max(w_max, 1e-9)
        lw = 0.6 + 5.0 * rel
        arrow = FancyArrowPatch(
            (x1, y1), (x2, y2),
            connectionstyle=f"arc3,rad={rad}",
            arrowstyle="-|>",
            mutation_scale=9 + 7 * rel,
            color=color, lw=lw, alpha=0.78,
            linestyle=FLIP_DASH if dashed else "-",
            zorder=3,
        )
        ax.add_patch(arrow)

    for m, (x, y) in positions.items():
        ax.add_patch(plt.Circle((x, y), node_radius,
                                facecolor="#fafafa",
                                edgecolor="#2c3e50", lw=0.8, zorder=10))
        ax.text(x, y, m, ha="center", va="center",
                fontsize=9, fontweight="bold",
                color="#2c3e50", zorder=11)


def draw_three_panel(W_stable, W_crisis, output_path,
                     mask_frac=0.10, top_pct=50.0):
    markets = CLOCKWISE_ORDER
    positions = circle_positions(markets)

    w_max = max(
        max((abs(w) for w in W_stable.values()), default=0.0),
        max((abs(w) for w in W_crisis.values()), default=0.0),
        1e-9,
    )
    mask_threshold = mask_frac * w_max
    print(f"  shared max|W| = {w_max:.3f}")
    print(f"  noise mask threshold = {mask_threshold:.3f}"
          f" ({mask_frac:.0%} of max|W|)")

    # Declutter stable/crisis panels: keep only the top `top_pct`% of
    # edges by |W|, using a single threshold derived from the pooled
    # distribution across both regimes so the two panels remain
    # directly comparable in what they show. The sign-flipped panel
    # (c) is intentionally NOT subject to this cutoff — its filter
    # is mask_threshold, and applying both would risk dropping
    # genuine flips whose magnitude is moderate but resolved.
    pooled_abs = ([abs(w) for w in W_stable.values()]
                  + [abs(w) for w in W_crisis.values()])
    top_pct = float(np.clip(top_pct, 1.0, 100.0))
    declutter_threshold = float(np.percentile(pooled_abs, 100.0 - top_pct))
    print(f"  declutter threshold = {declutter_threshold:.3f}"
          f" (keep top {top_pct:.0f}% by |W|, applies to panels a+b only)")

    stable_edges = [(s, t, w) for (s, t), w in W_stable.items()
                    if abs(w) >= declutter_threshold]
    crisis_edges = [(s, t, w) for (s, t), w in W_crisis.items()
                    if abs(w) >= declutter_threshold]

    flipped_edges = []
    n_raw_flips = 0
    for (s, t) in set(W_stable) & set(W_crisis):
        w0, w1 = W_stable[(s, t)], W_crisis[(s, t)]
        if (w0 > 0) == (w1 > 0):
            continue
        n_raw_flips += 1
        if abs(w0) < mask_threshold or abs(w1) < mask_threshold:
            continue
        # Show the crisis-regime weight in the flip panel — the sign the
        # edge moves *to* under crisis is the load-bearing piece of the claim.
        flipped_edges.append((s, t, w1))

    print(f"  edges:  stable={len(stable_edges)}  crisis={len(crisis_edges)}"
          f"  raw sign flips={n_raw_flips}"
          f"  masked sign flips={len(flipped_edges)}")

    fig, axes = plt.subplots(1, 3, figsize=(15.5, 5.6))
    draw_panel(axes[0], positions, stable_edges, w_max,
               f"(a) Stable regime (top {top_pct:.0f}% by |W|)")
    draw_panel(axes[1], positions, crisis_edges, w_max,
               f"(b) Crisis regime (top {top_pct:.0f}% by |W|)")
    draw_panel(axes[2], positions, flipped_edges, w_max,
               f"(c) Sign-flipped only (both $|W|>{mask_threshold:.2f}$)",
               dashed=True)

    legend_elems = [
        plt.Line2D([0], [0], color=POS_COLOR, lw=2.4,
                   label="Positive spillover"),
        plt.Line2D([0], [0], color=NEG_COLOR, lw=2.4,
                   label="Negative spillover"),
        plt.Line2D([0], [0], color="#7f7f7f", lw=2.0, ls=FLIP_DASH,
                   label="Sign-flipped across regimes"),
        plt.Line2D([0], [0], color="#7f7f7f", lw=0.8, alpha=0.6,
                   label=fr"Edge width $\propto |W|$ (shared max ${w_max:.2f}$)"),
    ]
    fig.suptitle(
        "CaRS inter-market spillover: stable vs. crisis regime comparison",
        fontsize=13, fontweight="bold", y=0.995,
    )
    fig.legend(handles=legend_elems, loc="upper center",
               bbox_to_anchor=(0.5, 0.955),
               ncol=4, fontsize=9, framealpha=0.95,
               borderaxespad=0.0)

    plt.tight_layout(rect=[0, 0, 1, 0.90])
    plt.savefig(output_path, dpi=200, bbox_inches="tight",
                facecolor="white", edgecolor="none")
    plt.close()
    print(f"  -> saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results_dir", type=Path,
                        default=Path(__file__).resolve().parent.parent
                                / "outputs" / "experiments"
                                / "12market_gat_spillover")
    parser.add_argument("--output", type=Path,
                        default=Path(__file__).resolve().parent.parent
                                / "paper" / "figs"
                                / "spillover_three_panel.pdf")
    parser.add_argument("--mask_frac", type=float, default=0.10,
                        help="Sign-flip mask: edges with min(|W_stable|,"
                             " |W_crisis|) < mask_frac * max|W| are dropped"
                             " from the flipped-only panel.")
    parser.add_argument("--top_pct", type=float, default=50.0,
                        help="Declutter panels (a) and (b): keep only edges"
                             " whose |W| is in the top X%% of the pooled"
                             " stable+crisis distribution. Use 100 to show"
                             " every edge. Does not affect panel (c).")
    parser.add_argument("--use_effective_weights", action="store_true",
                        help="Multiply each forbidden cross-border weight"
                             " by sigmoid(physical_prior_alpha_logit) loaded"
                             " from the checkpoint, so the figure shows what"
                             " the operational model emits rather than the"
                             " raw W parameter. No-op if the checkpoint was"
                             " trained without the soft prior.")
    args = parser.parse_args()

    if args.use_effective_weights:
        print("Loading EFFECTIVE spillover matrices "
              "(raw W gated by sigmoid(alpha) on forbidden edges) "
              f"from {args.results_dir} ...")
    else:
        print(f"Loading raw spillover matrices from {args.results_dir} ...")
    matrices = load_spillover_matrices(
        args.results_dir, CLOCKWISE_ORDER,
        use_effective_weights=args.use_effective_weights)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    draw_three_panel(matrices[0], matrices[1], args.output,
                     mask_frac=args.mask_frac, top_pct=args.top_pct)


if __name__ == "__main__":
    main()
