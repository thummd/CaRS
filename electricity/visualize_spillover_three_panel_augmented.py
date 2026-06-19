"""Augmented three-panel spillover figure: cross-border price spillovers
INSIDE each circular panel + domestic feature drivers OUTSIDE.

This extends `visualize_spillover_three_panel.py`. Each of the three
circular panels (Stable | Crisis | sign-flipped) keeps the 12-market
cross-border spillover network drawn *inside* the ring (curved arrows on
the `CLOCKWISE_ORDER` layout, red=positive / blue=negative, width
proportional to |W|). On top of that, for every market node we now draw
its top domestic causal drivers as satellite nodes radiating *outward*
from the ring (away from the circle centre), with a thin edge from each
domestic driver into its market node. Satellites are coloured by feature
type (weather / generation / forecast / load / calendar / price / ...),
exactly as in `visualize_european_network.draw_network`.

Data source
-----------
Unlike the original three-panel script (which reads a single seed42
checkpoint from the *hourly* `12market_gat_spillover/` layout), this
figure reads the **daily** CARGO experiment
`outputs/experiments/daily_dec8_confirm/`. Only seed 42 was retained for
that run (the other seeds were stopped early), so this is a **single
seed-42** figure, not a 5-seed mean: by default `--single_seed 42` is
applied and the per-source / per-feature signed weights come from that
one checkpoint. The price-canonicalisation below is still applied to that
single seed. Pass a different `--single_seed`, or `--single_seed -1` to
average over every seed found, if you point `--results_dir` at an
experiment that has more than one seed.

Regime semantics caveat (price-canonicalised)
---------------------------------------------
CaRS regime indices are arbitrary per (market, seed) -- label switching
means the model-internal regime 0/1 mixes low- and high-price states
across markets/seeds. To make the "Stable"/"Crisis" panels semantically
comparable, this script price-canonicalises each (market, seed) before
the W from its two regimes is read: for each (market, seed) it reads
`regime_assignments.npy` (per-timestep argmax regime) and `actuals.npy`
(that market's own standardised target price), computes the mean price
under each internal regime, and - following the established
`aggregate_regime_trajectories.relabel_by_price` convention - ensures
canonical regime 1 is always the higher-mean-price state (Crisis) and
canonical regime 0 the lower-mean-price state (Stable). If a seed's
internal labels are reversed it is swapped; seeds with missing files are
skipped (with a warning). When more than one seed is present the
canonicalised per-regime weights are averaged across seeds; with the
default single seed-42 source there is just the one canonicalised seed.
A seed that is degenerate (one regime essentially unoccupied, <5
timesteps) cannot be price-discriminated and is kept with its internal
labelling unchanged, exactly as `relabel_by_price` does -- for the dec8
seed-42 run most markets are dominated by a single occupied regime, so
their two panels show that one regime's W under the identity labelling.

Two normalisation layers
-------------------------
Cross-border (spillover) edge widths and domestic-satellite edge widths
are normalised **separately**: the spillover layer uses the shared
max|W| across both regimes' cross-border weights, and the domestic layer
uses the max|W| across all domestic driver weights. They are kept
separate because the two layers live at different magnitudes and a
single shared scale would render the (already weak) domestic edges
invisible. Each is documented in the legend.

Daily-W weak-identification caveat
----------------------------------
The daily structural W is currently only weakly identified (many edges
sit near random init), so the domestic satellites may look weak/noisy.
This is rendered faithfully. The only decluttering applied is the same
top-coverage rule the source scripts use (cumulative |W| coverage capped
at ~5 satellites per market) plus the cross-border top-pct / sign-flip
mask inherited from the original three-panel script.

Usage:
    python3 electricity/visualize_spillover_three_panel_augmented.py
    python3 electricity/visualize_spillover_three_panel_augmented.py \
        --use_effective_weights
    python3 electricity/visualize_spillover_three_panel_augmented.py \
        --single_seed 42
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
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch

from visualize_european_network import (
    _load_W_from_experiment, spillover_source, classify_feature,
    shorten_feature_name, FEATURE_TYPES, INTERCONNECTIONS,
)
from visualize_spillover_three_panel import (
    CLOCKWISE_ORDER, POS_COLOR, NEG_COLOR, FLIP_DASH,
    circle_positions, edge_endpoints, _load_trained_alpha,
)


# ----------------------------------------------------------------------
# Daily-experiment checkpoint discovery
# ----------------------------------------------------------------------
def discover_seed_dirs(results_dir, market, single_seed=None):
    """Return list of (seed, exp_dir) with a valid checkpoint+config for a
    market under the daily `<market>/h1/seed<N>/` layout. If `single_seed`
    is given, return at most that one seed.
    """
    h1 = results_dir / market / "h1"
    if not h1.is_dir():
        return []
    out = []
    for d in sorted(h1.iterdir()):
        if not d.is_dir() or not d.name.startswith("seed"):
            continue
        try:
            s = int(d.name.replace("seed", ""))
        except ValueError:
            continue
        if single_seed is not None and s != single_seed:
            continue
        # Accept either the full PyTorch checkpoint or the lightweight
        # W_tensors.npz extract (see electricity/extract_W_tensors.py), so
        # the figure reproduces from the small committed extract alone.
        has_weights = ((d / "checkpoints" / "final.tar").exists()
                       or (d / "W_tensors.npz").exists())
        if has_weights and (d / "config.json").exists():
            out.append((s, d))
    return out


# ----------------------------------------------------------------------
# Price-canonical regime relabelling (per market, per seed)
# ----------------------------------------------------------------------
# CaRS regime indices are arbitrary per (market, seed) -- label switching.
# We canonicalise so that canonical regime 1 = higher-mean-price (Crisis)
# and canonical regime 0 = lower-mean-price (Stable), following the same
# convention as `aggregate_regime_trajectories.relabel_by_price`.
#
# `internal_regime_for(exp_dir, canonical_regime)` returns the model-
# internal regime index whose `causal_emissions.{idx}` should be loaded
# for the requested canonical regime, or None if the seed must be skipped
# (missing canonicalisation files). A seed that is degenerate (one regime
# with <5 assigned timesteps) cannot be price-discriminated, so we keep
# its internal labelling unchanged (identity), matching `relabel_by_price`.
_RELABEL_CACHE = {}


def _seed_needs_swap(exp_dir):
    """Decide whether a seed's internal regimes are reversed vs canonical.

    Returns one of:
        False  -> internal labels already canonical (no swap)
        True   -> internal labels reversed (swap regime 0 <-> 1)
        None   -> cannot canonicalise (missing files) -> skip this seed
    Cached per `exp_dir`.
    """
    key = str(exp_dir)
    if key in _RELABEL_CACHE:
        return _RELABEL_CACHE[key]

    ra_path = exp_dir / "regime_assignments.npy"
    ac_path = exp_dir / "actuals.npy"
    if not ra_path.exists() or not ac_path.exists():
        print(f"    WARNING: cannot canonicalise {exp_dir.name} under"
              f" {exp_dir.parent.parent.name} (missing"
              f" {'regime_assignments.npy' if not ra_path.exists() else 'actuals.npy'})"
              " -- skipping seed from average")
        _RELABEL_CACHE[key] = None
        return None

    ra = np.asarray(np.load(ra_path)).ravel()
    ac = np.load(ac_path)
    # `actuals` is the market's own target price; take the target column if
    # it is multi-dimensional ([T, H] or [T, markets] -> column 0 = target).
    ac = np.asarray(ac)
    if ac.ndim > 1:
        ac = ac[:, 0]
    ac = ac.ravel()

    # Align on the trailing/overlapping window if lengths differ.
    n = min(len(ra), len(ac))
    ra, ac = ra[-n:], ac[-n:]

    m0 = ra == 0
    m1 = ra == 1
    # Degenerate seed: one regime essentially unoccupied -> cannot price-
    # discriminate. Keep internal labelling (no swap), as relabel_by_price.
    if m0.sum() < 5 or m1.sum() < 5:
        _RELABEL_CACHE[key] = False
        return False

    mean_p0 = float(np.nanmean(ac[m0]))
    mean_p1 = float(np.nanmean(ac[m1]))
    # If internal regime 0 has the HIGHER mean price, the internal labels
    # are reversed relative to canonical (canonical 1 = higher price).
    swap = mean_p0 > mean_p1
    _RELABEL_CACHE[key] = swap
    return swap


def internal_regime_for(exp_dir, canonical_regime):
    """Internal regime index to load for a given canonical regime, or None
    if the seed should be skipped (missing canonicalisation files).
    canonical_regime: 0 = Stable (lower price), 1 = Crisis (higher price).
    """
    swap = _seed_needs_swap(exp_dir)
    if swap is None:
        return None
    return (1 - canonical_regime) if swap else canonical_regime


def report_canonicalisation():
    """Summarise the per-(market, seed) canonicalisation decisions cached
    during loading: how many seeds were swapped / kept / skipped."""
    n_swap = sum(1 for v in _RELABEL_CACHE.values() if v is True)
    n_keep = sum(1 for v in _RELABEL_CACHE.values() if v is False)
    n_skip = sum(1 for v in _RELABEL_CACHE.values() if v is None)
    print(f"  price-canonicalisation: {len(_RELABEL_CACHE)} (market,seed)"
          f" pairs -> swapped={n_swap}, kept-as-is={n_keep},"
          f" skipped(missing files)={n_skip}")
    return n_swap, n_keep, n_skip


# ----------------------------------------------------------------------
# Cross-border spillover layer (seed-averaged)
# ----------------------------------------------------------------------
def _best_per_source(W, feature_cols, target, market_set,
                     use_effective, alpha, physical_pairs):
    """Per-source best-|W| signed cross-border weight into `target`,
    matching the aggregation rule in `load_spillover_matrices`."""
    best = {}
    for i, feat in enumerate(feature_cols):
        if i == 0:
            continue
        w = float(W[i, 0])
        if abs(w) <= 1e-6:
            continue
        src = spillover_source(feat)
        if src is None or src == target or src not in market_set:
            continue
        if use_effective and alpha is not None:
            if tuple(sorted((src, target))) not in physical_pairs:
                w = w * alpha
        prev = best.get(src)
        if prev is None or abs(w) > abs(prev):
            best[src] = w
    return best


def load_spillover_matrices_daily(results_dir, markets,
                                  use_effective_weights=False,
                                  single_seed=None):
    """{regime: {(source, target): seed_mean_weight}} over the daily
    experiment, averaging the per-source best-|W| weight across seeds."""
    physical_pairs = {tuple(sorted(p)) for p in INTERCONNECTIONS}
    market_set = set(markets)
    out = {0: {}, 1: {}}
    n_seeds_used = {}
    for target in markets:
        seed_dirs = discover_seed_dirs(results_dir, target, single_seed)
        if not seed_dirs:
            print(f"  WARNING: no daily checkpoint for {target} under"
                  f" {results_dir / target / 'h1'}")
            continue
        n_seeds_used[target] = len(seed_dirs)
        for regime in (0, 1):
            # `regime` is the CANONICAL regime (0=Stable/lower price,
            # 1=Crisis/higher price). Accumulate per-source weights across
            # seeds (loading each seed's price-canonicalised internal
            # regime), then mean.
            acc = {}
            for _, exp_dir in seed_dirs:
                internal = internal_regime_for(exp_dir, regime)
                if internal is None:
                    continue  # missing canonicalisation files -> skip seed
                W, fc = _load_W_from_experiment(exp_dir, internal)
                if W is None or fc is None:
                    continue
                alpha = (_load_trained_alpha(exp_dir, internal)
                         if use_effective_weights else None)
                best = _best_per_source(W, fc, target, market_set,
                                        use_effective_weights, alpha,
                                        physical_pairs)
                for src, w in best.items():
                    acc.setdefault(src, []).append(w)
            for src, vals in acc.items():
                out[regime][(src, target)] = float(np.mean(vals))
    if n_seeds_used:
        uniq = sorted(set(n_seeds_used.values()))
        print(f"  spillover layer: seeds per market = {uniq}")
    return out


# ----------------------------------------------------------------------
# Domestic satellite layer (seed-averaged)
# ----------------------------------------------------------------------
def _domestic_drivers_one_ckpt(W, feature_cols, market,
                               cumulative_threshold, max_satellites):
    """Top domestic drivers into price for one checkpoint, using the same
    cumulative-coverage selection as `visualize_european_network`.
    Returns dict feat -> signed weight (selected subset)."""
    domestic = []
    for i, feat in enumerate(feature_cols):
        if i == 0:
            continue
        w = float(W[i, 0])
        if abs(w) <= 1e-6:
            continue
        src = spillover_source(feat)
        if src is not None and src != market:
            continue  # cross-border -> handled by the arrow layer
        domestic.append((feat, w))
    domestic.sort(key=lambda x: abs(x[1]), reverse=True)
    if not domestic:
        return {}
    total = sum(abs(w) for _, w in domestic)
    selected = []
    cum = 0.0
    for feat, w in domestic:
        selected.append((feat, w))
        cum += abs(w)
        if total > 0 and cum / total >= cumulative_threshold:
            break
    selected = selected[:max_satellites]
    return dict(selected)


def load_domestic_satellites_daily(results_dir, markets, regime,
                                   cumulative_threshold=0.90,
                                   max_satellites=5, single_seed=None):
    """Per-market top domestic drivers, seed-averaged. For each market we
    average each domestic feature's signed weight across seeds (over seeds
    that have a checkpoint), then re-run the cumulative-coverage selection
    on the seed-mean weights. Returns {market: [ {feature, short_name,
    weight, type, color}, ... ]} ordered by |seed-mean weight| desc.
    """
    network = {}
    for market in markets:
        seed_dirs = discover_seed_dirs(results_dir, market, single_seed)
        if not seed_dirs:
            continue
        # Average each domestic feature's weight across seeds. We average
        # the *full* domestic weight vector (not the per-seed selection),
        # so the seed-mean is an honest point estimate before selection.
        acc = {}
        for _, exp_dir in seed_dirs:
            # `regime` is the CANONICAL regime; load the seed's price-
            # canonicalised internal regime (or skip if uncanonicalisable).
            internal = internal_regime_for(exp_dir, regime)
            if internal is None:
                continue
            W, fc = _load_W_from_experiment(exp_dir, internal)
            if W is None or fc is None:
                continue
            for i, feat in enumerate(fc):
                if i == 0:
                    continue
                src = spillover_source(feat)
                if src is not None and src != market:
                    continue
                acc.setdefault(feat, []).append(float(W[i, 0]))
        if not acc:
            continue
        mean_w = {f: float(np.mean(v)) for f, v in acc.items()}
        # Cumulative-coverage selection on the seed-mean weights.
        ranked = sorted(mean_w.items(), key=lambda x: abs(x[1]), reverse=True)
        total = sum(abs(w) for _, w in ranked) or 1.0
        selected, cum = [], 0.0
        for feat, w in ranked:
            if abs(w) <= 1e-6:
                break
            selected.append((feat, w))
            cum += abs(w)
            if cum / total >= cumulative_threshold:
                break
        selected = selected[:max_satellites]
        sats = []
        for feat, w in selected:
            ftype, color = classify_feature(feat)
            sats.append(dict(feature=feat,
                             short_name=shorten_feature_name(feat),
                             weight=w, type=ftype, color=color))
        network[market] = sats
    return network


# ----------------------------------------------------------------------
# Drawing
# ----------------------------------------------------------------------
RING_RADIUS = 1.0
NODE_RADIUS = 0.075
SAT_NODE_R = 0.032         # satellite node radius

# Satellite placement (anti-overlap).
# The 12 markets sit 30 deg apart on the ring, so to guarantee that two
# *adjacent* markets' satellite fans never collide we keep each fan strictly
# inside that market's own 30 deg angular slice: a half-width below 15 deg
# leaves a gap between neighbours. Within a single fan two anti-overlap
# devices are combined: (i) the satellite *markers* alternate between a near
# and a far radial band so consecutive markers are offset in angle *and*
# radius, and (ii) every satellite's *label* is pushed to its own distinct
# radius on a monotonically increasing "staircase" (inner satellite ->
# nearest label, outer satellite -> furthest label). Because no two labels
# in a fan ever share a radius, they cannot stack on one another even when
# the fan points almost straight up/down (the case that used to collide on
# SE / DK / NL).
SAT_FAN_HALF = np.radians(13.5)   # < 15 deg => no adjacent-market overlap
SAT_R_NEAR = 0.50                 # inner satellite marker band (dist. from node)
SAT_R_FAR = 0.72                  # outer satellite marker band
SAT_LABEL_R0 = 0.60              # first (innermost) label radius from the node
SAT_LABEL_DR = 0.135             # radial step between successive label radii


def _satellite_positions(cx, cy, n):
    """Place `n` satellites in a fan that points radially OUTWARD from the
    ring centre (0,0) through the market node, so satellites sit outside
    the ring rather than overlapping the inner network.

    Returns a list of ``(sx, sy, ang, r_marker, r_label)`` per satellite:
    ``ang`` is the radial bearing from the node, ``r_marker`` the radial
    distance of the satellite marker (near/far-staggered) and ``r_label``
    the radial distance of that satellite's text label, placed on a
    monotonically increasing staircase so labels never share a radius. The
    fan stays inside the market's own angular slice (see ``SAT_FAN_HALF``).
    """
    if n == 0:
        return []
    base = np.arctan2(cy, cx)              # outward radial direction
    # Spread the fan to the full per-market half-width once there are enough
    # satellites to need it; keep a tighter fan for just one or two so they
    # do not splay unnecessarily.
    half = SAT_FAN_HALF if n >= 3 else SAT_FAN_HALF * (0.5 if n == 2 else 0.0)
    if n == 1:
        angles = [base]
    else:
        angles = [base - half + 2 * half * k / (n - 1) for k in range(n)]
    out = []
    for k, a in enumerate(angles):
        # Alternate near/far marker band so adjacent satellites in the fan are
        # also radially offset (clears the dense-fan marker overlap)...
        r_marker = SAT_R_NEAR if (k % 2 == 0) else SAT_R_FAR
        # ...and give every satellite a distinct label radius (staircase), so
        # no two labels in the fan can occupy the same radius and collide.
        r_label = SAT_LABEL_R0 + SAT_LABEL_DR * k
        out.append((cx + r_marker * np.cos(a), cy + r_marker * np.sin(a),
                    a, r_marker, r_label))
    return out


def _separate_satellite_labels(ax, labels, max_iter=120, pad_px=1.2):
    """Push apart satellite labels whose rendered boxes overlap.

    `labels` is a list of ``(text_artist, bearing)``. The staircase radii in
    `_satellite_positions` already keep almost every fan clean, but very long
    feature names can still be wider than the radial step and clip a
    neighbour. This pass renders the labels, and for any overlapping pair it
    shifts the *outer-most* one a little further out along its own outward
    bearing (in display space) and repeats, so the cross-border arrows and
    marker geometry are untouched -- only label text positions move.
    """
    if len(labels) < 2:
        return
    fig = ax.figure
    fig.canvas.draw()                      # need a renderer for window extents
    rend = fig.canvas.get_renderer()
    inv = ax.transData.inverted()

    def boxes():
        return [t.get_window_extent(renderer=rend).expanded(1.0, 1.0)
                for t, _ in labels]

    # Outward display-space direction for each label's bearing (y axis may be
    # flipped between data and display, so derive it from the transform).
    dirs = []
    for _, ang in labels:
        p0 = ax.transData.transform((0.0, 0.0))
        p1 = ax.transData.transform((np.cos(ang), np.sin(ang)))
        v = np.array([p1[0] - p0[0], p1[1] - p0[1]], float)
        nrm = np.hypot(*v) or 1.0
        dirs.append(v / nrm)

    for _ in range(max_iter):
        bxs = boxes()
        moved = False
        for i in range(len(bxs)):
            for j in range(i + 1, len(bxs)):
                bi, bj = bxs[i], bxs[j]
                # Overlap test with a small padding.
                if (bi.x0 - pad_px < bj.x1 and bj.x0 - pad_px < bi.x1
                        and bi.y0 - pad_px < bj.y1 and bj.y0 - pad_px < bi.y1):
                    # Move whichever label is already further from the axes
                    # centre, so inner labels stay anchored near their node and
                    # only the outer one drifts outward along its bearing.
                    cen = ax.transData.transform((0.0, 0.0))
                    di = np.hypot(*(np.array(bi.get_points().mean(axis=0)) - cen))
                    dj = np.hypot(*(np.array(bj.get_points().mean(axis=0)) - cen))
                    k = i if di >= dj else j
                    txt = labels[k][0]
                    cur_disp = ax.transData.transform(txt.get_position())
                    step = (bi.height + bj.height) * 0.5 + pad_px
                    new_disp = cur_disp + dirs[k] * step
                    txt.set_position(tuple(inv.transform(new_disp)))
                    moved = True
        if not moved:
            break


def draw_panel(ax, positions, spill_edges, domestic, w_max_spill,
               w_max_dom, title, dashed=False):
    """One augmented circular panel.

    spill_edges : iterable of (src, tgt, signed_weight)  -> arrows inside ring
    domestic    : {market: [ {short_name, weight, type, color}, ... ]}
    """
    ax.set_xlim(-2.45, 2.45)
    ax.set_ylim(-2.45, 2.45)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title(title, fontsize=11, fontweight="bold", pad=6)

    # Faint guide ring
    ax.add_patch(plt.Circle((0, 0), RING_RADIUS, fill=False,
                            ec="#dcdcdc", lw=0.5, ls=(0, (2, 3)), zorder=1))

    # --- (1) Domestic satellites OUTSIDE the ring (drawn first, behind) ---
    def dom_lw(w):
        return 0.4 + 2.2 * (abs(w) / max(w_max_dom, 1e-9))

    sat_labels = []   # (text_artist, bearing_angle) for the de-collision pass
    sat_leaders = []  # (marker_x, marker_y, text_artist) -> thin leader lines
    for m, sats in domestic.items():
        if m not in positions:
            continue
        cx, cy = positions[m]
        sat_pos = _satellite_positions(cx, cy, len(sats))
        for (edge, (sx, sy, ang, r_marker, r_label)) in zip(sats, sat_pos):
            w = edge["weight"]
            # Edge from satellite boundary into the market-node boundary,
            # computed in data coords so it stops cleanly at the node ring.
            dx, dy = cx - sx, cy - sy
            seg = float(np.hypot(dx, dy)) or 1.0
            ux, uy = dx / seg, dy / seg
            ex1, ey1 = sx + SAT_NODE_R * ux, sy + SAT_NODE_R * uy
            ex2, ey2 = cx - NODE_RADIUS * ux, cy - NODE_RADIUS * uy
            ax.add_patch(FancyArrowPatch(
                (ex1, ey1), (ex2, ey2),
                arrowstyle="-",
                color=edge["color"], lw=dom_lw(w), alpha=0.6, zorder=2))
            ax.add_patch(plt.Circle((sx, sy), SAT_NODE_R,
                                    facecolor=edge["color"],
                                    edgecolor="#2c3e50", lw=0.4,
                                    alpha=0.9, zorder=6))
            # Label at this satellite's own bearing but on its dedicated
            # staircase radius (see `_satellite_positions`), so every label in
            # the fan sits at a distinct radius and cannot stack on another.
            lx = cx + r_label * np.cos(ang)
            ly = cy + r_label * np.sin(ang)
            ha = "left" if np.cos(ang) >= 0 else "right"
            txt = ax.text(lx, ly, edge["short_name"], ha=ha, va="center",
                          fontsize=4.6, color="#2c3e50", zorder=7,
                          bbox=dict(boxstyle="round,pad=0.08", facecolor="white",
                                    edgecolor="none", alpha=0.65))
            sat_labels.append((txt, ang))
            sat_leaders.append((sx, sy, txt))

    # Final safety net: nudge apart any satellite labels whose *rendered*
    # boxes still overlap (long feature names can be wider than the staircase
    # radial step). Each colliding label is pushed further out along its own
    # bearing until its box clears its neighbours.
    _separate_satellite_labels(ax, sat_labels)

    # Leader lines: connect every satellite marker to its (possibly displaced)
    # label so each label unambiguously maps to its own domestic-feature edge,
    # even though labels are offset onto the anti-overlap staircase.
    for (sx, sy, txt) in sat_leaders:
        lx, ly = txt.get_position()
        ax.plot([sx, lx], [sy, ly], color="#95a5a6", lw=0.3,
                alpha=0.55, zorder=5, solid_capstyle="round")

    # --- (2) Cross-border spillover arrows INSIDE the ring ---
    for (src, tgt, w) in spill_edges:
        if src not in positions or tgt not in positions:
            continue
        (x1, y1), (x2, y2), rad = edge_endpoints(positions, src, tgt,
                                                 NODE_RADIUS)
        color = POS_COLOR if w > 0 else NEG_COLOR
        rel = abs(w) / max(w_max_spill, 1e-9)
        ax.add_patch(FancyArrowPatch(
            (x1, y1), (x2, y2),
            connectionstyle=f"arc3,rad={rad}",
            arrowstyle="-|>", mutation_scale=8 + 6 * rel,
            color=color, lw=0.6 + 4.5 * rel, alpha=0.80,
            linestyle=FLIP_DASH if dashed else "-", zorder=4))

    # --- (3) Market nodes (on top) ---
    for m, (x, y) in positions.items():
        ax.add_patch(plt.Circle((x, y), NODE_RADIUS, facecolor="#fafafa",
                                edgecolor="#2c3e50", lw=0.8, zorder=10))
        ax.text(x, y, m, ha="center", va="center", fontsize=8,
                fontweight="bold", color="#2c3e50", zorder=11)


def draw_three_panel(W_stable, W_crisis, dom_stable, dom_crisis,
                     output_path, mask_frac=0.10, top_pct=50.0,
                     data_note=""):
    markets = CLOCKWISE_ORDER
    positions = circle_positions(markets, radius=RING_RADIUS)

    # Spillover-layer normalisation (shared across both regimes).
    w_max_spill = max(
        max((abs(w) for w in W_stable.values()), default=0.0),
        max((abs(w) for w in W_crisis.values()), default=0.0),
        1e-9)
    # Domestic-layer normalisation (separate; shared across both regimes).
    dom_vals = ([abs(e["weight"]) for sats in dom_stable.values() for e in sats]
                + [abs(e["weight"]) for sats in dom_crisis.values() for e in sats])
    w_max_dom = max(dom_vals) if dom_vals else 1e-9

    mask_threshold = mask_frac * w_max_spill
    print(f"  spillover shared max|W| = {w_max_spill:.3f}")
    print(f"  domestic shared max|W|  = {w_max_dom:.3f}")
    print(f"  sign-flip mask threshold = {mask_threshold:.3f}"
          f" ({mask_frac:.0%} of spillover max|W|)")

    # Declutter cross-border panels (a)+(b): keep top top_pct% by |W|.
    pooled = ([abs(w) for w in W_stable.values()]
              + [abs(w) for w in W_crisis.values()])
    top_pct = float(np.clip(top_pct, 1.0, 100.0))
    declutter = float(np.percentile(pooled, 100.0 - top_pct)) if pooled else 0.0
    print(f"  cross-border declutter threshold = {declutter:.3f}"
          f" (keep top {top_pct:.0f}% by |W|, panels a+b only)")

    stable_edges = [(s, t, w) for (s, t), w in W_stable.items()
                    if abs(w) >= declutter]
    crisis_edges = [(s, t, w) for (s, t), w in W_crisis.items()
                    if abs(w) >= declutter]

    flipped_edges = []
    n_raw = 0
    for (s, t) in set(W_stable) & set(W_crisis):
        w0, w1 = W_stable[(s, t)], W_crisis[(s, t)]
        if (w0 > 0) == (w1 > 0):
            continue
        n_raw += 1
        if abs(w0) < mask_threshold or abs(w1) < mask_threshold:
            continue
        flipped_edges.append((s, t, w1))
    print(f"  cross-border edges: stable={len(stable_edges)}"
          f" crisis={len(crisis_edges)} raw-flips={n_raw}"
          f" masked-flips={len(flipped_edges)}")

    fig, axes = plt.subplots(1, 3, figsize=(21, 8.4))
    draw_panel(axes[0], positions, stable_edges, dom_stable,
               w_max_spill, w_max_dom,
               f"(a) Stable (lower-price regime) (spillover top {top_pct:.0f}% by |W|)")
    draw_panel(axes[1], positions, crisis_edges, dom_crisis,
               w_max_spill, w_max_dom,
               f"(b) Crisis (higher-price regime) (spillover top {top_pct:.0f}% by |W|)")
    # Sign-flip panel keeps the cross-border flips; for the domestic layer
    # we show the crisis-regime satellites so the panel still carries the
    # outside-the-ring driver context.
    draw_panel(axes[2], positions, flipped_edges, dom_crisis,
               w_max_spill, w_max_dom,
               f"(c) Sign-flipped spillover only (both $|W|>{mask_threshold:.2f}$)",
               dashed=True)

    # --- Legend: spillover sign + domestic feature types ---
    spill_handles = [
        plt.Line2D([0], [0], color=POS_COLOR, lw=2.6,
                   label="Spillover +  (inside ring)"),
        plt.Line2D([0], [0], color=NEG_COLOR, lw=2.6,
                   label="Spillover -  (inside ring)"),
        plt.Line2D([0], [0], color="#7f7f7f", lw=2.0, ls=FLIP_DASH,
                   label="Sign-flipped across regimes"),
        plt.Line2D([0], [0], color="#7f7f7f", lw=0.8, alpha=0.6,
                   label=fr"Spillover width $\propto|W|$ (max {w_max_spill:.2f})"),
    ]
    dom_handles = []
    seen = set()
    for ftype, info in FEATURE_TYPES.items():
        if ftype == "spillover":
            continue  # spillover is the inside-ring arrow layer
        if info["label"] in seen:
            continue
        seen.add(info["label"])
        dom_handles.append(mpatches.Patch(facecolor=info["color"],
                                          edgecolor="#2c3e50",
                                          label=info["label"]))
    dom_handles.append(mpatches.Patch(facecolor="#bdc3c7",
                                      edgecolor="#2c3e50", label="Other"))
    dom_handles.append(
        plt.Line2D([0], [0], color="#7f7f7f", lw=1.4, alpha=0.7,
                   label=fr"Domestic width $\propto|W|$ (max {w_max_dom:.2f})"))

    fig.suptitle(
        "CaRS spillover (inside ring) + domestic drivers (outside ring):"
        " stable vs. crisis comparison",
        fontsize=14, fontweight="bold", y=0.985)
    leg1 = fig.legend(handles=spill_handles, loc="upper center",
                      bbox_to_anchor=(0.30, 0.935), ncol=2, fontsize=8.5,
                      title="Cross-border price spillover (inside ring)",
                      title_fontsize=9, framealpha=0.95, borderaxespad=0.0)
    fig.add_artist(leg1)
    fig.legend(handles=dom_handles, loc="upper center",
               bbox_to_anchor=(0.74, 0.935), ncol=4, fontsize=8.5,
               title="Domestic feature drivers (outside ring)",
               title_fontsize=9, framealpha=0.95, borderaxespad=0.0)

    if data_note:
        fig.text(0.5, 0.010, data_note, ha="center", va="bottom",
                 fontsize=7.5, color="#555555")

    plt.tight_layout(rect=[0, 0.02, 1, 0.85])
    plt.savefig(output_path, dpi=200, bbox_inches="tight",
                facecolor="white", edgecolor="none")
    plt.close()
    print(f"  -> saved to {output_path}")


def build_and_draw(results_dir, output_path, use_effective_weights,
                   mask_frac, top_pct, single_seed, cumulative_threshold,
                   max_satellites):
    if use_effective_weights:
        print("Loading EFFECTIVE spillover (raw W gated by sigmoid(alpha) on"
              f" forbidden edges) from {results_dir} ...")
    else:
        print(f"Loading raw spillover from {results_dir} ...")
    spill = load_spillover_matrices_daily(
        results_dir, CLOCKWISE_ORDER,
        use_effective_weights=use_effective_weights, single_seed=single_seed)
    dom_stable = load_domestic_satellites_daily(
        results_dir, CLOCKWISE_ORDER, regime=0,
        cumulative_threshold=cumulative_threshold,
        max_satellites=max_satellites, single_seed=single_seed)
    dom_crisis = load_domestic_satellites_daily(
        results_dir, CLOCKWISE_ORDER, regime=1,
        cumulative_threshold=cumulative_threshold,
        max_satellites=max_satellites, single_seed=single_seed)
    n_sat0 = sum(len(v) for v in dom_stable.values())
    n_sat1 = sum(len(v) for v in dom_crisis.values())
    print(f"  domestic satellites: stable={n_sat0} crisis={n_sat1}"
          f" (<= {max_satellites}/market, cum-coverage {cumulative_threshold:.0%})")
    report_canonicalisation()

    seed_note = (f"seed-{single_seed} weights" if single_seed is not None
                 else "multi-seed mean")
    wnote = "effective (soft-prior-gated)" if use_effective_weights else "raw"
    data_note = (f"Data: daily CARGO {results_dir.name} ({seed_note}, {wnote} W);"
                 " price-canonical regimes (1=higher-price/Crisis per seed,"
                 " relabel_by_price convention);"
                 " spillover & domestic widths normalised separately.")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    draw_three_panel(spill[0], spill[1], dom_stable, dom_crisis,
                     output_path, mask_frac=mask_frac, top_pct=top_pct,
                     data_note=data_note)


def main():
    repo = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results_dir", type=Path,
                        default=repo / "outputs" / "experiments"
                                / "12market_cargo_controls_daily",
                        help="Daily CARGO experiment dir (default: the"
                             " 5-seed controls_daily run used elsewhere in the"
                             " paper; pass daily_dec8_confirm for the dec8 run).")
    parser.add_argument("--output", type=Path,
                        default=repo / "paper" / "figs" / "cargo"
                                / "spillover_three_panel_augmented.pdf",
                        help="Raw-W output path. The effective-W variant is"
                             " written next to it with an _effective suffix"
                             " when --use_effective_weights is set.")
    parser.add_argument("--use_effective_weights", action="store_true",
                        help="Gate forbidden cross-border weights by"
                             " sigmoid(physical_prior_alpha_logit). Writes the"
                             " _effective output path.")
    parser.add_argument("--single_seed", type=int, default=None,
                        help="Use one seed instead of the seed-mean. Default"
                             " None = average over every seed found in"
                             " --results_dir (the paper's five-seed mean).")
    parser.add_argument("--mask_frac", type=float, default=0.10)
    parser.add_argument("--top_pct", type=float, default=50.0,
                        help="Keep top X%% cross-border edges by |W| in panels"
                             " (a)+(b). Does not affect panel (c).")
    parser.add_argument("--cumulative_threshold", type=float, default=0.90,
                        help="Domestic satellite selection: take top features"
                             " covering this fraction of incoming |W| to price,"
                             " capped at --max_satellites.")
    parser.add_argument("--max_satellites", type=int, default=5,
                        help="Max domestic satellites per market.")
    args = parser.parse_args()

    # `--single_seed -1` is the sentinel for "average over every seed found".
    single_seed = None if args.single_seed == -1 else args.single_seed

    output = args.output
    if args.use_effective_weights:
        # Default raw path -> _effective path.
        stem = output.stem
        if not stem.endswith("_effective"):
            output = output.with_name(stem + "_effective" + output.suffix)

    build_and_draw(
        args.results_dir, output, args.use_effective_weights,
        args.mask_frac, args.top_pct, single_seed,
        args.cumulative_threshold, args.max_satellites)


if __name__ == "__main__":
    main()
