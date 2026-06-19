r"""Plot day-ahead price time series with the CaRS-CAM discovered regime
structure overlaid.

For each market we draw:

  (i)   the full historical hourly day-ahead price line, coloured per-segment
        by the inferred regime — blue for Regime 0 (Stable), orange for
        Regime 1 (Crisis). Regime labels are re-mapped per market so that
        Regime 1 always denotes the higher-mean-price (= ``crisis'')
        state, making colours comparable across panels;

  (ii)  a thin regime-trajectory ``barcode'' strip at the bottom of each
        panel — a per-hour visualisation of argmax_d q(d_t=d | x_{1:T})
        that remains legible even when the multi-panel figure squeezes
        each market to a thin row;

  (iii) two dashed vertical lines marking the documented 2021-2023 European
        energy-crisis window (Russian gas curtailments -> gas price
        normalisation), as economic anchors for the reader;

  (iv)  a single held-out test window applied uniformly across all
        markets (computed once from the union of market timelines), so
        the grey marker at the top of every panel covers the same
        calendar period and is directly comparable.

Output: paper/figs/regime_timeseries_panel.pdf.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.collections import LineCollection
from matplotlib.patches import Patch

from unified_data_loader import load_unified_dataset

REGIME_LINE_COLOR = {0: "#1f77b4", 1: "#ff7f0e"}  # saturated blue / orange
REGIME_LABEL = {0: "Regime 0 (Stable)", 1: "Regime 1 (Crisis)"}
NA_LINE_COLOR = "#555555"  # neutral grey where no regime inference available
TEST_BAR_COLOR = "#555555"  # neutral grey for the held-out test marker

# Reference window: 2021--2023 natural-gas supply disruption in Europe
CRISIS_START = pd.Timestamp("2021-09-01")
CRISIS_END = pd.Timestamp("2023-06-30")


# Per-market idiosyncratic events visible in the time-series panel.
# Each entry is `(timestamp, label)` for the default high placement, or
# `(timestamp, label, y_frac)` to override the vertical position (axes
# fraction, 0=bottom, 1=top). Use a y_frac override when two events for
# the same market sit close in time and their labels would otherwise
# overlap horizontally — staggering them vertically resolves the
# collision without making the labels smaller. Default y_frac is the
# value of EVENT_LABEL_Y_DEFAULT.
EVENT_LABEL_Y_DEFAULT = 0.92
EVENT_LABEL_Y_LOW = 0.74  # used when alternating to avoid horizontal overlap
MARKET_EVENTS = {
    "DE": [
        ("2023-04-15", "Last 3 nuclear plants offline"),
    ],
    "FR": [
        ("2020-06-30", "Fessenheim closure"),
        ("2022-09-01", "Nuclear corrosion crisis"),
        # Winter 2025 cold snap + low-wind episodes (Dunkelflaute);
        # only post-2023 R1 episode >=1 week that maps cleanly to a
        # documented macro event in the crisis-aligned markets.
        ("2025-01-20", "Dunkelflaute (Jan/Feb 2025)", EVENT_LABEL_Y_LOW),
    ],
    "ES": [
        ("2022-06-15", "Iberian gas price cap"),
    ],
    "IT": [
        ("2022-07-01", "Po Valley drought"),
    ],
    "BE": [
        ("2022-09-23", "Doel 3 closure", EVENT_LABEL_Y_DEFAULT),
        ("2023-02-01", "Tihange 2 closure", EVENT_LABEL_Y_LOW),
    ],
    "SE": [
        ("2020-12-31", "Ringhals 1+2 closure"),
    ],
    "DK": [
        ("2023-12-29", "Viking Link UK cable online"),
    ],
    "PL": [
        ("2018-01-01", "Coal-driven PL–DE spread widens"),
    ],
    "HU": [],
    "NL": [
        ("2022-03-01", "Russian gas import halt"),
    ],
    "AT": [],
    "CZ": [],
}

# Behavioural grouping of markets by regime-trajectory statistics. Each
# market's bold left-margin badge is rendered with the group name below
# it so the panel layout itself makes the structural taxonomy visible.
# See electricity/extract_regime_trajectory.py for the per-market R1
# occupancy / mean-episode-length numbers that motivate this split.
MARKET_GROUPS = {
    # <1% R1 occupancy; CaRS-CAM converged to ~single-regime, which is
    # structurally consistent with these markets' price distribution
    # remaining close to unimodal even through the crisis (BE: large
    # nuclear baseload; ES: Iberian gas-price cap from June 2022).
    "BE": "Single-regime",
    "ES": "Single-regime",
    # 3-13% R1 occupancy; R1 episodes concentrate inside the 2021-2023
    # gas-crisis window; mean episode 7-12 hours. Classical macro-state
    # interpretation: stable vs. crisis.
    "AT": "Crisis-aligned (slow)",
    "DK": "Crisis-aligned (slow)",
    "NL": "Crisis-aligned (slow)",
    "DE": "Crisis-aligned (slow)",
    # Same crisis-aligned semantics but with substantially more intraday
    # toggling (high switch count, R1 occupancy 20-26%). Captures the
    # peak/off-peak alternation noted in the intra-day discussion.
    "CZ": "Crisis-aligned (fast)",
    "FR": "Crisis-aligned (fast)",
    # 84-93% R1 occupancy and multi-day episodes spanning pre-crisis
    # full years (e.g. IT 2016 is one continuous 8760-hour R1 episode).
    # The learned regime structure does NOT map to stable vs crisis for
    # these markets; it more likely encodes volatility regimes or
    # peak/off-peak persistence. Flagged in the paper as a structural
    # caveat.
    "HU": "Inverted-regime",
    "IT": "Inverted-regime",
    "PL": "Inverted-regime",
    "SE": "Inverted-regime",
}
MARKET_GROUP_COLOR = {
    "Single-regime":        "#7f7f7f",
    "Crisis-aligned (slow)": "#1f77b4",
    "Crisis-aligned (fast)": "#ff7f0e",
    "Inverted-regime":      "#9467bd",
}

# DAILY taxonomy (computed from the multi-seed CARGO+controls+soft-prior
# checkpoints in outputs/experiments/12market_cargo_controls_daily/).
# Daily training collapses the hourly "single-regime" (BE/ES) and
# "inverted-regime" (HU/IT/SE) categories into the crisis-aligned group:
# 10 of 12 markets land at R1 14-25%, episodes 7-12 days, with 54-78%
# of R1 days inside the documented 2021-2023 crisis window. The two
# residual exceptions are PL (R1 28.6% but only 21% inside the crisis
# window -- domestically driven by coal price dynamics) and DK
# (R1 55% with only 67% seed agreement -- the model can't converge
# to a stable regime partition).
MARKET_GROUPS_DAILY = {
    "SE": "Crisis-aligned",
    "AT": "Crisis-aligned",
    "BE": "Crisis-aligned",
    "FR": "Crisis-aligned",
    "DE": "Crisis-aligned",
    "IT": "Crisis-aligned",
    "HU": "Crisis-aligned",
    "NL": "Crisis-aligned",
    "ES": "Crisis-aligned",
    "CZ": "Crisis-aligned",
    # PL: after the PLN->EUR currency correction + retrain, PL's high-price
    # regime concentrates 97.6% inside the 2021-2023 crisis window (was a
    # spurious "domestically-driven" artifact of the PLN-inflated 2017-2019).
    "PL": "Crisis-aligned",
    "DK": "Seed-unstable",
}
MARKET_GROUP_COLOR_DAILY = {
    "Crisis-aligned":      "#1f77b4",
    "Domestically-driven": "#ff7f0e",
    "Seed-unstable":       "#d62728",
}


def get_market_groups(frequency):
    """Return (MARKET_GROUPS, MARKET_GROUP_COLOR) for the requested frequency.

    The hourly figures use the original four-group taxonomy derived
    from the 12market_gat_spillover hourly checkpoints; the daily
    figures use the simpler three-group taxonomy from the multi-seed
    daily CARGO checkpoints.
    """
    if frequency == "D":
        return MARKET_GROUPS_DAILY, MARKET_GROUP_COLOR_DAILY
    return MARKET_GROUPS, MARKET_GROUP_COLOR

# Super-grouping used by --split_by_group: rolls the four behavioural
# sub-groups into two figures so the paper has two clearly-themed
# multi-panel figures instead of four small ones.
SUPER_GROUPS = {
    "Crisis-aligned (slow)":  "Crisis-aligned",
    "Crisis-aligned (fast)":  "Crisis-aligned",
    "Inverted-regime":        "Inverted-or-single-regime",
    "Single-regime":          "Inverted-or-single-regime",
}
# Order in which markets appear within each super-group's figure.
SUPER_GROUP_ORDER = {
    "Crisis-aligned":            ["AT", "DE", "DK", "NL", "CZ", "FR"],
    "Inverted-or-single-regime": ["HU", "IT", "PL", "SE", "BE", "ES"],
}


def load_market_series(market, frequency="H"):
    df = load_unified_dataset(market, clean=True, frequency=frequency)
    df = df.copy()
    df.index = pd.to_datetime(df.index)
    return df.index.to_pydatetime(), df["Day_Ahead_Price"].values, df


def mask_synthetic_constants(prices, dates, min_window_hours=24 * 30 * 6):
    p = pd.Series(prices, index=pd.to_datetime(dates))
    diff = p.diff().abs()
    is_zero = (diff == 0).fillna(False).astype(int)
    run_id = (is_zero != is_zero.shift()).cumsum()
    run_lengths = is_zero.groupby(run_id).transform("sum")
    constant_run = (is_zero == 1) & (run_lengths >= min_window_hours)
    constant_run = constant_run | constant_run.shift(-1, fill_value=False)
    return ~constant_run.values


# Length of the CaRS-CAM recurrent context window (``timestep`` in the
# daily checkpoints). The regime posterior q(d_t | x_{1:T}) is only
# defined once this window has been consumed, so the first CONTEXT_WINDOW
# observations of EVERY chronological split segment (train, val, test)
# carry no regime label.
CONTEXT_WINDOW = 14  # days


def fill_context_warmups(regime_series, frequency):
    """Backfill the per-split context-window warmups with the next regime.

    The model emits no regime for the first ``timestep`` (=CONTEXT_WINDOW)
    observations of each train/val/test segment, which otherwise show up
    as grey strips at every split boundary (and at the very start of the
    series). We backfill each such leading-NaN run from the first inferred
    observation that follows it — i.e. we assign the warmup the regime of
    the model's 15th observation in that segment, the same state the model
    then reports once its input window is full. The fill limit is a little
    over one context window (scaled to the series frequency) so the ~14-day
    split warmups and the slightly longer initial warmup (from dropping
    leading feature-NaN rows) are filled, while any genuinely long gap is
    left as NaN. A short forward-fill closes the <=2-step tail left by the
    1-step forecast horizon at the end of the test segment.
    """
    if regime_series is None:
        return regime_series
    steps_per_day = 24 if frequency == "H" else 1
    bfill_limit = (3 * CONTEXT_WINDOW) * steps_per_day
    out = regime_series.bfill(limit=bfill_limit)
    out = out.ffill(limit=2 * steps_per_day)
    return out


def load_regime_trajectory(market, trajectory_dir, dates, frequency="H"):
    """Load discovered regime trajectory and align to plot timeline.

    Returns a Series indexed by `dates` whose values are the (potentially
    relabelled) regime indices in {0, 1}, or NaN where no inference is
    available.

    For hourly mode (default), the original trajectory is resampled to
    the hourly grid via dominant-regime-per-hour. For daily mode, the
    trajectory is reindexed directly onto the date axis. In both modes the
    per-split context-window warmups are backfilled (see
    ``fill_context_warmups``) so split boundaries do not render as grey
    strips.
    """
    path = Path(trajectory_dir) / f"regime_trajectory_{market}.csv"
    if not path.exists():
        return None
    raw = pd.read_csv(path, parse_dates=["timestamp"])
    prob_cols = [c for c in raw.columns if c.startswith("regime_prob_")]
    if not prob_cols:
        return None
    raw["regime"] = raw[prob_cols].values.argmax(axis=1)
    raw = raw.set_index("timestamp").sort_index()

    if frequency == "D":
        # Daily mode: nearest-day reindex (trajectory is already daily)
        s = raw["regime"].reindex(pd.DatetimeIndex(dates),
                                   method="nearest", limit=2)
        s = fill_context_warmups(s, frequency="D")
        return s.astype(float)

    # Hourly mode: resample 15-min trajectory to dominant regime per hour
    hourly = raw["regime"].resample("h").agg(lambda s: float(s.mode().iloc[0]) if len(s) else np.nan)
    hourly = hourly.reindex(pd.DatetimeIndex(dates), method="nearest", limit=1)
    hourly = fill_context_warmups(hourly, frequency="H")
    return hourly.astype(float)


def relabel_regimes_by_price_level(regime_series, prices, dates):
    """Re-map the regime IDs so that regime 1 always denotes the higher
    mean-price state for this market (i.e. the ``crisis''/expensive regime),
    making colours comparable across panels. Returns
    (relabelled_series, was_swapped).
    """
    if regime_series is None:
        return regime_series, False
    p = pd.Series(prices, index=pd.DatetimeIndex(dates))
    mask = regime_series.isin([0, 1])
    r0_mean = p[mask & (regime_series == 0)].mean()
    r1_mean = p[mask & (regime_series == 1)].mean()
    if pd.isna(r0_mean) or pd.isna(r1_mean):
        return regime_series, False
    if r0_mean > r1_mean:
        # Regime 0 is the higher-price (``crisis'') regime -> swap labels
        out = regime_series.copy()
        out.loc[regime_series == 0] = 1.0
        out.loc[regime_series == 1] = 0.0
        return out, True
    return regime_series, False


def relabel_regimes_by_volatility(regime_series, prices, dates):
    """Re-map the regime IDs so that regime 1 denotes the higher
    *absolute-volatility* state (larger std of daily price changes, in
    EUR/MWh) rather than the higher mean-price state. This is the
    volatility-based canonicalisation; comparing the resulting figure to
    the price-level one shows whether the latent partition tracks price
    level or price volatility (empirically, the two coincide for all 12
    markets -- see electricity/characterize_regimes.py). Returns
    (relabelled_series, was_swapped).
    """
    if regime_series is None:
        return regime_series, False
    dprice = pd.Series(prices, index=pd.DatetimeIndex(dates)).diff()
    mask = regime_series.isin([0, 1])
    r0_vol = dprice[mask & (regime_series == 0)].std()
    r1_vol = dprice[mask & (regime_series == 1)].std()
    if pd.isna(r0_vol) or pd.isna(r1_vol):
        return regime_series, False
    if r0_vol > r1_vol:
        out = regime_series.copy()
        out.loc[regime_series == 0] = 1.0
        out.loc[regime_series == 1] = 0.0
        return out, True
    return regime_series, False


# Canonicalisation strategies selectable via --canonicalize. ``none``
# keeps the raw latent index from the consensus trajectory (the model's
# own d_t, no relabel); ``price``/``volatility`` enforce that Regime 1 is
# the higher-mean-price / higher-absolute-volatility state respectively.
def _relabel_none(regime_series, prices, dates):
    return regime_series, False


CANONICALIZERS = {
    "price": relabel_regimes_by_price_level,
    "volatility": relabel_regimes_by_volatility,
    "none": _relabel_none,
}


def draw_regime_barcode(ax, dates, regime_series, y_lo=0.00, y_hi=0.06,
                        agg_freq="D"):
    """Draw a thin regime-trajectory ``barcode'' strip in axes-fraction
    coordinates at the bottom of the panel. Each contiguous run of regime
    `v` becomes a coloured rectangle spanning [y_lo, y_hi] of the panel
    height.

    The hourly regime trajectory is aggregated to ``agg_freq`` (default
    daily, dominant regime per bucket) before drawing — at hourly
    resolution, a 9-year panel rendered to a 13" PDF puts every hour at
    sub-pixel width, so isolated 1-h opposite-regime blips inside long
    runs rasterise to fractional coverage and create a visibly
    ``speckled'' lighter-blue appearance inside what should be a solid
    strip. Daily aggregation collapses those blips into one decision per
    visible pixel.

    Returns the number of regime switches at the *original* (hourly)
    resolution, for reporting purposes.
    """
    if regime_series is None:
        return 0
    coarse = regime_series.resample(agg_freq).agg(
        lambda s: float(s.mode().iloc[0]) if len(s.dropna()) else np.nan
    )
    ts = coarse.index
    r_c = coarse.values.astype(float)
    r_int = np.where(np.isnan(r_c), -1, r_c).astype(int)
    if (r_int >= 0).sum() == 0:
        return 0

    boundaries = np.where(np.diff(r_int) != 0)[0] + 1
    starts = np.concatenate([[0], boundaries])
    ends = np.concatenate([boundaries, [len(r_int)]])
    for s, e in zip(starts, ends):
        v = r_int[s]
        if v in (0, 1):
            ax.axvspan(ts[s], ts[min(e, len(ts) - 1)],
                       ymin=y_lo, ymax=y_hi,
                       color=REGIME_LINE_COLOR[v], alpha=1.0,
                       zorder=2, linewidth=0, antialiased=False)
    # Switch count is reported at original hourly resolution
    r_h = regime_series.values.astype(float)
    r_h_int = np.where(np.isnan(r_h), -1, r_h).astype(int)
    valid_seq = r_h_int[r_h_int >= 0]
    return int((np.diff(valid_seq) != 0).sum())


def plot_price_by_regime(ax, dates, prices, regime_series, usable_mask):
    """Plot the price as a SINGLE continuous polyline whose colour changes
    per-segment with the inferred regime.

    Earlier this drew one ``ax.plot`` pass per regime with the other
    regime NaN-masked; that left the connecting segment undrawn at every
    regime switch, so a switch coinciding with a sharp price level-shift
    rendered as a visible gap. Here we build one ``LineCollection`` over
    all consecutive sample pairs and colour each segment by its left
    endpoint's regime (blue/orange, or grey where no inference is
    available). Every pair of consecutive *usable* samples is joined, so
    there are no spurious gaps at regime boundaries; only genuine breaks
    (synthetic-constant fill masked out of ``usable_mask``, or NaN price)
    remain as gaps, because a segment is drawn only when both endpoints
    are finite.
    """
    x = mdates.date2num(dates)
    base = np.where(usable_mask, prices, np.nan)
    if regime_series is None:
        codes = np.full(len(base), -1, dtype=int)
    else:
        r = regime_series.values.astype(float)
        codes = np.where(np.isnan(r), -1, r).astype(int)
    color_for = {0: REGIME_LINE_COLOR[0], 1: REGIME_LINE_COLOR[1],
                 -1: NA_LINE_COLOR}

    pts = np.column_stack([x, base])
    segs = np.stack([pts[:-1], pts[1:]], axis=1)            # (N-1, 2, 2)
    finite = np.isfinite(base)
    drawable = finite[:-1] & finite[1:]                     # both endpoints finite
    if not drawable.any():
        return
    seg_codes = codes[:-1][drawable]
    seg_colors = [color_for[c] for c in seg_codes]
    lc = LineCollection(segs[drawable], colors=seg_colors,
                        linewidths=0.7, zorder=3)
    ax.add_collection(lc)
    # LineCollection does not extend the axes' data limits, so x-range is
    # set explicitly by the caller (plot_panel) from the global timeline.


def plot_panel(markets, trajectory_dir, output_path,
               shared_y=True, y_cap=None, y_floor=-80.0, title_suffix="",
               frequency="H", canonicalize="price"):
    n_mkt = len(markets)
    # Panel height: tall enough that adjacent y-tick labels don't
    # overlap. For 12-panel single figures this lifts the figure past
    # one page in height but is the price of readability; for 6-panel
    # sub-group figures the original 1.4" height is preserved.
    if n_mkt >= 8:
        panel_h = max(1.05, min(1.30, 13.0 / n_mkt))
        margin = 0.9
    else:
        panel_h = 1.4
        margin = 1.0
    fig, axes = plt.subplots(n_mkt, 1,
                             figsize=(13, panel_h * n_mkt + margin),
                             sharex=True)
    if n_mkt == 1:
        axes = [axes]

    # First pass: load all markets so we can apply a single, common
    # held-out test window to every panel AND derive a pooled y-axis
    # range when shared_y=True.
    loaded = {}
    all_ends = []
    all_starts = []
    all_max_obs = -np.inf  # pooled price max across all markets, for shared y-axis
    for market in markets:
        try:
            dates, prices, df_full = load_market_series(market, frequency=frequency)
        except Exception as e:
            print(f"  {market}: load failed {e!r}")
            continue
        loaded[market] = (dates, prices, df_full)
        all_starts.append(pd.to_datetime(df_full.index[0]))
        all_ends.append(pd.to_datetime(df_full.index[-1]))
        all_max_obs = max(all_max_obs, float(np.nanmax(prices)))

    if not loaded:
        print("  no markets loaded; nothing to plot")
        return

    global_start = min(all_starts)
    global_end = max(all_ends)
    global_test_start = global_start + 0.8 * (global_end - global_start)

    # Pooled y-range default: cap at the SECOND-largest per-market
    # maximum so one outlier market (e.g. FR's 2988 EUR/MWh
    # single-hour spike) doesn't squash every other panel. The outlier
    # market overflows the cap and is annotated with an arrow + peak.
    # If y_cap is set explicitly, use it instead.
    per_market_maxes = sorted(
        [float(np.nanmax(loaded[m][1])) for m in loaded], reverse=True)
    if shared_y:
        shared_ymin = y_floor
        if y_cap is not None:
            shared_ymax = min(y_cap, all_max_obs * 1.05)
            cap_note = f"capped at {y_cap:.0f}"
        elif len(per_market_maxes) > 1:
            shared_ymax = per_market_maxes[1] * 1.05
            cap_note = (f"capped at 2nd-largest per-market max "
                        f"({per_market_maxes[1]:.0f}); outlier {per_market_maxes[0]:.0f}")
        else:
            shared_ymax = all_max_obs * 1.05
            cap_note = "no cap, single market"
        print(f"  shared y-axis: [{shared_ymin:.0f}, {shared_ymax:.0f}]"
              f"  (pooled max = {all_max_obs:.0f}; {cap_note})")

    summary = []
    for ax, market in zip(axes, markets):
        if market not in loaded:
            continue
        dates, prices, df_full = loaded[market]

        # Discovered regime trajectory
        regime_raw = load_regime_trajectory(market, trajectory_dir, dates,
                                              frequency=frequency)
        relabel_fn = CANONICALIZERS.get(canonicalize, relabel_regimes_by_price_level)
        regime, swapped = relabel_fn(regime_raw, prices, dates)

        ymin, ymax = float(np.nanmin(prices)), float(np.nanmax(prices))
        if shared_y:
            plot_ymin, plot_ymax = shared_ymin, shared_ymax
        else:
            pad = 0.10 * (ymax - ymin)
            plot_ymin = max(-80, ymin - pad)
            plot_ymax = min(1000, ymax + pad)
        ax.set_ylim(plot_ymin, plot_ymax)
        # The price line is now a LineCollection, which does not extend the
        # axes' data limits, so fix the x-range to the shared global
        # timeline explicitly (sharex propagates it to every panel).
        ax.set_xlim(global_start, global_end)
        # Mark out-of-range overflow (e.g. FR's 2988 EUR/MWh single-hour
        # spike) with a small upward arrow + the actual peak value, so
        # the shared y-axis doesn't silently hide extremes.
        if shared_y and ymax > plot_ymax:
            peak_idx = int(np.nanargmax(prices))
            peak_x = pd.to_datetime(dates[peak_idx])
            ax.annotate(f" peak {ymax:.0f}",
                        xy=(peak_x, plot_ymax * 0.99),
                        xytext=(peak_x, plot_ymax * 0.87),
                        ha="left", va="bottom",
                        fontsize=6.5, color="#c0392b",
                        arrowprops=dict(arrowstyle="->",
                                        color="#c0392b", lw=0.6),
                        zorder=11)

        # Documented 2021-2023 crisis window — dashed vertical anchors
        for x in (CRISIS_START, CRISIS_END):
            ax.axvline(x, color="#7f7f7f", linestyle="--",
                       linewidth=0.7, alpha=0.55, zorder=1)

        # Mask synthetic constant-fill runs and plot regime-coloured price
        usable_mask = mask_synthetic_constants(prices, dates)
        plot_price_by_regime(ax, dates, prices, regime, usable_mask)

        # Regime barcode strip at the bottom of the panel
        n_switches = draw_regime_barcode(ax, dates, regime,
                                         y_lo=0.00, y_hi=0.06)
        if regime is not None:
            r0 = int((regime == 0).sum())
            r1 = int((regime == 1).sum())
            summary.append((market, r0, r1, n_switches, swapped))

        # Held-out test window — same calendar period on every panel
        ax.axvspan(global_test_start, global_end,
                   ymin=0.94, ymax=0.99,
                   color=TEST_BAR_COLOR, alpha=0.65, zorder=2,
                   linewidth=0)

        ax.set_ylabel("Price (EUR/MWh)", fontsize=8)
        ax.tick_params(axis="y", labelsize=7)
        ax.tick_params(axis="x", labelsize=8)
        # Bold market-code badge on the left margin, with a small
        # behavioural-group sub-label colour-coded by group so the
        # taxonomy is visible at a glance across the 12 panels.
        groups_dict, group_colour = get_market_groups(frequency)
        ax.text(-0.07, 0.62, market, transform=ax.transAxes,
                fontsize=15, fontweight="bold", ha="right", va="center",
                color="#2c3e50")
        group = groups_dict.get(market, "")
        if group:
            ax.text(-0.07, 0.32, group, transform=ax.transAxes,
                    fontsize=6.8, fontweight="bold",
                    ha="right", va="center",
                    color=group_colour.get(group, "#7f7f7f"))
        ax.grid(True, axis="y", alpha=0.12, linewidth=0.4)
        ax.grid(False, axis="x")

        # Per-market idiosyncratic event annotations: thin vertical tick
        # at the event date + short label tucked under the panel's top
        # spine so the event sits *inside* the panel but above the line.
        # The 3rd tuple element (if present) overrides the vertical
        # position so close-in-time events can be staggered (see BE).
        for ev in MARKET_EVENTS.get(market, []):
            ev_date, ev_label = ev[0], ev[1]
            y_frac = ev[2] if len(ev) > 2 else EVENT_LABEL_Y_DEFAULT
            ts = pd.Timestamp(ev_date)
            ax.axvline(ts, color="#444444", linestyle=":",
                       linewidth=0.6, alpha=0.55, zorder=1)
            ax.text(ts, y_frac, f" {ev_label}",
                    transform=ax.get_xaxis_transform(),
                    rotation=0, va="top", ha="left",
                    fontsize=6.5, color="#2c3e50",
                    bbox=dict(boxstyle="round,pad=0.15",
                              facecolor="white",
                              edgecolor="#cccccc", lw=0.3,
                              alpha=0.88),
                    zorder=12)

    # Crisis-window labels — placed just above the top panel (axes-
    # fraction y slightly > 1) with clip_on=False so they sit in the
    # figure margin instead of overlapping the price line. A short
    # downward tick connects each label to the dashed crisis-window
    # line that runs through every panel.
    top = axes[0]
    for ts, label in [(CRISIS_START, "Russian gas curtailments"),
                       (CRISIS_END, "Gas-price normalisation")]:
        top.annotate(label, xy=(ts, 1.0),
                     xytext=(ts, 1.002),
                     xycoords=("data", "axes fraction"),
                     ha="center", va="bottom",
                     fontsize=7.5, color="#555555", fontstyle="italic",
                     arrowprops=dict(arrowstyle="-", color="#7f7f7f",
                                     lw=0.6, alpha=0.7),
                     annotation_clip=False, zorder=20)

    axes[-1].set_xlabel("Time")
    legend_elems = [
        plt.Line2D([0], [0], color=REGIME_LINE_COLOR[0], lw=1.6,
                   label=REGIME_LABEL[0]),
        plt.Line2D([0], [0], color=REGIME_LINE_COLOR[1], lw=1.6,
                   label=REGIME_LABEL[1]),
        plt.Line2D([0], [0], color="#7f7f7f", lw=0.9, linestyle="--",
                   label="2021--2023 crisis window"),
        Patch(facecolor=TEST_BAR_COLOR, alpha=0.65,
              label="Held-out test split"),
    ]
    # Title at top, legend immediately below it, then the crisis-window
    # labels sitting flush with the top panel. Layout pulled tight so
    # the legend bottom and the crisis-label text top almost touch.
    base_title = "Day-ahead price history coloured by the CaRS-CAM discovered regime"
    if title_suffix:
        base_title = f"{base_title}  —  {title_suffix}"
    fig.suptitle(base_title, fontsize=11, fontweight="bold", y=0.965)
    fig.legend(handles=legend_elems, loc="upper center",
               bbox_to_anchor=(0.5, 0.920),
               fontsize=7.5, framealpha=0.95, ncol=4,
               borderaxespad=0.0)

    plt.tight_layout(rect=[0, 0, 1, 0.905])
    plt.savefig(output_path, dpi=150, bbox_inches="tight",
                facecolor="white", edgecolor="none")
    plt.close()
    print(f"  -> panel saved to {output_path}")

    # Per-market summary line
    print(f"\n{'Market':<8} {'Regime0 hrs':>12} {'Regime1 hrs':>12} {'Switches':>10}  Swapped?")
    for m, r0, r1, ns, sw in summary:
        print(f"{m:<8} {r0:>12} {r1:>12} {ns:>10}  {sw}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--trajectory_dir", type=Path,
                   default=Path(__file__).resolve().parent.parent / "outputs" / "regime_trajectories")
    p.add_argument("--output_dir", required=True, type=Path)
    p.add_argument("--markets", nargs="*",
                   default=["DE", "FR", "IT", "SE"])
    p.add_argument("--per_panel_y", action="store_true",
                   help="Use per-panel y-axis limits (legacy behaviour)."
                        " Default is a shared y-axis covering the pooled"
                        " maximum across all panels in the figure.")
    p.add_argument("--y_cap", type=float, default=None,
                   help="Optional upper bound for the shared y-axis"
                        " (EUR/MWh). Default None means the axis covers"
                        " the pooled maximum observation, so every spike"
                        " sits inside the panel. Set e.g. 800 to cap and"
                        " mark overflow with an arrow.")
    p.add_argument("--frequency", type=str, default="H", choices=["H", "D"],
                   help="Frequency of the unified data and regime"
                        " trajectory: H = hourly (default), D = daily.")
    p.add_argument("--canonicalize", type=str, default="price",
                   choices=["price", "volatility", "none"],
                   help="How to fix the arbitrary per-market regime label"
                        " for colouring: price (Regime 1 = higher mean"
                        " price, default), volatility (Regime 1 = higher"
                        " absolute price-change volatility), or none (raw"
                        " latent index from the consensus trajectory).")
    p.add_argument("--no_split_by_group", action="store_true",
                   help="Render every market in a single figure. By"
                        " default markets are partitioned by the"
                        " MARKET_GROUPS taxonomy and rendered as one PDF"
                        " per group, so each figure fits a single page"
                        " at the pooled-max y-axis scale.")
    args = p.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    shared_y = not args.per_panel_y

    if args.no_split_by_group:
        panel_path = args.output_dir / "regime_timeseries_panel.pdf"
        plot_panel(args.markets, args.trajectory_dir, panel_path,
                   shared_y=shared_y, y_cap=args.y_cap,
                   frequency=args.frequency, canonicalize=args.canonicalize)
        return

    # Partition markets by SUPER_GROUPS so the output is two figures —
    # one for the crisis-aligned markets (slow + fast switching) and
    # one for the inverted + single-regime markets — each consolidated
    # into a single page. Within each figure, markets are ordered per
    # SUPER_GROUP_ORDER (slow before fast in crisis-aligned; inverted
    # before single in the other) so the behavioural taxonomy is
    # visible top-to-bottom.
    from collections import OrderedDict
    super_groups = OrderedDict()
    for m in args.markets:
        g = MARKET_GROUPS.get(m, "Other")
        sg = SUPER_GROUPS.get(g, g)
        super_groups.setdefault(sg, []).append(m)
    # Reorder each super-group's markets by the canonical order
    for sg, ms in super_groups.items():
        if sg in SUPER_GROUP_ORDER:
            super_groups[sg] = [m for m in SUPER_GROUP_ORDER[sg] if m in ms]
    rendered = []
    for sg_name, sg_markets in super_groups.items():
        slug = (sg_name.lower()
                .replace(" ", "_").replace("(", "").replace(")", "")
                .replace(",", "").replace("-", "_"))
        out = args.output_dir / f"regime_timeseries_panel_{slug}.pdf"
        print(f"\n=== Super-group: {sg_name} ({len(sg_markets)} markets) ===")
        plot_panel(sg_markets, args.trajectory_dir, out,
                   shared_y=shared_y, y_cap=args.y_cap,
                   title_suffix=f"{sg_name} markets",
                   frequency=args.frequency, canonicalize=args.canonicalize)
        rendered.append((sg_name, sg_markets, out))
    print("\n--- per-super-group rendering complete ---")
    for g, ms, out in rendered:
        print(f"  {g:<30} ({','.join(ms):<22}) -> {out}")


if __name__ == "__main__":
    main()
