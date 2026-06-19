"""
Visualize the integrated European electricity market causal network.

Composes per-country CaRS DAGs into a single network showing:
- Each market as a node positioned geographically
- Top domestic causal drivers (colored by type) as satellite nodes
- Physical interconnections between markets
- Regime-specific edge strengths from learned W matrices

Usage:
    python3 visualize_european_network.py
    python3 visualize_european_network.py --regime 1 --threshold 0.1
"""

import sys
import json
import argparse
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    from matplotlib.patches import FancyArrowPatch
except ImportError:
    raise ImportError("matplotlib required: pip install matplotlib")


# Geographic coordinates (adjusted for poster readability — spread overlapping pairs)
MARKET_COORDS = {
    'DE': (10.4, 51.2),
    'FR': (2.2, 46.0),
    'NL': (4.0, 53.5),      # shifted left+up from BE
    'BE': (3.0, 50.0),      # shifted left from NL
    'AT': (14.0, 46.0),     # shifted down from CZ
    'IT': (12.5, 41.5),
    'ES': (-3.7, 40.4),
    'PL': (20.0, 52.5),     # shifted right from CZ
    'DK': (9.5, 57.0),
    'SE': (15.0, 62.0),
    'HU': (20.0, 46.5),     # shifted right
    'CZ': (16.0, 49.5),
}

# Physical interconnections
INTERCONNECTIONS = [
    ('DE', 'FR'), ('DE', 'NL'), ('DE', 'BE'), ('DE', 'AT'),
    ('DE', 'CZ'), ('DE', 'PL'), ('DE', 'DK'), ('DE', 'SE'),
    ('FR', 'BE'), ('FR', 'ES'), ('FR', 'IT'),
    ('NL', 'BE'), ('NL', 'DK'),
    ('AT', 'CZ'), ('AT', 'HU'), ('AT', 'IT'),
    ('CZ', 'PL'), ('PL', 'SE'), ('DK', 'SE'),
]

# Countries whose prefix in feature names indicates cross-border spillover
SPILLOVER_COUNTRIES = {'AT', 'BE', 'CZ', 'DE', 'DK', 'ES', 'FR', 'HU', 'IT', 'NL', 'PL', 'SE'}

# Feature type classification for coloring
FEATURE_TYPES = {
    'spillover': {'keywords': [f'{c}_price_lag' for c in SPILLOVER_COUNTRIES]
                              + [f'{c}_flow_lag' for c in SPILLOVER_COUNTRIES],
                  'color': '#e67e22', 'label': 'Cross-border spillover'},
    'generation': {'keywords': ['Solar', 'Wind', 'Nuclear', 'Gas', 'Coal', 'Lignite',
                                'Hydro', 'Biomass', 'Geothermal', 'Fossil'],
                   'color': '#2ecc71', 'label': 'Generation'},
    'forecast': {'keywords': ['forecast_', 'gen_forecast', 'load_forecast'],
                 'color': '#e74c3c', 'label': 'Forecast (new)'},
    'weather': {'keywords': ['temperature', 'wind_speed', 'radiation', 'cloud',
                             'precipitation', 'humidity', 'apparent_temp'],
                'color': '#3498db', 'label': 'Weather'},
    'load': {'keywords': ['Actual Load', 'load'],
             'color': '#f39c12', 'label': 'Load/Demand'},
    'price': {'keywords': ['price', 'Price'],
              'color': '#9b59b6', 'label': 'Price (autoregressive)'},
    'calendar': {'keywords': ['day_of', 'dow_', 'month', 'season', 'is_weekend',
                              'is_holiday', 'week_of'],
                 'color': '#95a5a6', 'label': 'Calendar'},
}


def classify_feature(feat_name):
    """Classify a feature into a type for coloring."""
    for ftype, info in FEATURE_TYPES.items():
        if any(kw in feat_name for kw in info['keywords']):
            return ftype, info['color']
    return 'other', '#bdc3c7'


def spillover_source(feat_name):
    """Return the source country code for a cross-border spillover feature,
    or None if the feature is not a spillover feature."""
    for c in SPILLOVER_COUNTRIES:
        if feat_name.startswith(f'{c}_price_lag') or feat_name.startswith(f'{c}_flow_lag'):
            return c
    return None


def shorten_feature_name(name):
    """Shorten feature names for display."""
    # Keep country prefix for spillover features (e.g. FR_price_lag1h -> FR lag1h)
    for country in SPILLOVER_COUNTRIES:
        if name.startswith(f'{country}_price_lag'):
            lag = name.replace(f'{country}_price_lag', '')
            return f'{country} lag{lag}'
        if name.startswith(f'{country}_flow_lag'):
            lag = name.replace(f'{country}_flow_lag', '')
            return f'{country} flow{lag}'
    # Remove country prefixes for domestic features
    for prefix in ['DE_', 'FR_', 'NL_', 'BE_', 'AT_', 'IT_', 'ES_',
                    'PL_', 'DK_', 'CZ_', 'HU_', 'SE_', 'DK_1_', 'DK_2_']:
        name = name.replace(prefix, '')
    replacements = {
        'forecast_Wind Onshore': 'Wind Fcst',
        'forecast_Wind Offshore': 'Offshore Fcst',
        'forecast_Solar': 'Solar Fcst',
        'gen_forecast_total': 'Gen Fcst Total',
        'load_forecast_Forecasted Load': 'Demand Fcst',
        'Actual Load': 'Load',
        'Day_Ahead_Price': 'Price',
        'price_lag1': 'Price(t-1)',
        'apparent_temperature': 'Apparent Temp',
        'temperature_2m': 'Temp 2m',
        'wind_speed_10m': 'Wind 10m',
        'wind_speed_100m': 'Wind 100m',
        'shortwave_radiation': 'Solar Rad',
        'relative_humidity_2m': 'Humidity',
        'cloud_cover': 'Cloud Cover',
        'precipitation': 'Precip',
        'day_of_week': 'DoW',
        'day_of_year_sin': 'DoY(sin)',
        'day_of_year_cos': 'DoY(cos)',
        'dow_sin': 'DoW(sin)',
        'dow_cos': 'DoW(cos)',
        'is_weekend': 'Weekend',
        'is_holiday': 'Holiday',
    }
    for old, new in replacements.items():
        if name == old:
            return new
    return name[:15]


def find_best_experiment(country_dir, prefer_seed=42):
    """Find the best experiment directory for a country.

    Prefers non-spillover experiments with the preferred seed.
    Falls back to spillover experiments if no non-spillover exists.
    """
    if not country_dir.is_dir():
        return None

    experiments = sorted(country_dir.iterdir())
    # Priority 1: shared_backbone non-spillover with preferred seed (latest timestamp)
    shared_seed = [e for e in experiments if e.is_dir()
                   and e.name.startswith('shared_backbone_d2')
                   and 'spillover' not in e.name
                   and f'seed{prefer_seed}' in e.name
                   and (e / 'checkpoints' / 'final.tar').exists()]
    if shared_seed:
        return shared_seed[-1]  # latest by timestamp suffix
    # Priority 2: any shared_backbone non-spillover with final.tar
    shared_any = [e for e in experiments if e.is_dir()
                  and e.name.startswith('shared_backbone')
                  and 'spillover' not in e.name
                  and (e / 'checkpoints' / 'final.tar').exists()]
    if shared_any:
        return shared_any[-1]
    # Priority 3: latest spillover with final.tar
    for exp in reversed(experiments):
        if not exp.is_dir():
            continue
        if 'spillover' in exp.name:
            ckpt = exp / 'checkpoints' / 'final.tar'
            if ckpt.exists():
                return exp
    # Priority 4: legacy h1/seed42 structure
    h1_path = country_dir / 'h1' / f'seed{prefer_seed}' / 'checkpoints' / 'final.tar'
    if h1_path.exists():
        return country_dir / 'h1' / f'seed{prefer_seed}'
    return None


def _load_W_from_experiment(exp_dir, regime):
    """Load W matrix and feature_cols from an experiment directory. Returns (W, feature_cols) or (None, None).

    Prefers the full PyTorch checkpoint (``checkpoints/final.tar``) when
    present. When it is absent, falls back to the lightweight
    ``W_tensors.npz`` extract produced by
    ``electricity/extract_W_tensors.py`` -- it holds the same
    ``causal_emissions.<regime>.icgnn.W`` tensor, so the spillover figures
    reproduce identically without shipping the multi-GB checkpoints.
    """
    cfg_path = exp_dir / 'config.json'
    if not cfg_path.exists():
        return None, None
    with open(cfg_path) as f:
        cfg = json.load(f)
    feature_cols = cfg['data']['feature_cols']

    ckpt_path = exp_dir / 'checkpoints' / 'final.tar'
    if ckpt_path.exists():
        import torch
        ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
        sd = ckpt['model_state_dict']
        W = sd[f'causal_emissions.{regime}.icgnn.W'][0].numpy()
        return W, feature_cols

    npz_path = exp_dir / 'W_tensors.npz'
    if npz_path.exists():
        data = np.load(npz_path)
        key = f'W_{regime}'
        if key not in data.files:
            return None, None
        return data[key][0], feature_cols

    return None, None


def load_network_data(results_dir, regime=0, threshold=0.08, top_n=3,
                      cumulative_threshold=None, min_w_max=0.03):
    """Load W matrices from checkpoints and extract top edges per market.

    Args:
        results_dir: Path to results directory with per-country subdirs.
        regime: Regime index (0=Stable, 1=Crisis).
        threshold: Minimum |W| to include an edge (used if cumulative_threshold is None).
        top_n: Maximum edges per market (used if cumulative_threshold is None).
        cumulative_threshold: If set (e.g. 0.90), select features that account for
            this fraction of total incoming causal weight to price. Overrides
            threshold and top_n.
        min_w_max: Minimum max|W| to consider a checkpoint valid (detect collapsed W).
    """
    import torch

    network = {}
    for country_dir in sorted(results_dir.iterdir()):
        if not country_dir.is_dir():
            continue
        country = country_dir.name
        if country not in MARKET_COORDS:
            continue

        exp_dir = find_best_experiment(country_dir)
        if exp_dir is None:
            print(f'  WARNING: No checkpoint found for {country}, skipping')
            continue

        W, feature_cols = _load_W_from_experiment(exp_dir, regime)
        # Check for collapsed W — try spillover fallback
        if W is not None and float(np.abs(W).max()) < min_w_max:
            print(f'  {country}: W collapsed in {exp_dir.name} (max={np.abs(W).max():.4f}), trying spillover...')
            spill = sorted([e for e in country_dir.iterdir()
                           if e.is_dir() and 'spillover' in e.name
                           and (e / 'checkpoints' / 'final.tar').exists()],
                          key=lambda e: e.name)
            for sp in reversed(spill):
                W2, fc2 = _load_W_from_experiment(sp, regime)
                if W2 is not None and float(np.abs(W2).max()) >= min_w_max:
                    W, feature_cols = W2, fc2
                    exp_dir = sp
                    print(f'    -> using {sp.name} (max={np.abs(W).max():.4f})')
                    break

        if W is None or feature_cols is None:
            print(f'  WARNING: Could not load data for {country}, skipping')
            continue

        # Extract all edges to price (index 0). Separate cross-border
        # spillover features (which become inter-market arrows in the
        # visualisation) from domestic features (which remain satellite
        # nodes). Aggregate spillover lags 1h/24h per source country by
        # keeping the horizon with the largest |W|.
        price_idx = 0
        domestic_edges = []
        best_per_source = {}  # source_country -> (w, feat_name)
        for i, feat in enumerate(feature_cols):
            if i == price_idx:
                continue
            w = float(W[i, price_idx])
            if abs(w) <= 1e-6:
                continue
            src = spillover_source(feat)
            if src is not None and src != country:
                prev = best_per_source.get(src)
                if prev is None or abs(w) > abs(prev[0]):
                    best_per_source[src] = (w, feat)
            else:
                ftype, color = classify_feature(feat)
                domestic_edges.append({
                    'feature': feat,
                    'short_name': shorten_feature_name(feat),
                    'weight': w,
                    'type': ftype,
                    'color': color,
                })

        spillover_candidates = [
            {
                'source': src,
                'feature': feat,
                'short_name': src,
                'weight': w,
                'type': 'spillover',
                'color': FEATURE_TYPES['spillover']['color'],
            }
            for src, (w, feat) in best_per_source.items()
        ]

        # Rank all candidate edges jointly; select top-N.
        all_edges = domestic_edges + spillover_candidates
        all_edges.sort(key=lambda x: abs(x['weight']), reverse=True)

        if cumulative_threshold is not None and all_edges:
            total_weight = sum(abs(e['weight']) for e in all_edges)
            if total_weight > 0:
                cumsum = 0.0
                edges = []
                for e in all_edges:
                    edges.append(e)
                    cumsum += abs(e['weight'])
                    if cumsum / total_weight >= cumulative_threshold:
                        break
                edges = edges[:max(1, min(len(edges), 5))]
            else:
                edges = []
        else:
            edges = [e for e in all_edges if abs(e['weight']) > threshold]
            edges = edges[:top_n]

        selected_domestic = [e for e in edges if e['type'] != 'spillover']
        selected_spillover = [e for e in edges if e['type'] == 'spillover']

        print(f'  {country}: {len(selected_domestic)} domestic + '
              f'{len(selected_spillover)} spillover (top {len(edges)}) '
              f'from {exp_dir.name}')
        network[country] = {
            'edges': edges,  # kept for backwards-compat detection
            'domestic_edges': selected_domestic,
            'spillover_edges': selected_spillover,
            'W_mean': float(np.abs(W).mean()),
            'n_features': len(feature_cols),
        }

    return network


def draw_network(network, regime=0, output_path=None):
    """Draw the integrated European market network."""
    fig, ax = plt.subplots(1, 1, figsize=(20, 24))

    # Compute tight axis limits from all node positions (markets + domestic
    # satellites). Spillover features are rendered as inter-market arrows
    # rather than satellite nodes, so they contribute no new positions.
    all_x, all_y = [], []
    radius = 2.2  # satellite radius
    for country, data in network.items():
        if country not in MARKET_COORDS:
            continue
        cx, cy = MARKET_COORDS[country]
        all_x.append(cx)
        all_y.append(cy)
        n_sat = len(data.get('domestic_edges', data.get('edges', [])))
        for idx in range(n_sat):
            angle = (2 * np.pi * idx / max(n_sat, 1)) - np.pi / 2
            all_x.append(cx + radius * np.cos(angle))
            all_y.append(cy + radius * np.sin(angle))

    pad_x = 1.5
    pad_y_bottom = 1.5
    pad_y_top = 3.0  # just enough for legend row above SE
    ax.set_xlim(min(all_x) - pad_x, max(all_x) + pad_x)
    ax.set_ylim(min(all_y) - pad_y_bottom, max(all_y) + pad_y_top)
    ax.set_aspect('equal')
    ax.axis('off')

    regime_label = 'Stable' if regime == 0 else 'Crisis'
    has_spillover = any(
        data.get('spillover_edges') for data in network.values()
    )
    if has_spillover:
        driver_desc = ('Top 5 drivers per market: satellite nodes are domestic'
                       ' features; curved arrows are inter-market spillover')
    else:
        driver_desc = 'Top domestic drivers per market'
    ax.set_title(
        f'CaRS Learned Causal Network: 12 European Electricity Markets\n'
        f'Regime {regime} ({regime_label}) — {driver_desc}',
        fontsize=18, fontweight='bold', pad=20
    )

    # Pre-scan all edge weights for a globally-consistent line-width scale
    # so edge widths are comparable across markets (domestic + spillover).
    all_weights = [abs(e['weight']) for data in network.values() for e in data['edges']]
    w_max = max(all_weights) if all_weights else 1.0

    def edge_linewidth(w):
        """Map |w| linearly to [0.6, 6.0] pt line width using the global max."""
        return 0.6 + 5.4 * (abs(w) / max(w_max, 1e-9))

    # --- (1) Inter-market spillover arrows, drawn behind nodes ---
    # Curved arrows (arc3 rad) stagger source->target and target->source so
    # bi-directional spillover pairs don't overlap. Ends are offset away
    # from the market node centres (shrink) so arrows don't disappear
    # under the node circles.
    for target, data in network.items():
        if target not in MARKET_COORDS:
            continue
        tx, ty = MARKET_COORDS[target]
        for edge in data.get('spillover_edges', []):
            src = edge['source']
            if src not in MARKET_COORDS:
                continue
            sx, sy = MARKET_COORDS[src]
            w = edge['weight']
            arrow_color = '#e74c3c' if w > 0 else '#3498db'
            linestyle = '-' if w > 0 else '--'
            lw = edge_linewidth(w)
            # Curvature sign based on source→target code order keeps reverse
            # pairs on opposite sides.
            rad = 0.18 if src < target else -0.18
            ax.annotate(
                '', xy=(tx, ty), xytext=(sx, sy),
                arrowprops=dict(
                    arrowstyle='-|>', color=arrow_color, lw=lw,
                    linestyle=linestyle, alpha=0.85,
                    shrinkA=12, shrinkB=12,
                    connectionstyle=f'arc3,rad={rad}',
                ),
                zorder=4,
            )
            # Weight label near the target end of the arrow
            lx = 0.35 * sx + 0.65 * tx
            ly = 0.35 * sy + 0.65 * ty
            # Nudge label off the arrow line in the curvature direction
            dx, dy = tx - sx, ty - sy
            norm = (dx * dx + dy * dy) ** 0.5 or 1.0
            nx, ny = -dy / norm, dx / norm  # unit normal
            label_offset = 0.45 * (1 if rad > 0 else -1)
            ax.text(
                lx + label_offset * nx, ly + label_offset * ny,
                f'{w:+.2f}', fontsize=6, ha='center', va='center',
                color=arrow_color, zorder=6,
                bbox=dict(boxstyle='round,pad=0.1', facecolor='white',
                          edgecolor='none', alpha=0.75),
            )

    # --- (2) Market nodes + domestic feature satellites ---
    for country, data in network.items():
        if country not in MARKET_COORDS:
            continue

        cx, cy = MARKET_COORDS[country]

        # Market node (large circle)
        ax.scatter(cx, cy, s=1200, c='#2c3e50', zorder=10,
                   edgecolors='white', linewidth=2)
        ax.text(cx, cy, country, ha='center', va='center',
                fontsize=14, fontweight='bold', color='white', zorder=11)

        # Draw domestic satellite driver nodes only. Spillover drivers are
        # rendered as inter-market arrows above.
        domestic = data.get('domestic_edges', [])
        n_sat = len(domestic)
        for idx, edge in enumerate(domestic):
            angle = (2 * np.pi * idx / max(n_sat, 1)) - np.pi / 2
            sx = cx + radius * np.cos(angle)
            sy = cy + radius * np.sin(angle)

            ax.scatter(sx, sy, s=400, c=edge['color'], zorder=8,
                       edgecolors='#2c3e50', linewidth=1, alpha=0.85)

            label = edge['short_name']
            ax.text(sx, sy - 0.7, label, ha='center', va='top',
                    fontsize=7, color='#2c3e50', zorder=9,
                    bbox=dict(boxstyle='round,pad=0.15', facecolor='white',
                              edgecolor='none', alpha=0.8))

            w = edge['weight']
            arrow_color = '#e74c3c' if w > 0 else '#3498db'
            linestyle = '-' if w > 0 else '--'
            linewidth = edge_linewidth(w)

            ax.annotate('', xy=(cx, cy), xytext=(sx, sy),
                        arrowprops=dict(arrowstyle='->', color=arrow_color,
                                        lw=linewidth, linestyle=linestyle),
                        zorder=7)

            mid_x = (cx + sx) / 2
            mid_y = (cy + sy) / 2
            ax.text(mid_x, mid_y, f'{w:+.2f}', fontsize=6,
                    ha='center', va='center', color=arrow_color,
                    bbox=dict(boxstyle='round,pad=0.1', facecolor='white',
                              edgecolor='none', alpha=0.7),
                    zorder=9)

    # Legend — domestic satellite colours only (spillover is rendered as
    # inter-market arrows, not satellite nodes, so its orange category
    # is suppressed here to avoid misleading readers).
    legend_elements = []
    for ftype, info in FEATURE_TYPES.items():
        if ftype == 'spillover':
            continue
        legend_elements.append(
            mpatches.Patch(facecolor=info['color'], edgecolor='#2c3e50',
                           label=info['label'])
        )
    legend_elements.append(
        plt.Line2D([0], [0], color='#e74c3c', lw=edge_linewidth(w_max),
                   label=f'Positive $+W$ (width $\\propto|W|$, max {w_max:.2f})')
    )
    legend_elements.append(
        plt.Line2D([0], [0], color='#3498db', lw=edge_linewidth(w_max),
                   linestyle='--',
                   label='Negative $-W$')
    )
    legend_elements.append(
        plt.Line2D([0], [0], color='#e74c3c', lw=edge_linewidth(w_max * 0.3),
                   label='Reference: $|W|\\!\\approx\\!0.3\\,w_{\\max}$')
    )

    # Place legend top-left, anchored at SE's y level
    es_x = MARKET_COORDS['ES'][0]
    se_y = MARKET_COORDS['SE'][1]
    legend = ax.legend(handles=legend_elements, fontsize=9,
                       framealpha=0.9, edgecolor='#bdc3c7',
                       loc='upper left',
                       bbox_to_anchor=(es_x, se_y + 1),
                       bbox_transform=ax.transData)

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight',
                    facecolor='white', edgecolor='none')
        print(f'Saved to {output_path}')
    else:
        plt.savefig('european_causal_network.pdf', dpi=150, bbox_inches='tight',
                    facecolor='white', edgecolor='none')
        print('Saved to european_causal_network.pdf')

    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Visualize European market causal network')
    parser.add_argument('--results_dir', type=str,
                        default='shared_backbone/results',
                        help='Directory with per-country results')
    parser.add_argument('--regime', type=int, default=0, choices=[0, 1])
    parser.add_argument('--threshold', type=float, default=0.08,
                        help='Min |W| to show an edge')
    parser.add_argument('--top_n', type=int, default=3,
                        help='Max edges per market')
    parser.add_argument('--cumulative_threshold', type=float, default=None,
                        help='Cumulative weight threshold (e.g. 0.90) for probabilistic feature selection')
    parser.add_argument('--output', type=str, default=None)
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    print(f'Loading from {results_dir}...')

    network = load_network_data(results_dir, regime=args.regime,
                                threshold=args.threshold, top_n=args.top_n,
                                cumulative_threshold=args.cumulative_threshold)
    print(f'Loaded {len(network)} markets')

    for country, data in sorted(network.items()):
        drivers = ', '.join(f"{e['short_name']}({e['weight']:+.2f})" for e in data['edges'])
        print(f'  {country}: {drivers}')

    output = args.output or f'european_causal_network_regime{args.regime}.pdf'
    draw_network(network, regime=args.regime, output_path=output)

    # Also generate regime 1 if regime 0 was requested
    if args.regime == 0:
        network1 = load_network_data(results_dir, regime=1,
                                     threshold=args.threshold, top_n=args.top_n,
                                     cumulative_threshold=args.cumulative_threshold)
        output1 = output.replace('regime0', 'regime1')
        draw_network(network1, regime=1, output_path=output1)


if __name__ == '__main__':
    main()
