#!/usr/bin/env python3
"""
Generate CaRS Architecture Flowchart as SVG.

Creates a horizontal flowchart showing:
Input (X, Y) --> GRU --> Shared Regime DAG --> Causal Emission --> Output (Y_hat)

With detailed GRU internals including reset/update gates, hidden state, regime, and latent.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle
from pathlib import Path

# Output directory
FIGURES_DIR = Path(__file__).parent.parent / 'figures'

# Colors matching presentation
COLORS = {
    'input': '#FFE066',       # elecyellow
    'output': '#FFE066',      # elecyellow
    'gru': '#E0E0E0',         # light gray
    'gru_gate': '#CCCCCC',    # gray
    'hidden': '#B0B0B0',      # medium gray
    'regime': '#FF9500',      # regimeorange
    'latent': '#4A90D9',      # causalblue
    'dag': '#4A90D9',         # causalblue
    'dag_shared': '#6BA3E0',  # lighter blue
    'dag_regime': '#FFB347',  # lighter orange
    'emission': '#7CB342',    # reprogreen
    'arrow': '#333333',       # dark gray
    'text': '#333333',        # dark gray
}


def draw_rounded_box(ax, x, y, width, height, color, label=None, label_pos='top',
                     fontsize=10, fontweight='bold', alpha=0.3, edgecolor='black'):
    """Draw a rounded rectangle with optional label."""
    box = FancyBboxPatch(
        (x - width/2, y - height/2), width, height,
        boxstyle="round,pad=0.02,rounding_size=0.1",
        facecolor=color, edgecolor=edgecolor, linewidth=1.5, alpha=alpha
    )
    ax.add_patch(box)

    if label:
        if label_pos == 'top':
            ax.text(x, y + height/2 + 0.15, label, ha='center', va='bottom',
                    fontsize=fontsize, fontweight=fontweight, color=COLORS['text'])
        elif label_pos == 'center':
            ax.text(x, y, label, ha='center', va='center',
                    fontsize=fontsize, fontweight=fontweight, color=COLORS['text'])
        elif label_pos == 'bottom':
            ax.text(x, y - height/2 - 0.15, label, ha='center', va='top',
                    fontsize=fontsize, fontweight=fontweight, color=COLORS['text'])

    return box


def draw_small_box(ax, x, y, width, height, color, label, fontsize=8, alpha=0.7):
    """Draw a small internal box."""
    box = FancyBboxPatch(
        (x - width/2, y - height/2), width, height,
        boxstyle="round,pad=0.01,rounding_size=0.05",
        facecolor=color, edgecolor='gray', linewidth=1, alpha=alpha
    )
    ax.add_patch(box)
    ax.text(x, y, label, ha='center', va='center', fontsize=fontsize, color=COLORS['text'])
    return box


def draw_circle(ax, x, y, radius, color, label, fontsize=8, alpha=0.7):
    """Draw a circle (for latent variables)."""
    circle = Circle((x, y), radius, facecolor=color, edgecolor='gray', linewidth=1, alpha=alpha)
    ax.add_patch(circle)
    ax.text(x, y, label, ha='center', va='center', fontsize=fontsize, color=COLORS['text'])
    return circle


def draw_arrow(ax, start, end, color=None, style='->', connectionstyle='arc3,rad=0'):
    """Draw an arrow between two points."""
    if color is None:
        color = COLORS['arrow']
    arrow = FancyArrowPatch(
        start, end,
        arrowstyle=style,
        connectionstyle=connectionstyle,
        color=color, linewidth=1.5,
        mutation_scale=12
    )
    ax.add_patch(arrow)
    return arrow


def create_cars_flowchart():
    """Create the CaRS architecture flowchart."""
    fig, ax = plt.subplots(figsize=(16, 10))
    ax.set_xlim(-1, 15)
    ax.set_ylim(-1, 8)
    ax.set_aspect('equal')
    ax.axis('off')

    # Main component positions (left to right)
    input_x = 0.5
    gru_x = 4.5
    dag_x = 9.5
    emission_x = 12.5
    output_x = 14.5
    main_y = 4

    # =========================================================================
    # 1. INPUT BOX
    # =========================================================================
    draw_rounded_box(ax, input_x, main_y, 1.5, 2, COLORS['input'], 'Input', alpha=0.5)
    draw_small_box(ax, input_x, main_y + 0.3, 0.8, 0.5, COLORS['input'], r'$X_t$', fontsize=10, alpha=0.8)
    draw_small_box(ax, input_x, main_y - 0.3, 0.8, 0.5, COLORS['input'], r'$Y_t$', fontsize=10, alpha=0.8)
    ax.text(input_x, main_y - 1.3, 'Features &\nTarget', ha='center', va='top', fontsize=8, style='italic')

    # =========================================================================
    # 2. GRU BOX (expanded with internal details)
    # =========================================================================
    gru_width = 4.5
    gru_height = 5.5
    gru_top = main_y + gru_height/2
    gru_bottom = main_y - gru_height/2

    draw_rounded_box(ax, gru_x, main_y, gru_width, gru_height, COLORS['gru'],
                     'GRU Backbone', label_pos='top', alpha=0.3)

    # Reset and Update gates
    gate_y = main_y + 1.8
    gate_width = 1.2
    gate_height = 0.7

    draw_small_box(ax, gru_x - 1.0, gate_y, gate_width, gate_height, COLORS['gru_gate'],
                   r'Reset $r_t$', fontsize=9)
    draw_small_box(ax, gru_x + 1.0, gate_y, gate_width, gate_height, COLORS['gru_gate'],
                   r'Update $u_t$', fontsize=9)

    # Gates formula annotation
    ax.text(gru_x, gate_y + 0.6, r'$r_t = \sigma(W_r[h_{t-1}, x_t])$', ha='center', va='bottom',
            fontsize=7, style='italic', color='gray')
    ax.text(gru_x, gate_y - 0.55, r'$u_t = \sigma(W_u[h_{t-1}, x_t])$', ha='right', va='top',
            fontsize=7, style='italic', color='gray')

    # Hidden state
    hidden_y = main_y + 0.3
    draw_small_box(ax, gru_x, hidden_y, 1.5, 0.8, COLORS['hidden'], r'$h_t$', fontsize=11)
    ax.text(gru_x + 1.0, hidden_y, 'hidden\nstate', ha='left', va='center', fontsize=7, style='italic')

    # Arrows from gates to hidden state
    draw_arrow(ax, (gru_x - 1.0, gate_y - gate_height/2), (gru_x - 0.3, hidden_y + 0.4))
    draw_arrow(ax, (gru_x + 1.0, gate_y - gate_height/2), (gru_x + 0.3, hidden_y + 0.4))

    # Regime (d_t) and Latent (z_t)
    regime_y = main_y - 1.2
    latent_y = main_y - 1.2

    draw_circle(ax, gru_x - 1.0, regime_y, 0.45, COLORS['regime'], r'$d_t$', fontsize=10)
    ax.text(gru_x - 1.0, regime_y - 0.7, 'regime', ha='center', va='top', fontsize=7, style='italic')

    draw_circle(ax, gru_x + 1.0, latent_y, 0.45, COLORS['latent'], r'$z_t$', fontsize=10)
    ax.text(gru_x + 1.0, latent_y - 0.7, 'latent', ha='center', va='top', fontsize=7, style='italic')

    # Arrows from hidden to regime and latent
    draw_arrow(ax, (gru_x - 0.4, hidden_y - 0.4), (gru_x - 1.0, regime_y + 0.45))
    draw_arrow(ax, (gru_x + 0.4, hidden_y - 0.4), (gru_x + 1.0, latent_y + 0.45))

    # Formulas for regime and latent
    ax.text(gru_x, main_y - 2.3, r'$d_t \sim \mathrm{Cat}(\mathrm{softmax}(W_d h_t))$',
            ha='center', va='top', fontsize=7, style='italic', color='gray')
    ax.text(gru_x, main_y - 2.7, r'$z_t \sim \mathcal{N}(\mu(h_t), \sigma(h_t))$',
            ha='center', va='top', fontsize=7, style='italic', color='gray')

    # =========================================================================
    # 3. SHARED REGIME DAG BOX
    # =========================================================================
    dag_width = 2.5
    dag_height = 4.0
    draw_rounded_box(ax, dag_x, main_y, dag_width, dag_height, COLORS['dag'],
                     'Shared\nRegime DAG', label_pos='top', alpha=0.3)

    # A_shared and A_regime boxes
    draw_small_box(ax, dag_x, main_y + 1.0, 1.8, 0.7, COLORS['dag_shared'],
                   r'$A_{\mathrm{shared}}$', fontsize=9)
    ax.text(dag_x, main_y + 0.45, 'backbone', ha='center', va='top', fontsize=7, style='italic')

    draw_small_box(ax, dag_x, main_y - 0.3, 1.8, 0.7, COLORS['dag_regime'],
                   r'$A_{\mathrm{regime}}^{(d)}$', fontsize=9)
    ax.text(dag_x, main_y - 0.85, 'sparse\ndeviation', ha='center', va='top', fontsize=7, style='italic')

    # Combined adjacency
    draw_small_box(ax, dag_x, main_y - 1.5, 2.0, 0.6, COLORS['dag'],
                   r'$A^{(d)}$', fontsize=9, alpha=0.9)
    ax.text(dag_x, main_y - 1.95, r'$= 1-(1-A_s)(1-A_r)$', ha='center', va='top',
            fontsize=7, style='italic', color='gray')

    # Arrows inside DAG box
    draw_arrow(ax, (dag_x, main_y + 0.65), (dag_x, main_y - 0.0))
    draw_arrow(ax, (dag_x, main_y - 0.65), (dag_x - 0.3, main_y - 1.2))
    draw_arrow(ax, (dag_x, main_y + 0.65), (dag_x + 0.3, main_y - 1.2), connectionstyle='arc3,rad=-0.2')

    # =========================================================================
    # 4. CAUSAL EMISSION BOX
    # =========================================================================
    emission_width = 2.2
    emission_height = 3.5
    draw_rounded_box(ax, emission_x, main_y, emission_width, emission_height, COLORS['emission'],
                     'Causal\nEmission', label_pos='top', alpha=0.3)

    # CausalICGNN
    draw_small_box(ax, emission_x, main_y + 0.5, 1.8, 0.8, COLORS['emission'],
                   'CausalICGNN', fontsize=8, alpha=0.8)

    # Conditioning
    draw_small_box(ax, emission_x, main_y - 0.6, 1.8, 0.7, COLORS['hidden'],
                   r'Cond: $[h_t, z_t]$', fontsize=8, alpha=0.6)

    # Arrow from conditioning to ICGNN
    draw_arrow(ax, (emission_x, main_y - 0.25), (emission_x, main_y + 0.1))

    # =========================================================================
    # 5. OUTPUT BOX
    # =========================================================================
    draw_rounded_box(ax, output_x, main_y, 1.2, 1.5, COLORS['output'], 'Output', alpha=0.5)
    draw_small_box(ax, output_x, main_y, 0.8, 0.6, COLORS['output'], r'$\hat{Y}_t$', fontsize=11, alpha=0.8)
    ax.text(output_x, main_y - 1.0, 'Predicted\nTarget', ha='center', va='top', fontsize=8, style='italic')

    # =========================================================================
    # MAIN FLOW ARROWS (between components)
    # =========================================================================
    arrow_y = main_y

    # Input -> GRU
    draw_arrow(ax, (input_x + 0.75, arrow_y), (gru_x - gru_width/2, arrow_y),
               style='-|>', color=COLORS['arrow'])

    # GRU -> DAG (from regime d_t)
    draw_arrow(ax, (gru_x + gru_width/2 - 0.5, regime_y), (dag_x - dag_width/2, main_y - 0.3),
               style='-|>', color=COLORS['regime'], connectionstyle='arc3,rad=0.1')
    ax.text((gru_x + dag_x)/2 + 0.3, regime_y + 0.5, r'$d_t$ selects', fontsize=7, style='italic',
            color=COLORS['regime'])

    # DAG -> Emission (adjacency matrix)
    draw_arrow(ax, (dag_x + dag_width/2, main_y - 1.5), (emission_x - emission_width/2, main_y + 0.5),
               style='-|>', color=COLORS['dag'], connectionstyle='arc3,rad=-0.1')
    ax.text((dag_x + emission_x)/2, main_y - 0.3, r'$A^{(d)}$', fontsize=8, color=COLORS['dag'])

    # GRU -> Emission (h_t and z_t for conditioning)
    draw_arrow(ax, (gru_x + gru_width/2 - 0.3, hidden_y - 0.3), (emission_x - emission_width/2, main_y - 0.4),
               style='-|>', color=COLORS['hidden'], connectionstyle='arc3,rad=-0.15')
    draw_arrow(ax, (gru_x + 1.0 + 0.45, latent_y), (emission_x - emission_width/2, main_y - 0.7),
               style='-|>', color=COLORS['latent'], connectionstyle='arc3,rad=0.1')

    # Emission -> Output
    draw_arrow(ax, (emission_x + emission_width/2, arrow_y), (output_x - 0.6, arrow_y),
               style='-|>', color=COLORS['arrow'])

    # =========================================================================
    # TITLE
    # =========================================================================
    ax.text(7, 7.5, 'CaRS: Causal Regime-Switching Model Architecture',
            ha='center', va='center', fontsize=14, fontweight='bold')

    # =========================================================================
    # LEGEND
    # =========================================================================
    legend_y = 0.3
    legend_items = [
        (COLORS['input'], 'Input/Output'),
        (COLORS['gru'], 'GRU Backbone'),
        (COLORS['regime'], 'Regime (discrete)'),
        (COLORS['latent'], 'Latent (continuous)'),
        (COLORS['dag'], 'Causal DAG'),
        (COLORS['emission'], 'Emission Model'),
    ]

    for i, (color, label) in enumerate(legend_items):
        x = 1.5 + i * 2.2
        rect = FancyBboxPatch((x - 0.15, legend_y - 0.15), 0.3, 0.3,
                              boxstyle="round,pad=0.02", facecolor=color,
                              edgecolor='gray', linewidth=0.5, alpha=0.7)
        ax.add_patch(rect)
        ax.text(x + 0.25, legend_y, label, ha='left', va='center', fontsize=8)

    plt.tight_layout()
    return fig


def create_icgnn_flowchart():
    """Create the CausalICGNN internal architecture flowchart."""
    fig, ax = plt.subplots(figsize=(14, 16))
    ax.set_xlim(-4, 12)
    ax.set_ylim(-1, 17)
    ax.set_aspect('equal')
    ax.axis('off')

    # Vertical flow (top to bottom)
    center_x = 4
    y_positions = {
        'input': 15.5,
        'embed': 13.5,
        'concat': 11.5,
        'encoder': 9,
        'agg': 6,
        'decoder': 3,
        'output': 0.5,
    }

    # Colors matching presentation
    data_color = '#FFE066'      # elecyellow
    net_color = '#4A90D9'       # causalblue
    op_color = '#7CB342'        # reprogreen
    regime_color = '#FF9500'    # regimeorange
    layer_color = '#E0E0E0'     # light gray
    var_color = '#BBBBBB'       # gray

    # =========================================================================
    # 1. INPUT
    # =========================================================================
    draw_rounded_box(ax, center_x, y_positions['input'], 5, 0.9, data_color,
                     alpha=0.4, edgecolor='black')
    ax.text(center_x, y_positions['input'], r'Input: $X \in \mathbb{R}^{B \times (L+1) \times n}$',
            ha='center', va='center', fontsize=11)
    ax.text(center_x + 3.5, y_positions['input'], r'$B$: batch, $L$: lag, $n$: nodes',
            ha='left', va='center', fontsize=8, style='italic', color='gray')

    # =========================================================================
    # 2. NODE EMBEDDINGS
    # =========================================================================
    draw_rounded_box(ax, center_x, y_positions['embed'], 4.5, 0.9, regime_color,
                     alpha=0.3, edgecolor='black')
    ax.text(center_x, y_positions['embed'], r'Node Embeddings $E$',
            ha='center', va='center', fontsize=11, fontweight='bold')
    ax.text(center_x + 3.5, y_positions['embed'],
            r'$E \in \mathbb{R}^{(L+1) \times n \times d}$ (learnable)',
            ha='left', va='center', fontsize=8, style='italic', color='gray')

    # =========================================================================
    # 3. CONCATENATION
    # =========================================================================
    draw_rounded_box(ax, center_x, y_positions['concat'], 4.5, 0.9, op_color,
                     alpha=0.3, edgecolor='black')
    ax.text(center_x, y_positions['concat'], r'Concatenate $[X_{\mathrm{exp}}, E]$',
            ha='center', va='center', fontsize=11)
    ax.text(center_x + 3.5, y_positions['concat'], r'Shape: $[B, L{+}1, n, n, d{+}n]$',
            ha='left', va='center', fontsize=8, style='italic', color='gray')

    # =========================================================================
    # 4. ENCODER NETWORK g
    # =========================================================================
    encoder_height = 2.8
    draw_rounded_box(ax, center_x, y_positions['encoder'], 5.5, encoder_height, net_color,
                     alpha=0.25, edgecolor='black')
    ax.text(center_x, y_positions['encoder'] + encoder_height/2 - 0.4,
            r'Encoder Network $g$', ha='center', va='center', fontsize=11, fontweight='bold')

    # Encoder layers
    layer_y = y_positions['encoder'] + 0.1
    for i, layer_text in enumerate([r'FC $\rightarrow$ LN $\rightarrow$ LeakyReLU',
                                    r'FC $\rightarrow$ LN $\rightarrow$ LeakyReLU',
                                    r'FC $\rightarrow$ Output$(d)$']):
        draw_rounded_box(ax, center_x, layer_y - i * 0.6, 4, 0.5, layer_color,
                         alpha=0.6, edgecolor='gray')
        ax.text(center_x, layer_y - i * 0.6, layer_text,
                ha='center', va='center', fontsize=8)

    ax.text(center_x + 3.8, y_positions['encoder'],
            r'$h = g(X, E)$' + '\n' + r'$\in \mathbb{R}^{B \times (L{+}1) \times n \times n \times d}$',
            ha='left', va='center', fontsize=8, style='italic', color='gray')

    # =========================================================================
    # 5. WEIGHTED MESSAGE AGGREGATION
    # =========================================================================
    agg_width = 6
    agg_height = 1.6
    draw_rounded_box(ax, center_x, y_positions['agg'], agg_width, agg_height, net_color,
                     alpha=0.4, edgecolor='black')
    ax.text(center_x, y_positions['agg'] + 0.35, r'Weighted Message Aggregation',
            ha='center', va='center', fontsize=11, fontweight='bold')
    ax.text(center_x, y_positions['agg'] - 0.35, r'$h_{\mathrm{agg}} = \sum_j (W \odot A)_{ij} \cdot h_j$',
            ha='center', va='center', fontsize=10)

    ax.text(center_x + 4, y_positions['agg'], r'$W$: learned weights' + '\n' + r'$A$: DAG (Gumbel)',
            ha='left', va='center', fontsize=8, style='italic', color='gray')

    # Side inputs (W and A^(d))
    w_x = center_x - 4.5
    a_x = center_x - 7
    side_y = y_positions['agg']

    draw_rounded_box(ax, w_x, side_y, 1.2, 0.8, net_color, alpha=0.5, edgecolor='gray')
    ax.text(w_x, side_y, r'$W$', ha='center', va='center', fontsize=10, fontweight='bold')

    draw_rounded_box(ax, a_x, side_y, 1.4, 0.8, regime_color, alpha=0.5, edgecolor='gray')
    ax.text(a_x, side_y, r'$A^{(d)}$', ha='center', va='center', fontsize=10, fontweight='bold')

    # Arrows for side inputs
    draw_arrow(ax, (a_x + 0.7, side_y), (w_x - 0.6, side_y), style='-|>')
    draw_arrow(ax, (w_x + 0.6, side_y), (center_x - agg_width/2, side_y), style='-|>')

    # =========================================================================
    # 6. DECODER NETWORK f
    # =========================================================================
    decoder_height = 2.8
    draw_rounded_box(ax, center_x, y_positions['decoder'], 5.5, decoder_height, net_color,
                     alpha=0.25, edgecolor='black')
    ax.text(center_x, y_positions['decoder'] + decoder_height/2 - 0.4,
            r'Decoder Network $f$', ha='center', va='center', fontsize=11, fontweight='bold')

    # Decoder layers
    layer_y = y_positions['decoder'] + 0.1
    for i, layer_text in enumerate([r'FC $\rightarrow$ LN $\rightarrow$ LeakyReLU',
                                    r'FC $\rightarrow$ LN $\rightarrow$ LeakyReLU',
                                    r'FC $\rightarrow$ Output$(n)$']):
        draw_rounded_box(ax, center_x, layer_y - i * 0.6, 4, 0.5, layer_color,
                         alpha=0.6, edgecolor='gray')
        ax.text(center_x, layer_y - i * 0.6, layer_text,
                ha='center', va='center', fontsize=8)

    ax.text(center_x + 3.8, y_positions['decoder'],
            r'$\hat{y} = f(E, h_{\mathrm{agg}})$' + '\n' + r'$\in \mathbb{R}^{B \times n}$',
            ha='left', va='center', fontsize=8, style='italic', color='gray')

    # =========================================================================
    # 7. OUTPUT
    # =========================================================================
    draw_rounded_box(ax, center_x, y_positions['output'], 5, 0.9, op_color,
                     alpha=0.4, edgecolor='black')
    ax.text(center_x, y_positions['output'], r'Predictions: $\hat{y} \in \mathbb{R}^{B \times n}$',
            ha='center', va='center', fontsize=11, fontweight='bold')

    # =========================================================================
    # 8. VARIANCE BRANCH (optional)
    # =========================================================================
    var_x = center_x + 6
    var_y = y_positions['decoder'] + 1

    draw_rounded_box(ax, var_x, var_y, 2, 1.2, var_color, alpha=0.3, edgecolor='gray')
    ax.text(var_x, var_y + 0.2, 'Variance', ha='center', va='center', fontsize=9)
    ax.text(var_x, var_y - 0.25, r'$g_{\mathrm{var}}, f_{\mathrm{var}}$',
            ha='center', va='center', fontsize=8)

    draw_rounded_box(ax, var_x, var_y - 1.5, 1.2, 0.7, var_color, alpha=0.5, edgecolor='gray')
    ax.text(var_x, var_y - 1.5, r'$\sigma^2$', ha='center', va='center', fontsize=10, fontweight='bold')

    ax.text(var_x + 1.2, var_y, '(heteroscedastic)', ha='left', va='center',
            fontsize=7, style='italic', color='gray')

    # Arrow from aggregation to variance branch
    draw_arrow(ax, (center_x + agg_width/2 - 0.5, y_positions['agg'] - 0.3),
               (var_x - 0.8, var_y + 0.5), style='-|>', color='gray',
               connectionstyle='arc3,rad=-0.2')
    draw_arrow(ax, (var_x, var_y - 0.6), (var_x, var_y - 1.15), style='-|>', color='gray')

    # =========================================================================
    # MAIN FLOW ARROWS
    # =========================================================================
    # Input -> Embed
    draw_arrow(ax, (center_x, y_positions['input'] - 0.45),
               (center_x, y_positions['embed'] + 0.45), style='-|>')

    # Embed -> Concat
    draw_arrow(ax, (center_x, y_positions['embed'] - 0.45),
               (center_x, y_positions['concat'] + 0.45), style='-|>')

    # Concat -> Encoder
    draw_arrow(ax, (center_x, y_positions['concat'] - 0.45),
               (center_x, y_positions['encoder'] + encoder_height/2), style='-|>')

    # Encoder -> Agg
    draw_arrow(ax, (center_x, y_positions['encoder'] - encoder_height/2),
               (center_x, y_positions['agg'] + agg_height/2), style='-|>')

    # Agg -> Decoder
    draw_arrow(ax, (center_x, y_positions['agg'] - agg_height/2),
               (center_x, y_positions['decoder'] + decoder_height/2), style='-|>')

    # Decoder -> Output
    draw_arrow(ax, (center_x, y_positions['decoder'] - decoder_height/2),
               (center_x, y_positions['output'] + 0.45), style='-|>')

    # Skip connection: Embed -> Decoder (dashed)
    skip_x = center_x - 3.5
    ax.annotate('', xy=(skip_x + 0.3, y_positions['decoder']),
                xytext=(skip_x + 0.3, y_positions['embed']),
                arrowprops=dict(arrowstyle='-|>', color='gray', linestyle='--',
                               connectionstyle='arc3,rad=0.1', linewidth=1.5))
    ax.text(skip_x - 0.3, (y_positions['embed'] + y_positions['decoder'])/2,
            r'$E$', ha='center', va='center', fontsize=9, color='gray', fontweight='bold')

    # =========================================================================
    # TITLE
    # =========================================================================
    ax.text(center_x, 16.5, 'CausalICGNN: Internal Architecture',
            ha='center', va='center', fontsize=14, fontweight='bold')

    plt.tight_layout()
    return fig


def main():
    """Generate and save the CaRS architecture flowchart."""
    print("=" * 70)
    print("Generating CaRS Architecture Flowchart")
    print("=" * 70)

    # Ensure output directory exists
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    # Create and save flowchart
    fig = create_cars_flowchart()

    output_path = FIGURES_DIR / 'cars_architecture_flowchart.svg'
    fig.savefig(output_path, format='svg', bbox_inches='tight', dpi=150)
    plt.close(fig)
    print(f"Saved: {output_path}")

    # Also save as PDF
    pdf_path = FIGURES_DIR / 'cars_architecture_flowchart.pdf'
    fig = create_cars_flowchart()
    fig.savefig(pdf_path, format='pdf', bbox_inches='tight', dpi=150)
    plt.close(fig)
    print(f"Saved: {pdf_path}")

    # =========================================================================
    # Generate CausalICGNN Internal Architecture
    # =========================================================================
    print("\n" + "=" * 70)
    print("Generating CausalICGNN Internal Architecture")
    print("=" * 70)

    # Create and save ICGNN flowchart (SVG)
    fig = create_icgnn_flowchart()
    output_path = FIGURES_DIR / 'icgnn_architecture.svg'
    fig.savefig(output_path, format='svg', bbox_inches='tight', dpi=150)
    plt.close(fig)
    print(f"Saved: {output_path}")

    # Also save as PDF
    pdf_path = FIGURES_DIR / 'icgnn_architecture.pdf'
    fig = create_icgnn_flowchart()
    fig.savefig(pdf_path, format='pdf', bbox_inches='tight', dpi=150)
    plt.close(fig)
    print(f"Saved: {pdf_path}")

    print("\n" + "=" * 70)
    print(f"All flowcharts saved to {FIGURES_DIR}/")
    print("=" * 70)


if __name__ == "__main__":
    main()
