"""
Quantitative Causal Validation for CaRS.

Provides three levels of causal validation:
1. Edge Stability: Are learned edges consistent across random seeds?
2. Merit Order Alignment: Do edge signs match energy economics?
3. Edge Ablation: Does removing specific edges degrade performance interpretably?
4. Regime-Conditional Effects: Do causal magnitudes vary by regime as expected?

These analyses support the paper's causal claims beyond forecasting accuracy.
"""

import sys
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from itertools import combinations
from typing import Dict, List, Optional, Tuple
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from electricity.evaluation.metrics import compute_all_metrics


# =============================================================================
# Merit Order Knowledge
# =============================================================================

# Expected sign of generation technology -> price causal effect
# Negative = price-suppressing (low marginal cost), Positive = price-increasing
MERIT_ORDER_SIGN = {
    'Nuclear': -1,
    'Wind Onshore': -1,
    'Wind Offshore': -1,
    'Solar': -1,
    'Hydro Pumped Storage': -1,
    'Hydro Run-of-river': -1,
    'Biomass': 0,            # Ambiguous: baseload, depends on subsidies
    'Fossil Brown coal': 0,  # Baseload in DE, ambiguous sign
    'Fossil Hard coal': +1,
    'Fossil Gas': +1,        # Typically price-setting, especially in crisis
    'Fossil Oil': +1,
}

# Approximate marginal cost ranking (lower = cheaper, more price-suppressing)
# Used for rank correlation analysis
MERIT_ORDER_RANK = {
    'Wind Onshore': 1,
    'Wind Offshore': 1,
    'Solar': 1,
    'Nuclear': 2,
    'Hydro Pumped Storage': 3,
    'Hydro Run-of-river': 3,
    'Biomass': 4,
    'Fossil Brown coal': 5,
    'Fossil Hard coal': 6,
    'Fossil Gas': 7,
    'Fossil Oil': 8,
}


def _match_feature_to_technology(feature_name: str) -> Optional[str]:
    """Map a feature column name to a generation technology."""
    feature_lower = feature_name.lower()
    for tech in MERIT_ORDER_SIGN:
        if tech.lower().replace(' ', '_') in feature_lower.replace(' ', '_'):
            return tech
        # Handle common column name patterns
        tech_parts = tech.lower().split()
        if all(p in feature_lower for p in tech_parts):
            return tech
    return None


def _get_generation_feature_indices(
    feature_names: List[str],
) -> Dict[str, int]:
    """Find indices of generation features in the feature list."""
    tech_indices = {}
    for i, name in enumerate(feature_names):
        tech = _match_feature_to_technology(name)
        if tech is not None:
            tech_indices[tech] = i
    return tech_indices


# =============================================================================
# 1. Edge Stability Across Seeds
# =============================================================================

def compute_edge_stability(
    checkpoint_paths: List[Path],
    model_class,
    model_kwargs: dict,
    threshold: float = 0.3,
    device: str = 'cuda'
) -> Dict:
    """
    Compute edge stability across multiple random seeds.

    Loads trained models from checkpoints, extracts DAGs, and measures
    how consistently edges appear across seeds. Stable edges (>= 80%
    frequency) represent robust causal claims.

    Args:
        checkpoint_paths: Paths to model checkpoints from different seeds
        model_class: DS3MCausal class for instantiation
        model_kwargs: kwargs to instantiate the model
        threshold: Edge probability threshold for binarization
        device: Torch device

    Returns:
        Dict with:
        - edge_frequency: [n_regimes, lag+1, n_nodes, n_nodes] frequency matrix
        - stable_edges: edges present in >= 80% of seeds per regime
        - unstable_edges: edges present in 20-80% of seeds
        - jaccard_similarity: pairwise Jaccard between seed DAGs
        - summary: aggregate statistics
    """
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    n_seeds = len(checkpoint_paths)

    # Load all DAGs
    all_dags = []  # List of [list of adjacency matrices per regime]
    for path in checkpoint_paths:
        model = model_class(**model_kwargs).to(device)
        state = torch.load(path, map_location=device, weights_only=False)
        if isinstance(state, dict) and 'model_state_dict' in state:
            model.load_state_dict(state['model_state_dict'])
        else:
            model.load_state_dict(state)
        model.eval()

        dags = model.get_causal_graphs()  # List of np arrays [lag+1, n, n]
        all_dags.append(dags)

    n_regimes = len(all_dags[0])
    dag_shape = all_dags[0][0].shape  # [lag+1, n_nodes, n_nodes]

    # Compute edge frequency per regime
    edge_frequency = np.zeros((n_regimes,) + dag_shape)
    for seed_dags in all_dags:
        for r, dag in enumerate(seed_dags):
            binary = (dag > threshold).astype(float)
            edge_frequency[r] += binary
    edge_frequency /= n_seeds

    # Classify edges
    stable_edges = {}
    unstable_edges = {}
    absent_edges = {}
    for r in range(n_regimes):
        stable = edge_frequency[r] >= 0.8
        unstable = (edge_frequency[r] >= 0.2) & (edge_frequency[r] < 0.8)
        absent = edge_frequency[r] < 0.2
        stable_edges[f'regime_{r}'] = {
            'count': int(stable.sum()),
            'indices': list(zip(*np.where(stable)))
        }
        unstable_edges[f'regime_{r}'] = {
            'count': int(unstable.sum()),
            'indices': list(zip(*np.where(unstable)))
        }
        absent_edges[f'regime_{r}'] = {'count': int(absent.sum())}

    # Pairwise Jaccard similarity between seeds
    jaccard_per_regime = {}
    for r in range(n_regimes):
        similarities = []
        for i, j in combinations(range(n_seeds), 2):
            dag_i = (all_dags[i][r] > threshold).flatten()
            dag_j = (all_dags[j][r] > threshold).flatten()
            intersection = (dag_i & dag_j).sum()
            union = (dag_i | dag_j).sum()
            jacc = float(intersection / union) if union > 0 else 1.0
            similarities.append(jacc)
        jaccard_per_regime[f'regime_{r}'] = {
            'mean': float(np.mean(similarities)) if similarities else 0.0,
            'std': float(np.std(similarities)) if similarities else 0.0,
            'all': similarities
        }

    # Summary
    summary = {
        'n_seeds': n_seeds,
        'n_regimes': n_regimes,
        'dag_shape': list(dag_shape),
        'threshold': threshold,
    }
    for r in range(n_regimes):
        summary[f'regime_{r}_stable'] = stable_edges[f'regime_{r}']['count']
        summary[f'regime_{r}_unstable'] = unstable_edges[f'regime_{r}']['count']
        summary[f'regime_{r}_jaccard_mean'] = jaccard_per_regime[f'regime_{r}']['mean']

    return {
        'edge_frequency': edge_frequency,
        'stable_edges': stable_edges,
        'unstable_edges': unstable_edges,
        'jaccard_similarity': jaccard_per_regime,
        'summary': summary
    }


# =============================================================================
# 2. Merit Order Alignment Score
# =============================================================================

def merit_order_alignment(
    W_weights: np.ndarray,
    feature_names: List[str],
    target_idx: int,
    dag: Optional[np.ndarray] = None,
    threshold: float = 0.3
) -> Dict:
    """
    Check if learned causal edge signs and magnitudes align with merit order.

    For generation technologies -> price edges, verifies:
    - Wind/Solar -> price has negative weight (price-suppressing)
    - Gas -> price has positive weight (price-increasing)
    - Magnitude ranking correlates with marginal cost ranking

    Args:
        W_weights: ICGNN weighted adjacency [lag+1, n_nodes, n_nodes]
        feature_names: List of feature column names
        target_idx: Index of the target (price) variable
        dag: Binary DAG adjacency (optional, for edge filtering)
        threshold: Probability threshold for considering an edge present

    Returns:
        Dict with alignment_score, rank_correlation, per_technology details
    """
    tech_indices = _get_generation_feature_indices(feature_names)

    if len(tech_indices) == 0:
        return {'alignment_score': None, 'error': 'No generation features found'}

    per_tech = {}
    correct_signs = 0
    total_with_expectation = 0

    for tech, idx in tech_indices.items():
        # Get the weight of edge idx -> target_idx (instantaneous, lag=0)
        w_inst = float(W_weights[0, idx, target_idx])

        # Also check lagged effects
        w_lag = float(W_weights[1, idx, target_idx]) if W_weights.shape[0] > 1 else 0.0

        # Check if edge exists in DAG
        edge_present = True
        if dag is not None:
            edge_present = dag[0, idx, target_idx] > threshold

        expected_sign = MERIT_ORDER_SIGN.get(tech, 0)
        actual_sign = np.sign(w_inst) if abs(w_inst) > 1e-6 else 0

        sign_correct = None
        if expected_sign != 0 and edge_present:
            sign_correct = (actual_sign == expected_sign)
            total_with_expectation += 1
            if sign_correct:
                correct_signs += 1

        per_tech[tech] = {
            'feature_idx': idx,
            'feature_name': feature_names[idx],
            'w_instantaneous': w_inst,
            'w_lagged': w_lag,
            'w_combined': w_inst + w_lag,
            'edge_present': edge_present,
            'expected_sign': expected_sign,
            'actual_sign': int(actual_sign),
            'sign_correct': sign_correct,
            'merit_rank': MERIT_ORDER_RANK.get(tech),
        }

    # Alignment score
    alignment_score = correct_signs / total_with_expectation if total_with_expectation > 0 else None

    # Rank correlation between |W| and merit order rank
    w_magnitudes = []
    merit_ranks = []
    for tech, info in per_tech.items():
        if info['merit_rank'] is not None and info['edge_present']:
            w_magnitudes.append(abs(info['w_combined']))
            merit_ranks.append(info['merit_rank'])

    rank_corr = None
    if len(w_magnitudes) >= 3:
        rank_corr, rank_pval = spearmanr(w_magnitudes, merit_ranks)
        rank_corr = float(rank_corr) if not np.isnan(rank_corr) else None

    return {
        'alignment_score': alignment_score,
        'n_correct_signs': correct_signs,
        'n_total_with_expectation': total_with_expectation,
        'rank_correlation': rank_corr,
        'per_technology': per_tech
    }


# =============================================================================
# 3. Edge Ablation Analysis
# =============================================================================

def edge_ablation_analysis(
    model,
    testX: torch.Tensor,
    testY: torch.Tensor,
    feature_names: List[str],
    target_idx: int = 0,
    edges_to_ablate: Optional[List[Tuple[int, int]]] = None,
    n_samples: int = 50,
    y_prev: Optional[np.ndarray] = None,
) -> Dict:
    """
    Measure prediction degradation when specific edges are removed.

    For each edge (source -> target_node), zeros it out in the DAG
    and re-runs prediction to measure the impact on forecasting.

    Args:
        model: Trained DS3MCausal model
        testX: Test input [timestep, batch, x_dim]
        testY: Test target [timestep, batch, y_dim]
        feature_names: Feature column names
        target_idx: Index of the price/target variable
        edges_to_ablate: List of (source_idx, target_idx) edges to ablate.
            If None, ablates all generation->price edges.
        n_samples: MC samples for prediction
        y_prev: Previous timestep values for directional accuracy

    Returns:
        Dict with baseline metrics, per-edge ablation impact, and ranking
    """
    device = next(model.parameters()).device
    model.eval()

    # Baseline prediction (full DAG)
    with torch.no_grad():
        baseline_result = model.predict(testX, n_samples=n_samples)

    baseline_preds = baseline_result['predictions'][-1].cpu().numpy()
    baseline_std = baseline_result['predictions_std'][-1].cpu().numpy()
    y_true = testY[-1].cpu().numpy()

    baseline_metrics = compute_all_metrics(
        y_true, baseline_preds, y_prev=y_prev, y_pred_std=baseline_std
    )

    # Determine edges to ablate
    if edges_to_ablate is None:
        tech_indices = _get_generation_feature_indices(feature_names)
        edges_to_ablate = [(idx, target_idx) for idx in tech_indices.values()]

    # Get DAG shape from model
    dags = model.get_causal_graphs()
    dag_shape = dags[0].shape  # [lag+1, n_nodes, n_nodes]

    ablation_results = []
    for src_idx, tgt_idx in edges_to_ablate:
        # Create mask: 1 everywhere except the ablated edge
        edge_mask = torch.ones(dag_shape, device=device)
        edge_mask[:, src_idx, tgt_idx] = 0.0  # Zero out all lags for this edge

        with torch.no_grad():
            ablated_result = model.predict(
                testX, n_samples=n_samples, edge_mask=edge_mask
            )

        ablated_preds = ablated_result['predictions'][-1].cpu().numpy()
        ablated_std = ablated_result['predictions_std'][-1].cpu().numpy()

        ablated_metrics = compute_all_metrics(
            y_true, ablated_preds, y_prev=y_prev, y_pred_std=ablated_std
        )

        # Compute deltas (positive = ablation made it worse)
        delta_rmse = ablated_metrics['rmse'] - baseline_metrics['rmse']
        delta_spearman = baseline_metrics['spearman'] - ablated_metrics['spearman']

        src_name = feature_names[src_idx] if src_idx < len(feature_names) else f'feature_{src_idx}'
        tgt_name = feature_names[tgt_idx] if tgt_idx < len(feature_names) else f'feature_{tgt_idx}'
        tech = _match_feature_to_technology(src_name)

        ablation_results.append({
            'source_idx': src_idx,
            'target_idx': tgt_idx,
            'source_name': src_name,
            'target_name': tgt_name,
            'technology': tech,
            'baseline_rmse': baseline_metrics['rmse'],
            'ablated_rmse': ablated_metrics['rmse'],
            'delta_rmse': delta_rmse,
            'delta_rmse_pct': 100 * delta_rmse / baseline_metrics['rmse'] if baseline_metrics['rmse'] > 0 else 0,
            'baseline_spearman': baseline_metrics['spearman'],
            'ablated_spearman': ablated_metrics['spearman'],
            'delta_spearman': delta_spearman,
            'ablated_metrics': ablated_metrics,
        })

    # Rank by impact (largest RMSE increase = most important edge)
    ablation_results.sort(key=lambda x: x['delta_rmse'], reverse=True)

    return {
        'baseline_metrics': baseline_metrics,
        'ablation_results': ablation_results,
        'n_edges_ablated': len(ablation_results),
    }


# =============================================================================
# 4. Regime-Conditional Causal Effects
# =============================================================================

def regime_conditional_effects(
    model,
    feature_names: List[str],
    target_idx: int = 0
) -> pd.DataFrame:
    """
    Extract and compare causal effect magnitudes across regimes.

    For each regime, extracts W weights from ICGNN and reports sign
    and magnitude for generation -> price edges.

    Args:
        model: Trained DS3MCausal model
        feature_names: Feature column names
        target_idx: Index of price variable

    Returns:
        DataFrame with columns: regime, technology, feature_name, w_weight, expected_sign
    """
    model.eval()
    tech_indices = _get_generation_feature_indices(feature_names)

    rows = []
    for d in range(model.d_dim):
        # Get W weights for this regime
        W = model.causal_emissions[d].icgnn.get_weighted_adjacency().detach().cpu().numpy()

        # Get DAG for edge presence check
        dags = model.get_causal_graphs()
        dag = dags[d]

        for tech, idx in tech_indices.items():
            w_inst = float(W[0, idx, target_idx])
            w_lag = float(W[1, idx, target_idx]) if W.shape[0] > 1 else 0.0
            edge_prob = float(dag[0, idx, target_idx])

            rows.append({
                'regime': d,
                'technology': tech,
                'feature_name': feature_names[idx],
                'feature_idx': idx,
                'w_instantaneous': w_inst,
                'w_lagged': w_lag,
                'w_combined': w_inst + w_lag,
                'edge_probability': edge_prob,
                'expected_sign': MERIT_ORDER_SIGN.get(tech, 0),
                'sign_correct': (
                    np.sign(w_inst) == MERIT_ORDER_SIGN[tech]
                    if tech in MERIT_ORDER_SIGN and MERIT_ORDER_SIGN[tech] != 0
                    else None
                ),
            })

    return pd.DataFrame(rows)


def cross_country_causal_table(
    models_by_country: Dict[str, object],
    feature_names_by_country: Dict[str, List[str]],
    target_idx: int = 0,
    stability_data: Optional[Dict[str, Dict]] = None
) -> pd.DataFrame:
    """
    Generate cross-country comparison of regime-conditional causal effects.

    Creates the core results table for the paper:
    Country x Regime x {Wind, Solar, Gas, Nuclear} -> Price weights.

    Args:
        models_by_country: Dict mapping country code to trained model
        feature_names_by_country: Dict mapping country code to feature names
        target_idx: Index of price variable
        stability_data: Optional edge stability data for significance indicators

    Returns:
        DataFrame ready for LaTeX table generation
    """
    key_technologies = ['Wind Onshore', 'Solar', 'Fossil Gas', 'Nuclear']

    rows = []
    for country, model in models_by_country.items():
        feature_names = feature_names_by_country[country]
        effects_df = regime_conditional_effects(model, feature_names, target_idx)

        for regime in effects_df['regime'].unique():
            row = {'country': country, 'regime': int(regime)}
            regime_df = effects_df[effects_df['regime'] == regime]

            for tech in key_technologies:
                tech_row = regime_df[regime_df['technology'] == tech]
                if len(tech_row) > 0:
                    w = tech_row.iloc[0]['w_combined']
                    edge_p = tech_row.iloc[0]['edge_probability']

                    # Add significance marker from stability data
                    sig = ''
                    if stability_data and country in stability_data:
                        freq = stability_data[country].get('edge_frequency')
                        if freq is not None:
                            r_idx = int(regime)
                            f_idx = tech_row.iloc[0]['feature_idx']
                            if r_idx < freq.shape[0]:
                                seed_freq = freq[r_idx, 0, f_idx, target_idx]
                                if seed_freq >= 0.8:
                                    sig = '***'
                                elif seed_freq >= 0.6:
                                    sig = '**'
                                elif seed_freq >= 0.4:
                                    sig = '*'

                    col_name = tech.replace(' ', '_').lower()
                    row[f'{col_name}_w'] = round(w, 4)
                    row[f'{col_name}_p'] = round(edge_p, 3)
                    row[f'{col_name}_sig'] = sig
                else:
                    col_name = tech.replace(' ', '_').lower()
                    row[f'{col_name}_w'] = None
                    row[f'{col_name}_p'] = None
                    row[f'{col_name}_sig'] = ''

            rows.append(row)

    return pd.DataFrame(rows)


# =============================================================================
# Convenience: Run All Validation
# =============================================================================

def run_full_causal_validation(
    model,
    testX: torch.Tensor,
    testY: torch.Tensor,
    feature_names: List[str],
    target_idx: int = 0,
    checkpoint_paths: Optional[List[Path]] = None,
    model_class=None,
    model_kwargs: Optional[dict] = None,
    n_samples: int = 50,
) -> Dict:
    """
    Run all causal validation analyses on a trained model.

    Args:
        model: Trained DS3MCausal model
        testX, testY: Test data tensors
        feature_names: Feature column names
        target_idx: Price variable index
        checkpoint_paths: Paths for edge stability (optional)
        model_class: Model class for edge stability (optional)
        model_kwargs: Model kwargs for edge stability (optional)
        n_samples: MC samples for ablation

    Returns:
        Dict with all validation results
    """
    results = {}

    # Merit order alignment (per regime)
    print("Computing merit order alignment...")
    dags = model.get_causal_graphs()
    merit_results = {}
    for d in range(model.d_dim):
        W = model.causal_emissions[d].icgnn.get_weighted_adjacency().detach().cpu().numpy()
        merit_results[f'regime_{d}'] = merit_order_alignment(
            W, feature_names, target_idx, dag=dags[d]
        )
    results['merit_order'] = merit_results

    # Edge ablation
    print("Running edge ablation analysis...")
    y_prev = testY[-2].cpu().numpy() if testY.shape[0] >= 2 else None
    results['ablation'] = edge_ablation_analysis(
        model, testX, testY, feature_names, target_idx,
        n_samples=n_samples, y_prev=y_prev
    )

    # Regime-conditional effects
    print("Extracting regime-conditional effects...")
    results['regime_effects'] = regime_conditional_effects(
        model, feature_names, target_idx
    ).to_dict('records')

    # Edge stability (if checkpoints provided)
    if checkpoint_paths and model_class and model_kwargs:
        print("Computing edge stability across seeds...")
        results['stability'] = compute_edge_stability(
            checkpoint_paths, model_class, model_kwargs
        )
        # Convert numpy arrays to lists for JSON serialization
        if 'edge_frequency' in results['stability']:
            results['stability']['edge_frequency'] = results['stability']['edge_frequency'].tolist()

    print("Causal validation complete.")
    return results
