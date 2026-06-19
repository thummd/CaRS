"""Physical-interconnect prior for the CARGO emission mechanism.

CARGO (Causal Additive Regime-Gated Output) is the causal emission
mechanism in CaRS — the renamed and improved successor of the original
ICGNN (Input Convex Graph Neural Network) emission. With cross-border
spillover features it can load arbitrary directed weights between any
pair of markets, including pairs with no electrical interconnection
(e.g. FR -> PL through DE and CZ). The post-hoc diagnostics in
electricity/analyze_physical_enrichment.py and
electricity/analyze_confounder_control.py show this freedom is largely
spent on common-driver confounding rather than directed flow.

This prior tightens CARGO by masking the cross-border rows of the
emission weight matrix W so only edges between markets with a real
physical interconnect can carry non-zero weight. The mask is applied
inside `CausalICGNN.get_weighted_adjacency()` (see causal_emission.py)
in one of two modes:

  hard - forbidden cross-border weights are exactly zero; the data
         cannot override the prior. Cleanest interpretation for a paper
         claim ("the model only sees physical edges"), at the cost of
         vanishing gradient on forbidden edges.

  soft - forbidden weights are multiplied by sigmoid(alpha) where alpha
         is a single learnable scalar shared across all forbidden edges.
         Initialised so the prior is strong (~5% pass-through). If the
         data really supports a non-physical edge, alpha will rise; if
         not, it stays small and the prior holds. Strictly preferred for
         exploratory work because it lets the model tell you when the
         prior is wrong.

This module provides `build_physical_mask` for use at model-construction
time, and `INTERCONNECTIONS` listing the 19 undirected European market
pairs used in the existing visualizations.
"""
from typing import Iterable, List, Optional, Set, Tuple

import torch


INTERCONNECTIONS: List[Tuple[str, str]] = [
    ("DE", "FR"), ("DE", "NL"), ("DE", "BE"), ("DE", "AT"),
    ("DE", "CZ"), ("DE", "PL"), ("DE", "DK"), ("DE", "SE"),
    ("FR", "BE"), ("FR", "ES"), ("FR", "IT"),
    ("NL", "BE"), ("NL", "DK"),
    ("AT", "CZ"), ("AT", "HU"), ("AT", "IT"),
    ("CZ", "PL"), ("PL", "SE"), ("DK", "SE"),
]

DEFAULT_MARKETS: Set[str] = {
    "AT", "BE", "CZ", "DE", "DK", "ES", "FR", "HU", "IT", "NL", "PL", "SE"
}


def _spillover_source(feature_name: str,
                       markets: Iterable[str]) -> Optional[str]:
    """If `feature_name` is a cross-border lag feature like
    `FR_price_lag1h` or `DE_flow_lag24h`, return the source country
    code; otherwise return None.
    """
    for c in markets:
        if (feature_name.startswith(f"{c}_price_lag")
                or feature_name.startswith(f"{c}_flow_lag")):
            return c
    return None


def build_physical_mask(
    feature_cols: List[str],
    target_country: str,
    lag: int = 1,
    markets: Optional[Iterable[str]] = None,
    interconnections: Optional[Iterable[Tuple[str, str]]] = None,
    apply_to_lag_zero: bool = True,
) -> torch.Tensor:
    """Build a [lag+1, num_features, num_features] binary mask for CARGO.

    Each feature in `feature_cols` is classified as either:
      - Domestic (default): mask row = 1 everywhere; the feature can
        carry signal to any target column.
      - Cross-border spillover from source country S: row = 1 iff
        (S, target_country) is in the physical-interconnect set.

    Note we mask entire rows (parent -> any child), not just
    parent -> price column 0. That guarantees a non-physical spillover
    feature is dead — it can neither drive the price nor any other
    domestic feature.

    Args:
        feature_cols: the model's `feature_cols` (e.g. from config.json).
        target_country: 2-letter code for the model's own market.
        lag: temporal lag used by the ICGNN. The returned mask has
            `lag + 1` leading slices.
        markets: candidate source markets; defaults to the 12-market set.
        interconnections: list of (a, b) tuples (order-insensitive);
            defaults to the 19 European interconnects.
        apply_to_lag_zero: whether to also mask the instantaneous slice.
            CaRS-GAT spillover features only live in the lag>=1 slice in
            practice, but masking lag-0 is safe and makes the prior
            symmetric.

    Returns:
        torch.Tensor of shape (lag+1, n_features, n_features) with
        dtype torch.float32 — 1.0 for allowed edges, 0.0 for forbidden.
    """
    markets = set(markets) if markets is not None else DEFAULT_MARKETS
    if interconnections is None:
        interconnections = INTERCONNECTIONS
    physical_pairs = {tuple(sorted(p)) for p in interconnections}
    n = len(feature_cols)

    # Per-feature: is this a cross-border spillover, and from which source?
    allowed_row = torch.ones(n, dtype=torch.float32)
    for i, feat in enumerate(feature_cols):
        src = _spillover_source(feat, markets)
        if src is None or src == target_country:
            continue
        if tuple(sorted((src, target_country))) not in physical_pairs:
            allowed_row[i] = 0.0

    # Broadcast a row-mask: mask[lag_idx, i, j] = allowed_row[i] for any j.
    row_mask_2d = allowed_row.unsqueeze(1).expand(n, n).contiguous()
    mask = torch.ones(lag + 1, n, n, dtype=torch.float32)
    for lag_idx in range(lag + 1):
        if lag_idx == 0 and not apply_to_lag_zero:
            continue
        mask[lag_idx] = row_mask_2d
    return mask


def summarize_mask(mask: torch.Tensor, feature_cols: List[str]) -> dict:
    """Return diagnostics on a built mask: number of forbidden rows,
    which features they correspond to, and the fraction of edges blocked.
    """
    # Mask is symmetric over lag and uses entire rows, so one slice is enough
    row_mask = mask[-1].sum(dim=1)  # n; > 0 iff row allowed
    forbidden = [feature_cols[i] for i, v in enumerate(row_mask) if v == 0]
    total_edges = mask.numel()
    forbidden_edges = int((mask == 0).sum().item())
    return {
        "n_features": len(feature_cols),
        "n_forbidden_features": len(forbidden),
        "forbidden_features": forbidden,
        "fraction_blocked_edges": forbidden_edges / total_edges,
    }
