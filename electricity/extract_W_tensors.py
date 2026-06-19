r"""Extract the small per-regime structural weight tensors that the
spillover figures need out of the large PyTorch checkpoints, so the
figures can be reproduced without shipping the multi-GB ``final.tar``
files.

For every daily CARGO seed directory
(``<results_dir>/<market>/h1/seed<N>/``) that has a ``checkpoints/final.tar``
and a ``config.json``, this writes a tiny ``W_tensors.npz`` next to the
config containing, for each regime index present in the checkpoint:

    W_<r>            : the structural weight tensor
                       ``model_state_dict['causal_emissions.<r>.icgnn.W']``
                       (shape ``(2, F, F)``; the figures use ``W[0]``).
    alpha_logit_<r>  : the scalar
                       ``causal_emissions.<r>.icgnn.physical_prior_alpha_logit``
                       used for the effective (soft-prior-gated) weights,
                       or NaN when the checkpoint was trained with the
                       soft prior off.

The companion small files the figures also read -- ``config.json`` (holds
``data.feature_cols``), ``regime_assignments.npy`` and ``actuals.npy``
(price-canonicalisation) -- already live in the same seed directory and
are committed as-is; only the heavy ``final.tar`` is replaced by this
extract.

``visualize_european_network._load_W_from_experiment`` and
``visualize_spillover_three_panel._load_trained_alpha`` transparently fall
back to ``W_tensors.npz`` when ``final.tar`` is absent, so the spillover
figures reproduce byte-for-byte from the extract alone.

Usage:
    python3 electricity/extract_W_tensors.py
    python3 electricity/extract_W_tensors.py --results_dir <dir> --regimes 0 1
"""
import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from visualize_spillover_three_panel import CLOCKWISE_ORDER


def _seed_dirs(results_dir, market):
    h1 = results_dir / market / "h1"
    if not h1.is_dir():
        return []
    out = []
    for d in sorted(h1.iterdir()):
        if not d.is_dir() or not d.name.startswith("seed"):
            continue
        if (d / "checkpoints" / "final.tar").exists() and (d / "config.json").exists():
            out.append(d)
    return out


def _regimes_in(sd):
    idxs = set()
    for k in sd:
        m = re.match(r"causal_emissions\.(\d+)\.icgnn\.W$", k)
        if m:
            idxs.add(int(m.group(1)))
    return sorted(idxs)


def extract_one(exp_dir, regimes, verify=True):
    """Write W_tensors.npz for one seed dir. Returns the output path."""
    import torch

    sd = torch.load(exp_dir / "checkpoints" / "final.tar",
                    map_location="cpu", weights_only=False)["model_state_dict"]
    present = _regimes_in(sd)
    want = [r for r in (regimes if regimes is not None else present) if r in present]

    arrays = {}
    for r in want:
        W = sd[f"causal_emissions.{r}.icgnn.W"].numpy()
        arrays[f"W_{r}"] = W
        ak = f"causal_emissions.{r}.icgnn.physical_prior_alpha_logit"
        arrays[f"alpha_logit_{r}"] = (
            np.float64(sd[ak].item()) if ak in sd else np.float64(np.nan))

    out_path = exp_dir / "W_tensors.npz"
    np.savez_compressed(out_path, **arrays)

    if verify:
        # Round-trip check: the W[0] the figure would read from final.tar
        # must equal the W[0] it now reads from the extract.
        chk = np.load(out_path)
        for r in want:
            ref = sd[f"causal_emissions.{r}.icgnn.W"][0].numpy()
            assert np.array_equal(chk[f"W_{r}"][0], ref), \
                f"W_{r}[0] mismatch in {exp_dir}"
    return out_path, want


def main():
    repo = Path(__file__).resolve().parent.parent
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--results_dir", type=Path,
                   default=repo / "outputs" / "experiments"
                           / "12market_cargo_controls_daily")
    p.add_argument("--markets", nargs="*", default=CLOCKWISE_ORDER)
    p.add_argument("--regimes", nargs="*", type=int, default=[0, 1],
                   help="Regime indices to extract (default 0 1). Pass"
                        " nothing after the flag to extract all present.")
    args = p.parse_args()

    regimes = args.regimes if args.regimes else None
    n_dirs = 0
    for market in args.markets:
        seed_dirs = _seed_dirs(args.results_dir, market)
        if not seed_dirs:
            print(f"  {market}: no seed dirs with final.tar+config under"
                  f" {args.results_dir / market / 'h1'}")
            continue
        for d in seed_dirs:
            out_path, want = extract_one(d, regimes)
            n_dirs += 1
            size_kb = out_path.stat().st_size / 1024
            print(f"  {market}/{d.name}: regimes {want} -> "
                  f"{out_path.relative_to(repo)} ({size_kb:.0f} KB)")
    print(f"\nDone: wrote W_tensors.npz for {n_dirs} seed dirs.")


if __name__ == "__main__":
    main()
