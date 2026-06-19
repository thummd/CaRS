"""Distil a trained \\CaRS{} model into a small student forecaster.

The teacher (\\CaRS) provides interpretable regime-conditional causal graphs
but is expensive at inference (encoder + DAG sampling + per-regime emissions).
For real-time deployment we train a small student MLP/LSTM head that mimics
the teacher's point predictions, then deploy the student for hot-path
inference while keeping the teacher for offline analysis.

Usage:
    python3 shared_backbone/distill_student.py \\
        --market SE \\
        --teacher_dir outputs/experiments/12market_cam/SE/h1/seed42 \\
        --output_dir outputs/distilled_students/SE \\
        --student_type mlp --hidden_dim 64

The student matches the teacher's per-window prediction on the train+val
windows; we evaluate on the held-out test split.
"""
import argparse
import json
import time
from pathlib import Path
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from shared_backbone.data_loader import prepare_unified_ds3m_data
from shared_backbone.models.ds3m_causal import DS3MCausal


class StudentMLP(nn.Module):
    """Flat MLP student: takes the 14*p flattened input window and outputs
    a single one-hour-ahead return prediction.
    """

    def __init__(self, timestep: int, x_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(timestep * x_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, p] (batch-first; the caller must permute if coming from
        # DS3M's [T, B, p] convention).
        if x.dim() != 3:
            raise ValueError(f"StudentMLP expects 3D batch-first input [B, T, p], got {tuple(x.shape)}")
        b = x.shape[0]
        return self.net(x.reshape(b, -1)).squeeze(-1)


class StudentLSTM(nn.Module):
    """Two-layer LSTM student matching CaRS's GRU encoder scale."""

    def __init__(self, x_dim: int, hidden_dim: int = 32):
        super().__init__()
        self.lstm = nn.LSTM(x_dim, hidden_dim, num_layers=2,
                            batch_first=True, dropout=0.1)
        self.head = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 3:
            raise ValueError(f"StudentLSTM expects 3D batch-first input [B, T, p], got {tuple(x.shape)}")
        out, _ = self.lstm(x)
        return self.head(out[:, -1, :]).squeeze(-1)


def load_teacher(teacher_dir: Path, x_dim: int, device: torch.device) -> DS3MCausal:
    cfg = json.load(open(teacher_dir / "config.json"))
    args = cfg["args"]
    model = DS3MCausal(
        x_dim=x_dim, y_dim=1,
        h_dim=args["h_dim"], z_dim=args["z_dim"], d_dim=args["d_dim"],
        device=device, num_nodes=x_dim, lag=args["lag"],
        sharing_mode=args["sharing_mode"],
        lambda_dag=args["lambda_dag"], lambda_sparse=args["lambda_sparse"],
        lambda_var_reg=args["lambda_var_reg"],
        use_attention=not args.get("no_attention", False),
        w_init_scale=args["w_init_scale"],
        aggregation_mode=args.get("aggregation_mode", "linear"),
        cam_hidden_dim=args.get("cam_hidden_dim", 32),
        elastic_threshold=args.get("elastic_threshold", 0.0),
        elastic_weight=args.get("elastic_weight", 0.0),
    ).to(device)
    ckpt = torch.load(teacher_dir / "checkpoints" / "final.tar",
                      map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model


def teacher_predict_all(model: DS3MCausal, X: torch.Tensor, batch_size: int = 1024) -> np.ndarray:
    """Run teacher on all windows in batches, return per-window predictions
    aligned with the last timestep of each window.
    """
    n = X.shape[1]
    preds = []
    with torch.no_grad():
        for s in range(0, n, batch_size):
            chunk = X[:, s:s + batch_size, :]
            out = model.predict(chunk, n_samples=1)
            preds.append(out["predictions"][-1].cpu().numpy())
    return np.concatenate(preds, axis=0).squeeze(-1)


def distill(
    market: str,
    teacher_dir: Path,
    output_dir: Path,
    student_type: str = "mlp",
    hidden_dim: int = 64,
    learning_rate: float = 1e-3,
    max_epochs: int = 100,
    batch_size: int = 256,
    patience: int = 10,
    device: str = None,
) -> dict:
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    output_dir.mkdir(parents=True, exist_ok=True)

    cfg = json.load(open(teacher_dir / "config.json"))
    args = cfg["args"]
    feature_groups = args.get("feature_groups",
                              "price,load,weather,calendar,gen_forecast,demand_forecast")
    if isinstance(feature_groups, str):
        feature_groups = feature_groups.split(",")

    data = prepare_unified_ds3m_data(
        country=market, timestep=args["timestep"],
        feature_groups=feature_groups, target_col=args.get("target_col"),
        task_type=args["task_type"], resampled=args.get("resampled", False),
        spillover=args.get("spillover", False), horizon=args["horizon"],
    )
    x_dim = data["trainX"].shape[-1]
    teacher = load_teacher(teacher_dir, x_dim, device)

    print(f"[{market}] x_dim={x_dim}, timestep={args['timestep']}")
    print(f"  Train windows: {data['trainX'].shape[1]:>7}  "
          f"Val: {data['valX'].shape[1]:>7}  "
          f"Test: {data['testX'].shape[1]:>7}")

    # Generate teacher targets (point predictions) on train+val
    print("  Generating teacher predictions...")
    t0 = time.time()
    teacher_train = teacher_predict_all(teacher, data["trainX"].to(device), batch_size=1024)
    teacher_val = teacher_predict_all(teacher, data["valX"].to(device), batch_size=1024)
    print(f"  Teacher inference: {time.time() - t0:.1f}s")

    # Convert to student-friendly shape [B, T, p]
    trainX = data["trainX"].permute(1, 0, 2).to(device)
    valX = data["valX"].permute(1, 0, 2).to(device)
    testX = data["testX"].permute(1, 0, 2).to(device)
    trainY = torch.from_numpy(teacher_train.astype(np.float32)).to(device)
    valY = torch.from_numpy(teacher_val.astype(np.float32)).to(device)

    # Build student
    if student_type == "mlp":
        student = StudentMLP(args["timestep"], x_dim, hidden_dim).to(device)
    elif student_type == "lstm":
        student = StudentLSTM(x_dim, hidden_dim).to(device)
    else:
        raise ValueError(f"Unknown student_type {student_type!r}")

    n_params = sum(p.numel() for p in student.parameters())
    teacher_n = sum(p.numel() for p in teacher.parameters() if p.requires_grad)
    print(f"  Student ({student_type}): {n_params:,} params  "
          f"({n_params / teacher_n * 100:.1f}% of teacher's {teacher_n:,})")

    # Train student
    opt = torch.optim.Adam(student.parameters(), lr=learning_rate)
    loss_fn = nn.MSELoss()
    best_val = float("inf")
    best_state = None
    bad = 0
    n = trainX.shape[0]
    print(f"  Training student for up to {max_epochs} epochs...")
    t0 = time.time()
    for ep in range(max_epochs):
        student.train()
        perm = torch.randperm(n, device=device)
        ep_loss = 0.0
        nb = 0
        for s in range(0, n, batch_size):
            idx = perm[s:s + batch_size]
            opt.zero_grad()
            pred = student(trainX[idx])
            loss = loss_fn(pred, trainY[idx])
            loss.backward()
            opt.step()
            ep_loss += float(loss); nb += 1
        student.eval()
        with torch.no_grad():
            val_pred = student(valX)
            val_loss = float(loss_fn(val_pred, valY))
        if val_loss < best_val - 1e-6:
            best_val = val_loss; bad = 0
            best_state = {k: v.detach().cpu().clone() for k, v in student.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                print(f"    early-stop at epoch {ep}  best val MSE={best_val:.5f}")
                break

    if best_state is not None:
        student.load_state_dict(best_state)
    student.eval()
    train_time = time.time() - t0
    print(f"  Training: {train_time:.1f}s")

    # Latency comparison on test
    test_actuals = data["testY"][-1, :, 0].cpu().numpy()
    teacher_test = teacher_predict_all(teacher, data["testX"].to(device), batch_size=1024)

    # Time teacher single-window inference (B=1) — deployment-relevant
    one_window = data["testX"][:, :1, :].to(device)
    torch.cuda.synchronize() if device.type == "cuda" else None
    t0 = time.time()
    for _ in range(200):
        with torch.no_grad():
            _ = teacher.predict(one_window, n_samples=1)
    if device.type == "cuda":
        torch.cuda.synchronize()
    teacher_lat = (time.time() - t0) / 200 * 1000

    # Time student single-window inference (B=1)
    one_window_s = trainX[:1]
    if device.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(1000):
        with torch.no_grad():
            _ = student(one_window_s)
    if device.type == "cuda":
        torch.cuda.synchronize()
    student_lat = (time.time() - t0) / 1000 * 1000

    # Test-set agreement
    with torch.no_grad():
        student_test = student(testX).cpu().numpy()
    rmse_st = float(np.sqrt(np.mean((student_test - test_actuals) ** 2)))
    rmse_te = float(np.sqrt(np.mean((teacher_test - test_actuals) ** 2)))
    rmse_agreement = float(np.sqrt(np.mean((student_test - teacher_test) ** 2)))
    da_st = float(np.mean(np.sign(student_test) == np.sign(test_actuals)))
    da_te = float(np.mean(np.sign(teacher_test) == np.sign(test_actuals)))

    metrics = {
        "market": market, "student_type": student_type,
        "n_params_student": int(n_params),
        "n_params_teacher": int(teacher_n),
        "param_ratio": float(n_params / teacher_n),
        "training_time_s": float(train_time),
        "test_rmse_teacher": rmse_te,
        "test_rmse_student": rmse_st,
        "test_rmse_agreement": rmse_agreement,
        "test_diracc_teacher": da_te,
        "test_diracc_student": da_st,
        "latency_ms_teacher_b1": float(teacher_lat),
        "latency_ms_student_b1": float(student_lat),
        "latency_speedup": float(teacher_lat / student_lat),
    }

    print()
    print(f"  Teacher  test RMSE: {rmse_te:.4f}   DirAcc: {da_te:.4f}   latency(B=1): {teacher_lat:.2f} ms")
    print(f"  Student  test RMSE: {rmse_st:.4f}   DirAcc: {da_st:.4f}   latency(B=1): {student_lat:.2f} ms")
    print(f"  Agreement RMSE: {rmse_agreement:.4f}   Speedup: {teacher_lat / student_lat:.1f}x")

    torch.save(student.state_dict(), output_dir / "student.pt")
    with open(output_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    return metrics


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--market", required=True)
    p.add_argument("--teacher_dir", required=True, type=Path)
    p.add_argument("--output_dir", required=True, type=Path)
    p.add_argument("--student_type", default="mlp", choices=["mlp", "lstm"])
    p.add_argument("--hidden_dim", type=int, default=64)
    p.add_argument("--max_epochs", type=int, default=100)
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--patience", type=int, default=10)
    args = p.parse_args()
    distill(args.market, args.teacher_dir, args.output_dir,
            student_type=args.student_type, hidden_dim=args.hidden_dim,
            max_epochs=args.max_epochs, batch_size=args.batch_size,
            patience=args.patience)


if __name__ == "__main__":
    main()
