"""Quick smoke test for CaRS training pipeline with extended data."""
import torch
from shared_backbone.data_loader import prepare_unified_ds3m_data
from shared_backbone.models.ds3m_causal import DS3MCausal
from shared_backbone.training.train_e2e import AugmentedLagrangianTrainer

print("Loading data...")
data = prepare_unified_ds3m_data('DE', timestep=14)
print(f"Data loaded: train={data['trainX'].shape}")

# Subsample for speed
trainX = data['trainX'][:, :1000, :]
trainY = data['trainY'][:, :1000, :]
valX = data['valX'][:, :200, :]
valY = data['valY'][:, :200, :]

n_features = trainX.shape[2]
print(f"Features: {n_features}")

device = torch.device('cpu')
model = DS3MCausal(
    x_dim=n_features,
    y_dim=1,
    d_dim=2,
    h_dim=32,
    z_dim=16,
    device=device,
    n_layers=1,
    num_nodes=n_features,
)
print(f"Model: {sum(p.numel() for p in model.parameters())} params")

optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

trainer = AugmentedLagrangianTrainer(
    model=model,
    optimizer=optimizer,
    device=device,
    max_auglag_steps=2,
    max_inner_epochs=3,
)

data_sub = {
    'trainX': trainX, 'trainY': trainY,
    'valX': valX, 'valY': valY,
    'testX': valX, 'testY': valY,
    'n_train': 1000, 'n_val': 200, 'n_test': 200,
    'feature_cols': data['feature_cols'],
    'Y_moments': data['Y_moments'], 'X_moments': data['X_moments'],
}

try:
    history = trainer.train(trainX, trainY, testX=valX, testY=valY)
    print("Training complete!")
    print(f"History keys: {list(history.keys())}")
    for k, v in history.items():
        if isinstance(v, list) and len(v) > 0:
            print(f"  {k}: last={v[-1]:.6f}" if isinstance(v[-1], float) else f"  {k}: {len(v)} entries")
except Exception as e:
    import traceback
    traceback.print_exc()
