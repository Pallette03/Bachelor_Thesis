import math
import torch
from botorch.models import SingleTaskGP
from botorch.models.transforms import Normalize, Standardize
from botorch.fit import fit_gpytorch_mll
from botorch.acquisition import ExpectedImprovement
from botorch.acquisition.analytic import LogExpectedImprovement
from botorch.optim import optimize_acqf
from gpytorch.mlls import ExactMarginalLogLikelihood
from KeypointDetector import main

torch.set_default_dtype(torch.float64)

def black_box_evaluate(params):
    b, lr, r = params
    main_params = {
        "model": "UNet",
        "dataset": "with_clutter",
        "batch_size": int(b.item()),
        "val_batch_size": int(b.item()),
        "learning_rate": float(lr.item()),
        "global_image_size": (int(r.item()), int(r.item())),
        "num_epochs": 8,
        "num_channels": 3,
        "gaussian_blur": True,
        "start_from_checkpoint": False,
        "bayes_optimization": True,
        "post_processing_threshold": 0.4,
        "distance_threshold": 5,
        "feature_extractor_lvl_amount": 8,
        "hourglass_stacks": 4
    }
    print(f"Evaluating with params: {main_params}")
    model, f1_score = main(main_params)
    
    return torch.tensor(f1_score)

bounds = torch.tensor([
    [8.0, 1e-6, 300.0],   # lower bounds: batch size, lr, resolution
    [32.0, 1e-1, 1000.0]   # upper bounds
])
# Number of initial points
n_init = 6

batch_lo, lr_lo, res_lo = bounds[0]
batch_hi, lr_hi, res_hi = bounds[1]

r = torch.rand(n_init, 3)

x_batch = r[:, 0] * (batch_hi - batch_lo) + batch_lo

log_lr_lo = math.log10(lr_lo)
log_lr_hi = math.log10(lr_hi)
x_lr = 10 ** ( r[:, 1] * (log_lr_hi - log_lr_lo) + log_lr_lo )

x_res = r[:, 2] * (res_hi - res_lo) + res_lo

train_X = torch.stack([x_batch, x_lr, x_res], dim=-1)
train_Y = torch.stack([black_box_evaluate(x) for x in train_X])

for iteration in range(10):
    gp = SingleTaskGP(
        train_X=train_X,
        train_Y=train_Y,
        input_transform=Normalize(d=train_X.shape[-1]),
        outcome_transform=Standardize(m=train_Y.shape[-1]),
    )
    mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
    fit_gpytorch_mll(mll)  # optimize GP hyperparameters :contentReference[oaicite:6]{index=6}

    acq_func = LogExpectedImprovement(model=gp, best_f=train_Y.max())

    candidate, _ = optimize_acqf(
        acq_function=acq_func,
        bounds=bounds,
        q=1,
        num_restarts=10,
        raw_samples=50,
    )

    new_y = black_box_evaluate(candidate.squeeze(0))
    train_X = torch.cat([train_X, candidate], dim=0)
    train_Y = torch.cat([train_Y, new_y.unsqueeze(-1)], dim=0)

best_idx = train_Y.argmax()
best_params = train_X[best_idx]
print(f"Best batch size: {best_params[0]:.0f}, "
      f"lr: {best_params[1]:.5f}, "
      f"resolution: {best_params[2]:.0f}, "
      f"F1: {train_Y[best_idx].item():.4f}")
