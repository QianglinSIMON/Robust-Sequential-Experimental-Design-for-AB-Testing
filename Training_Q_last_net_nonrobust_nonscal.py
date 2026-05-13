import warnings
warnings.filterwarnings("ignore", message=".*__array_wrap__.*", category=DeprecationWarning)

import os
import argparse
import logging
from datetime import datetime

# -------- server-safe matplotlib backend --------
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import random

# -------- reuse your utils_self.py --------
from utils_self import (
    set_seed,
    MinMaxScalerX,
    create_model,
    eval_metrics_log_space,
    baseline_metrics
)

# Limit BLAS/OpenMP threads (optional but good on servers)
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")


def _ts():
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def build_logger(log_dir, n_exp):
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f"train_robust_nexp{n_exp}_{_ts()}.log")

    logger = logging.getLogger("train_logger")
    logger.setLevel(logging.INFO)

    # avoid duplicated handlers
    if logger.handlers:
        logger.handlers.clear()

    fmt = logging.Formatter(
        fmt="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(fmt)

    fh = logging.FileHandler(log_file, encoding="utf-8")
    fh.setLevel(logging.INFO)
    fh.setFormatter(fmt)

    logger.addHandler(ch)
    logger.addHandler(fh)

    logger.info(f"Logger initialized. Log file: {log_file}")
    return logger, log_file


@torch.no_grad()
def eval_metrics_both_spaces(model, X_t, y_t_log, device):
    """
    Thin wrapper:
    - use utils_self.eval_metrics_log_space for log-space metrics
    - add orig-space metrics here (exp transform)
    """
    log_m = eval_metrics_log_space(model, X_t, y_t_log, device=device)

    # collect preds/targets in log space
    model.eval()
    preds_log, ys_log = [], []
    loader = DataLoader(TensorDataset(X_t, y_t_log), batch_size=2048, shuffle=False)
    for xb, yb in loader:
        xb = xb.to(device)
        yb = yb.to(device)
        preds_log.append(model(xb).detach().cpu())
        ys_log.append(yb.detach().cpu())

    p_log = torch.cat(preds_log, dim=0)
    y_log = torch.cat(ys_log, dim=0)

    p_orig = torch.exp(p_log)
    y_orig = torch.exp(y_log)

    mse_orig = torch.mean((p_orig - y_orig) ** 2).item()
    mae_orig = torch.mean(torch.abs(p_orig - y_orig)).item()
    var_orig = torch.var(y_orig, unbiased=False).item()
    r2_orig = 1.0 - (mse_orig / (var_orig + 1e-12))
    nrmse_orig = float(np.sqrt(mse_orig / (var_orig + 1e-12)))

    out = dict(**log_m)
    out.update(dict(
        mse_orig=mse_orig, mae_orig=mae_orig, r2_orig=r2_orig, nrmse_orig=nrmse_orig
    ))
    return out


def baseline_metrics_both_spaces(y_train_log, y_test_log):
    """
    Minimal baseline duplication:
    - log baseline computed here
    - orig baseline reused from utils_self.baseline_metrics
    """
    # log space baseline
    log_mu = float(np.mean(y_train_log))
    mse_log = float(np.mean((y_test_log - log_mu) ** 2))
    mae_log = float(np.mean(np.abs(y_test_log - log_mu)))
    var_log = float(np.var(y_test_log))
    r2_log = 1.0 - (mse_log / (var_log + 1e-12))
    nrmse_log = float(np.sqrt(mse_log / (var_log + 1e-12)))

    # orig baseline via utils function
    y_train_orig = np.exp(y_train_log)
    y_test_orig = np.exp(y_test_log)
    base_orig = baseline_metrics(y_train_orig, y_test_orig)

    out = dict(
        mse_log=mse_log, mae_log=mae_log, r2_log=r2_log, nrmse_log=nrmse_log,
        mse_orig=base_orig["mse"], mae_orig=base_orig["mae"],
        r2_orig=base_orig["r2"], nrmse_orig=base_orig["nrmse"]
    )
    return out


def main():
    parser = argparse.ArgumentParser()

    # core experiment
    parser.add_argument("--n_exp", type=int, default=20,
                        help="Real horizon is n_exp+1")
    parser.add_argument("--type_obj", type=str, default="obj",
                        choices=["obj", "var"])
    parser.add_argument("--data_path", type=str, default=None,
                        help="Default: Qval_dataset_{n_exp}_{type_obj}.txt")

    # training hyperparams
    parser.add_argument("--seed", type=int, default=2025)
    parser.add_argument("--test_ratio", type=float, default=0.2)
    parser.add_argument("--batch_size", type=int, default=2048)
    parser.add_argument("--epochs", type=int, default=2000)
    parser.add_argument("--patience", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--standardize_X", type=int, default=1,
                        help="1=True, 0=False")

    # output & logs
    parser.add_argument("--out_dir", type=str, default="plots_output_obj")
    parser.add_argument("--out_prefix", type=str, default="best_model_Q")
    parser.add_argument("--log_dir", type=str, default="logs")

    # GPU control
    parser.add_argument("--gpu", type=int, default=None,
                        help="Set CUDA_VISIBLE_DEVICES to this id")

    # keep plotting functionality but server-friendly default
    parser.add_argument("--show_plots", action="store_true",
                        help="If set, plt.show() will be called")

    args = parser.parse_args()

    # GPU selection (non-intrusive)
    if args.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    logger, log_file = build_logger(args.log_dir, args.n_exp)
    logger.info("device: %s", device)
    logger.info("args: %s", vars(args))

    # seed
    set_seed(args.seed)

    # dataset path
    if args.data_path is None:
        args.data_path = f"Qval_dataset_{args.n_exp}_{args.type_obj}.txt"

    if not os.path.exists(args.data_path):
        raise FileNotFoundError(f"Dataset not found: {args.data_path}")

    df = pd.read_csv(args.data_path, sep="\t", compression="infer", encoding="utf-8")
    assert "Out_Q_res" in df.columns

    # shuffle + clean
    df_shuf = df.sample(frac=1.0, random_state=args.seed).reset_index(drop=True)
    df_shuf = df_shuf.replace([np.inf, -np.inf], np.nan).dropna(axis=0)

    # safety for log
    bad = df_shuf["Out_Q_res"] <= 0
    if bad.any():
        logger.warning(f"Found {int(bad.sum())} rows with Out_Q_res <= 0. Dropped for log safety.")
        df_shuf = df_shuf.loc[~bad].copy()

    # features/labels
    y = df_shuf["Out_Q_res"].astype(np.float64).values.reshape(-1, 1)
    log_y = np.log(y)
    X = df_shuf.drop(columns=["Out_Q_res"]).astype(np.float64).values

    n_samples, n_features = X.shape
    logger.info(f"Samples: {n_samples}, Features: {n_features}")

    # split
    n_test = int(round(n_samples * args.test_ratio))
    idx = np.arange(n_samples)
    rng = np.random.RandomState(args.seed)
    rng.shuffle(idx)
    test_idx = idx[:n_test]
    train_idx = idx[n_test:]

    X_train, y_train_log = X[train_idx], log_y[train_idx]
    X_test, y_test_log = X[test_idx], log_y[test_idx]

    # scale X only
    standardize_X = bool(args.standardize_X)
    if standardize_X:
        x_scaler = MinMaxScalerX.fit(X_train)
        X_train_s = x_scaler.transform(X_train)
        X_test_s = x_scaler.transform(X_test)
    else:
        x_scaler = None
        X_train_s, X_test_s = X_train, X_test

    logger.info(f"[split] y std train/test (log): "
                f"{np.std(y_train_log, ddof=0):.4g} / {np.std(y_test_log, ddof=0):.4g}")

    # tensors
    X_train_t = torch.from_numpy(X_train_s).float().to(device)
    y_train_t_log = torch.from_numpy(y_train_log).float().to(device)
    X_test_t = torch.from_numpy(X_test_s).float().to(device)
    y_test_t_log = torch.from_numpy(y_test_log).float().to(device)

    train_loader = DataLoader(
        TensorDataset(X_train_t, y_train_t_log),
        batch_size=args.batch_size,
        shuffle=True
    )

    # model (reuse same structure as utils_self)
    in_dim = X_train_t.shape[1]
    model = create_model(in_dim, device=device)

    # optimizer/loss/scheduler
    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=30
    )

    # baseline
    base = baseline_metrics_both_spaces(y_train_log, y_test_log)
    logger.info(
        f"[baseline] Test MSE(log)={base['mse_log']:.4g} | MAE(log)={base['mae_log']:.4g} | "
        f"R2(log)={base['r2_log']:.4g} | NRMSE(log)={base['nrmse_log']:.4g}"
    )
    logger.info(
        f"[baseline] Test MSE(orig)={base['mse_orig']:.4g} | MAE(orig)={base['mae_orig']:.4g} | "
        f"R2(orig)={base['r2_orig']:.4g} | NRMSE(orig)={base['nrmse_orig']:.4g}"
    )

    # training curves
    train_curve_log, test_curve_log = [], []
    train_curve_orig, test_curve_orig = [], []

    # early stop
    best_test_loss = float("inf")
    patience_counter = 0
    best_model_state = None
    best_metrics = None

    # ---------------- train loop ----------------
    for epoch in range(1, args.epochs + 1):
        model.train()
        batch_losses_log = []

        for xb, yb_log in train_loader:
            xb = xb.to(device)
            yb_log = yb_log.to(device)

            optimizer.zero_grad(set_to_none=True)
            pred_log = model(xb)
            loss = criterion(pred_log, yb_log)
            loss.backward()
            optimizer.step()

            batch_losses_log.append(loss.item())

        train_m = eval_metrics_both_spaces(model, X_train_t, y_train_t_log, device)
        test_m = eval_metrics_both_spaces(model, X_test_t, y_test_t_log, device)

        train_curve_log.append(train_m["mse_log"])
        test_curve_log.append(test_m["mse_log"])
        train_curve_orig.append(train_m["mse_orig"])
        test_curve_orig.append(test_m["mse_orig"])

        logger.info(
            f"Epoch {epoch:03d} | "
            f"Train MSE(log,batch)={np.mean(batch_losses_log):.4g} | "
            f"Train MSE(log,full)={train_m['mse_log']:.4g} | "
            f"Test MSE(log,full)={test_m['mse_log']:.4g} || "
            f"Test(log): MSE={test_m['mse_log']:.4g}, MAE={test_m['mae_log']:.4g}, "
            f"R2={test_m['r2_log']:.4g}, NRMSE={test_m['nrmse_log']:.4g} | "
            f"Test(orig): MSE={test_m['mse_orig']:.4g}, MAE={test_m['mae_orig']:.4g}, "
            f"R2={test_m['r2_orig']:.4g}, NRMSE={test_m['nrmse_orig']:.4g} | "
            f"lr={optimizer.param_groups[0]['lr']:.2e}"
        )

        current_test_loss = test_m["mse_orig"]
        if current_test_loss < best_test_loss:
            best_test_loss = current_test_loss
            patience_counter = 0
            best_model_state = model.state_dict().copy()
            best_metrics = test_m.copy()
        else:
            patience_counter += 1

        scheduler.step(current_test_loss)

        if patience_counter >= args.patience:
            logger.info(f"Early stopping at epoch {epoch}")
            break

    # load best
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        logger.info(
            f"Loaded best model. Best Test MSE (orig): {best_test_loss:.6g}, "
            f"R2 (orig): {best_metrics['r2_orig']:.6g}, "
            f"MAE (orig): {best_metrics['mae_orig']:.6g}, "
            f"NRMSE (orig): {best_metrics['nrmse_orig']:.6g}"
        )
    else:
        best_metrics = eval_metrics_both_spaces(model, X_test_t, y_test_t_log, device)
        logger.info(
            f"Training completed without early stopping. Final Test MSE (orig): {best_metrics['mse_orig']:.6g}"
        )

    # ---------------- plotting (keep all your figures) ----------------
    os.makedirs(args.out_dir, exist_ok=True)
    timestamp = _ts()

    # 1) loss curves (log + orig)
    plt.figure(figsize=(10, 5))

    plt.subplot(1, 2, 1)
    plt.plot(train_curve_log, label="Train MSE (log space)")
    plt.plot(test_curve_log, label="Test MSE (log space)")
    plt.title(f"Loss Curve in Log Space (T = {args.n_exp + 1})")
    plt.xlabel("Epoch")
    plt.ylabel("MSE")
    plt.legend()
    plt.grid(alpha=0.3)

    plt.subplot(1, 2, 2)
    plt.plot(train_curve_orig, label="Train MSE (original space)")
    plt.plot(test_curve_orig, label="Test MSE (original space)")
    plt.title(f"Loss Curve in Original Space (T = {args.n_exp + 1})")
    plt.xlabel("Epoch")
    plt.ylabel("MSE")
    plt.legend()
    plt.grid(alpha=0.3)

    plt.tight_layout()
    combined_plot_path = os.path.join(
        args.out_dir,
        f"training_plots_combined_log_orig_spaces_n_exp{args.n_exp}_{timestamp}.png"
    )
    plt.savefig(combined_plot_path, dpi=200, bbox_inches="tight")
    logger.info(f"Combined plot saved to: {combined_plot_path}")
    if args.show_plots:
        plt.show()
    plt.close()

    # 2) pred vs true scatter (log)
    with torch.no_grad():
        model.eval()
        y_pred_log = model(X_test_t).detach().cpu().numpy().flatten()
        y_test_log_np = y_test_t_log.detach().cpu().numpy().flatten()

    plt.figure(figsize=(6, 6))
    plt.scatter(y_test_log_np, y_pred_log, alpha=0.6, s=20)
    min_val_log = min(y_test_log_np.min(), y_pred_log.min())
    max_val_log = max(y_test_log_np.max(), y_pred_log.max())
    plt.plot([min_val_log, max_val_log], [min_val_log, max_val_log],
             "r--", lw=2, label="Perfect prediction")
    plt.xlabel("True values (log space)")
    plt.ylabel("Predicted values (log space)")
    plt.title(f"Pred vs True in Log Space (R² = {best_metrics['r2_log']:.4f}, T = {args.n_exp + 1})")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    scatter_log_plot_path = os.path.join(
        args.out_dir, f"pred_vs_true_log_n_exp{args.n_exp}_{timestamp}.png"
    )
    plt.savefig(scatter_log_plot_path, dpi=200, bbox_inches="tight")
    logger.info(f"Prediction vs True (log) plot saved to: {scatter_log_plot_path}")
    if args.show_plots:
        plt.show()
    plt.close()

    # 3) pred vs true scatter (orig)
    y_pred_orig = np.exp(y_pred_log)
    y_test_orig_np = np.exp(y_test_log_np)

    plt.figure(figsize=(6, 6))
    plt.scatter(y_test_orig_np, y_pred_orig, alpha=0.6, s=20)
    min_val_orig = min(y_test_orig_np.min(), y_pred_orig.min())
    max_val_orig = max(y_test_orig_np.max(), y_pred_orig.max())
    plt.plot([min_val_orig, max_val_orig], [min_val_orig, max_val_orig],
             "r--", lw=2, label="Perfect prediction")
    plt.xlabel("True values (original space)")
    plt.ylabel("Predicted values (original space)")
    plt.title(f"Pred vs True in Original Space (R² = {best_metrics['r2_orig']:.4f}, T = {args.n_exp + 1})")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    scatter_orig_plot_path = os.path.join(
        args.out_dir, f"pred_vs_true_orig_n_exp{args.n_exp}_{timestamp}.png"
    )
    plt.savefig(scatter_orig_plot_path, dpi=200, bbox_inches="tight")
    logger.info(f"Prediction vs True (orig) plot saved to: {scatter_orig_plot_path}")
    if args.show_plots:
        plt.show()
    plt.close()

    # 4) residual plots (log + orig)
    residuals_log = y_test_log_np - y_pred_log
    residuals_orig = y_test_orig_np - y_pred_orig

    plt.figure(figsize=(10, 5))

    plt.subplot(1, 2, 1)
    plt.scatter(y_test_log_np, residuals_log, alpha=0.6, s=20)
    plt.axhline(0, color="r", linestyle="--", label="Zero residual")
    plt.xlabel("True values (log space)")
    plt.ylabel("Residual (true - pred)")
    plt.title(f"Residuals in Log Space (T = {args.n_exp + 1})")
    plt.legend()
    plt.grid(alpha=0.3)

    plt.subplot(1, 2, 2)
    plt.scatter(y_test_orig_np, residuals_orig, alpha=0.6, s=20)
    plt.axhline(0, color="r", linestyle="--", label="Zero residual")
    plt.xlabel("True values (original space)")
    plt.ylabel("Residual (true - pred)")
    plt.title(f"Residuals in Original Space (T = {args.n_exp + 1})")
    plt.legend()
    plt.grid(alpha=0.3)

    plt.tight_layout()
    residual_plot_path = os.path.join(
        args.out_dir, f"residuals_log_orig_spaces_n_exp{args.n_exp}_{timestamp}.png"
    )
    plt.savefig(residual_plot_path, dpi=200, bbox_inches="tight")
    logger.info(f"Residual plots saved to: {residual_plot_path}")
    if args.show_plots:
        plt.show()
    plt.close()

    # final logs
    logger.info(
        f"[final] Test(log): MSE={best_metrics['mse_log']:.6g}, MAE={best_metrics['mae_log']:.6g}, "
        f"R2={best_metrics['r2_log']:.6g}, NRMSE={best_metrics['nrmse_log']:.6g}"
    )
    logger.info(
        f"[final] Test(orig): MSE={best_metrics['mse_orig']:.6g}, MAE={best_metrics['mae_orig']:.6g}, "
        f"R2={best_metrics['r2_orig']:.6g}, NRMSE={best_metrics['nrmse_orig']:.6g}"
    )

    # save model (keep your payload style + reuse utils scaler/model)
    ckpt_name = f"{args.out_prefix}_{args.n_exp}_log_net.pth"
    ckpt_path = os.path.join(args.out_dir, ckpt_name)

    payload = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "final_metrics": best_metrics,
        "scheduler_state_dict": scheduler.state_dict(),
        "n_exp": args.n_exp,
        "type_obj": args.type_obj,
        "seed": args.seed,
        "standardize_X": standardize_X,
    }
    if standardize_X:
        payload.update({
            "x_scaler_min": x_scaler.min_val,
            "x_scaler_max": x_scaler.max_val,
        })

    torch.save(payload, ckpt_path)
    logger.info(f"Model saved to: {ckpt_path}")


if __name__ == "__main__":
    main()



# # By default, save figures without showing them.
# nohup python Training_Q_last_net_nonrobust_nonscal.py \
#   --gpu 1 \
#   --n_exp 20 \
#   --type_obj var \
#   --epochs 2000 \
#   --batch_size 2048 \
#   --patience 150 \
#   --lr 1e-2 \
#   --test_ratio 0.2 \
#   --out_dir plots_output_var \
#   --out_prefix best_model_Q \
#   --log_dir logs \
#   > train_Q_nexp20_gpu1.log 2>&1 &
