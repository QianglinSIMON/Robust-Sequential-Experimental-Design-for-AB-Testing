from utils_self import *

import os
import argparse
import inspect
import warnings
import numpy as np
import pandas as pd
import torch
import random
import logging

# Server-friendly matplotlib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from datetime import datetime


# ==============
# Small helpers
# ==============
def _ts():
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _setup_logger(log_dir: str, log_prefix: str, n_exp: int, obj_type: str):
    """
    Create a logger that logs to both console and file.
    """
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f"{log_prefix}_nexp{n_exp}_{obj_type}_{_ts()}.log")

    logger = logging.getLogger(f"train_logger_{log_file}")
    logger.setLevel(logging.INFO)
    logger.propagate = False  # avoid duplicate logs

    # avoid repeated handlers if re-imported
    if logger.handlers:
        return logger, log_file

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

    return logger, log_file


def _safe_read_dataset(output_dir: str, fname: str):
    """
    Try reading from output_dir first, fallback to current working dir.
    """
    cand1 = os.path.join(output_dir, fname) if output_dir else fname
    cand2 = fname

    if os.path.exists(cand1):
        return pd.read_csv(cand1, sep="\t", compression="infer", encoding="utf-8"), cand1
    if os.path.exists(cand2):
        return pd.read_csv(cand2, sep="\t", compression="infer", encoding="utf-8"), cand2

    raise FileNotFoundError(f"Dataset not found. Tried:\n- {cand1}\n- {cand2}")


def _maybe_call_Q_iter_ger_data(**kwargs):
    """
    Call Q_iter_ger_data with only supported kwargs.
    Compatible with multiple versions of utils_self.py.
    """
    sig = inspect.signature(Q_iter_ger_data)
    supported = set(sig.parameters.keys())
    filtered = {k: v for k, v in kwargs.items() if k in supported}
    return Q_iter_ger_data(**filtered)


def main():
    parser = argparse.ArgumentParser()

    # Core experiment controls
    parser.add_argument("--n_exp", type=int, default=20,
                        help="Backward starts from n_exp-1 down to 1. Real horizon is n_exp+1.")
    parser.add_argument("--obj_type", type=str, default="obj")
    parser.add_argument("--seed", type=int, default=2025)

    # GPU
    parser.add_argument("--gpu", type=int, default=None,
                        help="GPU id. If set, will set CUDA_VISIBLE_DEVICES.")

    # Monte Carlo / dataset sizes
    parser.add_argument("--B_share", type=int, default=8000)
    parser.add_argument("--B_iter", type=int, default=8000)

    # Training hyperparams per step
    parser.add_argument("--test_ratio", type=float, default=0.2)
    parser.add_argument("--batch_size", type=int, default=2048)
    parser.add_argument("--epochs", type=int, default=2000)
    parser.add_argument("--patience", type=int, default=200)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-5)

    # Output naming
    parser.add_argument("--out_dir", type=str, default=None,
                        help="Default: plots_output_{obj_type}")
    parser.add_argument("--model_prefix", type=str, default="best_model_Q")
    parser.add_argument("--data_prefix", type=str, default="Qval_dataset")

    # Logging
    parser.add_argument("--log_dir", type=str, default="logs")
    parser.add_argument("--log_prefix", type=str, default="backward_train")

    args = parser.parse_args()

    # ------------------------
    # GPU selection
    # ------------------------
    if args.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ------------------------
    # Output dir
    # ------------------------
    obj_type = args.obj_type
    output_dir = args.out_dir or f"plots_output_{obj_type}"
    os.makedirs(output_dir, exist_ok=True)

    # ------------------------
    # Logger
    # ------------------------
    logger, log_file = _setup_logger(args.log_dir, args.log_prefix, args.n_exp, obj_type)
    logger.info(f"Logger initialized. Log file: {log_file}")
    logger.info(f"device: {device}")
    logger.info(f"args: {vars(args)}")

    # ------------------------
    # Seeds
    # ------------------------
    seed = args.seed
    set_seed(seed)

    # ------------------------
    # Timers
    # ------------------------
    t_all = tic()

    # =========================
    # Stage 1: prespecified params
    # =========================
    logger.info("Stage 1: Computing prespecified parameters ...")
    t0 = tic()

    ATE_mc_true, C_hat_mean, Xi_mat, Sigma_mat, Utilde_mat, fixed_scaler, L_basis = \
        prespecified_params_fun(M_rept=100, print_every=10)

    toc(t0, "Stage 1 done")
    logger.info("Stage 1 done. ATE_mc_true=%.6g | L_basis=%s", ATE_mc_true, L_basis)

    # =========================
    # Stage 2: shared data
    # =========================
    logger.info("Stage 2: Building shared data ...")
    t0 = tic()

    B_share = int(args.B_share)
    S_exp_mat_share = sample_state(B_share, random_state=seed)
    Psi_exp_mat_share, _, _ = make_psi_legendre_tensor(
        S_exp_mat_share,
        include_intercept=True,
        scaler=fixed_scaler
    )

    act_share_pos = np.ones((B_share, 1), dtype=float)
    act_share_neg = -np.ones((B_share, 1), dtype=float)

    # Positive/negative versions
    S_exp_mat_share_pos = S_exp_mat_share
    S_exp_mat_share_neg = -S_exp_mat_share

    Psi_exp_mat_share_pos = Psi_exp_mat_share
    Psi_exp_mat_share_neg = -Psi_exp_mat_share

    toc(t0, "Stage 2 done")
    logger.info("Stage 2 done. B_share=%d", B_share)

    # =========================
    # Stage 3: backward loop
    # =========================
    n_exp = int(args.n_exp)
    n_exp_list = list(reversed(range(1, n_exp)))  # [n_exp-1, ..., 1]

    for n_exp_iter in n_exp_list:
        logger.info("=" * 70)
        logger.info("Backward step: n_exp = %d (T = %d)", n_exp_iter, n_exp_iter + 1)
        logger.info("=" * 70)

        # ------------------------
        # 3.0 Load last model (n_exp_iter+1)
        # ------------------------
        t0 = tic()

        last_model_path = os.path.join(
            output_dir,
            f"{args.model_prefix}_{n_exp_iter + 1}_log_net.pth"
        )
        if not os.path.exists(last_model_path):
            raise FileNotFoundError(f"Checkpoint not found: {last_model_path}")

        checkpoint = torch.load(last_model_path, weights_only=False)
        logger.info("Loaded checkpoint: %s", last_model_path)

        # ------------------------
        # 3.1 Generate dataset for current step
        # ------------------------
        B_iter = int(args.B_iter)
        logger.info("B_iter = %d", B_iter)

        _maybe_call_Q_iter_ger_data(
            checkpoint=checkpoint,
            n_exp=n_exp_iter,
            B=B_iter,
            seed=seed,
            scaler=fixed_scaler,
            act_share_pos=act_share_pos,
            act_share_neg=act_share_neg,
            S_exp_mat_share_pos=S_exp_mat_share_pos,
            S_exp_mat_share_neg=S_exp_mat_share_neg,
            Psi_exp_mat_share_pos=Psi_exp_mat_share_pos,
            Psi_exp_mat_share_neg=Psi_exp_mat_share_neg,
            obj_type=obj_type,
            # If your utils_self has added out_prefix/out_dir, this script will use them
            out_prefix=args.data_prefix,
            out_dir=output_dir
        )

        # ------------------------
        # 3.2 Read dataset
        # ------------------------
        fname = f"{args.data_prefix}_{n_exp_iter}_{obj_type}.txt"
        df_iter, used_path = _safe_read_dataset(output_dir, fname)
        logger.info("Loaded dataset: %s | shape=%s", used_path, df_iter.shape)

        toc(t0, f"Step {n_exp_iter}: dataset generation & load")

        # ------------------------
        # 3.3 Train network at this step
        # ------------------------
        t0 = tic()

        set_seed(seed)
        logger.info("Training device: %s", device)

        assert isinstance(df_iter, pd.DataFrame)
        assert "Out_Q_res" in df_iter.columns

        # Shuffle + clean
        df_shuf = df_iter.sample(frac=1.0, random_state=seed).reset_index(drop=True)
        df_shuf = df_shuf.replace([np.inf, -np.inf], np.nan).dropna(axis=0)

        # IMPORTANT:
        # Out_Q_res is assumed to be log space in your unified pipeline.
        y = df_shuf["Out_Q_res"].astype(np.float64).values.reshape(-1, 1)
        X = df_shuf.drop(columns=["Out_Q_res"]).astype(np.float64).values

        n_samples, n_features = X.shape
        logger.info("Samples=%d | Features=%d", n_samples, n_features)

        # Split
        test_ratio = float(args.test_ratio)
        n_test = int(round(n_samples * test_ratio))
        idx = np.arange(n_samples)
        rng = np.random.RandomState(seed)
        rng.shuffle(idx)
        test_idx = idx[:n_test]
        train_idx = idx[n_test:]

        X_train, y_train = X[train_idx], y[train_idx]
        X_test, y_test = X[test_idx], y[test_idx]

        # X scaling (Min-Max)
        x_scaler = MinMaxScalerX.fit(X_train)
        X_train_s = x_scaler.transform(X_train)
        X_test_s = x_scaler.transform(X_test)

        # Tensors
        X_train_t = torch.from_numpy(X_train_s).float().to(device)
        y_train_t = torch.from_numpy(y_train).float().to(device)
        X_test_t = torch.from_numpy(X_test_s).float().to(device)
        y_test_t = torch.from_numpy(y_test).float().to(device)

        train_loader = DataLoader(
            TensorDataset(X_train_t, y_train_t),
            batch_size=int(args.batch_size),
            shuffle=True
        )

        # Model
        in_dim = X_train_t.shape[1]
        model = create_model(in_dim, device=device)

        criterion = nn.MSELoss()
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=float(args.lr),
            weight_decay=float(args.weight_decay)
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=20
        )

        base = baseline_metrics(y_train, y_test)
        logger.info(
            "[baseline-log] Test MSE=%.4g | MAE=%.4g | R2=%.4g | NRMSE=%.4g",
            base["mse"], base["mae"], base["r2"], base["nrmse"]
        )

        epochs = int(args.epochs)
        patience = int(args.patience)

        train_curve_log, test_curve_log = [], []

        best_test_loss = float("inf")
        patience_counter = 0
        best_model_state = None
        best_metrics = None

        for epoch in range(1, epochs + 1):
            model.train()
            batch_losses = []

            for xb, yb in train_loader:
                xb = xb.to(device)
                yb = yb.to(device)
                optimizer.zero_grad(set_to_none=True)
                pred = model(xb)
                loss = criterion(pred, yb)
                loss.backward()
                optimizer.step()
                batch_losses.append(loss.item())

            train_m = eval_metrics_log_space(model, X_train_t, y_train_t, device=device)
            test_m = eval_metrics_log_space(model, X_test_t, y_test_t, device=device)

            train_curve_log.append(train_m["mse_log"])
            test_curve_log.append(test_m["mse_log"])

            if epoch % 50 == 0:
                logger.info(
                    "Step %d | Epoch %04d | Train MSE(batch)=%.4g | "
                    "Train MSE(log)=%.4g | Test MSE(log)=%.4g | R2(log)=%.4g | lr=%.2e",
                    n_exp_iter, epoch, float(np.mean(batch_losses)),
                    train_m["mse_log"], test_m["mse_log"], test_m["r2_log"],
                    optimizer.param_groups[0]["lr"]
                )

            current_test_loss = test_m["mse_log"]
            if current_test_loss < best_test_loss:
                best_test_loss = current_test_loss
                patience_counter = 0
                best_model_state = model.state_dict().copy()
                best_metrics = test_m.copy()
            else:
                patience_counter += 1

            scheduler.step(current_test_loss)

            if patience_counter >= patience:
                logger.info("Early stopping at epoch %d for step %d", epoch, n_exp_iter)
                break

        if best_model_state is not None:
            model.load_state_dict(best_model_state)

        logger.info(
            "[final-log] Step %d | MSE=%.6g | MAE=%.6g | R2=%.6g | NRMSE=%.6g",
            n_exp_iter,
            best_metrics["mse_log"], best_metrics["mae_log"],
            best_metrics["r2_log"], best_metrics["nrmse_log"]
        )

        toc(t0, f"Step {n_exp_iter}: training done")

        # ------------------------
        # 3.4 Plot & save
        # ------------------------
        t0 = tic()
        timestamp = _ts()

        os.makedirs(output_dir, exist_ok=True)

        # 3.4.1 Loss curve (log space)
        plt.figure(figsize=(9, 5))
        plt.plot(train_curve_log, label="Train MSE (log space)")
        plt.plot(test_curve_log, label="Test MSE (log space)")
        plt.title(f"Loss Curve in Log Space (T = {n_exp_iter + 1})")
        plt.xlabel("Epoch")
        plt.ylabel("MSE")
        plt.legend()
        plt.grid(alpha=0.3)
        plt.tight_layout()

        loss_path = os.path.join(
            output_dir, f"loss_curves_log_n_exp_{n_exp_iter}_{timestamp}.png"
        )
        plt.savefig(loss_path, dpi=200, bbox_inches="tight")
        plt.close()
        logger.info("Saved loss plot: %s", loss_path)

        # 3.4.2 Pred vs True scatter with R² (log space)
        with torch.no_grad():
            model.eval()
            y_pred_log = model(X_test_t).detach().cpu().numpy().flatten()
            y_true_log = y_test_t.detach().cpu().numpy().flatten()

        plt.figure(figsize=(6, 6))
        plt.scatter(y_true_log, y_pred_log, alpha=0.6, s=20)
        min_val = min(y_true_log.min(), y_pred_log.min())
        max_val = max(y_true_log.max(), y_pred_log.max())
        plt.plot([min_val, max_val], [min_val, max_val], "r--", lw=2, label="Perfect prediction")
        plt.xlabel("True values (log space)")
        plt.ylabel("Predicted values (log space)")
        plt.title(f"Pred vs True (log, R² = {best_metrics['r2_log']:.4f}, T = {n_exp_iter + 1})")
        plt.legend()
        plt.grid(alpha=0.3)
        plt.tight_layout()

        scatter_path = os.path.join(
            output_dir, f"pred_vs_true_log_n_exp_{n_exp_iter}_{timestamp}.png"
        )
        plt.savefig(scatter_path, dpi=200, bbox_inches="tight")
        plt.close()
        logger.info("Saved scatter plot: %s", scatter_path)

        # 3.4.3 Save model
        model_path = os.path.join(
            output_dir, f"{args.model_prefix}_{n_exp_iter}_log_net.pth"
        )
        torch.save({
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "x_scaler_min": x_scaler.min_val,
            "x_scaler_max": x_scaler.max_val,
            "final_metrics": best_metrics,
            "scheduler_state_dict": scheduler.state_dict(),
            "seed": seed,
            "n_exp": n_exp_iter,
            "obj_type": obj_type,
            "data_path_used": used_path
        }, model_path)
        logger.info("Model saved: %s", model_path)

        toc(t0, f"Step {n_exp_iter}: plots & model saved")

    # =========================
    # Done
    # =========================
    toc(t_all, "All steps finished")
    logger.info("All steps finished.")


if __name__ == "__main__":
    main()


# =========================
# Example nohup usage
# =========================
# nohup python Training_Q_net_iteration_noonrobust_nonscal.py \
#   --gpu 1 \
#   --n_exp 20 \
#   --obj_type var \
#   --B_share 8000 \
#   --B_iter 8000 \
#   --epochs 2000 \
#   --batch_size 2048 \
#   --patience 200 \
#   --lr 1e-3 \
#   --weight_decay 1e-5 \
#   --out_dir plots_output_var \
#   --model_prefix best_model_Q \
#   --data_prefix Qval_dataset \
#   --log_dir logs \
#   --log_prefix backward_train \
#   > backward_train_var_gpu1.stdout.log 2>&1 &