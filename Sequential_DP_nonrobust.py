# Sequential_DP_nonrobust.py
from utils_self import *
import argparse
import os
import numpy as np
import torch
import random


def build_shared_data(B_share, seed, fixed_scaler, degree=3,
                      include_intercept=True, interaction_order=1):
    """
    Build shared CPU NumPy samples for Cartesian products in Q_n_ger_data.
    The original positive/negative sample convention is preserved.
    """
    S_exp_mat_share = sample_state(B_share, random_state=seed)  # (B_share, d)
    Psi_exp_mat_share, _, _ = make_psi_legendre_tensor(
        S_exp_mat_share,
        degree=degree,
        scaler=fixed_scaler,
        include_intercept=include_intercept,
        interaction_order=interaction_order
    )  # (B_share, L)

    act_share_pos = np.ones((B_share, 1), dtype=float)
    act_share_neg = -np.ones((B_share, 1), dtype=float)

    # Keep the original positive/negative version convention.
    S_exp_mat_share_pos = S_exp_mat_share
    S_exp_mat_share_neg = -S_exp_mat_share

    Psi_exp_mat_share_pos = Psi_exp_mat_share
    Psi_exp_mat_share_neg = -Psi_exp_mat_share

    return (act_share_pos, act_share_neg,
            S_exp_mat_share_pos, S_exp_mat_share_neg,
            Psi_exp_mat_share_pos, Psi_exp_mat_share_neg)


def Q_n_ger_data(
        n_exp,
        B,
        seed,
        scaler,
        L_basis,
        Sigma_mat,
        Xi_mat,
        Utilde_mat,
        act_share_pos,
        act_share_neg,
        S_exp_mat_share_pos,
        S_exp_mat_share_neg,
        Psi_exp_mat_share_pos,
        Psi_exp_mat_share_neg,
        nu_factor=0.005,
        chunk_BK=2000,
        type_obj="obj",
        device=None,
        out_prefix="Qval_dataset",
        out_dir="."
):
    """
    Generate the Q-value dataset for day n_exp under the DP-sequence objective.
    - Supports BK chunking.
    - Supports explicit device selection.
    - Supports custom output prefixes and directories.
    """

    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # --------------------------
    # 1) Generate experimental samples on CPU as NumPy arrays.
    # --------------------------
    S_exp_array = sample_state_batch(B, n_exp, random_state=seed)  # (B, n_exp, d)

    Psi_exp_array, _, _ = make_psi_legendre_tensor_batch(
        S_exp_array,
        degree=3,
        scaler=scaler,
        include_intercept=True,
        interaction_order=1
    )  # (B, n_exp, L)

    # Use the caller-provided seed for reproducibility.
    act_exp_array = make_structured_action_space_batch(B, n_exp, random_state=seed)  # (B,K,n_exp)

    # --------------------------
    # 2) Compute delta / Delta / Gamma summaries.
    # --------------------------
    delta_a = act_exp_array.sum(axis=2)  # (B, K)
    Delta_a = np.matmul(act_exp_array, S_exp_array)     # (B, K, d)
    Gamma_a = np.matmul(act_exp_array, Psi_exp_array)  # (B, K, L)

    # --------------------------
    # 3) Flatten while preserving the current column-major construction style.
    # --------------------------
    delta_a_flat = delta_a.T.reshape(-1, 1)  # (B*K,1)
    Delta_a_flat = Delta_a.transpose(1, 0, 2).reshape(-1, Delta_a.shape[2])  # (B*K,d)
    Gamma_a_flat = Gamma_a.transpose(1, 0, 2).reshape(-1, Gamma_a.shape[2])  # (B*K,L)

    BK = delta_a_flat.shape[0]
    B_share = act_share_pos.shape[0]

    mean_obj_chunks = []
    Data_obj_chunks = []

    processed_BK = 0

    # --------------------------
    # 4) Chunked evaluation.
    # --------------------------
    for start in range(0, BK, chunk_BK):
        end = min(start + chunk_BK, BK)
        m = end - start

        delta_chunk = delta_a_flat[start:end]      # (m,1)
        Delta_chunk = Delta_a_flat[start:end, :]   # (m,d)
        Gamma_chunk = Gamma_a_flat[start:end, :]   # (m,L)

        # Combine candidate histories with shared one-step samples.
        delta_pos = (delta_chunk + act_share_pos.T).reshape(-1, 1)
        delta_neg = (delta_chunk + act_share_neg.T).reshape(-1, 1)

        Delta_pos = (Delta_chunk[:, None, :] + S_exp_mat_share_pos[None, :, :]) \
            .reshape(-1, Delta_chunk.shape[1])
        Delta_neg = (Delta_chunk[:, None, :] + S_exp_mat_share_neg[None, :, :]) \
            .reshape(-1, Delta_chunk.shape[1])

        Gamma_pos = (Gamma_chunk[:, None, :] + Psi_exp_mat_share_pos[None, :, :]) \
            .reshape(-1, Gamma_chunk.shape[1])
        Gamma_neg = (Gamma_chunk[:, None, :] + Psi_exp_mat_share_neg[None, :, :]) \
            .reshape(-1, Gamma_chunk.shape[1])

        # --------------------------
        # 5) Objective function.
        # --------------------------
        if type_obj == "obj":
            obj_pos, var_pos, bias_pos = Refine_Q_robust_obj_fun_v1_batch_flat(
                L_basis, n_exp + 1,
                delta_pos, Delta_pos, Gamma_pos,
                Sigma_mat, Xi_mat, Utilde_mat, nu_factor,
                device=device, return_numpy=True
            )
            obj_neg, var_neg, bias_neg = Refine_Q_robust_obj_fun_v1_batch_flat(
                L_basis, n_exp + 1,
                delta_neg, Delta_neg, Gamma_neg,
                Sigma_mat, Xi_mat, Utilde_mat, nu_factor,
                device=device, return_numpy=True
            )

            obj_vec_chunk = custom_selection(obj_pos, obj_neg)

            # Reshape back to (m, B_share).
            obj_vec_2d = obj_vec_chunk.reshape(m, B_share, order='C')

            obj_pos_2d = obj_pos.reshape(m, B_share, order='C')
            obj_neg_2d = obj_neg.reshape(m, B_share, order='C')
            var_pos_2d = var_pos.reshape(m, B_share, order='C')
            var_neg_2d = var_neg.reshape(m, B_share, order='C')
            bias_pos_2d = bias_pos.reshape(m, B_share, order='C')
            bias_neg_2d = bias_neg.reshape(m, B_share, order='C')

            # Average over shared samples.
            obj_mean = obj_vec_2d.mean(axis=1, keepdims=True)

            Data_obj_chunk = np.concatenate([
                obj_pos_2d.mean(axis=1, keepdims=True),
                obj_neg_2d.mean(axis=1, keepdims=True),
                var_pos_2d.mean(axis=1, keepdims=True),
                var_neg_2d.mean(axis=1, keepdims=True),
                bias_pos_2d.mean(axis=1, keepdims=True),
                bias_neg_2d.mean(axis=1, keepdims=True),
            ], axis=1)

        elif type_obj == "var":
            # Alternative non-DP-sequence imbalance objective for ablations.
            unb_pos = Refine_Q_nonrobust_obj_fun_v1_batch_flat(
                L_basis, n_exp + 1,
                delta_pos, Delta_pos, Gamma_pos,
                Sigma_mat, Xi_mat, Utilde_mat, nu_factor,
                device=device, return_numpy=True
            )
            unb_neg = Refine_Q_nonrobust_obj_fun_v1_batch_flat(
                L_basis, n_exp + 1,
                delta_neg, Delta_neg, Gamma_neg,
                Sigma_mat, Xi_mat, Utilde_mat, nu_factor,
                device=device, return_numpy=True
            )

            # Use the smaller objective value as the non-robust selection rule.
            obj_vec_chunk = np.minimum(unb_pos, unb_neg)
            obj_vec_2d = obj_vec_chunk.reshape(m, B_share, order='C')
            obj_mean = obj_vec_2d.mean(axis=1, keepdims=True)
            Data_obj_chunk = np.concatenate([
                obj_vec_2d.mean(axis=1, keepdims=True),
            ], axis=1)

        else:
            raise ValueError(f"Unknown type_obj: {type_obj}")

        mean_obj_chunks.append(obj_mean)
        Data_obj_chunks.append(Data_obj_chunk)

        processed_BK += m
        if processed_BK == BK or processed_BK % max(1000, chunk_BK) == 0:
            pct = processed_BK / BK * 100
            print(f"[Progress] Processed {processed_BK}/{BK} (b,k) pairs, about {pct:.2f}%")

    # --------------------------
    # 5) Concatenate chunks and save.
    # --------------------------
    mean_obj_bk = np.vstack(mean_obj_chunks)  # (BK,1)
    Data_obj = np.vstack(Data_obj_chunks)

    os.makedirs(out_dir, exist_ok=True)

    # Save features and labels.
    Out_Q_res = mean_obj_bk.reshape(-1, 1)
    filename_dataset = os.path.join(out_dir, f"{out_prefix}_{n_exp}_{type_obj}.txt")

    save_features_labels(
        delta_a_flat,
        Delta_a_flat,
        Gamma_a_flat,
        Out_Q_res,
        path=filename_dataset
    )
    print(f"[Saved] {filename_dataset}")

    return float(np.mean(mean_obj_bk))


def main():
    parser = argparse.ArgumentParser()

    # Core experiment parameters.
    parser.add_argument("--n_exp", type=int, default=20)
    parser.add_argument("--B", type=int, default=3200)
    parser.add_argument("--B_share", type=int, default=2500)
    parser.add_argument("--seed", type=int, default=2025)
    parser.add_argument("--nu_factor", type=float, default=0.001)

    # Monte Carlo stage for prespecified parameters.
    parser.add_argument("--M_rept", type=int, default=100)
    parser.add_argument("--print_every", type=int, default=10)

    # Compute and memory controls.
    parser.add_argument("--bk_batch_size", type=int, default=2000,
                        help="Chunk size along the BK dimension; maps to chunk_BK.")

    # Objective type.
    parser.add_argument("--type_obj", type=str, default="obj",
                        choices=["obj", "var"])

    # Output controls.
    parser.add_argument("--out_prefix", type=str, default="Qval_dataset")
    parser.add_argument("--out_dir", type=str, default=".")

    # GPU control.
    parser.add_argument("--gpu", type=int, default=None,
                        help="GPU id, e.g. 0 or 1. If omitted, use automatic device selection.")
    args = parser.parse_args()

    # ------------------------
    # 0) Configure visible GPU.
    # ------------------------
    if args.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
        # After this setting, PyTorch sees the selected device as cuda:0.
        print(f"[GPU] CUDA_VISIBLE_DEVICES = {args.gpu}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[Device] {device}")

    # ------------------------
    # 1) Overall timer.
    # ------------------------
    t_all = tic()

    # ------------------------
    # 2) Fix random seeds.
    # ------------------------
    set_seed(args.seed)

    # ------------------------
    # 3) Stage 1: estimate prespecified matrices.
    # ------------------------
    t0 = tic()
    ATE_mc_true, C_hat_mean, Xi_mat, Sigma_mat, Utilde_mat, fixed_scaler, L_basis = \
        prespecified_params_fun(
            M_rept=args.M_rept,
            print_every=args.print_every
        )
    toc(t0, "Prespecified parameter computation finished")

    # ------------------------
    # 4) Stage 2: shared samples.
    # ------------------------
    t0 = tic()
    (act_share_pos, act_share_neg,
     S_exp_mat_share_pos, S_exp_mat_share_neg,
     Psi_exp_mat_share_pos, Psi_exp_mat_share_neg) = \
        build_shared_data(
            B_share=args.B_share,
            seed=args.seed,
            fixed_scaler=fixed_scaler
        )
    toc(t0, "Shared sample construction finished")

    # ------------------------
    # 5) Stage 3: build the Q-value dataset.
    # ------------------------
    t0 = tic()
    mean_obj_bk_val = Q_n_ger_data(
        n_exp=args.n_exp,
        B=args.B,
        seed=args.seed,
        scaler=fixed_scaler,
        L_basis=L_basis,
        Sigma_mat=Sigma_mat,
        Xi_mat=Xi_mat,
        Utilde_mat=Utilde_mat,
        act_share_pos=act_share_pos,
        act_share_neg=act_share_neg,
        S_exp_mat_share_pos=S_exp_mat_share_pos,
        S_exp_mat_share_neg=S_exp_mat_share_neg,
        Psi_exp_mat_share_pos=Psi_exp_mat_share_pos,
        Psi_exp_mat_share_neg=Psi_exp_mat_share_neg,
        nu_factor=args.nu_factor,
        chunk_BK=args.bk_batch_size,
        type_obj=args.type_obj,
        device=device,
        out_prefix=args.out_prefix,
        out_dir=args.out_dir
    )
    print(f"[Result] mean_obj_bk_val = {mean_obj_bk_val:.6e}")
    toc(t0, "Generated dataset construction finished")

    # ------------------------
    # 6) Total elapsed time.
    # ------------------------
    toc(t_all, "Full pipeline finished")


if __name__ == "__main__":
    main()


# =========================
# Example usage with nohup and explicit GPU selection.
# =========================
# nohup python Sequential_DP_nonrobust.py \
#   --gpu 1 \
#   --bk_batch_size 256 \
#   --B 8000 \
#   --B_share 8000 \
#   --n_exp 20 \
#   --seed 2025 \
#   --nu_factor 0.005 \
#   --type_obj var \
#   --out_prefix Qval_dataset \
#   --out_dir . \
#   > qval_build_var_gpu1.log 2>&1 &
