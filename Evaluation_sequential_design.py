import os
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from datetime import datetime
from utils_self import (create_model, sample_state, make_psi_legendre_tensor,
                        prespecified_params_fun, set_seed, sample_dgp, ols_fit, MinMaxScalerX) # Ensure MinMaxScalerX is imported.


def _compute_terms(S_data, Psi_data, act_seq, Xi_mat, Utilde_mat, Sigma_mat_inv, Sigma_inv_aug, L_basis, nu_factor):
    """
    Helper function for computing closed-form terms for a given action sequence.
    """
    act_array = np.array(act_seq)
    n_days = S_data.shape[0] # Number of rows in S_data equals the number of days.

    delta_a_sum = np.sum(act_array, axis=0).reshape(1, -1)
    Delta_a_sum = (act_array.T @ S_data).reshape(1, -1)
    Gamma_a_sum = (act_array.T @ Psi_data).reshape(1, -1)

    Delta_a_norm = Delta_a_sum @ Sigma_mat_inv @ Delta_a_sum.T

    Delta_aug = np.vstack([delta_a_sum, Delta_a_sum.T])

    Gamma_tilde = Utilde_mat.T @ Gamma_a_sum.T - Utilde_mat.T @ Xi_mat @ Sigma_inv_aug @ Delta_aug
    Gamma_tilde_norm = Gamma_tilde.T @ Gamma_tilde

    denom = (n_days - (1 / n_days) * (delta_a_sum ** 2 + Delta_a_norm))

    # denom = 1e-6 if denom < 1e-6 else denom

    var_term = nu_factor / denom
    var_term_nonrobust = delta_a_sum ** 2 + Delta_a_norm

    bias_upper_term = L_basis * Gamma_tilde_norm / (denom) ** 2
    bias_upper_term_nonrobust = bias_upper_term * nu_factor

    obj_term = var_term + bias_upper_term

    return {
        'obj_term': obj_term,
        'var_term_nonrobust': var_term_nonrobust,
        'bias_upper_term_nonrobust': bias_upper_term_nonrobust,
        'act_array': act_array # Return the action sequence for subsequent sample_dgp calls.
    }


def run_sequential_experiments(Xi_mat, Sigma_mat, Utilde_mat, L_basis,
                               n_days=20, nu_factor=0.05, device=None, exp_random_state=None):
    """
    Run one sequential robust experiment.
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Set the random seed at function entry to keep experiments independent.
    set_seed(exp_random_state)

    # Generate experimental data with the same experiment seed.
    S_exp_mat_share = sample_state(n_days, random_state=exp_random_state)
    Psi_exp_mat_share, _, _ = make_psi_legendre_tensor(S_exp_mat_share, include_intercept=True)

    act_opt_list = []
    act_opt_list_var = []
    # act_opt_list_bias = []

    # --- Precompute invariants ---
    Sigma_mat_inv = np.linalg.inv(Sigma_mat)
    Sigma_inv_aug = np.block([
        [np.eye(1), np.zeros((1, Sigma_mat.shape[0]))],
        [np.zeros((Sigma_mat.shape[0], 1)), Sigma_mat_inv]])

    for n_exp_iter in range(n_days):  # 0, 1, 2, ..., n_days-1
        if n_exp_iter < n_days - 1:  # Use Q networks until the day before the final day.
            # --- Load the robust model and scaler ---
            last_model_path = os.path.join("plots_output_obj", f'best_model_Q_{n_exp_iter + 1}_log_net.pth')
            checkpoint = torch.load(last_model_path, weights_only=False)
            # Load scaler parameters from the checkpoint.
            x_scaler_min = checkpoint['x_scaler_min']
            x_scaler_max = checkpoint['x_scaler_max']
            # Reconstruct the scaler from checkpoint parameters.
            x_scaler_robust = MinMaxScalerX(min_val=x_scaler_min, max_val=x_scaler_max)

            # # --- Load the NonRobust Var model and scaler ---
            last_model_path_var = os.path.join("plots_output_var_unbalanced", f'best_model_Q_{n_exp_iter + 1}_log_net.pth')
            checkpoint_var = torch.load(last_model_path_var, weights_only=False)
            x_scaler_min_var = checkpoint_var['x_scaler_min']
            x_scaler_max_var = checkpoint_var['x_scaler_max']
            x_scaler_var = MinMaxScalerX(min_val=x_scaler_min_var, max_val=x_scaler_max_var)

            
            #
            # # --- Load the NonRobust Bias model and scaler ---
            # last_model_path_bias = os.path.join("plots_output_bias", f'best_model_Q_{n_exp_iter + 1}_net.pth')
            # checkpoint_bias = torch.load(last_model_path_bias, weights_only=False)
            # x_scaler_min_bias = checkpoint_bias['x_scaler_min']
            # x_scaler_max_bias = checkpoint_bias['x_scaler_max']
            # x_scaler_bias = MinMaxScalerX(min_val=x_scaler_min_bias, max_val=x_scaler_max_bias)

            # --- Build raw input data ---
            if n_exp_iter == 0:
                act_pos_seq = [1]
                act_neg_seq = [-1]
            else:
                act_pos_seq = act_opt_list +[1]
                act_neg_seq = act_opt_list +[-1]

            S_data = S_exp_mat_share[:(n_exp_iter + 1), :]  # Keep the first n_exp_iter+1 rows.
            Psi_data = Psi_exp_mat_share[:(n_exp_iter + 1), :]

            # Compute raw features before scaler transformation.
            act_pos_seq = np.array(act_pos_seq)
            act_neg_seq = np.array(act_neg_seq)

            delta_a_sum_pos = np.sum(act_pos_seq, axis=0)
            delta_a_sum_neg = np.sum(act_neg_seq, axis=0)

            Delta_a_sum_pos = act_pos_seq.T @ S_data
            Delta_a_sum_neg = act_neg_seq.T @ S_data

            Gamma_a_sum_pos = act_pos_seq.T @ Psi_data
            Gamma_a_sum_neg = act_neg_seq.T @ Psi_data

            delta_a_sum_pos = delta_a_sum_pos.reshape(1, -1)
            delta_a_sum_neg = delta_a_sum_neg.reshape(1, -1)

            Delta_a_sum_pos = Delta_a_sum_pos.reshape(1, -1)
            Delta_a_sum_neg = Delta_a_sum_neg.reshape(1, -1)

            Gamma_a_sum_pos = Gamma_a_sum_pos.reshape(1, -1)
            Gamma_a_sum_neg = Gamma_a_sum_neg.reshape(1, -1)

            # Build unscaled input features.
            X_input_data_pos = np.concatenate((delta_a_sum_pos, Delta_a_sum_pos, Gamma_a_sum_pos), axis=1)
            X_input_data_neg = np.concatenate((delta_a_sum_neg, Delta_a_sum_neg, Gamma_a_sum_neg), axis=1)

            # --- Apply the corresponding scalers ---
            # Robust
            X_input_s_pos_robust = x_scaler_robust.transform(X_input_data_pos)
            X_input_s_neg_robust = x_scaler_robust.transform(X_input_data_neg)

            # NonRobust Var
            X_input_s_pos_var = x_scaler_var.transform(X_input_data_pos)
            X_input_s_neg_var = x_scaler_var.transform(X_input_data_neg)


            #
            # # NonRobust Bias
            # X_input_s_pos_bias = x_scaler_bias.transform(X_input_data_pos)
            # X_input_s_neg_bias = x_scaler_bias.transform(X_input_data_neg)

            # --- Load models and run predictions ---
            in_dim = X_input_s_pos_robust.shape[1] # All scaled feature dimensions should match.

            # Robust Model
            Q_net_last_model = create_model(in_dim, device=device)
            Q_net_last_model.load_state_dict(checkpoint['model_state_dict'])
            Q_net_last_model.eval()

            # NonRobust Var Model
            Q_net_last_model_var = create_model(in_dim, device=device)
            Q_net_last_model_var.load_state_dict(checkpoint_var['model_state_dict'])
            Q_net_last_model_var.eval()

            
            #
            # # NonRobust Bias Model
            # Q_net_last_model_bias = create_model(in_dim, device=device)
            # Q_net_last_model_bias.load_state_dict(checkpoint_bias['model_state_dict'])
            # Q_net_last_model_bias.eval()

            with torch.no_grad():
                # Robust predictions using robust-scaled data.
                Q_pos = Q_net_last_model(torch.tensor(X_input_s_pos_robust, dtype=torch.float32,
                                                      device=device)).item()
                Q_neg = Q_net_last_model(torch.tensor(X_input_s_neg_robust, dtype=torch.float32,
                                                      device=device)).item()

                # --- Apply exponential transform ---
                Q_pos=np.exp(Q_pos) # Model predicts log(y_orig), so exp gives y_orig scale
                Q_neg=np.exp(Q_neg) # Model predicts log(y_orig), so exp gives y_orig scale
                # --- End transform ---

                # NonRobust Var predictions using var-scaled data.
                Q_pos_var = Q_net_last_model_var(
                    torch.tensor(X_input_s_pos_var, dtype=torch.float32, device=device)).item()
                Q_neg_var = Q_net_last_model_var(
                    torch.tensor(X_input_s_neg_var, dtype=torch.float32, device=device)).item()

                Q_pos_var =np.exp(Q_pos_var)
                Q_neg_var=np.exp(Q_neg_var) 

                # # NonRobust Bias predictions using bias-scaled data.
                # Q_pos_bias = Q_net_last_model_bias(
                #     torch.tensor(X_input_s_pos_bias, dtype=torch.float32, device=device)).item()
                # Q_neg_bias = Q_net_last_model_bias(
                #     torch.tensor(X_input_s_neg_bias, dtype=torch.float32, device=device)).item()

            # --- Select actions from predictions ---
            # Robust
            if Q_pos >= Q_neg:
                act_opt_list.append(-1)
            else:
                act_opt_list.append(1)

            # NonRobust Var
            if Q_pos_var >= Q_neg_var:
                act_opt_list_var.append(-1)
            else:
                act_opt_list_var.append(1)
            #
            # # NonRobust Bias
            # if Q_pos_bias >= Q_neg_bias:
            #     act_opt_list_bias.append(-1)
            # else:
            #     act_opt_list_bias.append(1)

        else:
            # Use the closed-form expression on the final day.
            # --- Compute terms with the helper function ---
            # Robust
            act_pos_seq_robust =  act_opt_list  +[1]
            act_neg_seq_robust =  act_opt_list  +[-1]
            S_data_final = S_exp_mat_share[:(n_exp_iter + 1), :]
            Psi_data_final = Psi_exp_mat_share[:(n_exp_iter + 1), :]

            terms_pos_robust = _compute_terms(S_data_final, Psi_data_final, act_pos_seq_robust, Xi_mat, Utilde_mat, Sigma_mat_inv, Sigma_inv_aug, L_basis, nu_factor)
            terms_neg_robust = _compute_terms(S_data_final, Psi_data_final, act_neg_seq_robust, Xi_mat, Utilde_mat, Sigma_mat_inv, Sigma_inv_aug, L_basis, nu_factor)

            # NonRobust Var
            act_pos_seq_var =  act_opt_list_var +[1]
            act_neg_seq_var =  act_opt_list_var +[-1]
            terms_pos_var = _compute_terms(S_data_final, Psi_data_final, act_pos_seq_var, Xi_mat, Utilde_mat, Sigma_mat_inv, Sigma_inv_aug, L_basis, nu_factor)
            terms_neg_var = _compute_terms(S_data_final, Psi_data_final, act_neg_seq_var, Xi_mat, Utilde_mat, Sigma_mat_inv, Sigma_inv_aug, L_basis, nu_factor)

            # # NonRobust Bias
            # act_pos_seq_bias = [1] + act_opt_list_bias
            # act_neg_seq_bias = [-1] + act_opt_list_bias
            # terms_pos_bias = _compute_terms(S_data_final, Psi_data_final, act_pos_seq_bias, Xi_mat, Utilde_mat, Sigma_mat_inv, Sigma_inv_aug, L_basis, nu_factor)
            # terms_neg_bias = _compute_terms(S_data_final, Psi_data_final, act_neg_seq_bias, Xi_mat, Utilde_mat, Sigma_mat_inv, Sigma_inv_aug, L_basis, nu_factor)

            # --- Select the final action using the closed-form expression ---
            # Robust
            Q_pos = terms_pos_robust['obj_term']
            Q_neg = terms_neg_robust['obj_term']
            if Q_pos >= Q_neg:
                act_opt_list.append(-1)
            else:
                act_opt_list.append(1)

            # NonRobust Var
            if terms_pos_var['var_term_nonrobust'] >= terms_neg_var['var_term_nonrobust']:
                act_opt_list_var.append(-1)
            else:
                act_opt_list_var.append(1)
            #
            # # NonRobust Bias
            # if terms_pos_bias['bias_upper_term_nonrobust'] >= terms_neg_bias['bias_upper_term_nonrobust']:
            #     act_opt_list_bias.append(-1)
            # else:
            #     act_opt_list_bias.append(1)

    # --- Generate data and estimate with the final action sequences ---
    act_opt_array = np.array(act_opt_list)
    act_opt_array_var = np.array(act_opt_list_var)
    # act_opt_array_bias = np.array(act_opt_list_bias)

    data_exp = sample_dgp(a_all=act_opt_array, s_all=S_exp_mat_share)
    R_outcome = data_exp["R"]
    X_mat_exp = data_exp["X"]
    beta_hat, _, _, _, _ = ols_fit(X_mat_exp, R_outcome, robust=True)
    ATE_est = 2 * beta_hat[0]

    data_exp_var = sample_dgp(a_all=act_opt_array_var, s_all=S_exp_mat_share)
    R_outcome_var = data_exp_var["R"]
    X_mat_exp_var = data_exp_var["X"]
    beta_hat_var, _, _, _, _ = ols_fit(X_mat_exp_var, R_outcome_var, robust=True)
    ATE_est_var = 2 * beta_hat_var[0]

    # data_exp_bias = sample_dgp(a_all=act_opt_array_bias, s_all=S_exp_mat_share)
    # R_outcome_bias = data_exp_bias["R"]
    # X_mat_exp_bias = data_exp_bias["X"]
    # beta_hat_bias, _, _, _, _ = ols_fit(X_mat_exp_bias, R_outcome_bias, robust=True)
    # ATE_est_bias = 2 * beta_hat_bias[0]

    return ATE_est , ATE_est_var #, ATE_est_bias


def run_randomized_experiment(n_days=20, exp_random_state=None):
    """
    Run one classical randomized-assignment experiment.
    """
    # Set the random seed at function entry to keep experiments independent.
    set_seed(exp_random_state)

    S_exp_mat_share = sample_state(n_days, random_state=exp_random_state)
    # Randomized action assignment.
    act_random = np.random.choice([-1, 1], size=n_days)
    data_exp = sample_dgp(a_all=act_random, s_all=S_exp_mat_share)
    R_outcome = data_exp["R"]
    X_mat_exp = data_exp["X"]
    beta_hat, _, _, _, _ = ols_fit(X_mat_exp, R_outcome, robust=True)
    ATE_est = 2 * beta_hat[0]
    return ATE_est

def main():
    """
    Main routine: repeat experiments, collect results, and create box plots.
    """
    n_reps = 100
    n_days = 14  # The real experimental horizon is n_exp + 1.
    nu_factor = 0.005
    # Seed once in the main routine to generate experiment-level seeds.
    set_seed(2025)

    # --- Preload fixed parameters ---
    true_ate, _, Xi_mat, Sigma_mat, Utilde_mat, _, L_basis = prespecified_params_fun(M_rept=100, print_every=10)
    print(
        f"Preloaded parameters: Xi_mat shape={Xi_mat.shape}, Sigma_mat shape={Sigma_mat.shape}, Utilde_mat shape={Utilde_mat.shape}, L_basis={L_basis}")

    sequential_robust_ates = []
    randomized_ates = []
    sequential_nonrobust_ates_var = [] # [Added back]

    print("Running Sequential Robust experiments...")
    for i in range(n_reps):
        # Generate a unique random seed for each experiment.
        exp_seed = np.random.randint(0, 100000)
        # Pass preloaded parameters into the experiment function.
        ate, ate_var = run_sequential_experiments( # [Updated to unpack two estimates]
            Xi_mat, Sigma_mat, Utilde_mat, L_basis,
            n_days=n_days, nu_factor=nu_factor, exp_random_state=exp_seed
        )
        sequential_robust_ates.append(ate)
        sequential_nonrobust_ates_var.append(ate_var) # [Added back]

        if (i + 1) % 20 == 0:
            print(f"  Completed {i + 1}/{n_reps} sequential_exps runs")

    print("Running Randomized experiments...")
    for i in range(n_reps):
        # Generate a unique random seed for each experiment.
        exp_seed = np.random.randint(0, 100000)
        ate = run_randomized_experiment(n_days=n_days, exp_random_state=exp_seed)
        randomized_ates.append(ate)
        if (i + 1) % 20 == 0:
            print(f"  Completed {i + 1}/{n_reps} randomized runs")

    sequential_robust_ates = np.array(sequential_robust_ates)
    sequential_nonrobust_ates_var = np.array(sequential_nonrobust_ates_var) # [Assigned back]
    randomized_ates = np.array(randomized_ates)

    # [IMPORTANT FIX] Assign empty arrays or None to unused variables to prevent errors later
    # This avoids NameError or UnboundLocalError if code below (incorrectly) tries to use them
    # sequential_nonrobust_ates_bias = np.array([]) # We are not using Bias anymore

    # --- Compute squared error for each experiment as an MSE component ---
    sequential_robust_mse_components = (sequential_robust_ates - true_ate) ** 2
    sequential_nonrobust_var_mse_components = (sequential_nonrobust_ates_var - true_ate) ** 2  # [Added back]
    randomized_mse_components = (randomized_ates - true_ate) ** 2

    # --- Compute summary statistics for printing ---
    methods = ['Sequential Robust', 'Sequential NonRobust Var', 'Randomized'] # [Updated methods list]
    # Organize estimates in the same order as methods.
    estimates = [sequential_robust_ates, sequential_nonrobust_ates_var, randomized_ates] # [Updated estimates list]
    results = {}

    for method, ests in zip(methods, estimates):
        mse = np.mean((ests - true_ate) ** 2)  # Average MSE.
        bias = np.mean(ests - true_ate)
        sd = np.std(ests)
        results[method] = {'MSE': mse, 'Bias': bias, 'SD': sd}
        print(f"\n{method}:")
        print(f"  MSE: {mse:.6f}")
        print(f"  Bias: {bias:.6f}")
        print(f"  SD: {sd:.6f}")

    # --- Create output directory ---
    output_dir = "results_output"
    os.makedirs(output_dir, exist_ok=True)

    # --- Save summary results to CSV ---
    results_df = pd.DataFrame(results).T  # Transpose so method names become row indices.
    results_df.index.name = 'Method'
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_filename = os.path.join(output_dir, f"experiment_results_{timestamp}.csv")
    results_df.to_csv(csv_filename)
    print(f"\nResults saved to CSV: {csv_filename}")

    # --- Save all estimates to CSV ---
    all_estimates_df = pd.DataFrame({
        'Sequential Robust': sequential_robust_ates,
        'Sequential NonRobust Var': sequential_nonrobust_ates_var, # [Added back]
        'Randomized': randomized_ates
    })
    estimates_csv_filename = os.path.join(output_dir, f"all_estimates_{timestamp}.csv")
    all_estimates_df.to_csv(estimates_csv_filename, index=False)
    print(f"All estimates saved to CSV: {estimates_csv_filename}")

    # --- Save squared-error components to CSV ---
    all_mse_components_df = pd.DataFrame({
        'Sequential Robust_MSE_Component': sequential_robust_mse_components,
        'Sequential NonRobust Var_MSE_Component': sequential_nonrobust_var_mse_components, # [Added back]
        'Randomized_MSE_Component': randomized_mse_components
    })
    mse_components_csv_filename = os.path.join(output_dir, f"mse_components_{timestamp}.csv")
    all_mse_components_df.to_csv(mse_components_csv_filename, index=False)
    print(f"All MSE components saved to CSV: {mse_components_csv_filename}")

    # --- Plot squared-error components and annotate mean/median values ---
    plt.style.use('default')

    fig, ax = plt.subplots(figsize=(12, 7))  # [Adjusted width for 3 methods]

    # Build plotting data from squared-error values and method labels.
    all_mse_data = np.concatenate([
        sequential_robust_mse_components,
        sequential_nonrobust_var_mse_components, # [Added back]
        randomized_mse_components
    ])
    all_method_labels = (['Sequential Robust'] * len(sequential_robust_mse_components) +
                         ['Sequential NonRobust Var'] * len(sequential_nonrobust_var_mse_components) + # [Added back]
                         ['Randomized'] * len(randomized_mse_components)
                         )

    df_plot_mse = pd.DataFrame({
        'Method': all_method_labels,
        'MSE_Component': all_mse_data
    })

    # Define custom colors for the three methods.
    custom_colors = ["#4C72B0", "#55A868", "#DD8452"]  # [Updated colors: blue, green, orange]

    # Draw a colored seaborn box plot.
    sns.boxplot(data=df_plot_mse, x='Method', y='MSE_Component', hue='Method', palette=custom_colors, ax=ax,
                legend=False)

    # Set title and axis labels.
    ax.set_title('Distribution of MSE Components\n(Sequential Robust vs NonRobust Var vs Randomized)', fontsize=16, # [Updated title]
                 fontweight='bold')
    ax.set_ylabel('MSE Component\n($(ATE_{est} - ATE_{true})^2$)', fontsize=14)
    ax.set_xlabel('Method', fontsize=14)
    ax.tick_params(axis='x', rotation=45)  # [Rotate labels for 3 methods]

    # Compute and annotate mean/median values.
    for i, method in enumerate(methods): # [Iterate over the 3 methods]
        # Select data for the current method.
        if method == 'Sequential Robust':
            data = sequential_robust_mse_components
        elif method == 'Sequential NonRobust Var': # [Added condition]
            data = sequential_nonrobust_var_mse_components
        else:  # method == 'Randomized'
            data = randomized_mse_components

        # Compute summary statistics.
        mean_val = np.mean(data)
        median_val = np.median(data)

        # Get the x-coordinate.
        x_pos = i
        # Place the annotation near the top of the observed data range.
        y_max_data = max(sequential_robust_mse_components.max(),
                         sequential_nonrobust_var_mse_components.max(), # [Added for max calc]
                         randomized_mse_components.max()
                         )
        y_pos = y_max_data * 0.8  # Place at 80% of the maximum data value.

        # Format the annotation text.
        stats_text = f"Mean: {mean_val:.6f}\nMed: {median_val:.6f}"

        # Add a text box above the corresponding box plot.
        ax.text(x_pos, y_pos, stats_text,
                verticalalignment='top', horizontalalignment='center',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8, edgecolor='gray'),
                fontsize=9, fontweight='normal')

    # Add grid lines for readability.
    ax.grid(axis='y', linestyle='--', alpha=0.7)

    # Adjust layout.
    # plt.tight_layout() # [Commented out due to rotation and 3 boxes]
    plt.subplots_adjust(bottom=0.15, top=0.9) # [Manual adjustment]

    # --- Save box plot ---
    plot_filename = os.path.join(output_dir, f"boxplot_MSE_components_{timestamp}.png")
    plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
    print(f"MSE components box plot saved to: {plot_filename}")

    # plt.show() # Comment out previous plot show if you want to display both

    # --- Plot ATE estimates ---
    plt.figure(figsize=(12, 7)) # [Adjusted width for 3 methods]
    # Prepare ATE estimate data for plotting.
    ate_plot_data = pd.DataFrame({
        'Sequential Robust': sequential_robust_ates,
        'Sequential NonRobust Var': sequential_nonrobust_ates_var, # [Added back]
        'Randomized': randomized_ates
    })

    # Convert DataFrame to long format for seaborn.
    ate_melted = ate_plot_data.melt(var_name='Method', value_name='ATE_Estimate')

    # Create box plot.
    #sns.boxplot(data=ate_melted, x='Method', y='ATE_Estimate', palette=custom_colors) # [Use updated colors]

    sns.boxplot(data=ate_melted, x='Method', y='ATE_Estimate', hue='Method', palette=custom_colors,
                legend=False)  # [Use updated colors, hue, and legend=False]
    # Add the true ATE reference line.
    plt.axhline(y=true_ate, color='red', linestyle='--', label=f'True ATE = {true_ate:.4f}')
    plt.title('Distribution of ATE Estimates\n(Sequential Robust vs NonRobust Var vs Randomized)', fontsize=16, fontweight='bold') # [Updated title]
    plt.xlabel('Method', fontsize=14)
    plt.ylabel('ATE Estimate', fontsize=14)
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.xticks(rotation=45) # [Rotate x-axis labels for 3 methods]

    plt.tight_layout()
    ate_plot_filename = os.path.join(output_dir, f"boxplot_ATE_estimates_{timestamp}.png")
    plt.savefig(ate_plot_filename, dpi=300, bbox_inches='tight')
    print(f"ATE estimates box plot saved to: {ate_plot_filename}")

    plt.show() # Show the ATE plot

    return results

if __name__ == "__main__":
    results = main()
