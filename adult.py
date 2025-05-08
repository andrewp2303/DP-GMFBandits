from plot import *
from data import *
from policies import *
from pathlib import Path
import numpy as np
import plot


def main(
    exp_suffix="trial",
    group="RAC1P",
    density=1,  # 1 to load the full folktables dataset, smaller than one for a smaller portion.
    n_samples_per_group=5000,  # minimum number of samples for each group: smaller groups are discarded
    T=5000,
    n_seeds=1,  # number of runs of each policy (same data)
    reg_param=0.01,  # regularization parameter for the ridge regression
    noise_magnitude=0.2,  # None to use the true rewards from the dataset
    expl_coeff_oful=0.01,  # OFUL exploration coefficient. O is equivalent to Greedy.
    plot_mult=1.2,  # higher value gives a larger plot
    plot_flag=False,
    plot_compare=False,  # Flag to indicate if we're in plot-compare mode
    epsilon=0.1,
    delta=1e-2,
    L_tilde=None,
    alpha_regression=None,
    alpha_delta=1/2,    # defines delta split between regression and relative rank
    alpha_eps=1/2,      # defines epsilon split between regression and relative rank
    delta_tilde=1e-3,
    noise_type_reg="gaussian",
    noise_type_rank="zcdp",
    rescale_bound=None,  # Bound for rescaling features, None for no rescaling
):

    exp_dir = run(
        exp_suffix=exp_suffix,
        plot_mult=plot_mult,
        group=group,
        density=density,
        n_samples_per_group=n_samples_per_group,
        reg_param=reg_param,
        noise_magnitude=noise_magnitude,
        n_seeds=n_seeds,
        algo_seeds=tuple(range(n_seeds)),
        expl_coeff_oful=expl_coeff_oful,
        T=T,
        compute_density=True,
        plot_flag=plot_flag,
        plot_compare=plot_compare,
        epsilon=epsilon,
        delta=delta,
        L_tilde=L_tilde,
        alpha_regression=alpha_regression,
        alpha_delta=alpha_delta,
        alpha_eps=alpha_eps,
        delta_tilde=delta_tilde,
        noise_type_reg=noise_type_reg,
        noise_type_rank=noise_type_rank,
        rescale_bound=rescale_bound,
    )

    if plot_flag:
        plot.main(dir=f"{exp_dir}plots/", mult=plot_mult, mode_histogram="percentage")

    return exp_dir


def run(
    exp_suffix="trial",
    # problem parameters,
    problem_seed=42,
    mu_noise_level=1e-8,
    compute_density=False,
    group="RAC1P",
    density=1,
    n_samples_per_group=50000,
    poly_degree=1,
    noise_magnitude=None,
    n_seeds=1,
    # algo parameters
    reg_param=0.1,
    expl_coeff_oful=0.1,
    T=5000,  # total number of rounds
    algo_seeds=None,
    plot_mult=0.8,
    plot_flag=False,
    plot_compare=False,  # Flag to indicate if we're in plot-compare mode
    epsilon=0.1,
    delta=1e-2,
    L_tilde=None,  # bound s.t. ||X||_2^2 + ||y||_2^2 <= L_tilde^2 -- normed to 1 in data.py
    alpha_regression=None,  # confidence parameter --- usual choice is 1/T according to Shariff & Sheffet 2018
    alpha_delta=1/2,
    alpha_eps=1/2,
    delta_tilde=1e-3,
    noise_type_reg="gaussian",
    noise_type_rank="zcdp",
    rescale_bound=None,  # Bound for rescaling features, None for no rescaling
):
    assert T <= n_samples_per_group
    assert algo_seeds is not None

    P = load_adult(
        group=group,
        density=density,
        poly_degree=poly_degree,
        n_samples_per_group=n_samples_per_group,
        seed=problem_seed,
        noise_magnitude=noise_magnitude,
        rescale_bound=rescale_bound,
        L_tilde=L_tilde,
    )
    L_tilde = P.L_tilde
    # Dynamically compute L_tilde if needed
    print(f"[INFO] Retrieved L_tilde: {L_tilde:.4f}...")
    if L_tilde is None or L_tilde <= 0:
        raise ValueError("[INFO] L_tilde could not be determined")
    
    n_arms = P.n_arms


    # find trial number manually by checking directories
    nth_trial = 1
    exp_dir = f"exps/adult/eps={epsilon}_T={T}_del={delta}_ns={n_seeds}_Lt={L_tilde:.4f}_nt={noise_type_reg}_nr={noise_type_rank}_ad={alpha_delta}_ae={alpha_eps}_{exp_suffix}={nth_trial}/"
    while Path(exp_dir).exists():
        nth_trial += 1
        exp_dir = f"exps/adult/eps={epsilon}_T={T}_del={delta}_ns={n_seeds}_Lt={L_tilde:.4f}_nt={noise_type_reg}_nr={noise_type_rank}_ad={alpha_delta}_ae={alpha_eps}_{exp_suffix}={nth_trial}/"
    Path(f"{exp_dir}plots/").mkdir(parents=True, exist_ok=True)

    params = dict(
        d=P.d,
        n_arms=n_arms,
        group=group,
        n_samples_per_group=n_samples_per_group,
        poly_degree=poly_degree,
        density=density,
    )

    plot.set_figsize_dpi(figsize=[plot_mult * i for i in (4, 3.5)], dpi=200)

    # plot rewards histograms
    mode_plot_rewards = "density" if compute_density else "hist"
    plot_rewards(
        P.true_rewards.astype(np.float64),
        mode=mode_plot_rewards,
        save=True,
        dir=f"{exp_dir}plots/",
        suffix="adult_",
        plot_flag=plot_flag,
    )

    if alpha_regression is None:
        alpha_regression = 1/T
    # Check if we're running in plot-compare mode (passed from main)
    plot_compare_mode = plot_compare
    
    if plot_compare_mode:
        # When plot-compare flag is used, only run PrivateFairGreedy since that's all that will be used
        policies_generators = [
            lambda: PrivateFairGreedy(
                T=T,
                epsilon=epsilon,
                delta=delta,
                delta_tilde=delta_tilde,
                L_tilde=L_tilde,
                alpha_regression=alpha_regression,
                alpha_delta=alpha_delta,
                alpha_eps=alpha_eps,
                noise_type_reg=noise_type_reg,
                noise_type_rank=noise_type_rank,
                reg_param=reg_param,
                d=P.d,
                n_arms=n_arms,
            ),
        ]
    else:
        # Normal mode - run all policies
        policies_generators = [
            # lambda: Random(),
            lambda: OFUL(reg_param, P.d, expl_coeff_oful),
            lambda: PrivateFairGreedy(
                T=T,
                epsilon=epsilon,
                delta=delta,
                delta_tilde=delta_tilde,
                L_tilde=L_tilde,
                alpha_regression=alpha_regression,
                alpha_delta=alpha_delta,
                alpha_eps=alpha_eps,
                noise_type_reg=noise_type_reg,
                noise_type_rank=noise_type_rank,
                reg_param=reg_param,
                d=P.d,
                n_arms=n_arms,
            ),
            lambda: FairGreedy(reg_param, P.d, mu_noise_level),
        ]

    total_ps, total_dfs = [], []
    for policy_gen in policies_generators:
        ps, dfs = [], []
        for s in algo_seeds:
            np.random.seed(s)
            p, results = test_policy(policy_gen, P=P, T=T)
            results["seed"] = s
            for k, v in params.items():
                results[k] = v

            dfs.append(results)
            ps.append(p)
        # df_dict[policy_class.__name__] = dfs
        policy_name = dfs[0]["policy"].drop_duplicates()[0]

        print(f"Results for {policy_name}")
        for df in dfs:
            df.hist(
                column="actions", bins=n_arms,
            )
            break
        
        if plot_flag:
            for df in dfs:
                df.hist(column="sel_group", bins=P.n_groups)
                break
            plt.title(f"sel_group_histo_{policy_name}")
            plt.show()

            for m in ["pseudo_regret", "pseudo_fair_regret"]:
                for df in dfs:
                    df[m].plot()
                plt.title(f"{m}_{policy_name}")
                plt.show()

        for p in ps:
            print(f"mu_est_MSE = {np.mean((p.get_mu_estimate() - P.mu_star) ** 2)}")
            print(f"mu_est = {p.get_mu_estimate()}")
            print(f"mu_star = {P.mu_star}")
            break

        total_ps.extend(ps)
        total_dfs.extend(dfs)

    result_df = pd.concat(total_dfs)

    # Print start time for CSV writing
    print(f"Writing {len(result_df)} rows to CSV using pandas...")
    result_df.to_csv(f"{exp_dir}results.csv")
    print(f"results.csv saved to {exp_dir}")

    return exp_dir


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--plot", action="store_true", help="Enable plotting (default: disabled)"
    )
    parser.add_argument(
        "--plot-compare",
        nargs=2,
        metavar=("FIELD", "METRIC"),
        help="Plot PrivateFairGreedy comparison across exp_dirs for FIELD and METRIC, e.g. --plot-compare ad pseudo_regret",
    )
    args = parser.parse_args()

    exp_dirs = []
    n_samples_per_groups = (50001,)
    # epsilons = (1, 5, 15, 50,)
    epsilons = (15,)
    # deltas = (0.1, 0.01, 0.001)
    deltas = (0.1,)
    # alpha_epsilons = (0.9, 0.7, 0.5)
    alpha_epsilons = (0.9,)
    alpha_deltas = (0.9, 0.7, 0.5)
    # alpha_deltas = (0.9,)
    for n_samples_per_group in n_samples_per_groups:
        for epsilon in epsilons:
            for delta in deltas:
                for alpha_eps in alpha_epsilons:
                    for alpha_delta in alpha_deltas:
                        exp_dir = main(
                            n_samples_per_group=n_samples_per_group,
                            n_seeds=10,
                            plot_flag=args.plot,
                            plot_compare=args.plot_compare is not None,  # Set to True if --plot-compare flag is used
                            T=100,
                            epsilon=epsilon,         # total epsilon budget for DPFairGreedy
                            delta=delta,          # total delta budget for DPFairGreedy
                            L_tilde=None,       # max row norm of X+Y, will be computed based on data
                            alpha_delta=alpha_delta,    # defines delta split between regression and relative rank
                            alpha_eps=alpha_eps,      # defines epsilon split between regression and relative rank
                            delta_tilde=0.005,   # slack on the relative rank delta for advanced composition
                            noise_type_rank="zcdp",
                            rescale_bound=None,
                        )
                        exp_dirs.append(exp_dir)  # Move this inside the innermost loop

    print("\n-------------------------- OUTPUT DIRS BY VARIABLES --------------------------\n")
    # Simplified printing - just iterate through the exp_dirs list directly
    dir_index = 0
    for n_samples_per_group in n_samples_per_groups:
        for epsilon in epsilons:
            for delta in deltas:
                for alpha_eps in alpha_epsilons:
                    for alpha_delta in alpha_deltas:
                        if dir_index < len(exp_dirs):
                            print(f"Output for n_samples={n_samples_per_group}, epsilon={epsilon}, delta={delta}, (alpha_eps, alpha_delta)=({alpha_eps}, {alpha_delta}): \n{exp_dirs[dir_index]}\n")
                            dir_index += 1

    # Plot comparison if requested
    if args.plot_compare:
        field, metric = args.plot_compare
        # Validate field and metric
        valid_fields = {'eps', 'ad', 'ae', 'nr', 'del'}
        valid_metrics = {
            'pseudo_regret',
            'pseudo_fair_regret',
            'pseudo_instant_regret',
            'pseudo_instant_fair_regret',
            'exp_reward_policy',
            'exp_reward_opt',
            'fair_reward_policy',
            'fair_reward_opt',
        }
        metric_aliases = {
            'fair_pseudo_regret': 'pseudo_fair_regret',
            'regret': 'pseudo_regret',
            'fair_regret': 'pseudo_fair_regret',
        }
        if field not in valid_fields:
            print(f"Error: Field '{field}' is not valid. Choose from: {sorted(valid_fields)}")
            exit(1)
        if metric in metric_aliases:
            print(f"[INFO] Interpreting metric '{metric}' as '{metric_aliases[metric]}'")
            metric = metric_aliases[metric]
        if metric not in valid_metrics:
            print(f"Error: Metric '{metric}' is not valid. Choose from: {sorted(valid_metrics)}")
            exit(1)
        if not exp_dirs:
            print("No experiment directories found for plotting.")
        else:
            print(f"\nPlotting PrivateFairGreedy comparison for varying {field} and metric {metric}...\n")
            plot_privatefairgreedy_across_fields(exp_dirs, field=field, metric=metric, save_fig=True, fig_path=f"{exp_dir}plots/{metric}_{field}.png")