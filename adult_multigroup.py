import plot
from data_multigroup import *
from plot import *
from policies_multigroup import *
import warnings
warnings.filterwarnings("error", category=RuntimeWarning)

def main(
    exp_prefix="trial_",
    group="RAC1P",
    density=1,  # 1 to load the full folktables dataset, smaller than one for a smaller portion.
    n_samples_per_group=5000,  # minimum number of samples for each group: smaller groups are discarded
    T=2500,
    n_arms=10,
    n_seeds=1,  # number of runs of each policy (same data)
    reg_param=0.01,  # regularization parameter for the ridge regression
    noise_magnitude=0.2,  # None to use the true rewards from the dataset
    expl_coeff_oful=0.01,  # OFUL exploration coefficient. O is equivalent to Greedy.
    plot_mult=1.5,  # higher value gives a larger plot
    plot_flag=False,
    epsilon=0.1,
    delta=1e-5,
    L_tilde=1,
    noise_type="gaussian",
):

    exp_dir = (
        f"exps/adult_multi/{exp_prefix}g_{group}_na_{n_arms}_d_{density}_n_{n_samples_per_group}_nm{noise_magnitude}"
        f"_lambda_{reg_param}_T={T}_ns_{n_seeds}_ecOFUL{expl_coeff_oful}/"
    )
    run(
        exp_dir=exp_dir,
        plot_mult=plot_mult,
        group=group,
        density=density,
        n_arms=n_arms,
        n_samples_per_group=n_samples_per_group,
        reg_param=reg_param,
        noise_magnitude=noise_magnitude,
        algo_seeds=tuple(range(n_seeds)),
        expl_coeff_oful=expl_coeff_oful,
        T=T,
        compute_density=True,
        plot_flag=plot_flag,
        epsilon=0.1,
        noise_type="gaussian",
    )
    if plot_flag:
        plot.main(dir=f"{exp_dir}plots/", mult=plot_mult, mode_histogram="percentage")


def run(
    # problem parameters,
    problem_seed=42,
    mu_noise_level=1e-8,
    n_arms=10,
    compute_density=False,
    group="RAC1P",
    density=1,
    n_samples_per_group=5000,
    poly_degree=1,
    noise_magnitude=None,
    # algo parameters
    reg_param=0.1,
    expl_coeff_oful=0.1,
    T=500,  # total number of rounds
    # T = 5000,
    algo_seeds=tuple(range(10)),
    plot_mult=0.8,
    exp_dir="exps/adult_multigroup/simple/",
    plot_flag=False,
    epsilon=0.1,
    delta=1e-5,
    L_tilde=1,          # bound s.t. ||X||_2^2 + ||y||_2^2 <= L_tilde^2 -- normed to 1 in data_multigroup.py
    alpha_param=None,  # confidence parameter --- usual choice is 1/T according to Shariff & Sheffet 2018
    noise_type="gaussian",
):
    from pathlib import Path

    Path(f"{exp_dir}plots/").mkdir(parents=True, exist_ok=True)

    P = load_adult(
        group=group,
        density=density,
        poly_degree=poly_degree,
        n_samples_per_group=n_samples_per_group,
        seed=problem_seed,
        noise_magnitude=noise_magnitude,
        n_arms=n_arms,
    )
    n_arms = P.n_arms

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
        P.true_rewards,
        mode=mode_plot_rewards,
        save=True,
        dir=f"{exp_dir}plots/",
        suffix="adult_",
        plot_flag=plot_flag,
    )

    # confidence parameter --- usual choice is 1/T according to Shariff & Sheffet 2018
    if alpha_param is None:
        alpha_param = 1 / T
    policies_generators = [
        # lambda: Random(),
        lambda: OFUL(reg_param, P.d, expl_coeff_oful),
        lambda: PrivateFairGreedy(T=T, 
                                epsilon=epsilon, 
                                delta=delta, 
                                L_tilde=L_tilde, 
                                alpha_param=alpha_param, 
                                noise_type=noise_type, 
                                reg_param=reg_param, 
                                d=P.d),
        lambda: FairGreedy(reg_param, P.d, mu_noise_level),
        # lambda: Greedy(reg_param, P.d),
        # lambda: FairGreedyKnownCDF(reg_param, P.d, mu_noise_level, P),
        # lambda: FairGreedyKnownMuStar(P)
        # lambda: FairGreedyNoNoise(reg_param, P.d),
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



# NOTE: THIS IS WHERE EXPERIMENTS START
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--plot', action='store_true', help='Enable plotting (default: disabled)')
    args = parser.parse_args()
    for n_arms in (2,):
        for n_samples_per_group in (5000,):
            main(n_arms=n_arms, 
                n_samples_per_group=n_samples_per_group, 
                n_seeds=2, 
                plot_flag=args.plot,
                T=10000,
                epsilon=10000,
                delta=10,
                L_tilde=0.0001,        # for test purposes - this is max row norm of X+Y
                )
