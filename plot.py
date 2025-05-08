import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import re


def set_figsize_dpi(figsize, dpi):
    plt.rcParams["figure.figsize"] = figsize
    plt.rcParams["figure.dpi"] = dpi


def plot_rewards(
    rewards,
    mode="density",
    linestyles=("solid", "dashed", "dotted", "dashdot"),
    save=True,
    dir="",
    suffix="",
    plot_flag=True,
):
    n_arms = len(rewards[0])

    for cdf in (True, False):
        cdf_string = "CDF" if cdf else "PDF"
        for i in range(n_arms):
            if mode == "density":
                sns.kdeplot(
                    rewards[:, i],
                    label=f"Group {i + 1}",
                    linestyle=linestyles[i % len(linestyles)],
                    cumulative=cdf,
                )

            elif mode == "hist":
                plt.hist(
                    rewards[:, i], alpha=0.3, label=f"g{i}", bins=1000, cumulative=cdf
                )
            else:
                raise NotImplementedError

        plt.title(f"Rewards {cdf_string}")
        plt.ylabel("")

        plt.xlabel("Reward")
        plt.legend()
        plt.tight_layout()
        if save:
            plt.savefig(f"{dir}{suffix}true_rewards_{mode}_{cdf_string}.png")
            plt.savefig(f"{dir}{suffix}true_rewards_{mode}_{cdf_string}.pdf")

        plt.clf()

    # figure(figsize=fig_size, dpi=dpi)
    # for i in range(n_arms):
    #     plt.hist((P.true_rewards[:, i] - muTc[i]) / nu_norm[i], alpha=0.3, label=f"g{i}", bins=1000)
    # plt.title("true normalized rewards histogram")
    # plt.xlabel("normalized rewards")
    # plt.legend()
    # plt.show()


def plot_results(
    result_df,
    title_dict=dict(
        pseudo_regret="Standard pseudo-regret", pseudo_fair_regret="Fair pseudo-regret"
    ),
    policy_dict=dict(
        Random="Uniform Random",
        OFUL="OFUL",
        FairGreedy="Fair-Greedy",
        PrivateFairGreedy="Private Fair-Greedy",
    ),
    line_style_dict=dict(
        Random="dashdot",
        OFUL="dotted",
        FairGreedy="solid",
        PrivateFairGreedy="solid",
    ),
    save_fig=True,
    dir="",
    selected_policies=None,
    y_lim_dict={},
    x_lim_dict={},
    prefix_name="",
    mode_histogram="percentage",  # number or percentage
):
    selected_policies = (
        result_df["policy"].drop_duplicates().values
        if selected_policies is None
        else selected_policies
    )

    g_col = "sel_group" if "sel_group" in result_df.columns else "actions"
    n_groups = len(result_df[g_col].drop_duplicates().values)
    n_arms = len(result_df["actions"].drop_duplicates().values)

    hist_dict = {}
    if "pseudo_fair_regret" in x_lim_dict.keys():
        N_rounds = x_lim_dict["pseudo_fair_regret"][-1]
    else:
        N_rounds = None
    for p in selected_policies:
        policy_df = result_df.loc[result_df["policy"] == p]
        label = policy_dict[p] if p in policy_dict else p

        hist_dict[label] = []

        def extract_ns_value(title):
            match = re.search(r"ns=(\d+)", title)
            return int(match.group(1)) if match else None

        nseeds = extract_ns_value(dir)
        for seed in range(nseeds):
            policy_df_0 = policy_df[policy_df["seed"] == seed]
            n_rounds_max = (
                N_rounds if N_rounds is not None else len(policy_df_0[g_col].values)
            )
            if "groups" in result_df.columns:
                all_groups = np.concatenate(
                    [
                        np.array([int(a) + 1 for a in b[1:-1].split(" ")])
                        for b in policy_df_0["groups"].values[:n_rounds_max]
                    ]
                )
            else:
                all_groups = np.concatenate(
                    [np.arange(1, n_groups + 1) for i in range(n_rounds_max)]
                )

            hist = np.histogram(
                policy_df_0[g_col].values[:n_rounds_max] + 1, bins=n_groups
            )[0]
            n_groups_total = np.histogram(all_groups, bins=n_groups)[0]

            if mode_histogram == "number":
                hist_dict[label].append(hist)
            elif mode_histogram == "percentage":
                hist_dict[label].append(100 * hist / n_groups_total)
            else:
                raise NotImplementedError

        hist_dict[label] = np.concatenate(
            [g[None, :] for g in hist_dict[label]], axis=0
        )
        # plt.hist(policy_df_0['sel_group'], histtype='step', label=label, alpha=0.5)

    # plt.legend()
    fig = plt.figure()
    ax = fig.add_subplot(111)
    width = 0.2
    for i, (label, hist) in enumerate(hist_dict.items()):

        ax.bar(
            height=hist.mean(axis=0),
            x=np.array((range(1, n_groups + 1))) + width * i,
            yerr=hist.std(axis=0),
            width=width,
            label=label,
        )
        ax.set_xticks(np.array((range(1, n_groups + 1))) + width * 1.7)

    # add titles
    if mode_histogram == "number":
        max_round = result_df["round"].max() if N_rounds is None else N_rounds
        ax.set_title(f"# of selected groups at T={max_round}")
        plt.yscale("log")
    elif mode_histogram == "percentage":
        max_round = result_df["round"].max() if N_rounds is None else N_rounds
        ax.set_title(f"Percentage of selected groups at T={max_round}")
        plt.axhline(y=100.0 / n_arms, linestyle="dotted", color="black")
    else:
        raise NotImplementedError

    ax.set_xticklabels((f"G{i+1}" for i in range(n_groups)))
    # ax.legend((rects1[0], rects2[0]), ('Men', 'Women'))
    plt.legend()
    if save_fig:
        plt.savefig(f"{dir}{prefix_name}{mode_histogram}hist_group.png")
        plt.savefig(f"{dir}{prefix_name}{mode_histogram}hist_group.pdf")
    plt.show()

    for m in [
        "pseudo_regret",
        "pseudo_fair_regret",
    ]:
        for p in selected_policies:
            policy_df = result_df.loc[result_df["policy"] == p]
            policy_df_seeds = policy_df.groupby(by=["round"], as_index=False)
            # Only calculate mean for numeric columns to avoid the error
            mean_df = policy_df_seeds.mean(numeric_only=True)
            std_df = policy_df_seeds.std(numeric_only=True)

            rounds = mean_df["round"]
            metric_mean = mean_df[m]
            metric_std = std_df[m]

            label = policy_dict[p] if p in policy_dict else p
            linestyle = line_style_dict[p] if p in line_style_dict else None

            plt.plot(rounds, metric_mean, label=label, linestyle=linestyle)
            plt.fill_between(
                rounds, metric_mean - metric_std, metric_mean + metric_std, alpha=0.3
            )

        title = title_dict[m] if m in title_dict else m
        plt.title(title)
        plt.xlabel("# of rounds")
        if m in y_lim_dict:
            plt.ylim(y_lim_dict[m])
        if m in x_lim_dict:
            plt.xlim(x_lim_dict[m])
        plt.legend()
        plt.tight_layout()
        if save_fig:
            plt.savefig(f"{dir}{prefix_name}{m}.png")
            plt.savefig(f"{dir}{prefix_name}{m}.pdf")
        plt.show()
        plt.close()


def plot_privatefairgreedy_across_fields(exp_dirs, field, metric="pseudo_regret", ax=None, save_fig=False, fig_path=None):
    """
    Plot PrivateFairGreedy results across varying exp_dirs for a given field.
    Args:
        exp_dirs: list of experiment directories (str)
        field: str, parameter to vary (e.g., 'ad', 'ae', 'eps', 'nr', 'del')
        metric: str, metric to plot (e.g., 'pseudo_regret', 'pseudo_fair_regret')
        ax: matplotlib axis (optional)
        save_fig: whether to save the figure
        fig_path: where to save the figure
    """
    import pandas as pd
    import sys

    def extract_field(exp_dir, field):
        # e.g. field='ad' or 'ae' or 'eps' etc.
        match = re.search(rf"{field}=([0-9.]+)", exp_dir)
        return float(match.group(1)) if match else None

    results = []
    for exp_dir in exp_dirs:
        csv_path = f"{exp_dir.rstrip('/')}/results.csv"
        try:
            df = pd.read_csv(csv_path)
        except Exception as e:
            print(f"Could not load {csv_path}: {e}")
            continue
        val = extract_field(exp_dir, field)
        # Only keep PrivateFairGreedy
        df = df[df['policy'] == 'PrivateFairGreedy'].copy()
        df[field] = val
        results.append(df)

    if not results:
        print("No results found.")
        return
    df_all = pd.concat(results, ignore_index=True)

    # Group by field and round, average over seeds
    grouped = df_all.groupby([field, 'round']).agg({metric: ['mean', 'std']}).reset_index()
    grouped.columns = [field, 'round', f'{metric}_mean', f'{metric}_std']

    if ax is None:
        set_figsize_dpi(figsize=[6,4], dpi=200)
        fig, ax = plt.subplots(figsize=(6,4))
    else:
        fig = None

    unique_vals = sorted(grouped[field].unique())
    # Mapping for fields to Greek symbols
    field_greek_map = {
        'eps': 'ε',
        'delta': 'δ',
        'ad': 'α_δ',  # alpha sub delta
        'ae': 'α_ε',    # alpha sub epsilon
        'nr': 'nr',
        'del': 'δ',
    }
    # Mapping for metrics to human-readable names
    metric_name_map = {
        'pseudo_regret': 'Pseudo-Regret',
        'pseudo_fair_regret': 'Fair Pseudo-Regret',
        'regret': 'Regret',
        'fair_regret': 'Fair Regret',
    }
    field_label = field_greek_map.get(field, field)
    metric_label = metric_name_map.get(metric, metric)

    for val in unique_vals:
        sub = grouped[grouped[field] == val]
        ax.plot(sub['round'], sub[f'{metric}_mean'], label=f"{field_label}={val}")
        ax.fill_between(sub['round'],
                        sub[f'{metric}_mean'] - sub[f'{metric}_std'],
                        sub[f'{metric}_mean'] + sub[f'{metric}_std'],
                        alpha=0.2)
    ax.set_title(f"PrivateFairGreedy: Varying {field_label}")
    ax.set_xlabel("Round")
    ax.set_ylabel(metric_label)
    ax.legend()
    plt.tight_layout()
    if save_fig and fig_path:

        plt.savefig(fig_path)
        print(f"Saved figure to {fig_path}")
    if fig is not None:
        plt.show()
        sys.exit()


def main_adult(dir="", mult=1, x_dim=4, y_dim=3.5):
    selected = [
        # "Random",
        "OFUL",
        "FairGreedy",
        "PrivateFairGreedy",
    ]
    # y_lim_dict = dict(pseudo_regret=[-1, 120], pseudo_fair_regret=[-1, 60])
    # x_lim_dict = dict(pseudo_regret=[-5, 2500], pseudo_fair_regret=[-5, 2500])
    prefix_name = "adult_"
    main(
        dir=dir,
        mult=mult,
        dpi=200,
        save_fig=True,
        selected_policies=selected,
        # y_lim_dict=y_lim_dict,
        prefix_name=prefix_name,
        # x_lim_dict=x_lim_dict,
        x_dim=x_dim,
        y_dim=y_dim,
    )


def main(mult=0.8, dpi=200, save_fig=True, dir="", x_dim=4, y_dim=3.5, **kwargs):
    set_figsize_dpi(figsize=[mult * i for i in (x_dim, y_dim)], dpi=dpi)
    import pandas as pd

    result_df = pd.read_csv(f"{dir}../results.csv")

    plot_results(result_df, save_fig=save_fig, dir=dir, **kwargs)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--plot-path', type=str, default="None", help='Directory containing plots and results.csv')
    args = parser.parse_args()
    if args.plot_path != "None":
        main_adult(dir=f"{args.plot_path}/plots/", x_dim=5, y_dim=3.8)

    exp_dirs_eps = [f"exps/adult/eps={eps}_T=50000_del=0.1_ns=10_Lt=21.6746_nt=gaussian_nr=zcdp_ad=0.9_ae=0.9_trial=1/" for eps in [1, 5, 15, 50]]
    exp_dirs_delta = [f"exps/adult/eps=15_T=50000_del={delta}_ns=10_Lt=21.6746_nt=gaussian_nr=zcdp_ad=0.9_ae=0.9_trial=1/" for delta in [0.1, 0.01, 0.001]]
    exp_dirs_alpha_delta = [f"exps/adult/eps=15_T=50000_del=0.1_ns=10_Lt=21.6746_nt=gaussian_nr=zcdp_ad={alpha_delta}_ae=0.9_trial=1/" for alpha_delta in [0.9, 0.7, 0.5]]
    exp_dirs_alpha_eps = [f"exps/adult/eps=15_T=50000_del=0.1_ns=10_Lt=21.6746_nt=gaussian_nr=zcdp_ad=0.9_ae={alpha_eps}_trial=1/" for alpha_eps in [0.9, 0.7, 0.5]]
    plot_privatefairgreedy_across_fields(exp_dirs_eps, field="eps", metric="pseudo_fair_regret", ax=None, save_fig=True, fig_path="exps/adult/varying_eps.png")
    # plot_privatefairgreedy_across_fields(exp_dirs_delta, field="del", metric="pseudo_fair_regret", ax=None, save_fig=True, fig_path="exps/adult/varying_delta.png")
    # plot_privatefairgreedy_across_fields(exp_dirs_alpha_delta, field="ad", metric="pseudo_fair_regret", ax=None, save_fig=True, fig_path="exps/adult/varying_ad.png")
    # plot_privatefairgreedy_across_fields(exp_dirs_alpha_eps, field="ae", metric="pseudo_fair_regret", ax=None, save_fig=True, fig_path="exps/adult/varying_ae.png")
