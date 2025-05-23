import folktables
import numpy as np
import pandas as pd
from folktables import ACSDataSource
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, PolynomialFeatures, MinMaxScaler

from policies import FairBanditProblem


class RealData(FairBanditProblem):
    def get_context(self):
        self.t += 1
        return self.xs[self.t]

    def get_noisy_reward(self, x, a):
        if self.noise_magnitude is not None:
            return (
                np.dot(x, self.mu_star) + self.noise_magnitude * np.random.randn(1)[0]
            )
        return self.ys[self.t][a]

    def get_reward(self, x, a):
        return np.dot(x, self.mu_star)

    def reset(self):
        self.t = 0
        shuffled_indexes = np.random.choice(range(len(self.ys)), size=len(self.ys))
        self.xs = self.xs[shuffled_indexes]
        self.ys = self.ys[shuffled_indexes]

    def __init__(self, xs, ys, mu_star, noise_magnitude=None, L_tilde=None):
        self.t = -1
        self.n_arms = xs.shape[1]
        self.d = xs.shape[2]
        self.xs = xs
        self.ys = ys
        self.noise_magnitude = noise_magnitude
        self.L_tilde = L_tilde
        true_rewards = np.einsum("ijk,k->ij", self.xs, mu_star)
        
        print(f"[INFO] Number of features: {self.d}")
        super().__init__(n_arms=xs.shape[1], mu_star=mu_star, true_rewards=true_rewards)


def get_dataset_string(group, density, n_samples_per_group, seed, poly_degree):
    return f"group_{group}den_{density}_n_{n_samples_per_group}_seed_{seed}_pd_{poly_degree}"


def load_adult(
    group="SEX",
    density=0.1,
    poly_degree=1,
    n_samples_per_group=50000,
    seed=42,
    data_dir="data/adult_proc/",
    noise_magnitude=None,
    rescale_bound=None,
    L_tilde=None,
):
    data_str = get_dataset_string(
        group, density, n_samples_per_group, seed, poly_degree
    )
    data_pre_dir = f"{data_dir}{data_str}/"
    data_dict = None
    
    # Check if rescaled data is requested and exists
    if rescale_bound is not None:
        try:
            data_dict = np.load(f"{data_pre_dir}data_rescaled_b={rescale_bound}.npz")
            print(f"Loading rescaled data with bound [-{rescale_bound},{rescale_bound}]")
        except:
            print(f"Rescaled data with bound [-{rescale_bound},{rescale_bound}] not found. Will create it.")
            
    # If rescaled data wasn't loaded, try to load original data
    if data_dict is None:
        try:
            data_dict = np.load(f"{data_pre_dir}data.npz")
            
            # If we need rescaled data but only have original data, create the rescaled version
            if rescale_bound is not None:
                print(f"Creating rescaled data with bound [-{rescale_bound},{rescale_bound}]")
                data_dict = rescale_data(data_dict, rescale_bound, data_pre_dir)
                
        except:
            print("Preprocessed data not found. Start preprocessing adult...")
            preprocess_folktables(
                group, density, poly_degree, n_samples_per_group, seed, data_dir
            )
            
            # After preprocessing, load the data
            data_dict = np.load(f"{data_pre_dir}data.npz")
            
            # If rescaling is needed, do it now
            if rescale_bound is not None:
                print(f"Creating rescaled data with bound [-{rescale_bound},{rescale_bound}]")
                data_dict = rescale_data(data_dict, rescale_bound, data_pre_dir)
    
    # Compute appropriate L_tilde if not manually set
    if L_tilde is None or L_tilde < 0:
        max_norm_sq = 0
        for x_group, y_group in zip(data_dict["x"], data_dict["y"]):
            for x_vec, y_val in zip(x_group, y_group):
                total_norm_sq = np.linalg.norm(x_vec) ** 2 + y_val ** 2
                if total_norm_sq > max_norm_sq:
                    max_norm_sq = total_norm_sq
        estimated_L_tilde = np.sqrt(max_norm_sq)
        print(f"[INFO] Computed L_tilde based on data: {estimated_L_tilde:.4f}")
    else:
        estimated_L_tilde = L_tilde

    return RealData(
        data_dict["x"],
        data_dict["y"],
        data_dict["mu_star"],
        noise_magnitude=noise_magnitude,
        L_tilde=estimated_L_tilde,
    )


def preprocess_folktables(
    group="SEX",
    density=0.1,
    poly_degree=1,
    n_samples_per_group=50000,
    seed=42,
    data_dir="data/adult_proc/",
):
    from pathlib import Path

    data_str = get_dataset_string(
        group, density, n_samples_per_group, seed, poly_degree
    )
    data_pre_dir = f"{data_dir}{data_str}/"
    Path(data_pre_dir).mkdir(parents=True, exist_ok=True)
    np.random.seed(seed)

    def adult_filter_cat(data):
        """Mimic the filters in place for Adult data.

        Adult documentation notes: Extraction was done by Barry Becker from
        the 1994 Census database. A set of reasonably clean records was extracted
        using the following conditions:
        ((AAGE>16) && (AGI>100) && (AFNLWGT>1)&& (HRSWK>0))
        """
        df = data
        df = df[df["AGEP"] > 16]
        df = df[df["PINCP"] > 100]
        df = df[df["WKHP"] > 0]
        df = df[df["PWGTP"] >= 1]

        columns = [
            "AGEP",
            "COW",
            "SCHL",
            "MAR",
            "POBP",  # place of birth, should be categorical
            "RELP",
            "OCCP",  # Occupation, should be categorical but has high number of values
            "WKHP",  # hours per week
            "SEX",
            "RAC1P",
            "PINCP",
        ]
        df = df[columns]
        for c in columns:
            print(f"{c} n_vals = {len(df[c].drop_duplicates())}")

        # https://www2.census.gov/programs-surveys/acs/tech_docs/pums/data_dict/PUMS_Data_Dictionary_2014-2018.pdf
        categorical_columns = [
            # 'COW',  #CLASS OF WORKER
            # 'SCHL', # EDUCATION,  almost ordered
            "MAR",  # MARITAL STATUS
            # "RELP",  # Relationship
            "SEX",  # Sex
            "RAC1P",  # RACE
        ]

        for column in categorical_columns:
            tempdf = pd.get_dummies(df[column], prefix=column)
            df = pd.merge(left=df, right=tempdf, left_index=True, right_index=True,)
        # print(df)

        return df

    ACSIncomeREG = folktables.BasicProblem(
        features=[
            "AGEP",
            "COW",
            # *[f'COW_{i+1}' for i in range(8)],
            "SCHL",
            # *[f'SCHL_{i}' for i in range(24)],
            # 'MAR',
            *[f"MAR_{i + 1}" for i in range(5)],
            # *[f'OCCP_{i}' for i in range(18)],
            "OCCP",
            "POBP",
            # *[f"RELP_{i + 1}" for i in range(17)],
            'RELP',
            "WKHP",
            # 'SEX',
            *[f"SEX_{i + 1}" for i in range(2)],
            # 'RAC1P',
            *[f"RAC1P_{i + 1}" for i in range(9)],
        ],
        target="PINCP",
        target_transform=lambda x: x,
        group=group,
        preprocess=adult_filter_cat,
        postprocess=lambda x: np.nan_to_num(x, -1),
    )

    def get_data_equal_groups(
        survey_year="2017",
        horizon="1-Year",
        survey="person",
        density=0.1,
        n_samples_per_group=50000,
        process_x=None,
        process_y=None,
    ):

        data_source = ACSDataSource(
            survey_year=survey_year, horizon=horizon, survey=survey
        )
        ca_data = data_source.get_data(download=True, density=density)
        x, y, group = ACSIncomeREG.df_to_numpy(ca_data)

        if process_x is not None:
            x = process_x(x)
        if process_y is not None:
            y = process_y(y)

        # add bias feature
        x_bias = np.ones((x.shape[0], x.shape[1] + 1))
        x_bias[:, 1:] = x
        x = x_bias

        group_numbers = np.unique(group)

        selected_groups = []
        for i in group_numbers:
            n_samples = np.sum((group == i).astype(np.float64))
            if n_samples >= n_samples_per_group:
                selected_groups.append(i)

        x_b = np.zeros((n_samples_per_group, len(selected_groups), x.shape[1]))
        y_b = np.zeros((n_samples_per_group, len(selected_groups)))
        g_b = np.zeros((n_samples_per_group, len(selected_groups)))

        for i in range(len(selected_groups)):
            sg = selected_groups[i]
            yg, xg, gg = y[group == sg], x[group == sg], group[group == sg]
            selected = np.random.choice(range(len(yg)), size=n_samples_per_group)
            x_b[:, i, :] = xg[selected]
            y_b[:, i] = yg[selected]
            g_b[:, i] = gg[selected]

        x = x_b.reshape(-1, x.shape[1])
        y = y_b.reshape(-1)
        group = g_b.reshape(-1)

        return x, y, group, x_b, y_b, selected_groups

    # Use 2017 data to set the scaler
    # Use data from 2018 to train the optimal mu

    x_pre, y_pre, _, _, _, _ = get_data_equal_groups(
        survey_year="2017",
        horizon="1-Year",
        survey="person",
        n_samples_per_group=n_samples_per_group,
        density=density,
    )

    poly_feat_proc = PolynomialFeatures(degree=poly_degree)
    scaler_x, scaler_y = StandardScaler(), StandardScaler()

    x_pre = scaler_x.fit_transform(poly_feat_proc.fit_transform(x_pre[:, 1:]))
    y_pre = scaler_y.fit_transform(y_pre.reshape(-1, 1)).reshape(-1)

    x, y, group, x_b, y_b, selected_groups = get_data_equal_groups(
        survey_year="2018",
        horizon="1-Year",
        survey="person",
        n_samples_per_group=n_samples_per_group,
        density=density,
        process_x=lambda i: scaler_x.transform(poly_feat_proc.transform(i)),
        process_y=lambda i: scaler_y.transform(i.reshape(-1, 1)).reshape(-1),
    )

    # Reward histograms
    y_groups = [y[group == i] for i in selected_groups]
    for i, yg in enumerate(y_groups):
        sg = selected_groups[i]
        n_samples = np.sum((group == sg).astype(np.float64))
        plt.hist(
            yg,
            cumulative=True,
            label=f"group {sg} #{n_samples}",
            bins=1000,
            alpha=0.3,
            density=True,
        )
    plt.title("reward density")
    plt.legend()
    plt.show()

    n_features = x.shape[1]
    print(f"n_features = {n_features}")
    x_train, x_test, y_train, y_test, group_train, group_test = train_test_split(
        x, y, group, test_size=0.2, random_state=seed, shuffle=True
    )

    mu_star = np.linalg.solve(
        (x_train.transpose() @ x_train) + 1e-8 * np.eye(n_features),
        x_train.transpose() @ y_train,
    )

    np.savez(
        f"{data_pre_dir}data.npz",
        mu_star=mu_star,
        x=x_b,
        y=y_b,
        selected_groups=np.array(selected_groups),
    )

    y_hat_train = x_train @ mu_star
    errors_train = y_hat_train - y_train
    mean_error_train = np.mean(errors_train ** 2)
    print(f"MSE_OLS_train, {mean_error_train}")
    print(f"MSE_0_predictor_train, {np.mean((y_train - np.zeros_like(y_train)) ** 2)}")

    y_hat_test = x_test @ mu_star
    errors_test = y_hat_test - y_test
    mean_error_test = np.mean(errors_test ** 2)
    print(f"MSE_OLS_test, {mean_error_test}")
    print(f"MSE_0_predictor_test, {np.mean((y_test - np.zeros_like(y_test)) ** 2)}")

    plt.hist(errors_train ** 2, label="train", alpha=0.3, bins=1000)
    plt.hist(errors_test ** 2, label="test", alpha=0.3, bins=1000)
    plt.xlim([-0.1, 12])
    plt.legend()
    plt.show()

    plt.hist(
        (y_train - np.zeros_like(y_train) ** 2), label="0_train", alpha=0.3, bins=100
    )
    plt.hist((y_test - np.zeros_like(y_test) ** 2), label="0_test", alpha=0.3, bins=100)
    plt.xlim([-0.1, 12])
    plt.legend()
    plt.show()


def rescale_data(data_dict, bound, data_pre_dir):
    """Rescale the data to the range [-bound, bound] feature-by-feature.
    
    Args:
        data_dict: Dictionary containing 'x', 'y', 'mu_star', and 'selected_groups'
        bound: The bound value for rescaling
        data_pre_dir: Directory to save the rescaled data
        
    Returns:
        Rescaled data dictionary
    """
    x = data_dict['x']
    y = data_dict['y']
    mu_star = data_dict['mu_star']
    selected_groups = data_dict['selected_groups']
    
    # Get original shapes to restore later
    original_x_shape = x.shape
    original_y_shape = y.shape
    
    # Reshape for rescaling (we need to rescale each feature separately)
    # x shape is (n_samples, n_groups, n_features)
    x_reshaped = x.reshape(-1, x.shape[2])  # Reshape to (n_samples*n_groups, n_features)
    
    # Now scale the features
    x_scaler = MinMaxScaler(feature_range=(-bound, bound))
    x_rescaled = x_scaler.fit_transform(x_reshaped)
    
    # Reshape x back to original shape
    x_rescaled = x_rescaled.reshape(original_x_shape)
    
    # Rescale y (output) to [-bound, bound]
    y_reshaped = y.reshape(-1, 1)  # Reshape to (n_samples*n_groups, 1)
    y_scaler = MinMaxScaler(feature_range=(-bound, bound))
    y_rescaled = y_scaler.fit_transform(y_reshaped)
    y_rescaled = y_rescaled.reshape(original_y_shape)
    
    # We need to adjust mu_star to account for the rescaling
    # This ensures that x @ mu_star still approximates y after rescaling
    # We can solve for new mu_star using least squares on the rescaled data
    x_flat = x_rescaled.reshape(-1, x_rescaled.shape[2])
    y_flat = y_rescaled.reshape(-1)
    mu_star_rescaled = np.linalg.solve(
        (x_flat.transpose() @ x_flat) + 1e-8 * np.eye(x_flat.shape[1]),
        x_flat.transpose() @ y_flat,
    )
    
    # Save the rescaled data
    np.savez(
        f"{data_pre_dir}data_rescaled_b={bound}.npz",
        mu_star=mu_star_rescaled,
        x=x_rescaled,
        y=y_rescaled,
        selected_groups=selected_groups
    )
    
    # Return the rescaled data dictionary
    return {
        'mu_star': mu_star_rescaled,
        'x': x_rescaled,
        'y': y_rescaled,
        'selected_groups': selected_groups
    }

if __name__ == "__main__":
    # Test the rescaling functionality
    load_adult(
        group="RAC1P",
        density=1,
        poly_degree=1,
        n_samples_per_group=5002,
        seed=42,
        rescale_bound=1
    )