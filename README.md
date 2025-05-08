# DP-GMFBandits

This repository contains the code for our experiments on a differentially private, group meritocratically fair linear contextual bandit algorithm. These experiments focus on comparing our algorithm (PrivateFairGreedy) with two baseline algorithms (Fair-Greedy and OFUL) on real-world US Census data.

## What are group meritocratic fairness and differential privacy?

This project builds on [the repository](https://github.com/CSML-IIT-UCL/GMFbandits) and *Fair-Greedy* algorithm of [Grazzi et al. (NeurIPS 2022)](https://arxiv.org/abs/2206.03150), which introduces **group meritocratic fairness (GMF)** in contextual linear bandits. GMF promotes fairness by selecting candidates based on their *relative rank* within their sensitive group — e.g., how well someone performs compared to others of the same race or gender. This approach addresses the challenge that candidates' rewards may not be directly comparable between groups, for example, when some groups have lower rewards due to discriminatory bias and/or social injustice.

We extend this framework with **differential privacy (DP)** to ensure that sensitive information about any single individual — such as their features or observed reward — cannot be inferred from the algorithm’s behavior. Since pure DP is too strict for interactive learning settings like bandits, we instead adopt **joint differential privacy (JDP)**, which protects each round’s data by requiring that changes to one candidate’s information have a quantifiably small effect on the actions taken in all other rounds. This is crucial in applications like hiring, where future decisions can inadvertently leak information about past individuals. Our *Private-Fair-Greedy* algorithm modifies *Fair-Greedy* by incorporating differentially private mechanisms for both regression and relative rank estimation, yielding a policy that is provably (ε, δ)-JDP while also learning group meritocratic fairness over time.

## Setup

To get started, install dependencies using:

```bash
pip install -r requirements.txt
````

## Running Experiments

The main entry point is `adult.py`, which runs bandit experiments on preprocessed [Folktables](https://github.com/zykls/folktables) ACS Census data.

```bash
python adult.py
```

### Key experimental parameters

* `n_samples_per_group`: Minimum number of samples per sensitive group (Must be > T)
* `n_seeds`: Number of random seeds for experiment runs (Higher `n_seeds`: slower, higher confidence)
* `T`: Number of rounds
* `epsilon`, `delta`: total privacy budget
* `alpha_eps`, `alpha_delta`: allocation of epsilon and delta between regression and relative rank estimation
* `delta_tilde`: Slack on the relative rank delta --- only used for advanced composition in rank estimation
* `noise_type_rank`: Noise type for rank estimation ("zcdp" or "approx")
* `group`: Sensitive census data attribute to define groups (e.g., `"RAC1P"`, `"SEX"`)

All experimental outputs are saved in an `exps/` subdirectory with automatic trial indexing and metadata-encoded paths (`adult.py` will never overwrite previous experimental outputs).

### Example Output Directory

```text
exps/adult/eps=10_T=5000_del=0.1_ns=1_Lt=1.3762_nt=gaussian_nr=zcdp_ad=0.9_ae=0.9_trial=1/
├── results.csv
└── plots/
    ├── adult_true_rewards_density_CDF.png
    ├── adult_pseudo_fair_regret.pdf
    ├── adult_percentagehist_group.pdf
    └── ...
```

## Repository Structure

* `adult.py`: Main experimental script for running and evaluating contextual bandits on real data.
* `data.py`: Data loading and preprocessing pipeline using Folktables (ACS PUMS 2017-2018).
* `policies.py`: Core algorithm implementations (FairGreedy, PrivateFairGreedy, OFUL, ridge regression, ...).
* `plot.py`: Evaluation and visualization of regret and group fairness metrics.
* `requirements.txt`: Python dependencies.
* `data/`: (Unpopulated) directory to store preprocessed ACS data files. Automatically populated on first run.
* `exps/`: (Unpopulated) directory to store experimental outputs, plots, and CSV results.
* `paper/`: (Optional) folder to include the paper or manuscript using this codebase.
