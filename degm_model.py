"""
DEGM: Decision Error Generative Model
- Simulation of decisions
- MLE estimation of parameters (kappa, beta_b, T0, tau, sigma)
- Calibration and plots
- Counterfactual error rates
Usage:
    python -m degm.degm_model --n 6000 --seed 42 --save
"""
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

try:
    from scipy.optimize import minimize
    SCIPY_OK = True
except Exception:
    SCIPY_OK = False

# ---------- Ground truth container ----------
class TrueParams:
    def __init__(self, kappa=0.75, beta_b=1.10, T0=0.70, tau=0.60, sigma=0.55):
        self.kappa = kappa
        self.beta_b = beta_b
        self.T0 = T0
        self.tau = tau
        self.sigma = sigma


def make_data(N: int, params: TrueParams, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    Delta = rng.normal(loc=1.0, scale=1.0, size=N)
    x = rng.normal(size=N)  # social pull toward wrong option
    z = rng.binomial(n=1, p=0.5, size=N)  # time pressure

    b = params.beta_b * x
    T = params.T0 + params.tau * z
    D = np.sqrt(T**2 + np.pi * params.sigma**2 / 8.0)

    eff = (params.kappa * Delta - b) / D
    p_correct = 1.0 / (1.0 + np.exp(-eff))
    y = rng.binomial(n=1, p=p_correct, size=N)

    return pd.DataFrame({
        "Delta": Delta,
        "x": x,
        "z_time_pressure": z,
        "T": T,
        "D": D,
        "p_correct": p_correct,
        "y": y
    })


# ---------- Likelihood ----------
def _softplus(t):
    return np.log1p(np.exp(t))

def neg_log_lik(theta, Delta, x, z, y):
    # theta = [kappa_raw, beta_b, T0_raw, tau_raw, sigma_raw]
    kappa = _softplus(theta[0])
    beta_b = theta[1]
    T0 = _softplus(theta[2])
    tau = _softplus(theta[3])
    sigma = _softplus(theta[4])

    T = T0 + tau * z
    D = np.sqrt(T**2 + np.pi * sigma**2 / 8.0)
    eff = (kappa * Delta - beta_b * x) / D
    ll = y * (-np.logaddexp(0.0, -eff)) + (1 - y) * (-np.logaddexp(0.0, eff))
    return -np.sum(ll)


def fit_mle(df: pd.DataFrame, theta0=None):
    Delta = df["Delta"].values
    x = df["x"].values
    z = df["z_time_pressure"].values
    y = df["y"].values

    if theta0 is None:
        theta0 = np.array([np.log(np.exp(0.5)-1), 0.5, np.log(np.exp(0.5)-1), np.log(np.exp(0.3)-1), np.log(np.exp(0.4)-1)])

    if SCIPY_OK:
        res = minimize(
            fun=neg_log_lik, x0=theta0, args=(Delta, x, z, y),
            method="L-BFGS-B", options={"maxiter": 400, "ftol": 1e-9}
        )
        theta_hat = res.x
        converged = res.success
        status = res.message
    else:
        # rough random search fallback
        converged = False
        status = "Scipy not available; using rough random search."
        best_val = np.inf
        theta_hat = theta0.copy()
        rng = np.random.default_rng(0)
        for _ in range(2500):
            cand = theta_hat + rng.normal(scale=0.1, size=theta_hat.shape)
            val = neg_log_lik(cand, Delta, x, z, y)
            if val < best_val:
                best_val = val
                theta_hat = cand

    def decode(theta):
        kappa = _softplus(theta[0])
        beta_b = theta[1]
        T0 = _softplus(theta[2])
        tau = _softplus(theta[3])
        sigma = _softplus(theta[4])
        return kappa, beta_b, T0, tau, sigma

    return decode(theta_hat), converged, status


# ---------- Plots ----------
def save_fig(path):
    plt.tight_layout()
    plt.savefig(path, bbox_inches="tight")
    plt.close()

def make_plots(df, params_hat, outdir="outputs"):
    kappa_hat, beta_b_hat, T0_hat, tau_hat, sigma_hat = params_hat
    Delta = df["Delta"].values
    x = df["x"].values
    z = df["z_time_pressure"].values
    y = df["y"].values

    # Calibration
    T_hat = T0_hat + tau_hat * z
    D_hat = np.sqrt(T_hat**2 + np.pi * sigma_hat**2 / 8.0)
    eff_hat = (kappa_hat * Delta - beta_b_hat * x) / D_hat
    p_hat = 1.0 / (1.0 + np.exp(-eff_hat))

    bins = np.linspace(0,1,11)
    bin_ids = np.digitize(p_hat, bins) - 1
    calib = []
    for b in range(10):
        m = bin_ids == b
        if np.any(m):
            calib.append(( (bins[b]+bins[b+1])/2.0, p_hat[m].mean(), y[m].mean() ))
    calib = np.array(calib)
    plt.figure()
    plt.plot(calib[:,1], calib[:,2], marker="o", linestyle="-", label="Calibration")
    plt.plot([0,1],[0,1], linestyle="--", label="Perfect")
    plt.xlabel("Predicted accuracy")
    plt.ylabel("Empirical accuracy")
    plt.title("Calibration curve")
    plt.legend()
    save_fig(os.path.join(outdir, "calibration.png"))

    # Distributions
    plt.figure()
    plt.hist(df["Delta"], bins=40, alpha=0.6, label="Δ")
    plt.hist(df["x"], bins=40, alpha=0.6, label="x")
    plt.xlabel("Value"); plt.ylabel("Frequency"); plt.title("Distributions: Δ and x"); plt.legend()
    save_fig(os.path.join(outdir, "distributions.png"))

    # Error vs Delta
    p_err_hat = 1.0 / (1.0 + np.exp(eff_hat))
    q = np.quantile(Delta, np.linspace(0,1,11))
    bins_d = np.digitize(Delta, q[1:-1], right=True)
    er = []
    for b in range(10):
        m = bins_d == b
        if np.any(m):
            er.append(( Delta[m].mean(), (1-y[m]).mean(), p_err_hat[m].mean() ))
    er = np.array(er)
    plt.figure()
    plt.plot(er[:,0], er[:,1], marker="o", linestyle="-", label="Empirical error")
    plt.plot(er[:,0], er[:,2], marker="s", linestyle="--", label="Predicted error")
    plt.xlabel("Decision difficulty (Δ)"); plt.ylabel("Error probability"); plt.title("Error vs Δ"); plt.legend()
    save_fig(os.path.join(outdir, "error_vs_delta.png"))

    # Accuracy by time pressure
    acc0 = y[z==0].mean() if np.any(z==0) else np.nan
    acc1 = y[z==1].mean() if np.any(z==1) else np.nan
    plt.figure()
    plt.bar([0,1],[acc0,acc1])
    plt.xticks([0,1], ["z=0 (no time pressure)", "z=1 (time pressure)"])
    plt.ylim(0,1); plt.ylabel("Accuracy"); plt.title("Accuracy under time pressure")
    save_fig(os.path.join(outdir, "accuracy_by_time_pressure.png"))

    # Partial dependence for each lever
    def mean_error(kappa=None, beta_b=None, T0=None, tau=None, sigma=None):
        k = kappa if kappa is not None else kappa_hat
        b = beta_b if beta_b is not None else beta_b_hat
        t0 = T0 if T0 is not None else T0_hat
        t = tau if tau is not None else tau_hat
        s = sigma if sigma is not None else sigma_hat
        Tcf = t0 + t * z
        Dcf = np.sqrt(Tcf**2 + np.pi * s**2 / 8.0)
        perr = 1.0 / (1.0 + np.exp((k * Delta - b * x) / Dcf))
        return perr.mean()

    grid = np.linspace(max(0.2, kappa_hat*0.4), min(1.5, kappa_hat*1.6), 25)
    errs = [mean_error(kappa=g) for g in grid]
    plt.figure(); plt.plot(grid, errs, marker="o"); plt.xlabel("κ"); plt.ylabel("Mean error"); plt.title("Error vs κ")
    save_fig(os.path.join(outdir, "partial_kappa.png"))

    grid = np.linspace(beta_b_hat*0.2, beta_b_hat*1.8, 25)
    errs = [mean_error(beta_b=g) for g in grid]
    plt.figure(); plt.plot(grid, errs, marker="o"); plt.xlabel("β_b"); plt.ylabel("Mean error"); plt.title("Error vs β_b")
    save_fig(os.path.join(outdir, "partial_beta_b.png"))

    grid = np.linspace(0.4, 1.6, 25)
    errs = [mean_error(T0=T0_hat*s, tau=tau_hat*s) for s in grid]
    plt.figure(); plt.plot(grid, errs, marker="o"); plt.xlabel("T-scale"); plt.ylabel("Mean error"); plt.title("Error vs T-scale")
    save_fig(os.path.join(outdir, "partial_Tscale.png"))

    grid = np.linspace(max(0.05, sigma_hat*0.3), sigma_hat*2.0, 25)
    errs = [mean_error(sigma=g) for g in grid]
    plt.figure(); plt.plot(grid, errs, marker="o"); plt.xlabel("σ"); plt.ylabel("Mean error"); plt.title("Error vs σ")
    save_fig(os.path.join(outdir, "partial_sigma.png"))


def counterfactual_table(df, params_hat, out_csv="data/degm_counterfactuals.csv"):
    kappa_hat, beta_b_hat, T0_hat, tau_hat, sigma_hat = params_hat
    Delta = df["Delta"].values
    x = df["x"].values
    z = df["z_time_pressure"].values

    def mean_error(kappa=None, beta_b=None, T0=None, tau=None, sigma=None):
        k = kappa if kappa is not None else kappa_hat
        b = beta_b if beta_b is not None else beta_b_hat
        t0 = T0 if T0 is not None else T0_hat
        t = tau if tau is not None else tau_hat
        s = sigma if sigma is not None else sigma_hat
        Tcf = t0 + t * z
        Dcf = np.sqrt(Tcf**2 + np.pi * s**2 / 8.0)
        perr = 1.0 / (1.0 + np.exp((k * Delta - b * x) / Dcf))
        return perr.mean()

    baseline = mean_error()
    kappa_up = mean_error(kappa=kappa_hat*1.2)
    bias_half = mean_error(beta_b=beta_b_hat*0.5)
    T_down = mean_error(T0=T0_hat*0.7, tau=tau_hat*0.7)
    sigma_down = mean_error(sigma=sigma_hat*0.7)

    df_cf = pd.DataFrame({
        "Scenario": [
            "Baseline (estimated)",
            "Increase information efficiency (kappa x1.2)",
            "Reduce social bias (beta_b x0.5)",
            "Lower time pressure (T0,tau x0.7)",
            "Lower internal noise (sigma x0.7)"
        ],
        "Mean error probability": [baseline, kappa_up, bias_half, T_down, sigma_down]
    })
    df_cf.to_csv(out_csv, index=False)
    return df_cf


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=6000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--save", action="store_true", help="save plots/tables to disk")
    ap.add_argument("--no-plots", action="store_true")
    ap.add_argument("--no-counterf", action="store_true")
    ap.add_argument("--csv-only", action="store_true")
    args = ap.parse_args()

    os.makedirs("data", exist_ok=True)
    os.makedirs("outputs", exist_ok=True)

    true = TrueParams()
    df = make_data(args.n, true, seed=args.seed)

    # Fit
    theta0 = None
    params_hat, converged, status = fit_mle(df, theta0)
    kappa_hat, beta_b_hat, T0_hat, tau_hat, sigma_hat = params_hat

    # Save parameter comparison
    summary = pd.DataFrame({
        "Parameter": ["kappa", "beta_b", "T0", "tau", "sigma"],
        "True": [true.kappa, true.beta_b, true.T0, true.tau, true.sigma],
        "Estimated": [kappa_hat, beta_b_hat, T0_hat, tau_hat, sigma_hat]
    })
    if args.save or args.csv_only:
        summary.to_csv("data/degm_parameters_summary.csv", index=False)

    # Counterfactuals
    if not args.no_counterf:
        cf = counterfactual_table(df, params_hat, out_csv="data/degm_counterfactuals.csv")

    # Plots
    if not args.no_plots and not args.csv_only:
        make_plots(df, params_hat, outdir="outputs")

    # Regret summary
    T_true = true.T0 + true.tau * df["z_time_pressure"].values
    D_true = np.sqrt(T_true**2 + np.pi * true.sigma**2 / 8.0)
    p_err_true = 1.0 / (1.0 + np.exp( (true.kappa * df["Delta"].values - true.beta_b * df["x"].values) / D_true ))
    avg_regret_true = (df["Delta"].values * p_err_true).mean()

    T_hat = T0_hat + tau_hat * df["z_time_pressure"].values
    D_hat = np.sqrt(T_hat**2 + np.pi * sigma_hat**2 / 8.0)
    p_err_hat = 1.0 / (1.0 + np.exp( (kappa_hat * df["Delta"].values - beta_b_hat * df["x"].values) / D_hat ))
    avg_regret_hat = (df["Delta"].values * p_err_hat).mean()

    print("Converged:", converged, "|", status)
    print("Estimated params:", params_hat)
    print("Average expected regret (true):", float(avg_regret_true))
    print("Average expected regret (estimated):", float(avg_regret_hat))
    if args.save or args.csv_only:
        print("Saved tables in data/. Figures in outputs/.")

if __name__ == "__main__":
    main()
