# Simulation Study

# %%
import math
import numpy as np
import functions as fn
import torch
from torch import Tensor
import math
import torch.nn.functional as F
import pyvinecopulib as pv
import pandas as pd

#%%
SEEDS = list(range(10))
FAMILIES = ["clayton", "gumbel", "gaussian", "student"]
TAUS = [0.3, 0.6, 0.9]
NS = [100, 500, 1000, 5000]

num_sims = len(SEEDS) * len(FAMILIES) * len(TAUS) * len(NS)

def q_of_n(n: int) -> float:
    return n ** (-0.5)

def copula_params(family: str, tau: float, nu: int = 4):
    if family == "clayton":
        theta = 2 * tau / (1 - tau)
        return {"theta": theta}
    if family == "gumbel":
        theta = 1 / (1 - tau)
        return {"theta": theta}
    if family in {"gaussian", "student"}:
        rho = math.sin(math.pi * tau / 2)
        out = {"rho": rho}
        if family == "student":
            out["nu"] = nu
        return out
    raise ValueError(f"Unknown family: {family}")

def set_bicop(family: str, params: dict) -> Tensor:

    
    if family == "clayton":
        theta = params["theta"]
        return pv.Bicop(
                    family=pv.BicopFamily.clayton, rotation = 0,
                    parameters=np.array([[theta]], dtype=float),
                )
    if family == "gumbel":
        theta = params["theta"]
        return pv.Bicop(
                    family=pv.BicopFamily.gumbel, rotation = 0,
                    parameters=np.array([[theta]], dtype=float),
                )
    if family == "gaussian":
        rho = params["rho"]
        return pv.Bicop(
                    family=pv.BicopFamily.gaussian,
                    parameters=np.array([[rho]], dtype=float),
                )
    if family == "student":
        rho = params["rho"]
        nu = params["nu"]
        return pv.Bicop(
            family=pv.BicopFamily.student,
            parameters=np.array([[rho], [nu]], dtype=float),
        )
    raise ValueError(f"Unknown family: {family}")
# %%

# Set up a 50x50 grid

# Integration grid
grid_size = 50
eps_grid = 1e-4
u_1d = torch.linspace(eps_grid, 1 - eps_grid, grid_size)
u_grid = torch.cartesian_prod(u_1d, u_1d)
cell_area = (u_1d[1] - u_1d[0]) ** 2
z_grid = fn.qnorm(u_grid)


# %%
results = []
completed = 0

for seed in SEEDS:
    # Set the seed for reproducibility
    torch.manual_seed(seed)

    for family in FAMILIES:
        for tau in TAUS:
            for n in NS:
                 
                # Simulate data
                bicop = set_bicop(family, copula_params(family, tau=tau))

                u_data_np = bicop.simulate(n, seeds = [seed])
                u_data = torch.tensor(u_data_np, dtype=torch.float64)

                z_data = fn.qnorm(u_data.clamp(1e-6, 1 - 1e-6))
                
                q = n ** (-1/2)  # lower-tail cutoff

                # Tail integration grid
                u_tail_grid = u_grid * q
                cell_area_tail = (q ** 2) * cell_area
                z_tail_grid = fn.qnorm(u_tail_grid)

                # --------------------------------------------------------------------------------------------------------------------------------------------- #

                ## True values
                c_true = torch.tensor(bicop.pdf(u_tail_grid.numpy()))
                r_true = q * c_true
                p_true = bicop.cdf(np.array([[q, q]]))[0]
                h_true = (q**2 / torch.tensor(p_true)) * c_true

                # --------------------------------------------------------------------------------------------------------------------------------------------- #

                ## Approach 1: Standard TLL estimator
                B_body = fn._select_bandwidth_constant(z_data)
                f_body = fn.fit_local_likelihood_constant(z_tail_grid.to(torch.float64), z_data.to(torch.float64), B_body)
                phi_body = fn.dnorm(z_tail_grid[:, 0]) * fn.dnorm(z_tail_grid[:, 1])

                # Estimation targets
                c_body = f_body / phi_body                       # copula density in the tail
                p_body = (c_body.sum() * cell_area_tail).item()  # probability mass in lower tail
                r_body = q * c_body                              # tail copula density
                h_body = (q**2 / p_body) * c_body                # conditional density on [0, 1]^2
                
                # --------------------------------------------------------------------------------------------------------------------------------------------- #

                ## Approach 2: Tail-adaptive TLL estimator

                # Extract data from the tail and get the tail mass
                u_tail_mask = (u_data[:, 0] <= q) & (u_data[:, 1] <= q)
                u_data_tail = u_data[u_tail_mask]
                k = u_data_tail.shape[0]
                
                if k < 5:
                    continue  # skip if too few tail observations for estimation

                # Rescale lower-left tail block to [0,1]^2
                s_data = (u_data_tail / q).clamp(1e-6, 1 - 1e-6)
                s_grid = (u_tail_grid / q).clamp(1e-6, 1 - 1e-6)

                # Transform rescaled tail observations
                zs_data = fn.qnorm(s_data)
                zs_grid = fn.qnorm(s_grid)

                # Bandwidth selection
                B_tail = (k ** (-2.0 / 6.0)) * fn.sample_covariance_2d(zs_data) + 1e-6 * torch.eye(
                            2, dtype=zs_data.dtype, device=zs_data.device
                )
                # B_tail = fn._select_bandwidth_constant(zs_data)

                # Fit on tail-rescaled data
                f_tail = fn.fit_local_likelihood_constant(
                            zs_grid.to(torch.float64),
                            zs_data.to(torch.float64),
                            B_tail,
                )
                phi_tail = fn.dnorm(zs_grid[:, 0]) * fn.dnorm(zs_grid[:, 1])

                # Estimation targets
                p_tail = k / n                        # probability mass in lower tail
                h_tail = f_tail / phi_tail            # conditional density on [0, 1]^2
                c_tail = (p_tail / q**2) * h_tail     # copula density approximation in the tail
                r_tail = q * c_tail                   # tail copula density

                # --------------------------------------------------------------------------------------------------------------------------------------------- #

                ## Evaluation
                ests = {
                    "Ordinary": {
                        "r": r_body,
                        "h": h_body,
                        "c": c_body,
                    },
                    "Tail": {
                        "r": r_tail,
                        "h": h_tail,
                        "c": c_tail,
                    },
                }

                truths = {
                    "r": r_true,
                    "h": h_true,
                    "c": c_true,
                }

                cell_areas = {
                    "r": cell_area,
                    "h": cell_area,
                    "c": cell_area_tail,
                }


                for model, model_ests in ests.items():
                    for target, est in model_ests.items():
                        metrics = fn.grid_metrics_density(
                            est=est,
                            truth=truths[target],
                            cell_area=cell_areas[target],
                        )
                        results.append({
                            "seed": seed,
                            "family": family,
                            "tau": tau,
                            "n": n,
                            "k": k,
                            "q": q,
                            "model": model,
                            "target": target,
                            **metrics,
                        })

                p_results = {
                    "Ordinary": p_body,
                    "Tail": p_tail,
                }

                for model, p_hat in p_results.items():
                    results.append({
                        "seed": seed,
                        "family": family,
                        "tau": tau,
                        "n": n,
                        "k": k,
                        "q": q,
                        "model": model,
                        "target": "p",
                        "p_hat": p_hat,
                        "p_true": p_true,
                        "AE": abs(p_hat - p_true),
                        "RE": abs(p_hat - p_true) / p_true,
                    })

                completed += 1
                pct = 100 * completed / num_sims
                print(
                    f"[{completed:3d}/{num_sims}] "
                    f"{pct:6.2f}% | "
                    f"{family:8s} | tau={tau:.1f} | n={n:4d} | seed={seed}"
                )

# %%
results_df = pd.DataFrame(results)
print(results_df)
results_df.to_csv("simulation_results_tll_bandwidth.csv", index=False)
# %%
B_tail = fn._select_bandwidth_constant(zs_data)
print(B_tail)
L = torch.linalg.cholesky(B_tail)
# %%
test_cop = copula_params("student", 0.6)
print(test_cop)
bicop = set_bicop("student", test_cop)
print(bicop)

# %%
n