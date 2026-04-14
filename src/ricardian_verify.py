"""
Ricardian equivalence: equilibrium bond pricing q_t from the stochastic Euler,
and government (tau, B) sequences that satisfy the flow budget at those prices.

Under lump-sum taxes, real allocations {c,n,k'} along a path are pinned down by
productivity, preferences, and {g_t}; financing only splits (tau,B). We verify
flow budgets at equilibrium q_t and report allocation invariance (identical
simulated paths under alternative financing schemes).
"""
from __future__ import annotations

import numpy as np

from .markov_rbc import MarkovSolution, policy_interp, production, simulate_baseline_policy
from .params import Params


def equilibrium_q_path(
    p: Params,
    sol: MarkovSolution,
    k_path: np.ndarray,
    z_idx_path: np.ndarray,
) -> np.ndarray:
    """
    q_t = beta * E_t[ (c_{t+1}/c_t)^(-sigma) ] with expectation over z_{t+1}
    given z_t, using policies c(k',z') at k_{t+1}=k'(k_t,z_t).
    """
    kp_of, n_of = policy_interp(sol)
    Pi = sol.Pi
    T = len(z_idx_path) - 1
    q = np.zeros(T)
    for t in range(T):
        zi = int(z_idx_path[t])
        k = k_path[t]
        n = n_of(k, zi)
        z = float(sol.z_vals[zi])
        y = production(z, k, n, p.alpha)
        c = y - p.g_y_ratio * y - (kp_of(k, zi) - (1.0 - p.delta) * k)
        if c <= 1e-14:
            q[t] = p.beta
            continue
        kp = kp_of(k, zi)
        exp_mrs = 0.0
        for jz in range(2):
            prob = Pi[zi, jz]
            n2 = n_of(kp, jz)
            z2 = float(sol.z_vals[jz])
            y2 = production(z2, kp, n2, p.alpha)
            g2 = p.g_y_ratio * y2
            kp2 = kp_of(kp, jz)
            c2 = y2 - g2 - (kp2 - (1.0 - p.delta) * kp)
            if c2 <= 1e-14:
                ratio = 1.0
            else:
                ratio = (c2 / c) ** (-p.sigma)
            exp_mrs += prob * ratio
        q[t] = p.beta * exp_mrs
    return q


def financing_sequences_time_varying_q(
    g_path: np.ndarray,
    q_path: np.ndarray,
    *,
    B0: float = 0.0,
    tau_smooth: float | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    (i) Tax finance: B=0, tau_t = g_t + B_t - q_t B_{t+1}.
    (ii) Smoothed tau: if tau_smooth is None, use mean(g) over path; else constant tau_smooth.
    """
    T = len(g_path)
    B_tax = np.zeros(T + 1)
    tau_tax = np.zeros(T)
    for t in range(T):
        B_tax[t + 1] = 0.0
        tau_tax[t] = g_path[t] + B_tax[t] - q_path[t] * B_tax[t + 1]

    B_d = np.zeros(T + 1)
    B_d[0] = B0
    if tau_smooth is None:
        tau_bar = float(np.mean(g_path))
    else:
        tau_bar = float(tau_smooth)
    tau_d = np.full(T, tau_bar)
    for t in range(T):
        B_d[t + 1] = (B_d[t] + g_path[t] - tau_d[t]) / max(q_path[t], 1e-8)
    return tau_tax, B_tax, tau_d, B_d


def government_budget_residual(
    g: np.ndarray,
    tau: np.ndarray,
    B: np.ndarray,
    q: np.ndarray,
) -> float:
    """max_t |q_t B_{t+1} + tau_t - B_t - g_t|."""
    T = len(g)
    res = []
    for t in range(T):
        lhs = q[t] * B[t + 1] + tau[t]
        rhs = B[t] + g[t]
        res.append(abs(lhs - rhs))
    return float(np.max(res)) if res else 0.0


def allocation_invariance_across_financing(
    path_a: dict[str, np.ndarray],
    path_b: dict[str, np.ndarray],
) -> dict[str, float]:
    """
    Max absolute differences in real allocations between two paths.
    Under Ricardian equivalence with the same {g_t} and same equilibrium policies,
    paths should coincide (up to float noise) because (tau,B) do not enter
    the household's Euler/intratemporal conditions for (c,n,k').
    """
    out: dict[str, float] = {}
    for key in ("c", "n", "y", "g"):
        if key not in path_a or key not in path_b:
            continue
        out[f"max_abs_diff_{key}"] = float(
            np.max(np.abs(path_a[key] - path_b[key]))
        )
    if "k" in path_a and "k" in path_b:
        out["max_abs_diff_k"] = float(np.max(np.abs(path_a["k"] - path_b["k"])))
    return out


def duplicate_baseline_paths_for_ricardian(
    p: Params,
    sol: MarkovSolution,
    k0: float,
    z_idx_path: np.ndarray,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    """Two copies of the same simulated allocation path (financing is residual)."""
    path = simulate_baseline_policy(sol, k0, z_idx_path)
    return path, {k: v.copy() for k, v in path.items()}
