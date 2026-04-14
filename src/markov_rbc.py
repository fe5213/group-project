"""
Stochastic Markov productivity RBC with government spending share g_y * Y.

Infinite-horizon rational expectations: value function iteration on a capital grid
for discrete productivity z in {z_L, z_H} with transition matrix Pi.

Budget: c + k' = Y - g + (1-delta)k,  g = g_y * Y, lump-sum balance (B=0).

Solver: continuous optimization over k' with intratemporal labor n solved by root-finding
(FOC), plus policy-based convergence and refined fixed points k'(k,z)=k.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
from scipy.interpolate import interp1d
from scipy.optimize import brentq

from .params import Params


@dataclass
class MarkovSolution:
    """Markov RBC solution on (k_grid, z_idx)."""

    k_grid: np.ndarray
    V: np.ndarray  # (nk, 2)
    k_policy: np.ndarray  # (nk, 2)
    n_policy: np.ndarray  # (nk, 2)
    c_policy: np.ndarray  # (nk, 2)
    y_policy: np.ndarray  # (nk, 2)
    g_policy: np.ndarray  # (nk, 2)
    Pi: np.ndarray
    z_vals: np.ndarray
    p: Params
    iterations: int
    sup_V_diff: float
    sup_kp_diff: float  # terminal max |k_pol_new - k_pol_old| on grid


def _u(c: float, sigma: float) -> float:
    if c <= 1e-14:
        return -1e12
    if abs(sigma - 1.0) < 1e-10:
        return np.log(c)
    return (c ** (1.0 - sigma) - 1.0) / (1.0 - sigma)


def _v_disutil(n: float, chi: float, phi: float) -> float:
    return chi * (n ** (1.0 + phi)) / (1.0 + phi)


def production(z: float, k: float, n: float, alpha: float) -> float:
    return z * (k**alpha) * (n ** (1.0 - alpha))


def budget_c(
    z: float,
    k: float,
    n: float,
    kp: float,
    p: Params,
    *,
    g_y_effective: float | None = None,
    g_level: float | None = None,
) -> float:
    y = production(z, k, n, p.alpha)
    if g_level is not None:
        g = g_level
    else:
        gy = p.g_y_ratio if g_y_effective is None else g_y_effective
        g = gy * y
    return y - g - (kp - (1.0 - p.delta) * k)


def expect_V(
    kp: float,
    z_idx: int,
    V_interp: list[Callable[[float], float]],
    Pi: np.ndarray,
) -> float:
    return float(Pi[z_idx, 0] * V_interp[0](kp) + Pi[z_idx, 1] * V_interp[1](kp))


def expect_V_np(
    kp: float,
    z_idx: int,
    k_grid: np.ndarray,
    V: np.ndarray,
    Pi: np.ndarray,
) -> float:
    """Fast continuation value using np.interp (no scipy interp1d overhead)."""
    v0 = float(np.interp(kp, k_grid, V[:, 0]))
    v1 = float(np.interp(kp, k_grid, V[:, 1]))
    return float(Pi[z_idx, 0] * v0 + Pi[z_idx, 1] * v1)


def build_V_interp(
    k_grid: np.ndarray, V_col: np.ndarray
) -> Callable[[float], float]:
    return interp1d(
        k_grid,
        V_col,
        kind="linear",
        bounds_error=False,
        fill_value=(float(V_col[0]), float(V_col[-1])),
    )


def _intratemporal_residual_n(
    n: float,
    z: float,
    k: float,
    kp: float,
    p: Params,
    *,
    g_y_eff: float | None,
    g_level: float | None,
) -> float:
    """FOC residual: chi n^phi - w c^{-sigma}."""
    n = float(n)
    if n <= 1e-6 or n >= 1.0 - 1e-6:
        return 1e6
    y = production(z, k, n, p.alpha)
    if g_level is not None:
        g = float(g_level)
    else:
        assert g_y_eff is not None
        g = g_y_eff * y
    inv = kp - (1.0 - p.delta) * k
    c = y - g - inv
    if c <= 1e-14:
        return 1e6
    w = (1.0 - p.alpha) * z * (k**p.alpha) * (n ** (-p.alpha))
    return p.chi * (n**p.phi) - w * (c ** (-p.sigma))


def solve_n_given_kp(
    z: float,
    k: float,
    kp: float,
    p: Params,
    *,
    g_y_eff: float | None = None,
    g_level: float | None = None,
    n_grid_pts: int = 24,
) -> float:
    """
    Labor n maximizing u(c)-v(n) given (z,k,k') (continuation depends only on kp).
    Vectorized grid + optional brentq polish.
    """
    lo, hi = 0.07, 0.95
    nv = np.linspace(lo, hi, n_grid_pts)
    y = z * (k**p.alpha) * (nv ** (1.0 - p.alpha))
    inv = kp - (1.0 - p.delta) * k
    if g_level is not None:
        c = y - float(g_level) - inv
    else:
        assert g_y_eff is not None
        g = g_y_eff * y
        c = y - g - inv
    mask = c > 1e-14
    flow = np.full_like(nv, -1e18, dtype=float)
    c_pos = c[mask]
    nv_pos = nv[mask]
    if c_pos.size:
        if abs(p.sigma - 1.0) < 1e-10:
            u_part = np.log(np.maximum(c_pos, 1e-14))
        else:
            u_part = (c_pos ** (1.0 - p.sigma) - 1.0) / (1.0 - p.sigma)
        v_part = p.chi * (nv_pos ** (1.0 + p.phi)) / (1.0 + p.phi)
        flow[mask] = u_part - v_part
    idx = int(np.argmax(flow))
    best_n = float(nv[idx])
    a = max(lo, best_n - 0.07)
    b = min(hi, best_n + 0.07)
    fa = _intratemporal_residual_n(a, z, k, kp, p, g_y_eff=g_y_eff, g_level=g_level)
    fb = _intratemporal_residual_n(b, z, k, kp, p, g_y_eff=g_y_eff, g_level=g_level)
    if fa * fb < 0:
        try:
            return float(
                brentq(
                    lambda nn: _intratemporal_residual_n(
                        nn, z, k, kp, p, g_y_eff=g_y_eff, g_level=g_level
                    ),
                    a,
                    b,
                    xtol=1e-7,
                    rtol=1e-7,
                )
            )
        except ValueError:
            pass
    return best_n


def maximize_bellman_continuous(
    z: float,
    k: float,
    z_idx: int,
    p: Params,
    Pi: np.ndarray,
    V_interp: list[Callable[[float], float]] | None,
    *,
    g_y_effective: float,
    k_min: float,
    k_max: float,
    n_kp_grid: int = 24,
    k_grid: np.ndarray | None = None,
    V_arr: np.ndarray | None = None,
) -> tuple[float, float, float, float, float, float]:
    """Maximize Bellman over k' with continuous n; dense k' grid + local refinement."""
    y_cap = production(z, k, 0.99, p.alpha)
    k_low = max((1.0 - p.delta) * k * 0.92, k_min * 0.5)
    k_high = min(
        y_cap * (1.0 - g_y_effective) + (1.0 - p.delta) * k,
        k_max * 1.05,
    )
    if k_high <= k_low + 1e-8:
        k_high = k_low + 0.05

    def ev_at(kp: float) -> float:
        if k_grid is not None and V_arr is not None:
            return expect_V_np(kp, z_idx, k_grid, V_arr, Pi)
        assert V_interp is not None
        return expect_V(kp, z_idx, V_interp, Pi)

    def bellman_at_kp(kp: float) -> float:
        n = solve_n_given_kp(z, k, kp, p, g_y_eff=g_y_effective, g_level=None)
        c = budget_c(z, k, n, kp, p, g_y_effective=g_y_effective)
        if c <= 1e-14:
            return -1e18
        ev = ev_at(kp)
        return _u(c, p.sigma) - _v_disutil(n, p.chi, p.phi) + p.beta * ev

    ks = np.linspace(k_low, k_high, n_kp_grid)
    best_val = -1e18
    best_kp = float(ks[len(ks) // 2])
    for kp in ks:
        kp = float(kp)
        v = bellman_at_kp(kp)
        if v > best_val:
            best_val, best_kp = v, kp
    # Local golden refinement on [k_a, k_b] around best grid point
    idx = int(np.argmin(np.abs(ks - best_kp)))
    i0 = max(0, idx - 1)
    i1 = min(len(ks) - 1, idx + 1)
    a, b = float(ks[i0]), float(ks[i1])
    if b <= a + 1e-12:
        b = min(k_high, a + 0.02)
    gr = 0.6180339887498949
    c1 = b - (b - a) * gr
    c2 = a + (b - a) * gr
    f1, f2 = bellman_at_kp(c1), bellman_at_kp(c2)
    for _ in range(6):
        if f1 >= f2:
            b, c2, f2 = c2, c1, f1
            c1 = b - (b - a) * gr
            f1 = bellman_at_kp(c1)
        else:
            a, c1, f1 = c1, c2, f2
            c2 = a + (b - a) * gr
            f2 = bellman_at_kp(c2)
    kp_star = float(0.5 * (c1 + c2))
    n_star = solve_n_given_kp(z, k, kp_star, p, g_y_eff=g_y_effective, g_level=None)
    c_star = budget_c(z, k, n_star, kp_star, p, g_y_effective=g_y_effective)
    y_star = production(z, k, n_star, p.alpha)
    g_star = g_y_effective * y_star
    val_star = bellman_at_kp(kp_star)
    return val_star, kp_star, n_star, c_star, y_star, g_star


def solve_markov_stationary(
    p: Params,
    *,
    nk: int = 42,
    tol_v: float = 5e-5,
    tol_kp: float = 5e-5,
    max_iter: int = 800,
    relax: float = 1.0,
) -> MarkovSolution:
    z_vals = np.array([p.z_L, p.z_H], dtype=float)
    Pi = np.array(p.Pi, dtype=float)

    from .steady_state import solve_steady_state

    z_bar, _ = p.ergodic_z()
    ss0 = solve_steady_state(p, z=z_bar)
    k_ss = ss0.k
    k_lo = max(0.03 * k_ss, 1e-4)
    k_hi = 4.0 * k_ss
    k_grid = np.linspace(k_lo, k_hi, nk)

    V = np.zeros((nk, 2))
    k_pol_old = np.zeros((nk, 2))
    for iz in range(2):
        z = z_vals[iz]
        for ik, k in enumerate(k_grid):
            y = production(z, k, 0.33, p.alpha)
            c = max(y * (1.0 - p.g_y_ratio) - p.delta * k * 0.5, 1e-6)
            V[ik, iz] = _u(c, p.sigma) / (1.0 - p.beta)
            k_pol_old[ik, iz] = (1.0 - p.delta) * k

    it = 0
    rel_v = 1.0
    sup_diff_kp = 1.0
    sup_diff_v = 1.0
    while it < max_iter:
        it += 1
        V_new = np.zeros_like(V)
        k_pol_new = np.zeros_like(k_pol_old)

        for iz in range(2):
            z = z_vals[iz]
            for ik, k in enumerate(k_grid):
                val, kp, _, _, _, _ = maximize_bellman_continuous(
                    z,
                    k,
                    iz,
                    p,
                    Pi,
                    None,
                    g_y_effective=p.g_y_ratio,
                    k_min=k_lo,
                    k_max=k_hi,
                    k_grid=k_grid,
                    V_arr=V,
                )
                V_new[ik, iz] = val
                k_pol_new[ik, iz] = kp

        sup_diff_v = float(np.max(np.abs(V_new - V)))
        sup_diff_kp = float(np.max(np.abs(k_pol_new - k_pol_old)))
        scale_v = max(float(np.max(np.abs(V_new))), 1.0)
        rel_v = sup_diff_v / scale_v
        V = relax * V_new + (1.0 - relax) * V
        k_pol_old = k_pol_new.copy()
        if rel_v < tol_v and sup_diff_kp < tol_kp:
            break

    k_pol = np.zeros_like(V)
    n_pol = np.zeros_like(V)
    c_pol = np.zeros_like(V)
    y_pol = np.zeros_like(V)
    g_pol = np.zeros_like(V)

    for iz in range(2):
        z = z_vals[iz]
        for ik, k in enumerate(k_grid):
            _, kp, n, c, y, g = maximize_bellman_continuous(
                z,
                k,
                iz,
                p,
                Pi,
                None,
                g_y_effective=p.g_y_ratio,
                k_min=k_lo,
                k_max=k_hi,
                k_grid=k_grid,
                V_arr=V,
            )
            k_pol[ik, iz] = kp
            n_pol[ik, iz] = n
            c_pol[ik, iz] = c
            y_pol[ik, iz] = y
            g_pol[ik, iz] = g

    return MarkovSolution(
        k_grid=k_grid,
        V=V,
        k_policy=k_pol,
        n_policy=n_pol,
        c_policy=c_pol,
        y_policy=y_pol,
        g_policy=g_pol,
        Pi=Pi,
        z_vals=z_vals,
        p=p,
        iterations=it,
        sup_V_diff=sup_diff_v,
        sup_kp_diff=sup_diff_kp,
    )


def policy_interp(sol: MarkovSolution) -> tuple[Callable, Callable]:
    kg = sol.k_grid

    def kp_of(k: float, z_idx: int) -> float:
        k = float(np.clip(k, kg[0], kg[-1]))
        return float(np.interp(k, kg, sol.k_policy[:, z_idx]))

    def n_of(k: float, z_idx: int) -> float:
        k = float(np.clip(k, kg[0], kg[-1]))
        return float(np.interp(k, kg, sol.n_policy[:, z_idx]))

    return kp_of, n_of


def simulate_baseline_policy(
    sol: MarkovSolution,
    k0: float,
    z_path: np.ndarray,
) -> dict[str, np.ndarray]:
    """Roll forward using interpolated stationary policy (Markov z_path length T+1)."""
    T = len(z_path) - 1
    p = sol.p
    kp_of, n_of = policy_interp(sol)
    k = np.zeros(T + 1)
    n = np.zeros(T)
    c = np.zeros(T)
    y = np.zeros(T)
    g = np.zeros(T)
    k[0] = k0
    for t in range(T):
        zi = int(z_path[t])
        z = float(sol.z_vals[zi])
        nt = n_of(k[t], zi)
        y_t = production(z, k[t], nt, p.alpha)
        g_t = p.g_y_ratio * y_t
        kp = kp_of(k[t], zi)
        c_t = y_t - g_t - (kp - (1.0 - p.delta) * k[t])
        n[t] = nt
        y[t] = y_t
        g[t] = g_t
        c[t] = c_t
        k[t + 1] = kp
    return {"k": k, "n": n, "c": c, "y": y, "g": g}


def _kp_minus_k(k: float, sol: MarkovSolution, z_idx: int) -> float:
    kp_of, _ = policy_interp(sol)
    return kp_of(k, z_idx) - float(k)


def find_k_fixed_point_refined(sol: MarkovSolution, z_idx: int) -> float:
    """Continuous root of k'(k,z)=k on [k_grid[0], k_grid[-1]]."""
    kg = sol.k_grid
    k_lo, k_hi = float(kg[0]), float(kg[-1])
    vals = np.array([_kp_minus_k(k, sol, z_idx) for k in kg])
    # Find sign change
    for i in range(len(kg) - 1):
        a, b = float(kg[i]), float(kg[i + 1])
        fa, fb = vals[i], vals[i + 1]
        if fa == 0:
            return a
        if fa * fb < 0:
            try:
                return float(brentq(lambda kk: _kp_minus_k(kk, sol, z_idx), a, b, xtol=1e-12))
            except ValueError:
                break
    # Fallback: grid argmin
    diff = np.abs(sol.k_policy[:, z_idx] - kg)
    return float(kg[int(np.argmin(diff))])


def find_k_fixed_point_grid(sol: MarkovSolution, z_idx: int) -> float:
    """Alias: refined fixed point (keeps API for callers)."""
    return find_k_fixed_point_refined(sol, z_idx)


def euler_residual_stats(sol: MarkovSolution) -> dict[str, float]:
    """Stochastic Euler: relative |lhs-rhs|/|lhs| at solved grid policies (no interpolation)."""
    p = sol.p
    Pi = sol.Pi
    res = []
    for iz in range(2):
        z = sol.z_vals[iz]
        for ik, k in enumerate(sol.k_grid):
            n = float(sol.n_policy[ik, iz])
            kp = float(sol.k_policy[ik, iz])
            c = budget_c(z, k, n, kp, p, g_y_effective=p.g_y_ratio)
            if c <= 1e-12:
                continue
            lhs = c ** (-p.sigma)
            rhs = 0.0
            for jz in range(2):
                prob = Pi[iz, jz]
                z2 = sol.z_vals[jz]
                n2 = float(np.interp(kp, sol.k_grid, sol.n_policy[:, jz]))
                kpp = float(np.interp(kp, sol.k_grid, sol.k_policy[:, jz]))
                c2 = budget_c(z2, kp, n2, kpp, p, g_y_effective=p.g_y_ratio)
                if c2 <= 1e-12:
                    continue
                r2 = p.alpha * z2 * (kp ** (p.alpha - 1.0)) * (n2 ** (1.0 - p.alpha))
                R = 1.0 - p.delta + r2
                rhs += prob * p.beta * (c2 ** (-p.sigma)) * R
            res.append(abs(lhs - rhs) / max(abs(lhs), 1e-12))
    arr = np.asarray(res, dtype=float)
    out = {
        "max_abs_euler_pct": float(np.max(arr)) if len(arr) else 0.0,
        "mean_abs_euler_pct": float(np.mean(arr)) if len(arr) else 0.0,
        "p95_abs_euler_pct": float(np.percentile(arr, 95)) if len(arr) else 0.0,
    }
    return out


def resource_residual_stats(sol: MarkovSolution) -> dict[str, float]:
    """max |c+g+k' - (1-d)k - Y| / max(|Y|,1e-12) on grid."""
    p = sol.p
    kp_of, n_of = policy_interp(sol)
    rel = []
    for iz in range(2):
        z = sol.z_vals[iz]
        for ik, k in enumerate(sol.k_grid):
            n = n_of(k, iz)
            kp = kp_of(k, iz)
            y = production(z, k, n, p.alpha)
            g = p.g_y_ratio * y
            c = budget_c(z, k, n, kp, p, g_y_effective=p.g_y_ratio)
            rc = c + g + (kp - (1.0 - p.delta) * k) - y
            rel.append(abs(rc) / max(abs(y), 1e-12))
    arr = np.asarray(rel, dtype=float)
    return {
        "max_resource_rel": float(np.max(arr)) if len(arr) else 0.0,
        "mean_resource_rel": float(np.mean(arr)) if len(arr) else 0.0,
    }


def solve_markov_stationary_alt_gy(p: Params, g_y_ratio: float, **kwargs) -> MarkovSolution:
    from dataclasses import replace

    return solve_markov_stationary(replace(p, g_y_ratio=g_y_ratio), **kwargs)
