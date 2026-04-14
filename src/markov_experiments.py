"""
Fiscal experiments: Markov productivity + RE continuation from stationary V(k,z).

- Unforeseen one-time level shock to g (additive), MC over future z paths.
- Foreseen one-time shock H quarters ahead: finite-horizon backward DP on the same k-grid.
- Permanent g/y increase: new stationary Markov solution; Monte Carlo averaging.

Shock sizing: for cross-state comparisons, the same absolute Delta g is used (scaled from
ergodic-mean productivity steady state) unless dg_absolute is omitted.
"""
from __future__ import annotations

from dataclasses import replace

import numpy as np

from .markov_rbc import (
    MarkovSolution,
    budget_c,
    euler_residual_stats,
    expect_V_np,
    find_k_fixed_point_grid,
    maximize_bellman_continuous,
    policy_interp,
    production,
    simulate_baseline_policy,
    solve_markov_stationary,
    solve_markov_stationary_alt_gy,
    solve_n_given_kp,
    _u,
    _v_disutil,
)
from .params import Params
from .steady_state import solve_steady_state


def optimal_one_period_fixed_g(
    p: Params,
    sol: MarkovSolution,
    k: float,
    z_idx: int,
    *,
    g_level: float,
) -> tuple[float, float, float, float]:
    """Returns (k_next, n, c, y). Continuous optimization over k'."""
    z = float(sol.z_vals[z_idx])
    Pi = sol.Pi
    k_lo = sol.k_grid[0]
    k_hi = sol.k_grid[-1]
    y_max = production(z, k, 0.99, p.alpha)
    k_low = max((1.0 - p.delta) * k * 0.92, k_lo * 0.5)
    k_high = min(y_max - g_level + (1.0 - p.delta) * k, k_hi * 1.05)
    if k_high <= k_low + 1e-8:
        k_high = k_low + 0.05

    def bell_at(kp: float) -> float:
        n = solve_n_given_kp(z, k, kp, p, g_level=g_level)
        c = budget_c(z, k, n, kp, p, g_level=g_level)
        if c <= 1e-14:
            return -1e18
        ev = expect_V_np(kp, z_idx, sol.k_grid, sol.V, Pi)
        return _u(c, p.sigma) - _v_disutil(n, p.chi, p.phi) + p.beta * ev

    ks = np.linspace(k_low, k_high, 48)
    best_kp, best_v = float(ks[24]), -1e18
    for kp in ks:
        v = bell_at(float(kp))
        if v > best_v:
            best_v, best_kp = v, float(kp)
    idx = int(np.argmin(np.abs(ks - best_kp)))
    i0, i1 = max(0, idx - 1), min(len(ks) - 1, idx + 1)
    a, b = float(ks[i0]), float(ks[i1])
    if b <= a + 1e-12:
        b = min(k_high, a + 0.02)
    gr = 0.6180339887498949
    c1, c2 = b - (b - a) * gr, a + (b - a) * gr
    f1, f2 = bell_at(c1), bell_at(c2)
    for _ in range(20):
        if f1 >= f2:
            b, c2, f2 = c2, c1, f1
            c1 = b - (b - a) * gr
            f1 = bell_at(c1)
        else:
            a, c1, f1 = c1, c2, f2
            c2 = a + (b - a) * gr
            f2 = bell_at(c2)
    kp_star = float(0.5 * (c1 + c2))
    n_star = solve_n_given_kp(z, k, kp_star, p, g_level=g_level)
    c_star = budget_c(z, k, n_star, kp_star, p, g_level=g_level)
    y_star = production(z, k, n_star, p.alpha)
    return kp_star, n_star, c_star, y_star


def _bellman_on_grid_gy(
    p: Params,
    sol: MarkovSolution,
    W_next: np.ndarray,
    *,
    g_y_eff: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """One Bellman step with g = g_y_eff * Y."""
    nk = len(sol.k_grid)
    W_new = np.zeros((nk, 2))
    k_pol = np.zeros((nk, 2))
    n_pol = np.zeros((nk, 2))
    c_pol = np.zeros((nk, 2))
    y_pol = np.zeros((nk, 2))
    Pi = sol.Pi
    k_lo = sol.k_grid[0]
    k_hi = sol.k_grid[-1]

    for iz in range(2):
        z = float(sol.z_vals[iz])
        for ik, k in enumerate(sol.k_grid):
            val, kp, n, c, y, _g = maximize_bellman_continuous(
                z,
                k,
                iz,
                p,
                Pi,
                None,
                g_y_effective=g_y_eff,
                k_min=k_lo,
                k_max=k_hi,
                k_grid=sol.k_grid,
                V_arr=W_next,
            )
            W_new[ik, iz] = val
            k_pol[ik, iz] = kp
            n_pol[ik, iz] = n
            c_pol[ik, iz] = c
            y_pol[ik, iz] = y

    return W_new, k_pol, n_pol, c_pol, y_pol


def _bellman_on_grid_fixed_g(
    p: Params,
    sol: MarkovSolution,
    W_next: np.ndarray,
    g_level_grid: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    nk = len(sol.k_grid)
    W_new = np.zeros((nk, 2))
    k_pol = np.zeros((nk, 2))
    n_pol = np.zeros((nk, 2))
    c_pol = np.zeros((nk, 2))
    y_pol = np.zeros((nk, 2))
    for iz in range(2):
        for ik, k in enumerate(sol.k_grid):
            glev = float(g_level_grid[ik, iz])
            kp, n, c, y = optimal_one_period_fixed_g(p, sol, k, iz, g_level=glev)
            ev = expect_V_np(kp, iz, sol.k_grid, W_next, sol.Pi)
            W_new[ik, iz] = _u(c, p.sigma) - _v_disutil(n, p.chi, p.phi) + p.beta * ev
            k_pol[ik, iz] = kp
            n_pol[ik, iz] = n
            c_pol[ik, iz] = c
            y_pol[ik, iz] = y
    return W_new, k_pol, n_pol, c_pol, y_pol


def backward_foreseen(
    p: Params,
    sol: MarkovSolution,
    *,
    H: int,
    g_high_level_grid: np.ndarray,
) -> tuple[list[np.ndarray], list[np.ndarray], list[np.ndarray], list[np.ndarray], list[np.ndarray]]:
    """
    tau=0: shock period (fixed g on grid). tau=1..H: normal g_y, W_{tau-1} continuation.
    Returns lists indexed by tau in 0..H where index H is farthest before shock.
    """
    W = sol.V.copy()
    policies_k: list[np.ndarray] = []
    policies_n: list[np.ndarray] = []
    policies_c: list[np.ndarray] = []
    policies_y: list[np.ndarray] = []
    policies_W: list[np.ndarray] = []

    W0, k0, n0, c0, y0 = _bellman_on_grid_fixed_g(p, sol, W, g_high_level_grid)
    policies_k.insert(0, k0)
    policies_n.insert(0, n0)
    policies_c.insert(0, c0)
    policies_y.insert(0, y0)
    policies_W.insert(0, W0)
    W_curr = W0

    for _ in range(H):
        W_curr, kpol, npol, cpol, ypol = _bellman_on_grid_gy(
            p, sol, W_curr, g_y_eff=p.g_y_ratio
        )
        policies_k.insert(0, kpol)
        policies_n.insert(0, npol)
        policies_c.insert(0, cpol)
        policies_y.insert(0, ypol)
        policies_W.insert(0, W_curr)

    return policies_W, policies_k, policies_n, policies_c, policies_y


def build_z_path_mc(Pi: np.ndarray, z0: int, T: int, rng: np.random.Generator) -> np.ndarray:
    z = np.zeros(T + 1, dtype=int)
    z[0] = z0
    for t in range(T):
        z[t + 1] = int(rng.choice(2, p=Pi[z[t], :]))
    return z


def pct_irf(base: np.ndarray, shock: np.ndarray) -> np.ndarray:
    return 100.0 * np.log(np.maximum(shock, 1e-14) / np.maximum(base, 1e-14))


def _mc_stderr_of_pct(
    base_draws: np.ndarray,
    shock_draws: np.ndarray,
) -> np.ndarray:
    """Approximate SE of mean of 100*log(shock/base) across draws (delta method shortcut)."""
    n = base_draws.shape[0]
    if n < 2:
        return np.zeros(base_draws.shape[1])
    pct_d = 100.0 * (
        np.log(np.maximum(shock_draws, 1e-14) / np.maximum(base_draws, 1e-14))
    )
    return np.std(pct_d, axis=0, ddof=1) / np.sqrt(n)


def simulate_foreseen_path(
    p: Params,
    sol: MarkovSolution,
    policies_k: list[np.ndarray],
    policies_n: list[np.ndarray],
    z_path: np.ndarray,
    k0: float,
    *,
    dg: float,
) -> dict[str, np.ndarray]:
    """policies_k has length H+1: index t in 0..H-1 anticipation, index H is shock quarter."""
    H = len(policies_k) - 1
    Ttot = len(z_path) - 1
    k = np.zeros(Ttot + 1)
    n = np.zeros(Ttot)
    c = np.zeros(Ttot)
    y = np.zeros(Ttot)
    g = np.zeros(Ttot)
    k[0] = k0
    kg = sol.k_grid
    for t in range(min(H + 1, Ttot)):
        zi = int(z_path[t])
        kk = k[t]
        ik = int(np.clip(np.searchsorted(kg, kk) - 1, 0, len(kg) - 2))
        denom = kg[ik + 1] - kg[ik]
        w0 = (kg[ik + 1] - kk) / denom if denom > 1e-14 else 1.0
        pol_k = policies_k[t]
        pol_n = policies_n[t]
        kp = w0 * pol_k[ik, zi] + (1.0 - w0) * pol_k[ik + 1, zi]
        nv = w0 * pol_n[ik, zi] + (1.0 - w0) * pol_n[ik + 1, zi]
        zv = float(sol.z_vals[zi])
        yv = production(zv, kk, nv, p.alpha)
        gv = p.g_y_ratio * yv + (dg if t == H else 0.0)
        inv = kp - (1.0 - p.delta) * kk
        cv = yv - gv - inv
        n[t] = nv
        y[t] = yv
        g[t] = gv
        c[t] = cv
        k[t + 1] = kp
    if Ttot > H + 1:
        zp = z_path[H + 1 : Ttot + 1]
        cont = simulate_baseline_policy(sol, k[H + 1], zp)
        tail = Ttot - H - 1
        for t in range(tail):
            n[H + 1 + t] = cont["n"][t]
            y[H + 1 + t] = cont["y"][t]
            c[H + 1 + t] = cont["c"][t]
            g[H + 1 + t] = cont["g"][t]
            k[H + 2 + t] = cont["k"][t + 1]
    return {"k": k, "n": n, "c": c, "y": y, "g": g}


def mc_irf_unforeseen(
    p: Params,
    sol: MarkovSolution,
    *,
    z0_idx: int,
    T: int,
    shock_frac: float,
    dg_absolute: float | None = None,
    n_draws: int = 3000,
    seed: int = 42,
    return_stderr: bool = True,
) -> dict:
    k0 = find_k_fixed_point_grid(sol, z0_idx)
    zf = float(sol.z_vals[z0_idx])
    ss_local = solve_steady_state(p, z=zf)
    if dg_absolute is not None:
        dg = float(dg_absolute)
    else:
        dg = shock_frac * ss_local.g
    rng = np.random.default_rng(seed)
    kp_of, n_of = policy_interp(sol)

    sum_b = {k: np.zeros(T) for k in ("y", "c", "n", "i")}
    sum_s = {k: np.zeros(T) for k in ("y", "c", "n", "i")}
    draws_b = {k: np.zeros((n_draws, T)) for k in ("y", "c", "n", "i")}
    draws_s = {k: np.zeros((n_draws, T)) for k in ("y", "c", "n", "i")}

    for d in range(n_draws):
        zp = build_z_path_mc(sol.Pi, z0_idx, T, rng)
        zi = z0_idx
        nb = n_of(k0, zi)
        yb0 = production(zf, k0, nb, p.alpha)
        gb0 = p.g_y_ratio * yb0
        k1b = kp_of(k0, zi)
        ib0 = k1b - (1.0 - p.delta) * k0
        cb0 = yb0 - gb0 - ib0
        sum_b["y"][0] += yb0
        sum_b["c"][0] += cb0
        sum_b["n"][0] += nb
        sum_b["i"][0] += ib0
        draws_b["y"][d, 0] = yb0
        draws_b["c"][d, 0] = cb0
        draws_b["n"][d, 0] = nb
        draws_b["i"][d, 0] = ib0
        # Continuation starts at t=1 with state z_{t=1}; z_path[0] has already been
        # used to compute period-0 objects above.
        cont_z = zp[1:]
        cont_b = simulate_baseline_policy(sol, k1b, cont_z)
        for t in range(1, T):
            sum_b["y"][t] += cont_b["y"][t - 1]
            sum_b["c"][t] += cont_b["c"][t - 1]
            sum_b["n"][t] += cont_b["n"][t - 1]
            inv = cont_b["k"][t] - (1.0 - p.delta) * cont_b["k"][t - 1]
            sum_b["i"][t] += inv
            draws_b["y"][d, t] = cont_b["y"][t - 1]
            draws_b["c"][d, t] = cont_b["c"][t - 1]
            draws_b["n"][d, t] = cont_b["n"][t - 1]
            draws_b["i"][d, t] = inv

        k1s, ns, cs, ys = optimal_one_period_fixed_g(
            p, sol, k0, z0_idx, g_level=gb0 + dg
        )
        is0 = k1s - (1.0 - p.delta) * k0
        sum_s["y"][0] += ys
        sum_s["c"][0] += cs
        sum_s["n"][0] += ns
        sum_s["i"][0] += is0
        draws_s["y"][d, 0] = ys
        draws_s["c"][d, 0] = cs
        draws_s["n"][d, 0] = ns
        draws_s["i"][d, 0] = is0
        cont_s = simulate_baseline_policy(sol, k1s, cont_z)
        for t in range(1, T):
            sum_s["y"][t] += cont_s["y"][t - 1]
            sum_s["c"][t] += cont_s["c"][t - 1]
            sum_s["n"][t] += cont_s["n"][t - 1]
            inv = cont_s["k"][t] - (1.0 - p.delta) * cont_s["k"][t - 1]
            sum_s["i"][t] += inv
            draws_s["y"][d, t] = cont_s["y"][t - 1]
            draws_s["c"][d, t] = cont_s["c"][t - 1]
            draws_s["n"][d, t] = cont_s["n"][t - 1]
            draws_s["i"][d, t] = inv

    for k in sum_b:
        sum_b[k] /= n_draws
        sum_s[k] /= n_draws

    out: dict = {
        "y_pct": pct_irf(sum_b["y"], sum_s["y"]),
        "c_pct": pct_irf(sum_b["c"], sum_s["c"]),
        "n_pct": pct_irf(sum_b["n"], sum_s["n"]),
        "i_pct": pct_irf(sum_b["i"], sum_s["i"]),
        "base": sum_b,
        "shock": sum_s,
        "dg_used": dg,
    }
    if return_stderr:
        out["y_pct_se"] = _mc_stderr_of_pct(draws_b["y"], draws_s["y"])
        out["c_pct_se"] = _mc_stderr_of_pct(draws_b["c"], draws_s["c"])
        out["n_pct_se"] = _mc_stderr_of_pct(draws_b["n"], draws_s["n"])
        out["i_pct_se"] = _mc_stderr_of_pct(draws_b["i"], draws_s["i"])
    return out


def mc_irf_foreseen(
    p: Params,
    sol: MarkovSolution,
    *,
    z0_idx: int,
    T: int,
    H: int,
    shock_frac: float,
    dg_absolute: float | None = None,
    n_draws: int = 2500,
    seed: int = 43,
    return_stderr: bool = True,
) -> dict:
    k0 = find_k_fixed_point_grid(sol, z0_idx)
    zb, _ = p.ergodic_z()
    ss_bar = solve_steady_state(p, z=zb)
    if dg_absolute is not None:
        dg_fixed = float(dg_absolute)
    else:
        dg_fixed = shock_frac * ss_bar.g

    g_high = np.zeros((len(sol.k_grid), 2))
    for iz in range(2):
        zv = float(sol.z_vals[iz])
        for ik, k in enumerate(sol.k_grid):
            nv = sol.n_policy[ik, iz]
            yv = production(zv, k, nv, p.alpha)
            g_high[ik, iz] = p.g_y_ratio * yv + dg_fixed

    _, pk, pn, _, _ = backward_foreseen(p, sol, H=H, g_high_level_grid=g_high)

    rng = np.random.default_rng(seed)
    sum_b = {k: np.zeros(T) for k in ("y", "c", "n", "i")}
    sum_s = {k: np.zeros(T) for k in ("y", "c", "n", "i")}
    draws_b = {k: np.zeros((n_draws, T)) for k in ("y", "c", "n", "i")}
    draws_s = {k: np.zeros((n_draws, T)) for k in ("y", "c", "n", "i")}

    kp_of, n_of = policy_interp(sol)
    for d in range(n_draws):
        zp = build_z_path_mc(sol.Pi, z0_idx, T, rng)
        kb = k0
        for t in range(T):
            zi = int(zp[t])
            zv = float(sol.z_vals[zi])
            nb = n_of(kb, zi)
            yb = production(zv, kb, nb, p.alpha)
            gb = p.g_y_ratio * yb
            k1b = kp_of(kb, zi)
            ib = k1b - (1.0 - p.delta) * kb
            cb = yb - gb - ib
            sum_b["y"][t] += yb
            sum_b["c"][t] += cb
            sum_b["n"][t] += nb
            sum_b["i"][t] += ib
            draws_b["y"][d, t] = yb
            draws_b["c"][d, t] = cb
            draws_b["n"][d, t] = nb
            draws_b["i"][d, t] = ib
            kb = k1b

        path_s = simulate_foreseen_path(
            p, sol, pk, pn, zp, k0, dg=dg_fixed
        )
        for t in range(T):
            inv = path_s["k"][t + 1] - (1.0 - p.delta) * path_s["k"][t]
            sum_s["y"][t] += path_s["y"][t]
            sum_s["c"][t] += path_s["c"][t]
            sum_s["n"][t] += path_s["n"][t]
            sum_s["i"][t] += inv
            draws_s["y"][d, t] = path_s["y"][t]
            draws_s["c"][d, t] = path_s["c"][t]
            draws_s["n"][d, t] = path_s["n"][t]
            draws_s["i"][d, t] = inv

    for k in sum_b:
        sum_b[k] /= n_draws
        sum_s[k] /= n_draws

    out = {
        "y_pct": pct_irf(sum_b["y"], sum_s["y"]),
        "c_pct": pct_irf(sum_b["c"], sum_s["c"]),
        "n_pct": pct_irf(sum_b["n"], sum_s["n"]),
        "i_pct": pct_irf(sum_b["i"], sum_s["i"]),
        "dg_used": dg_fixed,
    }
    if return_stderr:
        out["y_pct_se"] = _mc_stderr_of_pct(draws_b["y"], draws_s["y"])
        out["c_pct_se"] = _mc_stderr_of_pct(draws_b["c"], draws_s["c"])
        out["n_pct_se"] = _mc_stderr_of_pct(draws_b["n"], draws_s["n"])
        out["i_pct_se"] = _mc_stderr_of_pct(draws_b["i"], draws_s["i"])
    return out


def irf_permanent_shift(
    p: Params,
    sol_low: MarkovSolution,
    sol_high: MarkovSolution,
    *,
    z0_idx: int,
    T: int,
    n_draws: int = 2500,
    seed: int = 44,
    return_stderr: bool = True,
) -> dict:
    """Unforeseen permanent increase in g/y: switch to sol_high policy from t=0."""
    k0 = find_k_fixed_point_grid(sol_low, z0_idx)
    rng = np.random.default_rng(seed)
    kp_lo, n_lo = policy_interp(sol_low)
    kp_hi, n_hi = policy_interp(sol_high)

    sum_b = {k: np.zeros(T) for k in ("y", "c", "n", "i")}
    sum_s = {k: np.zeros(T) for k in ("y", "c", "n", "i")}
    draws_b = {k: np.zeros((n_draws, T)) for k in ("y", "c", "n", "i")}
    draws_s = {k: np.zeros((n_draws, T)) for k in ("y", "c", "n", "i")}

    for d in range(n_draws):
        zp = build_z_path_mc(sol_low.Pi, z0_idx, T, rng)
        kb = k0
        for t in range(T):
            zi = int(zp[t])
            zv = float(sol_low.z_vals[zi])
            nb = n_lo(kb, zi)
            yb = production(zv, kb, nb, p.alpha)
            gb = p.g_y_ratio * yb
            k1b = kp_lo(kb, zi)
            ib = k1b - (1.0 - p.delta) * kb
            cb = yb - gb - ib
            sum_b["y"][t] += yb
            sum_b["c"][t] += cb
            sum_b["n"][t] += nb
            sum_b["i"][t] += ib
            draws_b["y"][d, t] = yb
            draws_b["c"][d, t] = cb
            draws_b["n"][d, t] = nb
            draws_b["i"][d, t] = ib
            kb = k1b

        ks = k0
        for t in range(T):
            zi = int(zp[t])
            zv = float(sol_high.z_vals[zi])
            ns = n_hi(ks, zi)
            ys = production(zv, ks, ns, p.alpha)
            gs = sol_high.p.g_y_ratio * ys
            k1s = kp_hi(ks, zi)
            ins = k1s - (1.0 - p.delta) * ks
            cs = ys - gs - ins
            sum_s["y"][t] += ys
            sum_s["c"][t] += cs
            sum_s["n"][t] += ns
            sum_s["i"][t] += ins
            draws_s["y"][d, t] = ys
            draws_s["c"][d, t] = cs
            draws_s["n"][d, t] = ns
            draws_s["i"][d, t] = ins
            ks = k1s

    for k in sum_b:
        sum_b[k] /= n_draws
        sum_s[k] /= n_draws

    out = {
        "y_pct": pct_irf(sum_b["y"], sum_s["y"]),
        "c_pct": pct_irf(sum_b["c"], sum_s["c"]),
        "n_pct": pct_irf(sum_b["n"], sum_s["n"]),
        "i_pct": pct_irf(sum_b["i"], sum_s["i"]),
    }
    if return_stderr:
        out["y_pct_se"] = _mc_stderr_of_pct(draws_b["y"], draws_s["y"])
        out["c_pct_se"] = _mc_stderr_of_pct(draws_b["c"], draws_s["c"])
        out["n_pct_se"] = _mc_stderr_of_pct(draws_b["n"], draws_s["n"])
        out["i_pct_se"] = _mc_stderr_of_pct(draws_b["i"], draws_s["i"])
    return out


def run_all_markov_experiments(
    p: Params,
    *,
    T: int = 40,
    shock_quarter: int = 8,
    shock_frac: float = 0.05,
    permanent_delta_g_y: float = 0.02,
    mc_draws: int = 2500,
    sol: MarkovSolution | None = None,
    sol_hi: MarkovSolution | None = None,
) -> dict:
    if sol is None:
        sol = solve_markov_stationary(p)
    p_hi = replace(p, g_y_ratio=min(p.g_y_ratio + permanent_delta_g_y, 0.45))
    if sol_hi is None:
        sol_hi = solve_markov_stationary_alt_gy(p_hi, p_hi.g_y_ratio)

    zb, _ = p.ergodic_z()
    z_bar_idx = 0 if abs(p.z_L - zb) <= abs(p.z_H - zb) else 1
    ss_bar = solve_steady_state(p, z=zb)
    dg_common = shock_frac * ss_bar.g

    out: dict = {
        "sol": sol,
        "sol_high": sol_hi,
        "euler": euler_residual_stats(sol),
        "dg_common": dg_common,
        "z_bar": mc_irf_unforeseen(
            p,
            sol,
            z0_idx=z_bar_idx,
            T=T,
            shock_frac=shock_frac,
            dg_absolute=dg_common,
            n_draws=mc_draws,
        ),
        "foreseen_zbar": mc_irf_foreseen(
            p,
            sol,
            z0_idx=z_bar_idx,
            T=T,
            H=shock_quarter,
            shock_frac=shock_frac,
            dg_absolute=dg_common,
            n_draws=mc_draws,
        ),
        "permanent_zbar": irf_permanent_shift(
            p, sol, sol_hi, z0_idx=z_bar_idx, T=T, n_draws=mc_draws
        ),
        "z_L": mc_irf_unforeseen(
            p,
            sol,
            z0_idx=0,
            T=T,
            shock_frac=shock_frac,
            dg_absolute=dg_common,
            n_draws=mc_draws,
        ),
        "z_H": mc_irf_unforeseen(
            p,
            sol,
            z0_idx=1,
            T=T,
            shock_frac=shock_frac,
            dg_absolute=dg_common,
            n_draws=mc_draws,
        ),
    }
    return out
