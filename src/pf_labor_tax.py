"""
Perfect foresight with distortionary labor-income taxation financing government spending each period:

  tau_t * w_t * n_t = g_t  (balanced budget each period, no bonds)

Household budget (no lump-sum tax):
  c_t + k_{t+1} = (1 - tau_t) w_t n_t + rk_t k_t + (1-delta) k_t

Intratemporal:
  chi n_t^phi = (1 - tau_t) w_t c_t^{-sigma}
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np
from scipy.optimize import root

from . import equilibrium as eq
from .params import Params
from .pf_paths import PFPath, R_marginal as _R


def _intratemporal_labor_tax(
    n: float, w: float, c: float, g: float, p: Params
) -> float:
    wn = max(w * n, 1e-12)
    tau = min(max(g / wn, 0.0), 0.99)
    return eq.labor_mrs(n, p.chi, p.phi) - (1.0 - tau) * w * eq.marginal_utility_c(c, p.sigma)


@dataclass
class LaborTaxSS:
    """Steady state with labor-tax financing (same production side as baseline)."""

    z: float
    k: float
    n: float
    y: float
    c: float
    g: float
    tau: float


def solve_steady_state_labor_tax(p: Params, z: float, g_y: float) -> LaborTaxSS:
    """Solve steady state with g = g_y * y and tau * w * n = g."""

    rk_ss = eq.euler_k_target_rk(p)

    def residuals(x: np.ndarray) -> np.ndarray:
        k, n = float(x[0]), float(x[1])
        if k <= 0 or n <= 1e-6 or n >= 1.0 - 1e-6:
            return np.array([1e6, 1e6])
        w = eq.wage(z, k, n, p.alpha)
        y = eq.production(z, k, n, p.alpha)
        g = g_y * y
        wn = w * n
        tau = g / wn
        if tau >= 0.99:
            return np.array([1e6, 1e6])
        c = y - p.delta * k - g
        if c <= 1e-8:
            return np.array([1e6, 1e6])
        rk = eq.rental_rate(z, k, n, p.alpha)
        f0 = rk - rk_ss
        f1 = _intratemporal_labor_tax(n, w, c, g, p)
        return np.array([f0, f1])

    sol = root(residuals, np.array([40.0, 0.33]), method="hybr")
    if not sol.success:
        raise RuntimeError(f"Labor-tax SS failed: {sol.message}")
    k, n = float(sol.x[0]), float(sol.x[1])
    w = eq.wage(z, k, n, p.alpha)
    y = eq.production(z, k, n, p.alpha)
    g = g_y * y
    tau = g / (w * n)
    c = y - p.delta * k - g
    return LaborTaxSS(z=z, k=k, n=n, y=y, c=c, g=g, tau=tau)


def solve_pf_path_labor_tax(
    p: Params,
    z: float,
    k0: float,
    g_path: np.ndarray,
    ss_terminal: LaborTaxSS,
    *,
    x0_hint: np.ndarray | None = None,
) -> PFPath:
    T = len(g_path)
    k_term = ss_terminal.k
    c_term = ss_terminal.c
    n_term = ss_terminal.n

    def unpack(x: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        c = x[0:T].copy()
        n = x[T : 2 * T].copy()
        kp = x[2 * T : 3 * T].copy()
        return c, n, kp

    def residuals(x: np.ndarray) -> np.ndarray:
        c, n, kp = unpack(x)
        k = np.empty(T + 1)
        k[0] = k0
        k[1:] = kp
        res = np.zeros(3 * T)
        for t in range(T):
            y_t = eq.production(z, k[t], n[t], p.alpha)
            inv = k[t + 1] - (1.0 - p.delta) * k[t]
            res[t] = y_t - c[t] - inv - g_path[t]

        for t in range(T):
            w_t = eq.wage(z, k[t], n[t], p.alpha)
            res[T + t] = _intratemporal_labor_tax(n[t], w_t, c[t], g_path[t], p)

        for t in range(T - 1):
            R_tp = _R(k[t + 2], n[t + 1], z, p)
            euler = p.beta * ((c[t + 1] / c[t]) ** (-p.sigma)) * R_tp - 1.0
            res[2 * T + t] = euler

        R_T = _R(k_term, n_term, z, p)
        res[3 * T - 1] = p.beta * ((c_term / c[T - 1]) ** (-p.sigma)) * R_T - 1.0
        return res

    if x0_hint is not None:
        x0 = x0_hint
    else:
        c = np.full(T, ss_terminal.c)
        n = np.full(T, ss_terminal.n)
        kp = np.linspace(k0, k_term, T + 1)[1:]
        x0 = np.concatenate([c, n, kp])
    sol = root(residuals, x0, method="hybr", options={"maxfev": 100000})
    if not sol.success:
        raise RuntimeError(f"PF labor-tax solver failed: {sol.message}")
    c, n, kp = unpack(sol.x)
    k = np.empty(T + 1)
    k[0] = k0
    k[1:] = kp
    y = np.array([eq.production(z, k[t], n[t], p.alpha) for t in range(T)])
    return PFPath(k=k, c=c, n=n, y=y, g=g_path.copy())
