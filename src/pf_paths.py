"""Lump-sum-tax perfect-foresight paths (used only for the labor-tax extension baseline)."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import root

from . import equilibrium as eq
from .params import Params
from .steady_state import SteadyState


@dataclass
class PFPath:
    k: np.ndarray
    c: np.ndarray
    n: np.ndarray
    y: np.ndarray
    g: np.ndarray


def R_marginal(kp: float, np_: float, z: float, p: Params) -> float:
    rk = eq.rental_rate(z, kp, np_, p.alpha)
    return 1.0 - p.delta + rk


def investment_series(path: PFPath, delta: float) -> np.ndarray:
    return path.k[1:] - (1.0 - delta) * path.k[:-1]


def solve_pf_path(
    p: Params,
    z: float,
    k0: float,
    g_path: np.ndarray,
    ss_terminal: SteadyState,
    *,
    x0_hint: np.ndarray | None = None,
) -> PFPath:
    T = int(len(g_path))
    if T < 2:
        raise ValueError("Need horizon T >= 2")

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
            res[T + t] = eq.intratemporal_residual(n[t], w_t, c[t], p)

        # Euler at time t:  u'(c_t) = beta * u'(c_{t+1}) * [rk(k_{t+1}, n_{t+1}) + 1 - delta]
        # In array terms:   c_next = c[t+1], and the rental is evaluated at k[t+1], n[t+1].
        for t in range(T - 1):
            R_tp = R_marginal(k[t + 1], n[t + 1], z, p)
            euler = p.beta * ((c[t + 1] / c[t]) ** (-p.sigma)) * R_tp - 1.0
            res[2 * T + t] = euler

        # Terminal anchor (period T): use k[T] (final choice) with terminal labor n_term.
        R_T = R_marginal(k[T], n_term, z, p)
        euler_Tm1 = p.beta * ((c_term / c[T - 1]) ** (-p.sigma)) * R_T - 1.0
        res[3 * T - 1] = euler_Tm1

        return res

    if x0_hint is None:
        ss0 = ss_terminal
        c = np.full(T, ss0.c)
        n = np.full(T, ss0.n)
        kp = np.linspace(k0, k_term, T + 1)[1:]
        x0 = np.concatenate([c, n, kp])
    else:
        x0 = x0_hint

    sol = root(residuals, x0, method="hybr", options={"maxfev": 50000})
    if not sol.success:
        raise RuntimeError(f"PF solver failed: {sol.message}")

    c, n, kp = unpack(sol.x)
    k = np.empty(T + 1)
    k[0] = k0
    k[1:] = kp
    y = np.array([eq.production(z, k[t], n[t], p.alpha) for t in range(T)])
    return PFPath(k=k, c=c, n=n, y=y, g=g_path.copy())
