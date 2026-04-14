"""
Deterministic steady state at reference productivity z_bar (ergodic mean).
Government: one-period bonds; baseline uses B/Y ratio and g/Y ratio from Params.
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np
from scipy.optimize import root

from . import equilibrium as eq
from .params import Params


@dataclass
class SteadyState:
    z: float
    k: float
    n: float
    y: float
    c: float
    g: float
    i: float
    rk: float
    w: float
    q: float
    B: float
    tau: float
    a: float  # household bonds (= B in equilibrium)


def solve_steady_state(p: Params, z: float | None = None) -> SteadyState:
    """
    Solve deterministic steady state with constant z, constant g/Y, and constant B/Y.
    Lump-sum tax tau clears government budget: q*B' + tau = B + g with B'=B.
    """
    if z is None:
        z_bar, _ = p.ergodic_z()
        z = z_bar

    rk_ss = eq.euler_k_target_rk(p)
    g_y = p.g_y_ratio

    def residuals(x: np.ndarray) -> np.ndarray:
        k, n = float(x[0]), float(x[1])
        if k <= 0 or n <= 1e-6 or n >= 1.0 - 1e-6:
            return np.array([1e6, 1e6])
        rk = eq.rental_rate(z, k, n, p.alpha)
        w = eq.wage(z, k, n, p.alpha)
        y = eq.production(z, k, n, p.alpha)
        g = g_y * y
        c = y - p.delta * k - g
        if c <= 1e-8:
            return np.array([1e6, 1e6])
        f0 = rk - rk_ss
        f1 = eq.intratemporal_residual(n, w, c, p)
        return np.array([f0, f1])

    # Initial guess: typical RBC magnitudes
    x0 = np.array([40.0, 0.33])
    sol = root(residuals, x0, method="hybr")
    if not sol.success:
        raise RuntimeError(f"Steady state failed: {sol.message}")

    k, n = float(sol.x[0]), float(sol.x[1])
    rk = eq.rental_rate(z, k, n, p.alpha)
    w = eq.wage(z, k, n, p.alpha)
    y = eq.production(z, k, n, p.alpha)
    g = g_y * y
    c = y - p.delta * k - g
    i = p.delta * k
    q = p.beta  # risk-free pricing in deterministic SS
    B = p.b_y_ratio * y
    tau = B + g - q * B  # = g + B(1-q)

    return SteadyState(
        z=z,
        k=k,
        n=n,
        y=y,
        c=c,
        g=g,
        i=i,
        rk=rk,
        w=w,
        q=q,
        B=B,
        tau=tau,
        a=B,
    )


def print_steady_state(ss: SteadyState, name: str = "steady state") -> None:
    print(f"--- {name} ---")
    print(f"z={ss.z:.4f} k={ss.k:.4f} n={ss.n:.4f} y={ss.y:.4f}")
    print(f"c={ss.c:.4f} g={ss.g:.4f} i={ss.i:.4f} g/y={ss.g/ss.y:.3f}")
    print(f"rk={ss.rk:.4f} w={ss.w:.4f} q={ss.q:.4f} B={ss.B:.4f} tau={ss.tau:.4f}")
