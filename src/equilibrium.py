"""
Static equilibrium conditions: production, factor prices, and household intratemporal FOC.
"""
from __future__ import annotations

import numpy as np

from .params import Params


def production(z: float, k: float, n: float, alpha: float) -> float:
    """Output Y = z K^alpha N^(1-alpha)."""
    return z * (k**alpha) * (n ** (1.0 - alpha))


def rental_rate(z: float, k: float, n: float, alpha: float) -> float:
    return alpha * z * (k ** (alpha - 1.0)) * (n ** (1.0 - alpha))


def wage(z: float, k: float, n: float, alpha: float) -> float:
    return (1.0 - alpha) * z * (k**alpha) * (n ** (-alpha))


def marginal_utility_c(c: float, sigma: float) -> float:
    return c ** (-sigma)


def labor_mrs(n: float, chi: float, phi: float) -> float:
    """Marginal disutility v'(n) = chi n^phi."""
    return chi * (n**phi)


def intratemporal_residual(
    n: float, w: float, c: float, p: Params
) -> float:
    """FOC: chi n^phi = w * c^(-sigma)."""
    return labor_mrs(n, p.chi, p.phi) - w * marginal_utility_c(c, p.sigma)


def euler_k_target_rk(p: Params) -> float:
    """Steady-state rental rate from capital Euler: 1 = beta (1-delta + rk)."""
    return (1.0 / p.beta) - 1.0 + p.delta
