"""
IRFs and labor-tax vs lump-sum comparison using minimal perfect-foresight paths.

Core stochastic model: ``markov_rbc`` / ``markov_experiments``.
"""
from __future__ import annotations

import numpy as np

from .params import Params
from .pf_labor_tax import LaborTaxSS, solve_pf_path_labor_tax, solve_steady_state_labor_tax
from .pf_paths import investment_series, solve_pf_path
from .steady_state import SteadyState, solve_steady_state


def irf_from_g_paths(
    p: Params,
    z: float,
    ss: SteadyState,
    g_lo: np.ndarray,
    g_hi: np.ndarray,
    ss_terminal_lo: SteadyState,
    ss_terminal_hi: SteadyState | None = None,
) -> tuple[dict[str, np.ndarray], float]:
    if g_lo.shape != g_hi.shape:
        raise ValueError("g_lo and g_hi must have same shape")
    if ss_terminal_hi is None:
        ss_terminal_hi = ss_terminal_lo
    path_lo = solve_pf_path(p, z, ss.k, g_lo, ss_terminal_lo)
    path_hi = solve_pf_path(p, z, ss.k, g_hi, ss_terminal_hi)
    dg = float(np.max(np.abs(g_hi - g_lo)))
    if dg < 1e-16:
        raise ValueError("g paths identical")

    def pct(x_lo: np.ndarray, x_hi: np.ndarray) -> np.ndarray:
        return 100.0 * np.log(np.maximum(x_hi, 1e-14) / np.maximum(x_lo, 1e-14))

    i_lo = investment_series(path_lo, p.delta)
    i_hi = investment_series(path_hi, p.delta)

    out = {
        "y_pct": pct(path_lo.y, path_hi.y),
        "c_pct": pct(path_lo.c, path_hi.c),
        "n_pct": pct(path_lo.n, path_hi.n),
        "i_pct": pct(i_lo, i_hi),
        "g_pct": pct(path_lo.g, path_hi.g),
    }
    return out, dg


def irf_from_g_paths_labor_tax(
    p: Params,
    z: float,
    ss: LaborTaxSS,
    g_lo: np.ndarray,
    g_hi: np.ndarray,
    ss_terminal_lo: LaborTaxSS,
    ss_terminal_hi: LaborTaxSS | None = None,
) -> tuple[dict[str, np.ndarray], float]:
    if g_lo.shape != g_hi.shape:
        raise ValueError("g_lo and g_hi must have same shape")
    if ss_terminal_hi is None:
        ss_terminal_hi = ss_terminal_lo
    path_lo = solve_pf_path_labor_tax(p, z, ss.k, g_lo, ss_terminal_lo)
    path_hi = solve_pf_path_labor_tax(p, z, ss.k, g_hi, ss_terminal_hi)
    dg = float(np.max(np.abs(g_hi - g_lo)))
    if dg < 1e-16:
        raise ValueError("g paths identical")

    def pct(x_lo: np.ndarray, x_hi: np.ndarray) -> np.ndarray:
        return 100.0 * np.log(np.maximum(x_hi, 1e-14) / np.maximum(x_lo, 1e-14))

    i_lo = investment_series(path_lo, p.delta)
    i_hi = investment_series(path_hi, p.delta)

    out = {
        "y_pct": pct(path_lo.y, path_hi.y),
        "c_pct": pct(path_lo.c, path_hi.c),
        "n_pct": pct(path_lo.n, path_hi.n),
        "i_pct": pct(i_lo, i_hi),
        "g_pct": pct(path_lo.g, path_hi.g),
    }
    return out, dg


def one_time_g_shock_paths(ss: SteadyState, T: int, dg: float) -> tuple[np.ndarray, np.ndarray]:
    lo = np.full(T, ss.g)
    hi = lo.copy()
    hi[0] = ss.g + dg
    return lo, hi


def fiscal_shock_size(ss: SteadyState, frac_of_g: float = 0.05) -> float:
    return frac_of_g * ss.g


def run_labor_tax_vs_lumpsum_unforeseen(
    p: Params,
    *,
    T: int = 80,
    g_shock_frac: float = 0.05,
    match_absolute_dg_from_lumpsum: bool = False,
) -> dict:
    z_bar, _ = p.ergodic_z()
    ss_ls = solve_steady_state(p, z=z_bar)
    ss_lt = solve_steady_state_labor_tax(p, z_bar, p.g_y_ratio)
    dg_ls = fiscal_shock_size(ss_ls, frac_of_g=g_shock_frac)
    dg_lt = dg_ls if match_absolute_dg_from_lumpsum else g_shock_frac * ss_lt.g
    g_lo_ls, g_hi_ls = one_time_g_shock_paths(ss_ls, T, dg_ls)
    g_lo_lt = np.full(T, ss_lt.g)
    g_hi_lt = g_lo_lt.copy()
    g_hi_lt[0] = ss_lt.g + dg_lt
    irf_ls, _ = irf_from_g_paths(p, z_bar, ss_ls, g_lo_ls, g_hi_ls, ss_ls, ss_ls)
    irf_lt, _ = irf_from_g_paths_labor_tax(
        p, z_bar, ss_lt, g_lo_lt, g_hi_lt, ss_lt, ss_lt
    )
    return {
        "z_bar": z_bar,
        "irf_lumpsum": irf_ls,
        "irf_labor_tax": irf_lt,
    }
