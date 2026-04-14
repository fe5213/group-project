"""Write CSV / text tables for LaTeX and grading."""
from __future__ import annotations

from pathlib import Path

import numpy as np

from .markov_rbc import MarkovSolution, find_k_fixed_point_grid, policy_interp, production
from .params import Params
from .steady_state import solve_steady_state


def steady_state_rows_markov(
    p: Params,
    sol: MarkovSolution,
) -> list[dict[str, float]]:
    rows = []
    kp_of, n_of = policy_interp(sol)
    for iz, z in enumerate(sol.z_vals):
        zf = float(z)
        k = find_k_fixed_point_grid(sol, iz)
        n = n_of(k, iz)
        y = production(zf, k, n, p.alpha)
        g = p.g_y_ratio * y
        rk = p.alpha * zf * (k ** (p.alpha - 1.0)) * (n ** (1.0 - p.alpha))
        w = (1.0 - p.alpha) * zf * (k**p.alpha) * (n ** (-p.alpha))
        kp = kp_of(k, iz)
        i = kp - (1.0 - p.delta) * k
        c = y - g - i
        tau = g
        rows.append(
            {
                "z": zf,
                "k": k,
                "y": y,
                "c": c,
                "n": n,
                "i": i,
                "g": g,
                "w": w,
                "rk": rk,
                "tau": tau,
            }
        )
    zb, _ = p.ergodic_z()
    ssb = solve_steady_state(p, z=zb)
    rows.append(
        {
            "z": zb,
            "k": ssb.k,
            "y": ssb.y,
            "c": ssb.c,
            "n": ssb.n,
            "i": ssb.i,
            "g": ssb.g,
            "w": ssb.w,
            "rk": ssb.rk,
            "tau": ssb.tau,
        }
    )
    return rows


def write_steady_state_csv(path: Path, rows: list[dict[str, float]], labels: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cols = ["label", "z", "k", "y", "c", "n", "i", "g", "w", "rk", "tau"]
    lines = [",".join(cols)]
    for lab, row in zip(labels, rows):
        lines.append(
            ",".join(
                [lab]
                + [f"{row[c]:.8f}" for c in cols[1:]]
            )
        )
    path.write_text("\n".join(lines), encoding="utf-8")


def irf_summary_table(
    horizons: tuple[int, ...],
    y_pct: np.ndarray,
    c_pct: np.ndarray,
    n_pct: np.ndarray,
    i_pct: np.ndarray,
) -> str:
    lines = ["horizon,y_pct,c_pct,n_pct,i_pct"]
    for h in horizons:
        if h < len(y_pct):
            lines.append(
                f"{h},{y_pct[h]:.6f},{c_pct[h]:.6f},{n_pct[h]:.6f},{i_pct[h]:.6f}"
            )
    return "\n".join(lines)


def cumulative_multiplier_pct(
    y_pct: np.ndarray,
    g_pct: np.ndarray | None,
    H: int,
) -> float:
    """Rough cumulative output response / fiscal impulse in pct points (same units as IRF)."""
    return float(np.sum(y_pct[: H + 1]))
