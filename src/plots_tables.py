"""
Figures and tables for the report (saved under output/).
"""
from __future__ import annotations

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

from .params import Params
from .pf_labor_tax import solve_steady_state_labor_tax
from .steady_state import SteadyState


def ensure_output_dir(root: Path | str = "output") -> Path:
    p = Path(root)
    p.mkdir(parents=True, exist_ok=True)
    return p


def _style():
    plt.rcParams.update(
        {
            "figure.figsize": (9, 5.5),
            "axes.grid": True,
            "grid.alpha": 0.3,
            "font.size": 10,
        }
    )


def plot_irf_comparison(
    horizons: np.ndarray,
    series: dict[str, np.ndarray],
    title: str,
    outfile: Path,
) -> None:
    _style()
    fig, ax = plt.subplots()
    for name, y in series.items():
        ax.plot(horizons, y, label=name, linewidth=2)
    ax.axhline(0.0, color="black", linewidth=0.8)
    ax.set_xlabel("Quarter")
    ax.set_ylabel("Approx. % change (100·log ratio)")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    fig.savefig(outfile, dpi=150)
    plt.close(fig)


def plot_state_compare(
    h: np.ndarray,
    yL: np.ndarray,
    yH: np.ndarray,
    varname: str,
    title: str,
    outfile: Path,
) -> None:
    _style()
    fig, ax = plt.subplots()
    ax.plot(h, yL, label=r"$z=z_L$", linewidth=2)
    ax.plot(h, yH, label=r"$z=z_H$", linewidth=2)
    ax.axhline(0.0, color="black", linewidth=0.8)
    ax.set_xlabel("Quarter")
    ax.set_ylabel(varname)
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    fig.savefig(outfile, dpi=150)
    plt.close(fig)


def plot_unforeseen_zoom_2x2(
    h: np.ndarray,
    y: np.ndarray,
    c: np.ndarray,
    n: np.ndarray,
    i: np.ndarray,
    outfile: Path,
) -> None:
    """Zoom into post-impact dynamics for the unforeseen one-time shock."""
    _style()
    fig, axes = plt.subplots(2, 2, figsize=(9, 6.5), sharex=True)
    panels = [
        ("Output", y),
        ("Consumption", c),
        ("Labor", n),
        ("Investment", i),
    ]
    for ax, (title, series) in zip(axes.ravel(), panels):
        ax.plot(h, series, linewidth=2)
        ax.axhline(0.0, color="black", linewidth=0.8)
        ax.set_title(title)
        ax.set_ylabel("Approx. % change")
    for ax in axes[-1, :]:
        ax.set_xlabel("Quarter")
    fig.suptitle("Unforeseen one-time shock: post-impact zoom (h=1..12)")
    fig.tight_layout()
    fig.savefig(outfile, dpi=150)
    plt.close(fig)


def plot_financing_regime_irf_overlay(
    h: np.ndarray,
    series_a: np.ndarray,
    series_b: np.ndarray,
    label_a: str,
    label_b: str,
    ylabel: str,
    title: str,
    outfile: Path,
) -> None:
    """Overlay two IRF series (e.g. lump-sum vs labor-tax financing)."""
    _style()
    fig, ax = plt.subplots()
    ax.plot(h, series_a, label=label_a, linewidth=2)
    ax.plot(h, series_b, label=label_b, linewidth=2)
    ax.axhline(0.0, color="black", linewidth=0.8)
    ax.set_xlabel("Quarter")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    fig.savefig(outfile, dpi=150)
    plt.close(fig)


def plot_ricardian_financing(
    tau_tax: np.ndarray,
    tau_debt: np.ndarray,
    B_debt: np.ndarray,
    outfile: Path,
) -> None:
    _style()
    fig, ax1 = plt.subplots()
    ax1.plot(tau_tax, label=r"$\tau$ (tax-financed, $B=0$)", color="C0")
    ax1.plot(tau_debt, label=r"$\tau$ (smoothed + debt)", color="C1")
    ax1.set_xlabel("Quarter")
    ax1.set_ylabel("Lump-sum tax / transfers")
    ax2 = ax1.twinx()
    ax2.plot(B_debt, label=r"$B$ (debt path)", color="C2", linestyle="--")
    ax1.set_title("Ricardian financing: same $\\{g_t\\}$, different $(\\tau,B)$ timing")
    lines, labels = [], []
    for ax in (ax1, ax2):
        for line in ax.get_lines():
            lines.append(line)
            labels.append(line.get_label())
    ax1.legend(lines, labels, loc="best")
    fig.tight_layout()
    fig.savefig(outfile, dpi=150)
    plt.close(fig)


def plot_breakdown_labor_tax_steady_state(
    p: Params,
    z: float,
    ss: SteadyState,
    outfile: Path,
) -> None:
    """
    Ricardian non-equivalence illustration: same g/Y target, distortionary labor-tax finance
    lowers labor and consumption relative to lump-sum finance (steady-state comparison).
    """
    lt = solve_steady_state_labor_tax(p, z, p.g_y_ratio)
    _style()
    fig, ax = plt.subplots(figsize=(7, 4))
    labels = ["Consumption", "Labor", "Output"]
    x = np.arange(len(labels))
    wbar = 0.35
    ls_vals = np.array([ss.c / ss.y, ss.n, 1.0])
    lt_vals = np.array([lt.c / lt.y, lt.n, 1.0])
    ax.bar(x - wbar / 2, ls_vals, width=wbar, label="Lump-sum taxes")
    ax.bar(x + wbar / 2, lt_vals, width=wbar, label="Labor-income tax (distortionary)")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Normalized (Y=1 for output)")
    ax.set_title("Breakdown example: distortionary financing vs lump-sum (steady state)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(outfile, dpi=150)
    plt.close(fig)
