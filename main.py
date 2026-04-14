#!/usr/bin/env python3
"""
FE5213 Group Project — reproduce all model figures, tables, and literature outputs.

Usage:
    python main.py

Outputs: ./output/  (figures PNG + CSV/text tables)
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import pickle
import time
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np

from src.export_tables import (
    irf_summary_table,
    steady_state_rows_markov,
    write_steady_state_csv,
)
from src.pf_extension import run_labor_tax_vs_lumpsum_unforeseen
from src.literature_empirical import write_literature_summary
from src.markov_experiments import mc_irf_unforeseen, run_all_markov_experiments
from src.markov_rbc import (
    MarkovSolution,
    find_k_fixed_point_grid,
    resource_residual_stats,
    simulate_baseline_policy,
    solve_markov_stationary,
)
from src.params import default_params
from src.plots_tables import (
    ensure_output_dir,
    plot_breakdown_labor_tax_steady_state,
    plot_financing_regime_irf_overlay,
    plot_irf_comparison,
    plot_ricardian_financing,
    plot_state_compare,
    plot_unforeseen_zoom_2x2,
)
from src.ricardian_verify import (
    allocation_invariance_across_financing,
    duplicate_baseline_paths_for_ricardian,
    equilibrium_q_path,
    financing_sequences_time_varying_q,
    government_budget_residual,
)
from src.steady_state import solve_steady_state


def _sha1_text(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8")).hexdigest()


def _file_sha1(path: Path) -> str:
    if not path.is_file():
        return "missing"
    return hashlib.sha1(path.read_bytes()).hexdigest()


def _solver_cache_file(
    out: Path,
    p,
    *,
    nk: int,
    tol_v: float,
    tol_kp: float,
    max_iter: int,
    relax: float,
    tag: str,
) -> Path:
    payload = {
        "tag": tag,
        "params": asdict(p),
        "solver": {
            "nk": nk,
            "tol_v": tol_v,
            "tol_kp": tol_kp,
            "max_iter": max_iter,
            "relax": relax,
        },
        "solver_src": _file_sha1(Path("src/markov_rbc.py")),
    }
    h = _sha1_text(json.dumps(payload, sort_keys=True))
    cache_dir = out / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir / f"markov_{tag}_{h[:12]}.pkl"


def _load_or_solve_markov(
    out: Path,
    p,
    *,
    nk: int,
    tol_v: float,
    tol_kp: float,
    max_iter: int,
    relax: float,
    tag: str,
    use_cache: bool,
    force_recompute: bool,
) -> tuple[MarkovSolution, bool]:
    cache_file = _solver_cache_file(
        out,
        p,
        nk=nk,
        tol_v=tol_v,
        tol_kp=tol_kp,
        max_iter=max_iter,
        relax=relax,
        tag=tag,
    )
    if use_cache and (not force_recompute) and cache_file.is_file():
        with cache_file.open("rb") as f:
            sol = pickle.load(f)
        return sol, True

    sol = solve_markov_stationary(
        p, nk=nk, tol_v=tol_v, tol_kp=tol_kp, max_iter=max_iter, relax=relax
    )
    if use_cache:
        with cache_file.open("wb") as f:
            pickle.dump(sol, f, protocol=pickle.HIGHEST_PROTOCOL)
    return sol, False


def _build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="FE5213 full replication pipeline")
    ap.add_argument("--mc-draws", type=int, default=1200, help="MC draws for main IRFs")
    ap.add_argument(
        "--wide-mc-draws",
        type=int,
        default=800,
        help="MC draws for wide z spread sensitivity",
    )
    ap.add_argument("--skip-sensitivity", action="store_true", help="Skip wide-z sensitivity")
    ap.add_argument("--skip-empirical", action="store_true", help="Skip empirical block")
    ap.add_argument("--quick", action="store_true", help="Fast dev run preset")
    ap.add_argument("--use-frozen", action="store_true", help="Force frozen empirical dataset")
    ap.add_argument("--no-cache", action="store_true", help="Disable solver cache usage")
    ap.add_argument(
        "--force-recompute-solvers",
        action="store_true",
        help="Ignore cache and recompute VFI solutions",
    )
    ap.add_argument("--nk", type=int, default=42, help="Capital grid size for VFI")
    ap.add_argument("--max-iter", type=int, default=800, help="Max iterations for VFI")
    ap.add_argument("--tol-v", type=float, default=5e-5, help="Relative V tolerance")
    ap.add_argument("--tol-kp", type=float, default=5e-5, help="Policy tolerance")
    ap.add_argument(
        "--clean-output",
        action="store_true",
        help="Delete previously generated output files before running",
    )
    return ap


def _clean_generated_outputs(out: Path) -> None:
    patterns = ("*.png", "*.csv", "*.txt", "*.tex")
    removed = 0
    for pat in patterns:
        for p in out.glob(pat):
            # Keep cache and user-provided data snapshots.
            if p.name == "macro_quarterly_frozen.csv":
                continue
            p.unlink(missing_ok=True)
            removed += 1
    print(f"Cleaned {removed} generated files in {out}.")


def _parse_first_stage_h0(out: Path) -> float:
    p = out / "iv_lp_first_stage.txt"
    if not p.is_file():
        return float("nan")
    for line in p.read_text(encoding="utf-8").splitlines():
        key = "first_stage_t_stat_sq_on_instrument_h0="
        if line.startswith(key):
            try:
                return float(line[len(key) :].strip())
            except ValueError:
                return float("nan")
    return float("nan")


def _tex_escape(s: str) -> str:
    return (
        s.replace("\\", r"\textbackslash{}")
        .replace("_", r"\_")
        .replace("%", r"\%")
        .replace("&", r"\&")
        .replace("#", r"\#")
        .replace("{", r"\{")
        .replace("}", r"\}")
    )


def _write_report_body_tables_auto(out: Path, p, results: dict, emp: dict) -> None:
    """Export main-body LaTeX tables so the PDF stays synced to the run."""

    def fmt3(x: float) -> str:
        try:
            if np.isnan(float(x)):
                return "---"
        except (TypeError, ValueError):
            return "---"
        return f"{float(x):.3f}"

    sol = results["sol"]
    rows_m = steady_state_rows_markov(p, sol)
    zbar = results["z_bar"]
    h0 = h1 = h4 = h8 = None
    for h in (0, 1, 4, 8):
        row = (
            h,
            float(zbar["y_pct"][h]),
            float(zbar["c_pct"][h]),
            float(zbar["n_pct"][h]),
            float(zbar["i_pct"][h]),
        )
        if h == 0:
            h0 = row
        elif h == 1:
            h1 = row
        elif h == 4:
            h4 = row
        elif h == 8:
            h8 = row

    if emp.get("ok") and "irf_df" in emp and len(emp["irf_df"]) > 0:
        var0 = emp["irf_df"].iloc[0]
        var_g = float(var0.get("g", float("nan")))
        var_y = float(var0.get("y", float("nan")))
        var_c = float(var0.get("c", float("nan")))
        var_i = float(var0.get("i", float("nan")))
        var_h = float(var0.get("h", float("nan")))
    else:
        var_g = var_y = var_c = var_i = var_h = float("nan")

    if emp.get("ok") and "lp_df" in emp and len(emp["lp_df"]) > 0:
        lp0 = emp["lp_df"].iloc[0]
        lp_y = float(lp0.get("y", float("nan")))
        lp_c = float(lp0.get("c", float("nan")))
        lp_i = float(lp0.get("i", float("nan")))
        lp_h = float(lp0.get("h", float("nan")))
    else:
        lp_y = lp_c = lp_i = lp_h = float("nan")

    if emp.get("ok") and "iv_df" in emp and len(emp["iv_df"]) > 0:
        iv0 = emp["iv_df"].iloc[0]
        iv_y = float(iv0.get("y", float("nan")))
        iv_c = float(iv0.get("c", float("nan")))
        iv_i = float(iv0.get("i", float("nan")))
        iv_h = float(iv0.get("h", float("nan")))
    else:
        iv_y = iv_c = iv_i = iv_h = float("nan")

    fs_h0 = _parse_first_stage_h0(out)

    lines_markov = [
        "% Auto-generated by main.py; do not hand-edit.",
        r"\begin{table}[htbp]",
        r"  \centering",
        r"  \caption{Long-run objects: Markov fixed points vs.\ deterministic steady state at $\bar z$.}",
        r"  \label{tab:markov_fp}",
        r"  \begin{tabular}{@{}lrrrrrr@{}}",
        r"    \toprule",
        r"    Case & $k$ & $y$ & $c$ & $n$ & $i$ & $g$ \\",
        r"    \midrule",
        f"    Markov FP, $z_L={rows_m[0]['z']:.2f}$ & {fmt3(rows_m[0]['k'])} & {fmt3(rows_m[0]['y'])} & {fmt3(rows_m[0]['c'])} & {fmt3(rows_m[0]['n'])} & {fmt3(rows_m[0]['i'])} & {fmt3(rows_m[0]['g'])} \\\\",
        f"    Markov FP, $z_H={rows_m[1]['z']:.2f}$ & {fmt3(rows_m[1]['k'])} & {fmt3(rows_m[1]['y'])} & {fmt3(rows_m[1]['c'])} & {fmt3(rows_m[1]['n'])} & {fmt3(rows_m[1]['i'])} & {fmt3(rows_m[1]['g'])} \\\\",
        f"    Det.\\ SS at $\\bar z={rows_m[2]['z']:.2f}$ & {fmt3(rows_m[2]['k'])} & {fmt3(rows_m[2]['y'])} & {fmt3(rows_m[2]['c'])} & {fmt3(rows_m[2]['n'])} & {fmt3(rows_m[2]['i'])} & {fmt3(rows_m[2]['g'])} \\\\",
        r"    \bottomrule",
        r"  \end{tabular}",
        r"  \medskip",
        "",
        r"  \raggedright\small",
        r"  \textit{Note:} Values correspond to the numerical replication underlying all figures and appendix tables. Per-capita or aggregate consistency follows the model specification; all objects are in level units.",
        r"\end{table}",
        "",
    ]
    (out / "report_table_markov_fp_auto.tex").write_text(
        "\n".join(lines_markov), encoding="utf-8"
    )

    lines_irf = [
        "% Auto-generated by main.py; do not hand-edit.",
        r"\begin{table}[htbp]",
        r"  \centering",
        r"  \caption{Unforeseen one-time $g$ shock at $\bar z$ neighborhood: IRF summary ($100\times\log$ ratio, MC mean).}",
        r"  \label{tab:irf_impact}",
        r"  \begin{tabular}{@{}rcccc@{}}",
        r"    \toprule",
        r"    Horizon $h$ & $Y$ & $C$ & $N$ & $I$ \\",
        r"    \midrule",
        f"    ${int(h0[0])}$ & {fmt3(h0[1])} & {fmt3(h0[2])} & {fmt3(h0[3])} & {fmt3(h0[4])} \\\\",
        f"    ${int(h1[0])}$ & {fmt3(h1[1])} & {fmt3(h1[2])} & {fmt3(h1[3])} & {fmt3(h1[4])} \\\\",
        f"    ${int(h4[0])}$ & {fmt3(h4[1])} & {fmt3(h4[2])} & {fmt3(h4[3])} & {fmt3(h4[4])} \\\\",
        f"    ${int(h8[0])}$ & {fmt3(h8[1])} & {fmt3(h8[2])} & {fmt3(h8[3])} & {fmt3(h8[4])} \\\\",
        r"    \bottomrule",
        r"  \end{tabular}",
        r"  \medskip",
        "",
        r"  \raggedright\small",
        r"  \textit{Note:} These are log-percentage IRFs, $100\times \log(x_t^{\text{shock}}/x_t^{\text{base}})$, not fiscal multipliers. The same-run shock normalization appears in Appendix Table~\ref{tab:auto_shock_irf_diag}.",
        r"\end{table}",
        "",
    ]
    (out / "report_table_irf_impact_auto.tex").write_text(
        "\n".join(lines_irf), encoding="utf-8"
    )

    lines_var = [
        "% Auto-generated by main.py; do not hand-edit.",
        r"\begin{table}[htbp]",
        r"  \centering",
        r"  \caption{Data VAR: impact responses to a one s.d.\ orthogonalized $G$ innovation (baseline Cholesky $G,Y,C,I,H$).}",
        r"  \label{tab:var_impact}",
        r"  \begin{tabular}{@{}lrrrrr@{}}",
        r"    \toprule",
        r"    $h=0$ & $G$ & $Y$ & $C$ & $I$ & $H$ \\",
        r"    \midrule",
        f"    Response & {fmt3(var_g)} & {fmt3(var_y)} & {fmt3(var_c)} & {fmt3(var_i)} & {fmt3(var_h)} \\\\",
        r"    \bottomrule",
        r"  \end{tabular}",
        r"  \medskip",
        "",
        r"  \raggedright\small",
        r"  \textit{Note:} Units match the transformed data (approximate quarterly percentage change). The frozen-data sample and first-stage diagnostics are reported in Appendix Table~\ref{tab:auto_emp_diag}.",
        r"\end{table}",
        "",
    ]
    (out / "report_table_var_impact_auto.tex").write_text(
        "\n".join(lines_var), encoding="utf-8"
    )

    lines_model_data = [
        "% Auto-generated by main.py; do not hand-edit.",
        r"\begin{table}[htbp]",
        r"  \centering",
        r"  \caption{Impact responses: model and data side by side (native units).}",
        r"  \label{tab:model_data_impact}",
        r"  \small",
        r"  \begin{tabular}{@{}lrrrr@{}}",
        r"    \toprule",
        r"    Specification & $Y$ & $C$ & $I$ & $N/H$ \\",
        r"    \midrule",
        f"    Model: Markov RBC, unforeseen one-time $g$ shock & {fmt3(h0[1])} & {fmt3(h0[2])} & {fmt3(h0[4])} & {fmt3(h0[3])} \\\\",
        f"    Data: recursive VAR & {fmt3(var_y)} & {fmt3(var_c)} & {fmt3(var_i)} & {fmt3(var_h)} \\\\",
        f"    Data: VAR-residual LP & {fmt3(lp_y)} & {fmt3(lp_c)} & {fmt3(lp_i)} & {fmt3(lp_h)} \\\\",
        f"    Data: IV-LP & {fmt3(iv_y)} & {fmt3(iv_c)} & {fmt3(iv_i)} & {fmt3(iv_h)} \\\\",
        r"    \bottomrule",
        r"  \end{tabular}",
        r"  \medskip",
        "",
        r"  \raggedright\small",
        (
            r"  \textit{Note:} The model row reports $100\times\log(x_t^{\mathrm{shock}}/x_t^{\mathrm{base}})$"
            r" at $h=0$ for the unforeseen one-time spending shock; empirical rows report impact coefficients"
            r" from the frozen-sample VAR, VAR-residual local projections, and IV local projections."
            r" The model uses labor $N$, while empirical rows use hours $H$."
            rf" Scales are therefore not directly comparable; the table is intended for sign and pattern comparison."
            rf" The IV first-stage statistic at $h=0$ is {fs_h0:.4f}, so the IV row should be read with the weak-instrument caveat."
        ),
        r"\end{table}",
        "",
    ]
    (out / "report_table_model_data_impact_auto.tex").write_text(
        "\n".join(lines_model_data), encoding="utf-8"
    )


def _write_report_appendix_auto(
    out: Path,
    p,
    results: dict,
    res_diag: dict,
    res_tax: float,
    res_debt: float,
    inv: dict,
    emp: dict,
) -> None:
    """Export compact diagnostics tables for direct inclusion in the PDF."""
    sol = results["sol"]
    eul = results["euler"]
    zb, _ = p.ergodic_z()
    ss_bar = solve_steady_state(p, z=zb)
    dg = float(results.get("dg_common", float("nan")))
    dg_y = 100.0 * dg / max(ss_bar.y, 1e-12)
    dg_c = 100.0 * dg / max(ss_bar.c, 1e-12)
    dg_i = 100.0 * dg / max(ss_bar.i, 1e-12)
    dg_g = 100.0 * dg / max(ss_bar.g, 1e-12)

    zbar = results["z_bar"]
    zL = results["z_L"]
    zH = results["z_H"]

    h_list = [0, 1, 4, 8]
    irf_rows = []
    for h in h_list:
        irf_rows.append(
            (
                h,
                float(zbar["y_pct"][h]),
                float(zbar["c_pct"][h]),
                float(zbar["n_pct"][h]),
                float(zbar["i_pct"][h]),
            )
        )

    fs_h0 = _parse_first_stage_h0(out)
    n_levels = int(len(emp["df"])) if emp.get("ok") and "df" in emp else -1
    n_dlog = int(len(emp["dx"])) if emp.get("ok") and "dx" in emp else -1
    if emp.get("ok") and "df" in emp and len(emp["df"]) > 0:
        sample_start = str(emp["df"].index.min().date())
        sample_end = str(emp["df"].index.max().date())
    else:
        sample_start = "n/a"
        sample_end = "n/a"
    if emp.get("ok") and "irf_df" in emp and len(emp["irf_df"]) > 0:
        h0 = emp["irf_df"].iloc[0]
        var_g = float(h0.get("g", float("nan")))
        var_y = float(h0.get("y", float("nan")))
        var_c = float(h0.get("c", float("nan")))
        var_i = float(h0.get("i", float("nan")))
        var_h = float(h0.get("h", float("nan")))
    else:
        var_g = var_y = var_c = var_i = var_h = float("nan")

    lines: list[str] = []
    lines.append("% Auto-generated by main.py; do not hand-edit.")
    lines.append(r"\section{Appendix: numerical and empirical diagnostics}")
    lines.append("")
    lines.append(r"\begin{table}[htbp]")
    lines.append(r"  \centering")
    lines.append(r"  \caption{Numerical solution diagnostics (same replication as figures and tables).}")
    lines.append(r"  \label{tab:auto_solver_diag}")
    lines.append(r"  \begin{tabular}{@{}lr@{}}")
    lines.append(r"    \toprule")
    lines.append(r"    Metric & Value \\")
    lines.append(r"    \midrule")
    lines.append(f"    VFI iterations & {sol.iterations:d} \\\\")
    lines.append(f"    VFI $\\|\\Delta V\\|_\\infty$ (abs) & {sol.sup_V_diff:.6e} \\\\")
    lines.append(f"    VFI $\\|\\Delta k'\\|_\\infty$ & {sol.sup_kp_diff:.6e} \\\\")
    lines.append(f"    Euler max abs residual (pct) & {eul['max_abs_euler_pct']:.6e} \\\\")
    lines.append(f"    Euler mean abs residual (pct) & {eul['mean_abs_euler_pct']:.6e} \\\\")
    lines.append(f"    Euler p95 abs residual (pct) & {eul['p95_abs_euler_pct']:.6e} \\\\")
    lines.append(f"    Resource max relative residual & {res_diag['max_resource_rel']:.6e} \\\\")
    lines.append(f"    Resource mean relative residual & {res_diag['mean_resource_rel']:.6e} \\\\")
    lines.append(f"    Gov budget residual (tax finance), max abs & {res_tax:.6e} \\\\")
    lines.append(f"    Gov budget residual (debt smoothing), max abs & {res_debt:.6e} \\\\")
    lines.append(
        "    Allocation invariance, max abs diff across $(c,n,k,y,g)$"
        f" & {max(float(v) for v in inv.values()):.6e} \\\\"
    )
    lines.append(r"    \bottomrule")
    lines.append(r"  \end{tabular}")
    lines.append(r"\end{table}")
    lines.append("")
    lines.append(r"\begin{table}[htbp]")
    lines.append(r"  \centering")
    lines.append(r"  \caption{Shock normalization and main model moments (MC means).}")
    lines.append(r"  \label{tab:auto_shock_irf_diag}")
    lines.append(r"  \begin{tabular}{@{}lrrrr@{}}")
    lines.append(r"    \toprule")
    lines.append(r"    Object & $Y$ & $C$ & $N$ & $I$ \\")
    lines.append(r"    \midrule")
    lines.append(
        f"    $\\Delta g$ as \\% of baseline ($\\bar z$ fixed point proxy) & {dg_y:.3f} & {dg_c:.3f} & --- & {dg_i:.3f} \\\\"
    )
    lines.append(
        f"    Unforeseen one-time, $h=0$ ($100\\log$) & {irf_rows[0][1]:.3f} & {irf_rows[0][2]:.3f} & {irf_rows[0][3]:.3f} & {irf_rows[0][4]:.3f} \\\\"
    )
    lines.append(
        f"    Unforeseen one-time, $h=1$ ($100\\log$) & {irf_rows[1][1]:.3f} & {irf_rows[1][2]:.3f} & {irf_rows[1][3]:.3f} & {irf_rows[1][4]:.3f} \\\\"
    )
    lines.append(
        f"    Unforeseen one-time, $h=4$ ($100\\log$) & {irf_rows[2][1]:.3f} & {irf_rows[2][2]:.3f} & {irf_rows[2][3]:.3f} & {irf_rows[2][4]:.3f} \\\\"
    )
    lines.append(
        f"    Unforeseen one-time, $h=8$ ($100\\log$) & {irf_rows[3][1]:.3f} & {irf_rows[3][2]:.3f} & {irf_rows[3][3]:.3f} & {irf_rows[3][4]:.3f} \\\\"
    )
    lines.append(
        f"    State dependence at $h=0$: $z_H-z_L$ ($100\\log$) & {float(zH['y_pct'][0]-zL['y_pct'][0]):.3f} & {float(zH['c_pct'][0]-zL['c_pct'][0]):.3f} & {float(zH['n_pct'][0]-zL['n_pct'][0]):.3f} & {float(zH['i_pct'][0]-zL['i_pct'][0]):.3f} \\\\"
    )
    lines.append(r"    \bottomrule")
    lines.append(r"  \end{tabular}")
    lines.append(r"\end{table}")
    lines.append("")
    lines.append(r"\begin{table}[htbp]")
    lines.append(r"  \centering")
    lines.append(r"  \caption{Empirical diagnostics (same replication as the empirical figures).}")
    lines.append(r"  \label{tab:auto_emp_diag}")
    lines.append(r"  \begin{tabular}{@{}lr@{}}")
    lines.append(r"    \toprule")
    lines.append(r"    Item & Value \\")
    lines.append(r"    \midrule")
    data_source_tex = _tex_escape(str(emp.get("data_source", "unknown")))
    lines.append(f"    Data source tag & \\texttt{{{data_source_tex}}} \\\\")
    lines.append(f"    Sample start & \\texttt{{{_tex_escape(sample_start)}}} \\\\")
    lines.append(f"    Sample end & \\texttt{{{_tex_escape(sample_end)}}} \\\\")
    lines.append(f"    Effective levels sample length & {n_levels:d} \\\\")
    lines.append(f"    Effective $100\\Delta\\log$ sample length & {n_dlog:d} \\\\")
    lines.append(f"    VAR impact response $h=0$ for $G$ & {var_g:.3f} \\\\")
    lines.append(f"    VAR impact response $h=0$ for $Y$ & {var_y:.3f} \\\\")
    lines.append(f"    VAR impact response $h=0$ for $C$ & {var_c:.3f} \\\\")
    lines.append(f"    VAR impact response $h=0$ for $I$ & {var_i:.3f} \\\\")
    lines.append(f"    VAR impact response $h=0$ for $H$ & {var_h:.3f} \\\\")
    lines.append(f"    IV first-stage statistic at $h=0$ ($t^2$) & {fs_h0:.4f} \\\\")
    lines.append(r"    \bottomrule")
    lines.append(r"  \end{tabular}")
    lines.append(r"\end{table}")
    lines.append("")
    lines.append(
        rf"\noindent\textit{{Note:}} These tables are produced in the same numerical replication as the paper's figures and main tables. Common one-time shock size is $\Delta g={dg:.8f}$ ({dg_g:.3f}\% of baseline $g$)."
    )

    (out / "report_appendix_auto.tex").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def main(argv: list[str] | None = None) -> None:
    args = _build_arg_parser().parse_args(argv)
    if args.quick:
        args.mc_draws = min(args.mc_draws, 300)
        args.wide_mc_draws = min(args.wide_mc_draws, 300)
        args.skip_sensitivity = True
        args.skip_empirical = True

    if args.use_frozen:
        os.environ["FE5213_USE_FROZEN"] = "1"

    use_cache = not args.no_cache
    out = ensure_output_dir("output")
    if args.clean_output:
        _clean_generated_outputs(out)
    p = default_params()

    t0 = time.perf_counter()
    print("Preparing Markov RE solves (cache-aware)...")
    sol, hit_base = _load_or_solve_markov(
        out,
        p,
        nk=args.nk,
        tol_v=args.tol_v,
        tol_kp=args.tol_kp,
        max_iter=args.max_iter,
        relax=1.0,
        tag="baseline",
        use_cache=use_cache,
        force_recompute=args.force_recompute_solvers,
    )
    p_hi = replace(p, g_y_ratio=min(p.g_y_ratio + 0.02, 0.45))
    sol_hi, hit_hi = _load_or_solve_markov(
        out,
        p_hi,
        nk=args.nk,
        tol_v=args.tol_v,
        tol_kp=args.tol_kp,
        max_iter=args.max_iter,
        relax=1.0,
        tag="high_gy",
        use_cache=use_cache,
        force_recompute=args.force_recompute_solvers,
    )
    print(
        f"Solver cache status: baseline={'hit' if hit_base else 'miss'}, high_gy={'hit' if hit_hi else 'miss'}"
    )
    results = run_all_markov_experiments(
        p,
        T=40,
        shock_quarter=8,
        shock_frac=0.05,
        permanent_delta_g_y=0.02,
        mc_draws=args.mc_draws,
        sol=sol,
        sol_hi=sol_hi,
    )
    sol = results["sol"]
    res_diag = resource_residual_stats(sol)
    print("Euler diagnostic:", results["euler"])
    print("Resource residual:", res_diag)
    print(f"Core model block done in {time.perf_counter() - t0:.1f}s")
    (out / "solver_diagnostics.txt").write_text(
        f"VFI_iterations={sol.iterations}\n"
        f"VFI_sup_V_diff_abs={sol.sup_V_diff:.6e}\n"
        f"VFI_sup_kp_diff={sol.sup_kp_diff:.6e}\n"
        f"Euler_max_abs_pct_on_grid={results['euler']['max_abs_euler_pct']:.6e}\n"
        f"Euler_mean_abs_pct_on_grid={results['euler']['mean_abs_euler_pct']:.6e}\n"
        f"Euler_p95_abs_pct_on_grid={results['euler']['p95_abs_euler_pct']:.6e}\n"
        f"Resource_max_rel={res_diag['max_resource_rel']:.6e}\n"
        f"Resource_mean_rel={res_diag['mean_resource_rel']:.6e}\n"
        f"Note: V stopping uses relative change in V plus policy stability; see report.\n",
        encoding="utf-8",
    )

    zb, _ = p.ergodic_z()
    z_bar_idx = 0 if abs(p.z_L - zb) <= abs(p.z_H - zb) else 1

    # Steady-state / fixed-point table (Markov policy at z_L, z_H + deterministic SS at z_bar)
    rows_m = steady_state_rows_markov(p, sol)
    labels_m = ["z_L_MarkovFP", "z_H_MarkovFP", "det_SS_ergodic_z"]
    write_steady_state_csv(out / "steady_state_markov.csv", rows_m, labels_m)

    h = np.arange(40)

    def trim(d: np.ndarray) -> np.ndarray:
        return np.asarray(d)[: len(h)]

    zu = results["z_bar"]
    zf = results["foreseen_zbar"]
    zp = results["permanent_zbar"]

    plot_irf_comparison(
        h,
        {
            "Unforeseen one-time": trim(zu["y_pct"]),
            "Foreseen one-time": trim(zf["y_pct"]),
            "Permanent g/y shift": trim(zp["y_pct"]),
        },
        "Output IRFs: Markov RE + fiscal experiments (approx. %, MC mean)",
        out / "irf_output_compare.png",
    )
    plot_irf_comparison(
        h,
        {
            "Unforeseen": trim(zu["c_pct"]),
            "Foreseen": trim(zf["c_pct"]),
            "Permanent": trim(zp["c_pct"]),
        },
        "Consumption IRFs (Markov RE, MC mean)",
        out / "irf_consumption_compare.png",
    )
    plot_irf_comparison(
        h,
        {
            "Unforeseen one-time": trim(zu["n_pct"]),
            "Foreseen one-time": trim(zf["n_pct"]),
            "Permanent shift": trim(zp["n_pct"]),
        },
        "Labor IRFs (Markov RE, MC mean)",
        out / "irf_labor_compare.png",
    )
    plot_irf_comparison(
        h,
        {
            "Unforeseen one-time": trim(zu["i_pct"]),
            "Foreseen one-time": trim(zf["i_pct"]),
            "Permanent shift": trim(zp["i_pct"]),
        },
        "Investment IRFs (Markov RE, MC mean)",
        out / "irf_investment_compare.png",
    )
    h_zoom = np.arange(1, 13)
    plot_unforeseen_zoom_2x2(
        h_zoom,
        trim(zu["y_pct"])[1:13],
        trim(zu["c_pct"])[1:13],
        trim(zu["n_pct"])[1:13],
        trim(zu["i_pct"])[1:13],
        out / "irf_unforeseen_zoom_h1_12.png",
    )

    zLu = results["z_L"]
    zHu = results["z_H"]
    plot_state_compare(
        h,
        trim(zLu["y_pct"]),
        trim(zHu["y_pct"]),
        "Approx. % change",
        "Output: unforeseen g shock, z_L vs z_H (Markov RE)",
        out / "irf_output_zL_zH.png",
    )
    plot_state_compare(
        h,
        trim(zLu["c_pct"]),
        trim(zHu["c_pct"]),
        "Approx. % change",
        "Consumption: z_L vs z_H",
        out / "irf_consumption_zL_zH.png",
    )
    plot_state_compare(
        h,
        trim(zLu["n_pct"]),
        trim(zHu["n_pct"]),
        "Approx. % change",
        "Labor: z_L vs z_H",
        out / "irf_labor_zL_zH.png",
    )
    plot_state_compare(
        h,
        trim(zLu["i_pct"]),
        trim(zHu["i_pct"]),
        "Approx. % change",
        "Investment: z_L vs z_H",
        out / "irf_investment_zL_zH.png",
    )

    # Wider productivity spread (optional; expensive)
    if args.skip_sensitivity:
        print("Sensitivity block skipped (--skip-sensitivity).")
        for name in (
            "irf_output_zL_zH_wide.png",
            "irf_consumption_zL_zH_wide.png",
            "irf_labor_zL_zH_wide.png",
            "irf_investment_zL_zH_wide.png",
        ):
            (out / name).unlink(missing_ok=True)
    else:
        p_wide = replace(p, z_L=0.90, z_H=1.10)
        print("Sensitivity: wider z_L, z_H spread...")
        t_w = time.perf_counter()
        sol_w, hit_w = _load_or_solve_markov(
            out,
            p_wide,
            nk=args.nk,
            tol_v=args.tol_v,
            tol_kp=args.tol_kp,
            max_iter=args.max_iter,
            relax=1.0,
            tag="wide_z",
            use_cache=use_cache,
            force_recompute=args.force_recompute_solvers,
        )
        print(f"Sensitivity solver cache: {'hit' if hit_w else 'miss'}")
        zLw = mc_irf_unforeseen(
            p_wide, sol_w, z0_idx=0, T=40, shock_frac=0.05, n_draws=args.wide_mc_draws
        )
        zHw = mc_irf_unforeseen(
            p_wide, sol_w, z0_idx=1, T=40, shock_frac=0.05, n_draws=args.wide_mc_draws
        )
        plot_state_compare(
            h,
            trim(zLw["y_pct"]),
            trim(zHw["y_pct"]),
            "Approx. % change",
            r"Output: z_L=0.90 vs z_H=1.10 (sensitivity)",
            out / "irf_output_zL_zH_wide.png",
        )
        plot_state_compare(
            h,
            trim(zLw["c_pct"]),
            trim(zHw["c_pct"]),
            "Approx. % change",
            r"Consumption: wide spread sensitivity",
            out / "irf_consumption_zL_zH_wide.png",
        )
        plot_state_compare(
            h,
            trim(zLw["n_pct"]),
            trim(zHw["n_pct"]),
            "Approx. % change",
            r"Labor: wide spread sensitivity",
            out / "irf_labor_zL_zH_wide.png",
        )
        plot_state_compare(
            h,
            trim(zLw["i_pct"]),
            trim(zHw["i_pct"]),
            "Approx. % change",
            r"Investment: wide spread sensitivity",
            out / "irf_investment_zL_zH_wide.png",
        )
        print(f"Sensitivity block done in {time.perf_counter() - t_w:.1f}s")

    # Ricardian: equilibrium q(c), financing paths, budget residuals
    Tq = 50
    z_fix = np.full(Tq + 1, z_bar_idx, dtype=int)
    k0r = find_k_fixed_point_grid(sol, z_bar_idx)
    path_r = simulate_baseline_policy(sol, k0r, z_fix)
    q_path = equilibrium_q_path(p, sol, path_r["k"], z_fix)
    tau_tax, B_tax, tau_d, B_d = financing_sequences_time_varying_q(
        path_r["g"], q_path, tau_smooth=float(np.mean(path_r["g"]))
    )
    plot_ricardian_financing(
        tau_tax,
        tau_d,
        B_d,
        out / "ricardian_financing.png",
    )
    res_tax = government_budget_residual(
        path_r["g"], tau_tax, B_tax, q_path
    )
    res_debt = government_budget_residual(path_r["g"], tau_d, B_d, q_path)
    (out / "ricardian_budget_residuals.txt").write_text(
        f"max_abs_flow_residual_tax_finance={res_tax:.3e}\n"
        f"max_abs_flow_residual_debt_smoothing={res_debt:.3e}\n"
        "Bond prices q_t from stochastic Euler (expectation over z').\n",
        encoding="utf-8",
    )
    path_a, path_b = duplicate_baseline_paths_for_ricardian(p, sol, k0r, z_fix)
    inv = allocation_invariance_across_financing(path_a, path_b)
    lines_inv = ["Ricardian allocation invariance (two copies of same simulated path):"]
    for k, v in sorted(inv.items()):
        lines_inv.append(f"  {k}={v:.3e}")
    lines_inv.append(
        "Interpretation: lump-sum financing does not enter household FOCs; "
        "only {g_t} and productivity matter for (c,n,k)."
    )
    (out / "ricardian_allocation_invariance.txt").write_text(
        "\n".join(lines_inv) + "\n", encoding="utf-8"
    )
    (out / "fiscal_shock_common.txt").write_text(
        f"dg_absolute_common={results.get('dg_common', float('nan')):.8f}\n"
        "(Same absolute one-time Delta g for z_L vs z_H and foreseen experiment; "
        "scaled as shock_frac * g_ss at ergodic-mean z.)\n",
        encoding="utf-8",
    )

    # IRF summary CSV (grading)
    (out / "irf_summary_unforeseen_zbar.csv").write_text(
        irf_summary_table((0, 1, 4, 8), zu["y_pct"], zu["c_pct"], zu["n_pct"], zu["i_pct"]),
        encoding="utf-8",
    )
    (out / "state_dependence_h0.txt").write_text(
        f"h=0 unforeseen one-time g shock (approx pct points, MC mean):\n"
        f"  dY  z_H - z_L = {zHu['y_pct'][0] - zLu['y_pct'][0]:.4f}\n"
        f"  dC  z_H - z_L = {zHu['c_pct'][0] - zLu['c_pct'][0]:.4f}\n"
        f"  dN  z_H - z_L = {zHu['n_pct'][0] - zLu['n_pct'][0]:.4f}\n"
        f"  dI  z_H - z_L = {zHu['i_pct'][0] - zLu['i_pct'][0]:.4f}\n",
        encoding="utf-8",
    )

    # Extension: labor tax (PF transition — distortionary wedge; kept for breakdown demo)
    ss_bar = solve_steady_state(p, z=zb)
    plot_breakdown_labor_tax_steady_state(p, zb, ss_bar, out / "breakdown_labor_tax_steady_state.png")
    fin = run_labor_tax_vs_lumpsum_unforeseen(p, T=80, g_shock_frac=0.05)
    h_fin = np.arange(40)
    plot_financing_regime_irf_overlay(
        h_fin,
        fin["irf_lumpsum"]["c_pct"][: len(h_fin)],
        fin["irf_labor_tax"]["c_pct"][: len(h_fin)],
        "Lump-sum taxes (baseline PF)",
        "Labor-income tax, balanced budget",
        "Approx. % change (100·log ratio)",
        "Consumption IRF: financing regimes (PF auxiliary)",
        out / "irf_consumption_financing_regimes.png",
    )
    plot_financing_regime_irf_overlay(
        h_fin,
        fin["irf_lumpsum"]["n_pct"][: len(h_fin)],
        fin["irf_labor_tax"]["n_pct"][: len(h_fin)],
        "Lump-sum taxes (baseline PF)",
        "Labor-income tax, balanced budget",
        "Approx. % change (100·log ratio)",
        "Labor IRF: financing regimes (PF auxiliary)",
        out / "irf_labor_financing_regimes.png",
    )

    write_literature_summary(out)

    from src.empirical import run_empirical_block

    if args.skip_empirical:
        emp = {"ok": False, "data_source": "skipped", "message": "Empirical block skipped"}
        print("Empirical block skipped (--skip-empirical).")
        for name in (
            "empirical_var_irf_g_shock.png",
            "empirical_var_irf_ordering_compare_i_h.png",
            "empirical_local_projection.png",
            "empirical_iv_lp.png",
            "var_orth_irf_shock_g.csv",
            "var_orth_irf_shock_g_y_before_g.csv",
            "var_orth_irf_stderr_g_first.csv",
            "var_orth_irf_stderr_y_first.csv",
            "lp_irf_beta.csv",
            "lp_irf_stderr.csv",
            "iv_lp_beta.csv",
            "iv_lp_stderr.csv",
            "iv_lp_first_stage.txt",
            "empirical_meta.txt",
            "macro_quarterly.csv",
            "macro_quarterly_dlog.csv",
        ):
            (out / name).unlink(missing_ok=True)
    else:
        t_e = time.perf_counter()
        emp = run_empirical_block(out)
        print(f"Empirical block done in {time.perf_counter() - t_e:.1f}s")
    _write_report_body_tables_auto(out, p, results, emp)
    _write_report_appendix_auto(out, p, results, res_diag, res_tax, res_debt, inv, emp)
    print("Empirical block:", emp.get("data_source", ""), emp.get("message", ""))

    print(f"Done in {time.perf_counter() - t0:.1f}s. Outputs in {out.resolve()}")


if __name__ == "__main__":
    main()
