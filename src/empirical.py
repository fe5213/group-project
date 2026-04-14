"""
U.S. quarterly fiscal empirical block: FRED data (latest by default), VAR, and local projections.

Default behavior downloads **current** FRED vintages (requires network). For offline reproducibility:
  set environment variable FE5213_USE_FROZEN=1 to use data/macro_quarterly_frozen.csv instead.

Series (FRED): GDPC1, PCECC96, GPDIC1, GCEC1, HOANBS; USRECQ (NBER recession indicator, quarterly).
"""
from __future__ import annotations

import os
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

try:
    import matplotlib.pyplot as plt
except ImportError:  # pragma: no cover
    plt = None

try:
    import pandas_datareader.data as web
except ImportError:  # pragma: no cover
    web = None

try:
    import statsmodels.api as sm
    from statsmodels.tsa.api import VAR
except ImportError:  # pragma: no cover
    sm = None
    VAR = None


FRED_CORE = {
    "y": "GDPC1",
    "c": "PCECC96",
    "i": "GPDIC1",
    "g": "GCEC1",
    "h": "HOANBS",
}
FRED_REC = "USRECQ"


def _project_root() -> Path:
    return Path(__file__).resolve().parent.parent


def default_frozen_macro_path() -> Path:
    return _project_root() / "data" / "macro_quarterly_frozen.csv"


def load_macro_from_csv(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path, parse_dates=["DATE"], index_col="DATE")
    df.columns = [str(c).lower() for c in df.columns]
    return df.sort_index()


def fetch_fred_series(sid: str, start: str, end: str | None) -> pd.DataFrame:
    if web is None:
        raise RuntimeError("pandas_datareader not installed")
    kwargs: dict = {"start": start}
    if end is not None:
        kwargs["end"] = end
    s = web.DataReader(sid, "fred", **kwargs)
    return s


def fetch_macro(
    start: str = "1980-01-01",
    end: str | None = None,
    *,
    include_recession: bool = True,
) -> pd.DataFrame:
    """Align quarterly macro + optional USRECQ on inner join."""
    if web is None:
        raise RuntimeError("pandas_datareader not installed")
    frames = []
    for name, sid in FRED_CORE.items():
        s = fetch_fred_series(sid, start, end)
        s = s.rename(columns={sid: name})
        frames.append(s)
    df = pd.concat(frames, axis=1).dropna()
    if include_recession:
        r = fetch_fred_series(FRED_REC, start, end)
        r = r.rename(columns={FRED_REC: "rec"})
        df = df.join(r, how="inner").dropna()
    return df.sort_index()


def build_var_sample(df: pd.DataFrame) -> pd.DataFrame:
    """100 * first difference of logs ~ quarterly % change."""
    cols = [c for c in ["y", "c", "i", "g", "h"] if c in df.columns]
    x = np.log(df[cols])
    dx = 100.0 * x.diff()
    out = dx.dropna()
    if "rec" in df.columns:
        out = out.join(df["rec"].shift(1).loc[out.index].rename("rec_l1"), how="left")
    return out


def estimate_var_irf_ordered(
    dx: pd.DataFrame,
    order: list[str],
    *,
    shock_var: str,
    maxlags: int = 4,
    periods: int = 20,
) -> tuple[pd.DataFrame, pd.DataFrame | None, list[str], int]:
    if VAR is None:
        raise RuntimeError("statsmodels not installed")
    if shock_var not in order:
        raise ValueError(f"shock_var {shock_var!r} not in order")
    shock_idx = order.index(shock_var)
    use = [c for c in order if c in dx.columns]
    data = dx[use].dropna().copy()
    data.index = pd.RangeIndex(len(data))
    model = VAR(data)
    res = model.fit(maxlags=maxlags, ic=None)
    irf = res.irf(periods)
    orth = irf.orth_irfs
    out: dict[str, np.ndarray] = {}
    for i, name in enumerate(use):
        out[name] = orth[:, i, shock_idx]
    irf_df = pd.DataFrame(out)

    stderr_df: pd.DataFrame | None = None
    try:
        se = irf.stderr(orth=True)
        out_se: dict[str, np.ndarray] = {}
        for i, name in enumerate(use):
            out_se[name] = se[:, i, shock_idx]
        stderr_df = pd.DataFrame(out_se)
    except Exception:
        stderr_df = None

    return irf_df, stderr_df, use, shock_idx


def local_projection_hac(
    dx: pd.DataFrame,
    order: list[str],
    *,
    var_lags: int = 4,
    max_h: int = 20,
    hac_maxlags: int | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Jordà-style LP: shock = innovation to G from recursive VAR (same ordering as ``order``).
    y_{i,t+h} = const + beta_h * e_{g,t} + sum_l Gamma_l x_{t-l} + u.
    """
    if sm is None or VAR is None:
        raise RuntimeError("statsmodels not installed")
    use = [c for c in order if c in dx.columns]
    data = dx[use].dropna().copy()
    data.index = pd.RangeIndex(len(data))
    T = len(data)
    fit = VAR(data).fit(maxlags=var_lags, ic=None)
    resid_g = fit.resid["g"]

    betas = pd.DataFrame(index=range(max_h + 1), columns=use, dtype=float)
    ses = pd.DataFrame(index=range(max_h + 1), columns=use, dtype=float)

    for h in range(max_h + 1):
        for outv in use:
            y_list: list[float] = []
            X_rows: list[list[float]] = []
            for t in range(var_lags, T - h):
                ei = float(resid_g.iloc[t - var_lags])
                yi = float(data[outv].iloc[t + h])
                xrow = [1.0, ei]
                for lag in range(1, var_lags + 1):
                    for v in use:
                        xrow.append(float(data[v].iloc[t - lag]))
                y_list.append(yi)
                X_rows.append(xrow)
            y_arr = np.asarray(y_list)
            X_arr = np.asarray(X_rows)
            if y_arr.size < X_arr.shape[1] + 5:
                betas.loc[h, outv] = np.nan
                ses.loc[h, outv] = np.nan
                continue
            hac_lags = hac_maxlags if hac_maxlags is not None else min(
                12, max(4, h + var_lags)
            )
            ols = sm.OLS(y_arr, X_arr).fit(
                cov_type="HAC", cov_kwds={"maxlags": hac_lags}
            )
            betas.loc[h, outv] = float(ols.params[1])
            ses.loc[h, outv] = float(ols.bse[1])

    return betas, ses


def iv_local_projection_predetermined_g(
    dx: pd.DataFrame,
    order: list[str],
    *,
    max_h: int = 20,
    inst_lag: int = 2,
    var_lags: int = 4,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """
    IV local projections: treat quarterly growth in G as endogenous; instrument with
    its own value lagged ``inst_lag`` quarters (strictly predetermined in Δlog units).

    This is weaker than narrative defense shocks but transparent and replicable; see meta file.
    """
    if sm is None:
        raise RuntimeError("statsmodels not installed")
    use = [c for c in order if c in dx.columns]
    d = dx[use].dropna().copy()
    d.index = pd.RangeIndex(len(d))
    T = len(d)
    betas = pd.DataFrame(index=range(max_h + 1), columns=use, dtype=float)
    ses = pd.DataFrame(index=range(max_h + 1), columns=use, dtype=float)
    first_stage: dict[int, float] = {}

    t_lo = max(var_lags, inst_lag)
    for h in range(max_h + 1):
        t_hi = T - 1 - h
        if t_hi < t_lo + 10:
            continue
        n_obs = t_hi - t_lo + 1
        g_end = np.zeros(n_obs)
        z_inst = np.zeros(n_obs)
        y_by_out: dict[str, np.ndarray] = {v: np.zeros(n_obs) for v in use}
        lag_block = np.zeros((n_obs, var_lags * len(use)))
        row = 0
        for t in range(t_lo, t_hi + 1):
            g_end[row] = float(d["g"].iloc[t])
            z_inst[row] = float(d["g"].iloc[t - inst_lag])
            for outv in use:
                y_by_out[outv][row] = float(d[outv].iloc[t + h])
            col = 0
            for lag in range(1, var_lags + 1):
                for v in use:
                    lag_block[row, col] = float(d[v].iloc[t - lag])
                    col += 1
            row += 1

        ones = np.ones((n_obs, 1))
        X_fs = np.hstack([ones, z_inst.reshape(-1, 1), lag_block])
        fs = sm.OLS(g_end, X_fs).fit()
        g_hat = fs.fittedvalues
        try:
            first_stage[h] = float(fs.tvalues[1] ** 2)
        except Exception:
            first_stage[h] = float("nan")

        hac_m = min(12, max(4, h + var_lags))
        for outv in use:
            yv = y_by_out[outv]
            X2 = np.hstack([ones, g_hat.reshape(-1, 1), lag_block])
            try:
                ss = sm.OLS(yv, X2).fit(
                    cov_type="HAC", cov_kwds={"maxlags": hac_m}
                )
                betas.loc[h, outv] = float(ss.params[1])
                ses.loc[h, outv] = float(ss.bse[1])
            except Exception:
                betas.loc[h, outv] = np.nan
                ses.loc[h, outv] = np.nan

    meta_iv = {
        "instrument": f"L{inst_lag}_g_in_dlog_space",
        "first_stage_chi2_instrument_coef_sq_h0": first_stage.get(0, float("nan")),
        "note": "Predetermined instrument (weak-IV caveat); contrast with Cholesky VAR.",
    }
    return betas, ses, meta_iv


def save_iv_lp_figure(
    betas: pd.DataFrame,
    ses: pd.DataFrame,
    outfile: Path,
    *,
    title: str,
    z: float = 1.96,
) -> None:
    if plt is None:
        raise RuntimeError("matplotlib not installed")
    hq = np.arange(len(betas))
    fig, ax = plt.subplots(figsize=(9, 5.5))
    cols = [c for c in ["g", "y", "c", "i", "h"] if c in betas.columns]
    for col in cols:
        y = betas[col].values
        ax.plot(hq, y, label=col.upper(), linewidth=2)
        if col in ses.columns:
            se = ses[col].values
            lo = y - z * se
            hi = y + z * se
            ax.fill_between(hq, lo, hi, alpha=0.15)
    ax.axhline(0.0, color="black", linewidth=0.7)
    ax.set_title(title)
    ax.set_xlabel("Horizon (quarters)")
    ax.set_ylabel("IV-LP coefficient (Δlog ×100 units)")
    ax.legend()
    fig.tight_layout()
    outfile.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(outfile, dpi=150)
    plt.close(fig)


def save_lp_figure(
    betas: pd.DataFrame,
    ses: pd.DataFrame,
    outfile: Path,
    *,
    title: str,
    z: float = 1.96,
) -> None:
    if plt is None:
        raise RuntimeError("matplotlib not installed")
    hq = np.arange(len(betas))
    fig, ax = plt.subplots(figsize=(9, 5.5))
    cols = [c for c in ["g", "y", "c", "i", "h"] if c in betas.columns]
    for col in cols:
        y = betas[col].values
        ax.plot(hq, y, label=col.upper(), linewidth=2)
        if col in ses.columns:
            se = ses[col].values
            lo = y - z * se
            hi = y + z * se
            ax.fill_between(hq, lo, hi, alpha=0.15)
    ax.axhline(0.0, color="black", linewidth=0.7)
    ax.set_title(title)
    ax.set_xlabel("Horizon (quarters)")
    ax.set_ylabel("LP coefficient (Δlog units, ×100 per shock)")
    ax.legend()
    fig.tight_layout()
    outfile.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(outfile, dpi=150)
    plt.close(fig)


def save_var_irf_figure(
    irf_df: pd.DataFrame,
    outfile: Path,
    *,
    stderr_df: pd.DataFrame | None = None,
    title: str = "VAR orthogonalized IRF",
    z_score: float = 1.96,
) -> None:
    if plt is None:
        raise RuntimeError("matplotlib not installed")
    hq = np.arange(len(irf_df))
    fig, ax = plt.subplots(figsize=(9, 5.5))
    cols = [c for c in ["g", "y", "c", "i", "h"] if c in irf_df.columns]
    for col in cols:
        y = irf_df[col].values
        ax.plot(hq, y, label=col.upper(), linewidth=2)
        if stderr_df is not None and col in stderr_df.columns:
            se = stderr_df[col].values
            lo = y - z_score * se
            hi = y + z_score * se
            ax.fill_between(hq, lo, hi, alpha=0.18)
    ax.axhline(0.0, color="black", linewidth=0.7)
    ax.set_title(title)
    ax.set_xlabel("Horizon (quarters)")
    ax.set_ylabel("Response (Δlog ×100)")
    ax.legend()
    fig.tight_layout()
    outfile.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(outfile, dpi=150)
    plt.close(fig)


def save_var_irf_ordering_compare_i_h(
    irf_g_first: pd.DataFrame,
    irf_y_first: pd.DataFrame,
    outfile: Path,
    *,
    stderr_g: pd.DataFrame | None = None,
    stderr_y: pd.DataFrame | None = None,
    z_score: float = 1.96,
) -> None:
    if plt is None:
        raise RuntimeError("matplotlib not installed")
    fig, axes = plt.subplots(2, 1, figsize=(9, 6.5), sharex=True)
    hq = np.arange(len(irf_g_first))
    for ax, var, ylab in zip(axes, ("i", "h"), ("Investment I", "Hours H")):
        if var not in irf_g_first.columns:
            continue
        y1 = irf_g_first[var].values
        y2 = irf_y_first[var].values
        ax.plot(hq, y1, label="G first", linewidth=2, color="C0")
        ax.plot(hq, y2, label="Y before G", linewidth=2, color="C1", linestyle="--")
        if stderr_g is not None and var in stderr_g.columns:
            lo = y1 - z_score * stderr_g[var].values
            hi = y1 + z_score * stderr_g[var].values
            ax.fill_between(hq, lo, hi, alpha=0.12, color="C0")
        if stderr_y is not None and var in stderr_y.columns:
            lo2 = y2 - z_score * stderr_y[var].values
            hi2 = y2 + z_score * stderr_y[var].values
            ax.fill_between(hq, lo2, hi2, alpha=0.12, color="C1")
        ax.axhline(0.0, color="black", linewidth=0.7)
        ax.set_ylabel(f"Δlog {var.upper()}")
        ax.set_title(f"{ylab}: ordering robustness")
        ax.legend(loc="best", fontsize=8)
    axes[-1].set_xlabel("Horizon (quarters)")
    fig.suptitle("VAR: Cholesky ordering (I, H)", fontsize=11, y=1.02)
    fig.tight_layout()
    outfile.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(outfile, dpi=150, bbox_inches="tight")
    plt.close(fig)


def write_empirical_meta(
    outdir: Path,
    df: pd.DataFrame,
    dx: pd.DataFrame,
    data_source: str,
    *,
    fred_start: str,
) -> None:
    lines = [
        f"data_source={data_source}",
        f"fred_sample_requested_start={fred_start}",
        f"fred_sample_end={df.index.max().date()}",
        f"n_quarters_levels={len(df)}",
        f"n_quarters_dlog={len(dx)}",
        "variables_y_c_i_g_h=GDPC1,PCECC96,GPDIC1,GCEC1,HOANBS",
        "recession_series=USRECQ (if present in levels df)",
        "transform_levels=100*diff(log) quarterly approx pct change",
        "var_order_baseline=G,Y,C,I,H Cholesky",
        "lp_shock=first_equation_VAR_residual_recursive_order",
        "iv_lp=2SLS_predetermined_instrument_L2_g",
        "",
        "Identification: Cholesky/LP are reduced-form; IV-LP uses lagged G (predetermined, weak-IV caveat).",
    ]
    (outdir / "empirical_meta.txt").write_text("\n".join(lines), encoding="utf-8")


def write_model_var_comparison_stub(outdir: Path) -> None:
    text = """Model vs empirical comparison (qualitative)

Stochastic Markov RBC (course model): fiscal IRFs from the solved Markov RE policies; signs depend
on calibration and shock definition (see report).

Empirical outputs (see empirical_meta.txt for sample dates):
- Cholesky VAR and VAR-residual LP: reduced-form benchmarks.
- IV local projections (iv_lp_*.csv / empirical_iv_lp.png): predetermined lag of G as instrument.

Compare Y, C, I, and hours responses across these objects and the literature summary file.
"""
    (outdir / "model_var_comparison_notes.txt").write_text(text, encoding="utf-8")


def run_empirical_block(
    outdir: Path,
    *,
    start: str = "1980-01-01",
    end: str | None = None,
) -> dict:
    """
    Prefer live FRED unless FE5213_USE_FROZEN=1. Falls back to frozen CSV if download fails.
    """
    outdir.mkdir(parents=True, exist_ok=True)
    result: dict = {"ok": False, "message": "", "data_source": ""}
    use_frozen = os.environ.get("FE5213_USE_FROZEN", "").strip().lower() in (
        "1",
        "true",
        "yes",
    )
    frozen_path = default_frozen_macro_path()

    try:
        if use_frozen and frozen_path.is_file():
            df = load_macro_from_csv(frozen_path)
            result["data_source"] = f"frozen:{frozen_path.name}"
        else:
            try:
                df = fetch_macro(start=start, end=end, include_recession=True)
                result["data_source"] = f"fred_live start={start}" + (
                    f" end={end}" if end else " end=latest"
                )
            except Exception as e1:
                if frozen_path.is_file():
                    warnings.warn(f"FRED fetch failed ({e1}); using frozen CSV.")
                    df = load_macro_from_csv(frozen_path)
                    result["data_source"] = f"fallback_frozen:{frozen_path.name}"
                else:
                    raise

        df.to_csv(outdir / "macro_quarterly.csv")
        dx = build_var_sample(df)
        dx.to_csv(outdir / "macro_quarterly_dlog.csv")

        order = ["g", "y", "c", "i", "h"]
        irf_g, stderr_g, use_o, _ = estimate_var_irf_ordered(
            dx, order, shock_var="g", maxlags=4, periods=20
        )
        irf_g.to_csv(outdir / "var_orth_irf_shock_g.csv", index_label="quarter")
        if stderr_g is not None:
            stderr_g.to_csv(
                outdir / "var_orth_irf_stderr_g_first.csv", index_label="quarter"
            )

        irf_y, stderr_y, _, _ = estimate_var_irf_ordered(
            dx, ["y", "g", "c", "i", "h"], shock_var="g", maxlags=4, periods=20
        )
        irf_y.to_csv(outdir / "var_orth_irf_shock_g_y_before_g.csv", index_label="quarter")

        save_var_irf_figure(
            irf_g,
            outdir / "empirical_var_irf_g_shock.png",
            stderr_df=stderr_g,
            title=(
                "VAR IRF: Cholesky G shock (G,Y,C,I,H) — 95% asymptotic bands"
            ),
        )
        save_var_irf_ordering_compare_i_h(
            irf_g,
            irf_y,
            outdir / "empirical_var_irf_ordering_compare_i_h.png",
            stderr_g=stderr_g,
            stderr_y=stderr_y,
        )

        betas_lp, ses_lp = local_projection_hac(dx, order, var_lags=4, max_h=20)
        betas_lp.to_csv(outdir / "lp_irf_beta.csv", index_label="horizon")
        ses_lp.to_csv(outdir / "lp_irf_stderr.csv", index_label="horizon")
        save_lp_figure(
            betas_lp,
            ses_lp,
            outdir / "empirical_local_projection.png",
            title="Local projections (HAC): response to VAR innovation in G (recursive)",
        )

        betas_iv, ses_iv, meta_iv = iv_local_projection_predetermined_g(
            dx, order, max_h=20, inst_lag=2, var_lags=4
        )
        betas_iv.to_csv(outdir / "iv_lp_beta.csv", index_label="horizon")
        ses_iv.to_csv(outdir / "iv_lp_stderr.csv", index_label="horizon")
        save_iv_lp_figure(
            betas_iv,
            ses_iv,
            outdir / "empirical_iv_lp.png",
            title="IV local projections (HAC-2SLS): G instrumented by L2(g)",
        )
        (outdir / "iv_lp_first_stage.txt").write_text(
            "first_stage_t_stat_sq_on_instrument_h0="
            f"{meta_iv.get('first_stage_chi2_instrument_coef_sq_h0', float('nan')):.4f}\n"
            f"instrument={meta_iv.get('instrument', '')}\n"
            f"{meta_iv.get('note', '')}\n",
            encoding="utf-8",
        )

        write_empirical_meta(outdir, df, dx, result["data_source"], fred_start=start)
        write_model_var_comparison_stub(outdir)

        result.update(
            {
                "ok": True,
                "message": "VAR + LP + IV-LP complete.",
                "df": df,
                "dx": dx,
                "irf_df": irf_g,
                "lp_df": betas_lp,
                "lp_se": ses_lp,
                "iv_df": betas_iv,
                "iv_se": ses_iv,
                "iv_meta": meta_iv,
            }
        )
    except Exception as exc:
        warnings.warn(f"Empirical block failed: {exc}")
        result["message"] = str(exc)
    return result
