# FE5213 Group Project (Spring 2026)

**Stochastic Markov productivity** RBC with fiscal policy. Core numerical method: **value function iteration** on a capital grid for each productivity state $z\in\{z_L,z_H\}$, with labor optimized conditional on $k'$ and continuation values integrated with $\bm{\Pi}$ (fast `numpy.interp` continuation). Fiscal IRFs are **Monte Carlo means** over future productivity paths. The same **absolute** $\Delta g$ (from $g_{\mathrm{ss}}$ at ergodic-mean $z$) is used across $z_L$ vs.\ $z_H$ and for the foreseen shock realization.

## Requirements

- **Python 3.10+** recommended  
- See [`requirements.txt`](requirements.txt)

## Reproduce everything (figures + tables)

```powershell
cd "path\to\Group Project"
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
python main.py --use-frozen
```

Outputs are written to [`output/`](output/). For the exact report numbers and figures, use the bundled frozen dataset via `--use-frozen` (or `FE5213_USE_FROZEN=1`) so the empirical block does not drift with live FRED updates.  
Runtime has two regimes:

- **First full run (cold cache):** typically minutes (VFI solves dominate).
- **Subsequent runs (solver cache hit):** much faster, usually driven by Monte Carlo + empirical block.

`main.py` now caches solved Markov objects under `output/cache/`.

### Fast iteration vs full reproduction

```powershell
# Fast local check (no empirical + no wide-z sensitivity + lower MC)
python main.py --quick

# Full paper replication with frozen empirical data
$env:FE5213_USE_FROZEN=1; python main.py

# Full replication but skip costly wide-z sensitivity
python main.py --skip-sensitivity

# Force recompute VFI (ignore cache)
python main.py --force-recompute-solvers
```

Useful knobs:

- `--mc-draws <int>`: main IRF MC draws (default 1200)
- `--wide-mc-draws <int>`: wide-z sensitivity draws (default 800)
- `--skip-empirical`: skip VAR/LP/IV-LP block
- `--nk`, `--max-iter`, `--tol-v`, `--tol-kp`: VFI controls

### Empirical block (default)

`main.py` always runs [`src/empirical.py`](src/empirical.py):

- **Default:** downloads current **FRED** quarterly series from `1980-01-01` through the latest available observation (requires network and `pandas_datareader`).
- **Exact report replication / offline fixed snapshot:** set `FE5213_USE_FROZEN=1` or pass `--use-frozen` to use [`data/macro_quarterly_frozen.csv`](data/macro_quarterly_frozen.csv) instead.

```powershell
$env:FE5213_USE_FROZEN=1; python main.py
```

Writes VAR, LP, **IV-LP** figures and CSVs, plus [`output/empirical_meta.txt`](output/empirical_meta.txt).

### Key output files

| File | Description |
|------|-------------|
| `steady_state_markov.csv` | Fixed-point capital & allocations ($z_L$, $z_H$) + deterministic SS at $\bar z$ |
| `solver_diagnostics.txt` | VFI iterations, relative $V$ change, policy diff, Euler & resource residuals |
| `irf_summary_unforeseen_zbar.csv` | IRF numbers at selected horizons |
| `state_dependence_h0.txt` | $z_H-z_L$ at $h=0$ (common $\Delta g$) |
| `fiscal_shock_common.txt` | Absolute $\Delta g$ used across experiments |
| `ricardian_budget_residuals.txt` | Flow-budget residuals at equilibrium $q_t$ |
| `ricardian_allocation_invariance.txt` | Max abs diff $(c,n,k,y)$ across financing (should be $0$) |
| `literature_fiscal_summary.txt` | Literature-based empirical comparison |
| `empirical_meta.txt` | FRED/frozen source, sample, VAR/LP/IV definitions |
| `iv_lp_beta.csv`, `iv_lp_stderr.csv`, `iv_lp_first_stage.txt` | IV local projections |
| `macro_quarterly.csv`, `macro_quarterly_dlog.csv` | Levels and growth rates used in estimation |
| `empirical_var_irf_g_shock.png`, `empirical_var_irf_ordering_compare_i_h.png` | VAR IRFs |
| `empirical_local_projection.png`, `lp_irf_beta.csv`, `lp_irf_stderr.csv` | VAR-residual LP |
| `empirical_iv_lp.png` | IV local projections figure |
| `irf_*.png` | Model IRF figures for the report |

## Build the PDF report

After `python main.py` (so `output/*.png` exists):

```powershell
cd report
pdflatex fe5213_report.tex
bibtex fe5213_report
pdflatex fe5213_report.tex
pdflatex fe5213_report.tex
```

Or: `latexmk -pdf fe5213_report.tex` if installed.

Sources: [`report/fe5213_report.tex`](report/fe5213_report.tex), [`report/references.bib`](report/references.bib).

## One-shot reproduce script (Windows)

[`scripts/reproduce.ps1`](scripts/reproduce.ps1) runs `python main.py --use-frozen` then attempts LaTeX if `pdflatex` is on PATH.

## Project layout

| Path | Role |
|------|------|
| [`main.py`](main.py) | Single entry point |
| [`src/markov_rbc.py`](src/markov_rbc.py) | Markov VFI + simulation helpers |
| [`src/markov_experiments.py`](src/markov_experiments.py) | Fiscal IRFs (MC, foreseen backward DP, permanent) |
| [`src/ricardian_verify.py`](src/ricardian_verify.py) | Equilibrium $q_t$, financing paths, allocation invariance |
| [`src/export_tables.py`](src/export_tables.py) | CSV / summary tables |
| [`src/literature_empirical.py`](src/literature_empirical.py) | Literature summary text |
| [`src/steady_state.py`](src/steady_state.py) | Deterministic steady state (calibration target) |
| [`src/empirical.py`](src/empirical.py) | FRED/frozen data, VAR, LP, IV-LP |
| [`src/pf_paths.py`](src/pf_paths.py) | **Auxiliary** perfect-foresight path primitives (extension) |
| [`src/pf_extension.py`](src/pf_extension.py) | PF fiscal IRFs (lump-sum vs labor tax) |
| [`src/pf_labor_tax.py`](src/pf_labor_tax.py) | Labor-tax allocation on PF paths |
| [`slides/presentation.md`](slides/presentation.md) | Slide outline |

## Submission checklist

1. Run `python main.py` (with network for latest FRED, or `FE5213_USE_FROZEN=1` for the bundled snapshot).
2. Build `report/fe5213_report.pdf`.
3. Add **group member names** on the report title page.
4. Include `output/` figures (or confirm grader runs `main.py`).
5. Zip source + PDF + (optional) exported slides.
