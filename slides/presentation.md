# FE5213 Presentation Plan (strictly from report PDF)

## Slide 1 — Title and research question

- **Title:** Fiscal Policy in a Stochastic Markov RBC Model
- **Question:** How does a government spending shock transmit in a Markov RBC economy, and how does it compare with U.S. reduced-form evidence?
- **Scope in this talk:** benchmark model, three fiscal experiments, state dependence, Ricardian test, empirical comparison, distortionary-tax extension.

---

## Slide 2 — Main findings (one-slide preview)

- Unforeseen one-quarter spending shock ($\Delta g = 0.05\,g_{\mathrm{ss}}$): impact responses are $Y=4.752$, $C=-5.077$, $N=7.424$, $I=29.875$ (log points).
- Same absolute $\Delta g$ across $z_L$ and $z_H$: state dependence is present but moderate in output/labor; strongest in investment.
- Ricardian equivalence holds in the lump-sum benchmark numerically: zero flow-budget residuals and zero allocation differences across financing regimes (reported precision).
- Data comparison: recursive VAR/LP imply negative impact $Y$ and $H$; IV-LP has weak first stage (horizon-0 statistic $2.1307$).

---

## Slide 3 — Model environment

- Two-state productivity Markov chain: $z_t\in\{z_L,z_H\}$ with transition matrix $\Pi$.
- Representative household with CRRA utility and disutility from labor; firm with Cobb-Douglas production.
- Government buys $g_t$, uses lump-sum taxes and one-period bonds in benchmark.
- Computational state is $(k,z)$; resource constraint in solved problem: $c_t+k_{t+1}-(1-\delta)k_t+g_t=Y_t$.

---

## Slide 4 — Calibration and long-run objects

- Quarterly baseline: $\beta=0.99$, $\sigma=2$, $\phi=1$, $\alpha=0.36$, $\delta=0.025$, $g/y=0.20$, $(z_L,z_H)=(0.98,1.02)$.
- Labor scale $\chi=15.5$ targets roughly one-third time worked in deterministic benchmark.
- Show **Table:** Markov fixed points vs deterministic steady state (`report_table_markov_fp_auto.tex` in report).
- Speak to key contrast: higher $z$ raises $y,c$ and slightly lowers $n$ at fixed point.

---

## Slide 5 — Solution method and numerical quality

- Value-function iteration on capital grid ($n_k=42$, range $[0.03k_{ss},4k_{ss}]$), labor optimized conditional on $k'$.
- Continuation values integrated over Markov transitions using interpolation.
- Convergence diagnostics (reported): 464 iterations; policy sup-distance 0; very small Euler residuals; resource residuals at machine precision.
- IRFs are Monte Carlo means over stochastic future productivity paths (baseline 1200 draws).

---

## Slide 6 — Ricardian equivalence: theory + quantitative check

- Proposition: with lump-sum taxes, complete markets, solvency, financing timing does not change $(c,n,k')$ for fixed $\{g_t\}$.
- Bond pricing is computed from stochastic Euler equation: $q_t=\beta E_t[(c_{t+1}/c_t)^{-\sigma}]$ (not imposed as constant).
- Show **Figure:** `ricardian_financing.png` (different $(\tau_t,B_t)$ paths for same $\{g_t\}$).
- Reported test result: max government budget residual = 0; max allocation difference across financing schemes = 0.

---

## Slide 7 — Fiscal experiment design

- Experiment 1: unforeseen one-time level shock at $t=0$.
- Experiment 2: foreseen one-time shock, realization at horizon $H=8$ (anticipation allowed).
- Experiment 3: permanent policy shift, $g/y$ increases by 2 percentage points.
- Important normalization: same absolute $\Delta g=0.05\,g_{ss}(\bar z)$ used in baseline/state-comparison exercises.

---

## Slide 8 — Unforeseen shock results (core IRFs)

- Show **IRF figures:** `irf_output_compare.png`, `irf_consumption_compare.png`, `irf_labor_compare.png`, `irf_investment_compare.png`.
- Show impact table values from report (h=0,h=1,h=4,h=8).
- Interpretation from report: $C$ crowding-out with higher $Y,N$ on impact; large % response in $I$ partly reflects denominator effect.
- Mention post-impact smoothness with zoom figure: `irf_unforeseen_zoom_h1_12.png`.

---

## Slide 9 — State dependence: $z_L$ vs $z_H$

- Same absolute spending increment applied at each state-specific Markov fixed point.
- Show four figures: `irf_output_zL_zH.png`, `irf_consumption_zL_zH.png`, `irf_labor_zL_zH.png`, `irf_investment_zL_zH.png`.
- Impact gaps reported in text: $z_H-z_L=$ 0.53 (Y), -0.54 (C), 0.82 (N), 4.12 (I) log points.
- Sensitivity slide option (backup): wider spread figure `irf_*_zL_zH_wide`.

---

## Slide 10 — Empirical comparison (fixed U.S. sample)

- Data: U.S. quarterly 1980Q1–2025Q4, transformed as $100\Delta\log(\cdot)$.
- Methods shown in report: recursive VAR(4), VAR-residual LP, IV-LP.
- Show figures: `empirical_var_irf_g_shock.png`, `empirical_local_projection.png`, `empirical_iv_lp.png`.
- Key caveat from report: IV-LP first-stage at horizon 0 is 2.1307 (weak instrument).

---

## Slide 11 — Model vs data scorecard

- Show **Table:** model-data impact comparison (`report_table_model_data_impact_auto.tex`).
- Core contrast: model predicts positive impact $Y,N,I$ and negative $C$; recursive VAR/LP show negative impact in $Y,C,I,H$.
- Limited alignment: impact consumption crowding-out sign.
- Literature anchors in report: Blanchard-Perotti, Ramey, Auerbach-Gorodnichenko (state dependence stronger in recession designs).

---

## Slide 12 — Extension, limitations, and conclusion

- Extension: distortionary labor-tax financing (auxiliary perfect-foresight stack) breaks Ricardian neutrality.
- Show figures: `breakdown_labor_tax_steady_state.png`, `irf_consumption_financing_regimes.png`, `irf_labor_financing_regimes.png`.
- Limitations (as stated): two-state productivity only, no nominal rigidities/HTM households, weak IV identification, denominator effects in investment IRFs.
- Final takeaway: model is a reproducible benchmark for Ricardian logic and stochastic fiscal propagation, but broader empirical fit needs richer mechanisms.

---

## Backup Q&A (strictly report-consistent)

| Q | A |
|---|---|
| Why Monte Carlo IRFs? | In the Markov setup, future productivity is stochastic; IRFs are expectations over simulated paths. |
| Why does investment jump so much on impact? | Responses are log-percent by variable; shock is larger relative to baseline investment denominator. |
| Is Ricardian equivalence only theoretical here? | No, numerical checks report zero budget residuals and zero allocation differences across financing schemes. |
| Does empirical evidence validate the benchmark? | Partially: consumption crowding-out sign aligns on impact in recursive designs, but output-hours comovement does not. |
