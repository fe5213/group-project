---
title: "Cross-Implementation Validation Report"
subtitle: "Fiscal RBC — Ricardian equivalence and numerical consistency"
status: Final
version: "1.2"
date: "2026-04-17"
language: en
---

# Cross-Implementation Validation Report

**FE5213 — Fiscal RBC with Ricardian Equivalence**

**Purpose**: Independent audit of numerical consistency between this group’s replication and a separately prepared reference implementation (deterministic perfect-foresight notebook). 

---

## Abstract

Two independent implementations of the same frictionless fiscal RBC structure can disagree in levels even when both are correct, because steady states and impulse responses depend on calibration (frequency, parameter values, and how government spending enters the resource constraint). This report separates **calibration effects** from **coding errors** using analytical steady states and equation-by-equation residual checks along transition paths. Both implementations pass steady-state and Ricardian-equivalence checks. A one-period indexing mistake was found and corrected **only** in the auxiliary perfect-foresight routines used for the labour-tax extension; the stochastic Markov benchmark and Ricardian verification based on equilibrium bond prices are unaffected. After correction, dynamic residuals match machine precision. Numerical disagreement with the reference implementation is overwhelmingly explained by different calibrations, not by inconsistent equilibrium conditions.

---

## Table of contents

1. [Executive summary](#1-executive-summary)
2. [Shared economic structure](#2-shared-economic-structure)
3. [Why headline numbers differ](#3-why-headline-numbers-differ)
4. [Analytical steady-state cross-check](#4-analytical-steady-state-cross-check)
5. [Transition path: residual audit](#5-transition-path-residual-audit)
6. [Ricardian equivalence](#6-ricardian-equivalence)
7. [Technical correction applied](#7-technical-correction-applied)
8. [What was left unchanged](#8-what-was-left-unchanged)
9. [Notes on the reference implementation](#9-notes-on-the-reference-implementation)
10. [Reproducibility](#10-reproducibility)
11. [Which graphical outputs depend on the correction](#11-which-graphical-outputs-depend-on-the-correction)
12. [Conclusion](#12-conclusion)

---

## 1. Executive summary

The reference implementation and this group’s package produce different impulse-response levels and steady-state magnitudes. That gap is **not**, by itself, evidence that either model is misspecified: two valid programs solving the same economic structure at **different calibrations** will generally produce different numbers.

This report uses checks that do **not** depend on matching calibrations:

- Closed-form Cobb–Douglas steady states as independent benchmarks.
- Maximum absolute residuals for the resource constraint, intratemporal labour condition, and intertemporal Euler equation along each solved transition path.
- Government flow budgets and present-value relationships under alternative financing.

**Verdict**

| Object | Reference impl. | This package (before fix) | This package (after fix) |
|--------|-----------------|---------------------------|---------------------------|
| Steady state vs. closed form | Match | Match | Match |
| Transition residuals | \(\lesssim 10^{-8}\) on Euler | Euler \(\approx 2\times 10^{-4}\) (systematic) | \(\lesssim 10^{-13}\) |
| Ricardian budgets / PV | Consistent | Consistent (Markov path) | Consistent |

The pre-correction Euler gap traced to a **single timing error** in the capital return entering the perfect-foresight Euler loop in the **lump-sum** and **labour-tax** auxiliary transition solvers. The main stochastic solver’s Euler conditions were already indexed correctly.

---

## 2. Shared economic structure

Both implementations embed:

- CRRA preferences and convex labour disutility.
- Cobb–Douglas production with competitive factor prices.
- Lump-sum taxes, one-period government debt, and a flow budget linking taxes, debt, and spending.
- Goods and capital accumulation clearing the resource constraint.

**Conditions checked numerically**

1. Intratemporal: marginal rate of substitution between leisure and consumption equals the real wage times marginal utility of consumption.
2. Intertemporal: consumption Euler equation with rental rate evaluated at the capital stock **beginning** the next period (standard timing).
3. Ricardian proposition: for a fixed spending path under lump-sum finance, real allocations do not depend on how taxes are timed relative to debt issuance.

---

## 3. Why headline numbers differ

The reference notebook uses an **annual-style** calibration (higher depreciation and discounting per period in a way consistent with a yearly frequency) and **absolute** government spending in levels. This group’s published benchmark uses **quarterly** parameters and spending as a **fixed share of output** in steady state, plus a **two-state productivity Markov chain** for the main experiments.

Representative differences include discount factors, depreciation, capital share, labour-disutility scale, and the mapping from \(g\) to resources. Any one of these shifts the steady state and the entire IRF. **Comparing figures without harmonising calibrations compares different economies**, not two solutions of the same numerical problem.

---

## 4. Analytical steady-state cross-check

Imposing steady state in the Cobb–Douglas economy yields the usual rental rate from the Euler equation, a unique capital–labour ratio from marginal productivity, and then labour and aggregates from intratemporal and resource conditions. That algebra is independent of either team’s nonlinear solver.

For the reference calibration, closed-form objects (e.g. \(k\approx 2.426\), \(n\approx 0.547\), \(c\approx 0.500\)) coincide with the reference solver to many digits. This group’s quarterly calibration matches **its** corresponding closed form to machine precision as well.

---

## 5. Transition path: residual audit

After a perfect-foresight path is computed, we re-evaluate the three core equations **as written in economic notation** (not merely trust the solver’s internal tolerance).

**Findings**

- The reference transition solver exhibits small residuals on the order of \(10^{-8}\) or better—consistent with a correctly specified system up to floating-point limits.
- Before correction, this package’s auxiliary perfect-foresight solver showed a **large, systematic** Euler residual (order \(10^{-4}\)) while resource and intratemporal conditions were tight. That pattern indicates a **specification slip in one equation** (wrong timing in the rental rate), not generic poor convergence.
- After correction, Euler residuals fall to the order of \(10^{-14}\).

**Root cause (conceptual)**  
The rental rate in the Euler equation for period \(t\) must be evaluated using capital at date \(t+1\) together with labour at \(t+1\). The erroneous code advanced the capital index one period too far in the array representation of the path.

**Typical size of the mistake**  
On a representative test path, level differences between mistaken and corrected paths for \(c\), \(n\), \(y\), and \(k\) are small as a share of steady state (often well below one-tenth of a percent) but **not** negligible for publication-quality equation checks.

**Scope**  
Only the deterministic perfect-foresight stacks used for the **labour-tax vs. lump-sum dynamic overlays** were affected. The value-function iteration solution, Monte Carlo IRFs, and Ricardian exercise built from **stochastic** policies and equilibrium bond prices do **not** use the faulty routine.

---

## 6. Ricardian equivalence

Both teams recover:

- **Flow consistency:** \(q_t B_{t+1} + \tau_t = B_t + g_t\) at numerical precision for constructed financing schemes.
- **Present-value consistency** of tax streams across schemes when bond prices follow the household’s stochastic discount factor.

This group’s main Ricardian figure is based on **simulated paths under the Markov solution** and **equilibrium** \(q_t\); it does not rely on the auxiliary perfect-foresight Euler that was corrected.

---

## 7. Technical correction applied

The correction consists of aligning the capital index in the Euler residual with standard timing: use the rental rate at \((k_{t+1}, n_{t+1})\), and align the terminal Euler condition with the **last** simulated capital stock rather than an exogenous steady-state anchor alone where the solver already pins down \(k_T\). The same logical fix applies to the labour-tax variant of the perfect-foresight problem.

No change was made to economic assumptions, preferences, or the stochastic core of the model.

---

## 8. What was left unchanged

- Quarterly calibration and Markov productivity structure for the main paper.
- Monte Carlo fiscal IRFs, state-dependence experiments, and empirical block.
- The reference notebook itself (read-only comparison).

---

## 9. Notes on the reference implementation

These are documentation and clarity points, not refutations of the economics:

1. Text in one cell describes utility with a symbol that clashes with risk aversion in the code—purely notational.
2. The debt path under geometric retirement should state explicitly how the terminal period is closed.
3. Plotting calls at import time can block automated testing unless a non-interactive backend is selected.
4. A reduced-form “hand-to-mouth” illustration is useful for intuition but is not a full heterogeneous-agent equilibrium.

---

## 10. Reproducibility

All residual diagnostics and steady-state benchmarks described here can be regenerated by the **automated validation suite** distributed with this group’s replication package. The suite recomputes closed-form steady states, re-evaluates transition residuals, and prints government-budget checks.

**How to run:** from the replication environment, execute the validation entry point documented in the **project replication instructions** (same document that describes full figure and table reproduction). No manual copying of equations is required.

---

## 11. Which graphical outputs depend on the correction

Re-running the **full numerical pipeline** after applying the Euler correction changes **only** the two dynamic panels that overlay **lump-sum** and **labour-income tax** financing in the extension section of the main PDF report:

| Role in the main report | Affected by Euler fix? |
|---------------------------|------------------------|
| Consumption: lump-sum vs. labour-tax financing (dynamic overlay) | **Yes** |
| Labour: same comparison | **Yes** |
| Ricardian financing diagram (stochastic \(q_t\), allocation invariance) | No |
| Steady-state labour-tax vs. lump-sum bar-style comparison | No (static first-order conditions only) |
| All Markov Monte Carlo IRFs (unforeseen / foreseen / permanent; \(z_L\) vs. \(z_H\)) | No |
| Empirical VAR / local-projection figures | No |

**Recommendation for final submission:** regenerate those two overlays once after the code correction, then rebuild the PDF so figures and algebra are mutually consistent.

---

## 12. Conclusion

1. The reference implementation is **internally coherent** under its own calibration.  
2. This group’s stochastic benchmark and Ricardian verification are **internally coherent** under the quarterly calibration.  
3. Apparent conflicts in levels between the two sets of results are **primarily calibration comparisons**, not proof of a wrong model on either side.  
4. The only substantive coding issue identified was a **timing bug in the auxiliary perfect-foresight Euler**, now fixed; its footprint is limited to the two financing-regime dynamic figures listed above.
