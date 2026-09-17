# Spline-empirical

A synthetic, reproducible testbed for answering one question:

> **Which spline parameterization, knot placement, fitting method, time
> parameterization, regularization and sampling strategy give the most
> sample-efficient and robust reconstruction of complex / snappy motion from
> sparse noisy observations?**

This repository is the companion instrument to
`SplineGS-empirical`, whose README documents
that the per-Gaussian cubic Hermite spline in **SplineGS** (Park et al.,
[`park2024splinegs`]) and its successor WebSpline fail on (H2) velocity
discontinuities, snap-rest-snap and near-Nyquist motion, and that no method in
that study has an *adaptive* temporal budget (MACP only prunes; AD-GS fixes
knots at `frames/3`; OriGS anchors are frame-discrete). That study also found
spline-class methods are the fastest to evaluate and the only ones with
editability, so the open slot is a **sample-efficient, noise-robust,
non-uniform/growth-capable spline** — exactly what this benchmark measures.

> **Status: v0.3, Iteration 2 in.** All registries, the synthetic motion
> suite, the runner and **131 tests** are implemented and green. Section 6–7
> are the first-pass results (2 703 deduplicated conditions, 3 seeds; see §5).
> **Section 8 is the Iteration 2 stress test** of the first-pass winners
> (GP-Matérn, adaptive knots, penalized cubic B-splines, Sobol): realistic
> structured noise, time-parameterization, extrapolation, adversarial
> conditioning and auto-tuned knot/penalty counts, over 4 455 conditions ×
> 3–5 seeds. Tables in §6 and §8 are reproducible from `results/*.jsonl` and
> `results/iteration2/*.jsonl` via `splinebench.report`; the `--full` grids
> remain to be run.

---

## 1. Research questions

| ID | Question | Primary metrics |
|---|---|---|
| RQ1 | Sample efficiency: which parameterizations reach a target error at 5–20 observations? | `pos_rmse` vs `budget`, AULC |
| RQ2 | Snappy / high-frequency fidelity: which survive kinks (bounce), snap-rest (staccato, double_step), bang-bang and chirps? | `pos_rmse`, `vel_rmse`, `jerk_rmse`, `peak_time_err` |
| RQ3 | Knot placement: does data-adaptive placement beat uniform at fixed `n_sites`? Which placement wins per motion class? | `pos_rmse`, `jerk_rmse` at fixed `n_params` |
| RQ4 | Fitting / regularization: when do robust losses and difference penalties pay off under noise and outliers? | error under `noise_std`, `outlier_frac`, `missing_frac` |
| RQ5 | Sampling: do derivative-peak / importance / active samplers beat random and LHS at equal budget? | AULC, final `pos_rmse` |
| RQ6 | Cost: accuracy per stored parameter and per second (Pareto) | `n_params_total`, `fit_time_s`, `pos_rmse` |

Orientation (`SO(3)`/`SE(3)`) is deliberately **out of scope in v1** (the
trajectory is per-axis translation). The representation registry is designed
so quaternion / Lie-group entries can be added without touching the runner
(see Appendix A).

## 2. Experimental unit

Every experiment is a tuple

```
(representation, knot placement, fitting/optimization, time parameterization,
 regularization/constraints, sampling strategy)
```

plus the nuisance axes we explicitly control and report:

```
motion, budget, seed, noise_std, outlier_frac/scale, missing_frac, bias
```

The runner (`splinebench/experiment.py`) executes one tuple as:

1. build a deterministic synthetic motion with **analytic derivatives up to
   snap** (`splinebench/motions.py`);
2. draw observation times with a sampling strategy (possibly adaptive,
   querying an `ObservationOracle` that adds noise/outliers/missing values);
3. fit a monotone time parameterization `u(t)` from the observations (linear,
   chord, centripetal, acceleration- or jerk-weighted; optionally oracle);
4. place `n_sites` knots/control sites in `u` (uniform, quantile, chord,
   curvature, derivative peaks, RDP, split-merge, greedy, k-means,
   farthest-point, Bayesian optimization, active residual);
5. fit the representation in `u` with the chosen fitter and regularizer;
6. evaluate position/velocity/acceleration/jerk/snap against the analytic
   ground truth on a dense grid via the chain rule through `u(t)`, and record
   timing, parameter count and flags.

**Oracle vs realistic.** Methods may use ground-truth derivatives (e.g.
`jerk_importance` sampling, `curvature` knots) or only the observed noisy
tracks (Savitzky–Golay/FD derivative estimates, active learning on the current
fit). Every record stores `sampling_uses_gt`, `knots_uses_gt`,
`time_param_uses_gt`, so oracle and realistic comparisons can always be
separated.

## 3. What is implemented

### 3.1 Representations

`basis_derivs(u)` is analytic for every linear representation, so metrics
never rely on finite differences.

| Registry name | Family | Params | Status | Citation |
|---|---|---:|---|---|
| `linear` | piecewise linear | `K·D` | implemented | [`deboor1978`] |
| `hermite` | cubic Hermite, central-difference tangents (SplineGS-style) | `K·D` | implemented | [`park2024splinegs`, `deboor1978`] |
| `catmull_rom` | uniform Catmull-Rom | `K·D` | implemented | [`catmull1974`] |
| `bspline2/3/5/7` | clamped B-spline, degree 2/3/5/7 | `(m+k+1)·D` | implemented | [`deboor1978`] |
| `pspline` | B-spline + difference penalty | `(m+k+1)·D` | implemented | [`eilers1996`] |
| `pspline_gcv` | P-spline with GCV-selected penalty | `(m+k+1)·D` | implemented (I2) | [`wahba1990`] |
| `chebyshev` | Chebyshev polynomial basis | `(d+1)·D` | implemented | [`trefethen2013`] |
| `fourier` | truncated Fourier series | `(2H+1)·D` | implemented | — |
| `nurbs` | rational B-spline, fixed or optimized weights | `K·D (+K)` | implemented | [`piegl1995`] |
| `akima`, `pchip` | Akima / monotone cubic interpolation through knot values | `K·D` | implemented | [`akima1970`, `fritsch1980`] |
| `gp_rbf`, `gp_matern52` | GP regression (hyperparams by marginal likelihood) | `3 (+N)` | implemented | [`rasmussen2006`] |
| `dmp` | discrete DMP, forcing weights by LS | `M (+x0,g)` | implemented | [`ijspeert2013`] |
| `mlp` | Fourier-feature MLP, Adam, autograd derivatives | `~10^4` | implemented | [`sitzmann2020`] |
| `bspline_mlp` | B-spline backbone + MLP residual (hybrid) | backbone + MLP | implemented | — |

Parameter counts are recorded per record (`n_params`, `n_data_params`,
`n_params_total`); `gp_*` report their `N` support points as data parameters
because they store the samples, mirroring SplineGS's `O(G·Nc)` storage
concern.

### 3.2 Knot / control-site placement

Implemented: `uniform`, `quantile`, `chord`, `centripetal`, `curvature`,
`feature_peaks` (velocity/acceleration/jerk extrema + acceleration reversals),
`rdp`, `split_merge` (recursive max-residual bisection), `cv` (K-fold
cross-validation over the **number** of knots, Iteration 2), `clustered`
(adversarial cluster / near-duplicate / one-gap / endpoint layouts, Iteration
2), `greedy` (forward selection with a linear/real-representation objective),
`kmeans` (weighted on derivative density), `farthest_point`, `bayesopt`
(GP-EI over knot locations), `active_residual` (iterative residual-driven
insertion).

### 3.3 Fitting / optimization

Linear: `least_squares`, `weighted`, `ridge`, `huber` (IRLS), `ransac`,
`lasso`, `elastic_net`, `tv`, `adam`.
Nonlinear: `lm`, `trf`, `dogbox` (scipy `least_squares`), `nelder_mead`,
`powell`, `differential_evolution`, `bayes_opt` (GP-EI).
Regularizers: `ridge` (L2), `diff` (2nd/3rd difference → P-spline), `tv`.

### 3.4 Time parameterization

`linear`, `chord`, `centripetal`, `accel`, `jerk`, each with an optional
`oracle=True` mode that uses ground-truth derivatives instead of the noisy
observations. This is the "spatial path vs timing law" separation: the
representation is fit in `u`, and metrics compose `d^k/du^k` with `u(t)` by
the chain rule.

### 3.5 Sampling strategies

`random`, `grid`, `stratified`, `lhs`, `sobol`, `halton`, `farthest_point`,
`jerk_importance` (inverse-CDF sampling of a derivative-density),
`derivative_peaks`, `hybrid` (uniform + importance), `active_gp`
(posterior-variance sampling), `active_residual`, `query_by_committee`.
`*_oracle=True` uses ground-truth derivative densities; the non-oracle modes
estimate them from a pilot sample.

### 3.6 Observation / noise model

Gaussian i.i.d. (`noise_std`), heavy-tail outliers
(`outlier_frac` × `outlier_scale`), missing observations (`missing_frac`,
encoded as NaN and masked in fitting), constant per-axis bias (`bias`).
**Iteration 2 adds the structured processes real GS/mocap tracks have**,
configured per condition through `noise_kwargs`:

| key | process |
|---|---|
| `ar1_rho` | temporally correlated AR(1) track noise (replaces the i.i.d. draw) |
| `hetero` | observation std scales with ground-truth speed |
| `speed_outlier` | outlier probability proportional to speed |
| `missing_bursts` / `burst_width` | contiguous missing intervals, not i.i.d. gaps |
| `jitter` | timestamp jitter (smooth random shift of the queried time) |
| `quantize` | amplitude quantization |
| `axis_corr` | equicorrelated per-axis noise |
| `bias_drift` | smooth low-frequency bias per axis |

When every key is absent the model is bit-identical to the original i.i.d.
one. The oracle builds deterministic continuous noise fields over `[0, 1]`, so
adaptive samplers and repeated queries see a consistent realization.

Iteration 2 also adds **extrapolation protocols** (`split`): `early`, `late`,
`middle` and `interp` restrict sampling to a train window (or union of
windows) and evaluate on a disjoint hold-out window.

### 3.7 Metrics

`pos_rmse`, `pos_mae`, `pos_max`, `vel_rmse`, `acc_rmse`, `jerk_rmse`,
`snap_rmse`, `jerk_max_fit`, `snap_max_fit`, `peak_time_err`, `overshoot`,
`settle_time_err`, `isj_fit`, `isj_ratio`, `cstr_{v,a,j,s}_{mean,max}`,
`train_rmse`, `fit_time_s`, `eval_time_s`, `n_params*`,
plus aggregation metrics `aulc` (area under the error-vs-log-budget curve) and
`pareto_front`.

**Iteration 2 adds conditioning telemetry and timing/window metrics.**

- Per-fit telemetry: `design_cond`, `design_rank`, `n_coeff`, `eff_dof`,
  `eff_dof_ratio`, `penalty_rank`, `penalty_nullity` (linear bases);
  `gp_kernel_cond`, `gp_lengthscale`, `gp_signal_var`, `gp_noise_var`,
  `gp_n_support`, `gp_nll` (GPs); `mlp_param_norm`, `mlp_grad_norm` (MLP);
  `dmp_weight_norm`; `resid_acf1`; and `knot_{min,median,max}_gap`,
  `knot_near_dupes`, `knot_endpoint_span`, `n_selected_sites`.
- `holdout_*` / `trainwin_*` metrics on the extrapolation windows.
- `phase_err` (cross-correlation lag) and `dtw` (dynamic time warping),
  because position RMSE hides timing errors.

## 4. Synthetic snappy motions

All motions are normalized to unit peak displacement and have closed-form
derivatives to 4th order (sine, chirp, min-jerk step, bang-bang, Gaussian
bump, impulse, damped step). Evaluation is therefore bandwidth-limited only
by the sampling, never by derivative estimation.

| Scene | Stress | Snappy? |
|---|---|---|
| `orbit` | smooth multi-frequency reference | no |
| `staccato` | snap → dwell → reverse snap | yes |
| `double_step` | two snaps 60 ms apart (snap-rest-snap) | yes |
| `bang_bang` | trapezoidal velocity, acceleration jumps | yes |
| `bounce` | repeated damped impacts (C0-ish kinks) | yes |
| `contact_drop` | second-order impact response | yes |
| `pulses` | narrow Gaussian position/velocity/acceleration spikes | yes |
| `wobble` | 18 cycles over 48-frame-equivalent window (near Nyquist) | high-freq |
| `chirp` | 1→20 cycle frequency sweep | high-freq |
| `combo` | steps + chirp + impacts simultaneously | yes |

## 5. Running experiments

```bash
pip install -r requirements.txt          # numpy scipy matplotlib pandas pytest
pip install -e .                         # optional, enables `splinebench` import anywhere

pytest -q                                # full unit/conformance suite (no GPU needed)

# tiny end-to-end check (6 conditions)
python -m splinebench.run --suite smoke --out results/smoke.jsonl

# starter matrix, quick variant (3 motions, 3 seeds, budgets 5..50)
python -m splinebench.run --suite starter --out results/starter.jsonl \
    --summary results/starter_summary.csv --plots results/starter

# one-axis studies
python -m splinebench.run --suite placement --full --out results/placement.jsonl
python -m splinebench.run --suite fitters   --out results/fitters.jsonl
python -m splinebench.run --suite sampling  --out results/sampling.jsonl
python -m splinebench.run --suite robustness --out results/robustness.jsonl

# aggregate an existing run only
python -m splinebench.run --report results/starter.jsonl \
    --summary results/starter_summary.csv --plots results/starter

# Iteration 2 stress grids (structured noise, time parameterization,
# extrapolation, adversarial knots, auto-tuned knot/penalty counts)
python scripts/run_iteration2.py                 # full: 4 455 conditions, 5 seeds
python scripts/run_iteration2.py --quick         # 3 seeds
python -m splinebench.run --suite i2_noise --out results/iteration2/i2_noise.jsonl

# print the registry / backlog
python -m splinebench.run --list
```

Runs append JSONL and resume automatically (condition keys are hashed);
interrupting is safe. `suites.py` defines `smoke`, `starter`, `placement`,
`fitters`, `sampling`, `robustness`, `full`; `--full` switches to the larger
seed/motion/budget grid.

### Condition counts

| Suite | quick | full |
|---|---:|---:|
| smoke | 6 | — |
| starter | 324 | 3 780 |
| placement | 504 | 21 840 |
| fitters | 120 | 7 560 |
| sampling | 240 | 10 920 |
| robustness | 120 | 12 600 |
| full (union) | — | 56 700 |
| i2_core | 540 | 900 |
| i2_noise | 810 | 1 350 |
| i2_timeparam | 540 | 900 |
| i2_extrap | 540 | 900 |
| i2_adversarial | 99 | 165 |
| i2_knotcount | 144 | 240 |
| iteration2 (union) | 2 673 | 4 455 |

(Counts are generated by `suites.py`; run
`python -c "from splinebench import suites; print(len(suites.build('starter', quick=True)))"`.)

### How the Section 6 tables were produced

The quick suites do not cover every column (budgets 100/200, `bounce`/
`chirp`/`double_step`, active samplers, the full noise grid). Two commands
close the gaps and then generate every table deterministically:

```bash
for s in starter placement fitters sampling robustness; do
  python -m splinebench.run --suite $s --out results/$s.jsonl
done
python scripts/run_extra.py          # targeted supplements (budgets 100/200,
                                     # bounce/chirp, active samplers, 7-noise grid,
                                     # global-basis and penalty sweeps)
python -m splinebench.report --results results --out results/ANALYSIS.md \
    --readme README.md --plots --facts
```

`report.py` replaces the content between the `<!-- AUTO:6.x -->` markers and
prints the decision-relevant comparisons (`--facts`), so re-running the full
grids later refreshes the README without hand-editing.

## 6. Results (quick + targeted grids, n=3 seeds)

> Generated by `splinebench.report` from `results/*.jsonl`; all rows are
> non-oracle (samplers/placers see only noisy observations) unless a flag
> column says otherwise. Means ± sd over seeds; bold = best per column.

### 6.1 Sample efficiency — starter matrix (pos RMSE, mean ± sd over seeds)

<!-- AUTO:6.1 START -->
| motion | representation | knots | sampler | 5 | 10 | 20 | 50 | 100 | 200 | AULC ↓ |
|---|---|---|---|---|---|---|---|---|---|---|
| staccato | gp_rbf | curvature | random | 0.151 ± 0.012 | 0.162 ± 0.0084 | 0.14 ± 0.0269 | 0.0931 ± 0.0428 | 0.0523 ± 0.0129 | 0.0421 ± 0.00535 | 0.175 |
| staccato | bspline_mlp | uniform | random | 0.182 ± 0.0171 | 0.168 ± 0.0481 | 0.161 ± 0.0414 | 0.0779 ± 0.0234 | 0.0433 ± 0.00656 | 0.0408 ± 0.00603 | 0.18 |
| staccato | catmull_rom | uniform | random | 0.172 ± 0.00984 | 0.163 ± 0.00135 | 0.156 ± 0.0275 | 0.0838 ± 0.00928 | 0.0727 ± 0.00224 | 0.0716 ± 0.000705 | 0.191 |
| staccato | mlp | curvature | random | 0.168 ± 0.00468 | 0.17 ± 0.0343 | 0.2 ± 0.149 | 0.0959 ± 0.0396 | 0.0443 ± 0.00572 | 0.0408 ± 0.00602 | 0.199 |
| staccato | pspline | split_merge | random | 0.173 ± 0.0102 | 0.159 ± 0.0273 | 0.165 ± 0.064 | 0.112 ± 0.0357 | 0.0732 ± 0.0104 | 0.0684 ± 0.00636 | 0.203 |
| staccato | hermite | feature_peaks | random | 0.172 ± 0.00984 | 0.162 ± 0.00166 | 0.136 ± 0.018 | 1.14 ± 10.5 | 0.0535 ± 0.0103 | 0.0489 ± 0.00373 | 0.545 |
| staccato | dmp | curvature | random | 0.196 ± 0.0367 | 0.312 ± 0.452 | 1.46 ± 4.06 | 0.562 ± 1.05 | 0.171 ± 0.00328 | 0.168 ± 0.00163 | 0.905 |
| staccato | bspline3 | uniform | random | 0.285 ± 0.115 | 21.4 ± 107 | 0.281 ± 0.503 | 0.113 ± 0.0301 | 0.0541 ± 0.00268 | 0.051 ± 0.00564 | 6.65 |
| staccato | bspline5 | curvature | random | 0.681 ± 0.552 | 356 ± 2.27e+03 | 8.4e+06 ± 5.69e+07 | 525 ± 4.68e+03 | 0.0576 ± 0.00701 | 0.0508 ± 0.00997 | 2.94e+06 |
| bounce | gp_rbf | uniform | random | 0.172 ± 0.0209 | 0.201 ± 0.0477 | 0.11 ± 0.0209 | 0.0578 ± 0.0481 | 0.00846 ± 0.000367 | 0.00548 ± 0.00024 | 0.148 |
| bounce | mlp | uniform | random | 0.166 ± 0.0111 | 0.147 ± 0.019 | 0.179 ± 0.0238 | 0.0404 ± 0.0139 | 0.00896 ± 0.0012 | 0.00784 ± 0.00395 | 0.15 |
| bounce | catmull_rom | uniform | random | 0.161 ± 0.00865 | 0.155 ± 0.00851 | 0.133 ± 0.00292 | 0.0764 ± 0.00455 | 0.068 ± 0.00297 | 0.0646 ± 0.000607 | 0.174 |
| bounce | bspline_mlp | uniform | random | 0.205 ± 0.0228 | 0.172 ± 0.038 | 0.209 ± 0.0336 | 0.0502 ± 0.0168 | 0.00837 ± 0.000816 | 0.00523 ± 0.000228 | 0.176 |
| bounce | pspline | split_merge | random | 0.19 ± 0.026 | 0.156 ± 0.0191 | 0.14 ± 0.00882 | 0.0981 ± 0.029 | 0.0617 ± 0.00729 | 0.0583 ± 0.00749 | 0.186 |
| bounce | hermite | feature_peaks | random | 0.161 ± 0.00865 | 0.149 ± 0.00842 | 0.122 ± 0.0117 | 0.185 ± 0.323 | 0.0339 ± 0.00686 | 0.0223 ± 0.00273 | 0.19 |
| bounce | dmp | uniform | random | 0.208 ± 0.0503 | 0.156 ± 0.00736 | 0.163 ± 0.0107 | 0.164 ± 0.00332 | 0.163 ± 0.00191 | 0.161 ± 0.00167 | 0.266 |
| bounce | bspline5 | curvature | random | 0.365 ± 0.143 | 0.368 ± 0.327 | 2.11 ± 0.815 | 0.109 ± 0.0467 | 0.0471 ± 0.00636 | 0.0444 ± 0.00639 | 0.961 |
| bounce | bspline3 | chord | random | 0.17 ± 0.021 | 0.149 ± 0.0105 | 0.151 ± 0.0345 | 15.6 ± 79.3 | 0.0463 ± 0.00298 | 0.0506 ± 0.01 | 5.61 |
| wobble | gp_rbf | curvature | random | 0.692 ± 0.101 | 0.62 ± 0.0683 | 0.588 ± 0.0718 | 0.441 ± 0.095 | 0.0397 ± 0.0118 | 0.00971 ± 0.00432 | 0.664 |
| wobble | mlp | curvature | random | 0.832 ± 0.0279 | 1.02 ± 0.119 | 1.04 ± 0.144 | 0.688 ± 0.397 | 0.0904 ± 0.0212 | 0.0278 ± 0.0174 | 1.07 |
| wobble | catmull_rom | uniform | random | 0.806 ± 0.0457 | 0.703 ± 0.0309 | 0.802 ± 0.0913 | 0.789 ± 0.0175 | 0.647 ± 0.00633 | 0.632 ± 0.00587 | 1.18 |
| wobble | bspline_mlp | uniform | random | 1.04 ± 0.0777 | 1.12 ± 0.19 | 1.46 ± 0.0798 | 0.856 ± 0.281 | 0.192 ± 0.0933 | 0.0842 ± 0.0679 | 1.37 |
| wobble | hermite | split_merge | random | 0.806 ± 0.0457 | 0.733 ± 0.0427 | 0.791 ± 0.136 | 1.45 ± 3.78 | 0.623 ± 0.000718 | 0.605 ± 0.00784 | 1.4 |
| wobble | pspline | split_merge | random | 1 ± 0.0473 | 0.895 ± 0.311 | 1.3 ± 0.625 | 0.842 ± 0.231 | 0.719 ± 0.0877 | 0.647 ± 0.0302 | 1.48 |
| wobble | bspline3 | chord | random | 1.05 ± 0.215 | 0.8 ± 0.0293 | 1.33 ± 0.877 | 4.23 ± 17.1 | 0.641 ± 0.00656 | 0.628 ± 0.00704 | 2.63 |
| wobble | dmp | curvature | random | 0.868 ± 0.0371 | 8.87 ± 20.8 | 12.7 ± 25 | 3.17 ± 6.51 | 0.66 ± 0.00933 | 0.657 ± 0.00774 | 8.64 |
| wobble | bspline5 | curvature | random | 3.5 ± 2.33 | 1.42e+04 ± 8.96e+04 | 4.9e+03 ± 1.8e+04 | 6.1e+03 ± 4.41e+04 | 4.19 ± 2.54 | 11.5 ± 15.4 | 8.13e+03 |

Mean ± sd over seeds (`n=3`); AULC is the area under the error-vs-log-budget curve over the budgets with data.
<!-- AUTO:6.1 END -->

### 6.2 Peak timing and higher derivatives

<!-- AUTO:6.2 START -->
| motion | method | `vel_nrmse` | `acc_nrmse` | `jerk_nrmse` | `peak_time_err` | `overshoot` |
|---|---|---|---|---|---|---|
| bounce | hermite | 3 | 13.5 | 137 | 0.179 | 1.7 |
| bounce | bspline5 | 1.77 | 14.9 | 282 | 0.124 | 0.799 |
| bounce | pspline | 1.27 | 5.31 | 45.2 | 0.111 | 0.45 |
| bounce | gp_rbf | 0.543 | 0.702 | **0.944** | 0.124 | 0.052 |
| bounce | gp_matern52 | **0.363** | **0.555** | 0.946 | **0.00545** | 0.105 |
| bounce | dmp | 1.03 | 1.12 | 3.41 | 0.422 | **0.0155** |
| double_step | hermite | 1.21 | 6.06 | 47.3 | **0.0397** | 0.0585 |
| double_step | bspline5 | 98.6 | 344 | 856 | 0.0628 | 61.6 |
| double_step | pspline | 1.35 | 1.84 | 5.05 | 0.126 | 0.451 |
| double_step | gp_rbf | 1.32 | 2.03 | 2.39 | 0.072 | 0.169 |
| double_step | gp_matern52 | 1.27 | 2 | 2.88 | 0.072 | 0.117 |
| double_step | dmp | **0.958** | **1.11** | **1.95** | 0.0722 | **0.0127** |
| wobble | hermite | 3.19 | 10.8 | 75.2 | 0.367 | 18.7 |
| wobble | bspline5 | 6.82e+04 | 9.78e+05 | 1.51e+07 | 0.585 | 2.36e+05 |
| wobble | pspline | 1.3 | 2.49 | 8.56 | 0.514 | 7.24 |
| wobble | gp_rbf | 0.834 | 3.77 | 45.4 | 0.483 | 0.227 |
| wobble | gp_matern52 | **0.569** | **0.677** | **1.13** | 0.408 | **0.0582** |
| wobble | dmp | 1.33 | 1.44 | 3.63 | **0.323** | 12.3 |
| chirp | hermite | 3.41 | 16.2 | 124 | 0.436 | 12.5 |
| chirp | bspline5 | 1.31e+03 | 1.58e+03 | 1.51e+03 | 0.512 | 1.19e+04 |
| chirp | pspline | 1.25 | 2.5 | 8.93 | 0.528 | 4.11 |
| chirp | gp_rbf | **0.45** | **0.491** | **0.567** | 0.65 | 0.336 |
| chirp | gp_matern52 | 0.543 | 0.626 | 0.859 | 0.77 | 0.215 |
| chirp | dmp | 1.04 | 1.71 | 5.03 | **0.0683** | **0.0571** |

Budget 50, mean over 3 seeds; derivatives normalized by the ground-truth derivative RMS (`*_nrmse`, lower is better); best per motion/column in bold.
<!-- AUTO:6.2 END -->

### 6.3 Placement ablation (fixed `n_sites`, `pos_rmse` / `jerk_rmse`)

<!-- AUTO:6.3 START -->
**bspline3** (n_sites=8, budget=50, `pos_rmse` / `jerk_rmse`)

| knots | staccato | bounce | wobble | chirp |
|---|---|---|---|---|
| uniform | 0.142 / 7.23e+03 | 0.121 / 1.6e+04 | 0.733 / 8.37e+05 | 0.586 / 3.96e+05 |
| chord | **0.104 / 4.9e+05** | 0.094 / 1.18e+06 | 0.689 / 5.33e+06 | **0.523 / 3.18e+06** |
| curvature | 0.13 / 2.22e+04 | 0.0958 / 1.55e+04 | 0.718 / 8.38e+05 | 0.581 / 4.07e+05 |
| feature_peaks | 0.107 / 9.66e+03 | 0.0972 / 1.52e+04 | 0.681 / 8.37e+05 | 0.542 / 3.96e+05 |
| split_merge | 0.141 / 7.56e+05 | 0.0988 / 1.21e+06 | **0.669 / 6.86e+06** | 0.532 / 3.27e+06 |
| greedy | 0.146 / 3.43e+04 | **0.0904 / 1.57e+04** | 44.4 / 8.04e+06 | 0.687 / 4.64e+05 |
| bayesopt | 0.151 / 1.68e+05 | 140 / 5.86e+07 | 2.01 / 1.02e+06 | 0.754 / 1.05e+07 |
| active_residual | 0.114 / 1.13e+06 | 0.109 / 6.32e+05 | 0.718 / 3.66e+06 | 0.542 / 2.46e+06 |

Mean over 3 seeds; best position RMSE per motion in bold.

**bspline5** (n_sites=8, budget=50, `pos_rmse` / `jerk_rmse`)

| knots | staccato | bounce | wobble | chirp |
|---|---|---|---|---|
| uniform | 0.201 / 1.82e+04 | 0.146 / 2.13e+04 | 0.987 / 8.39e+05 | 0.673 / 3.99e+05 |
| chord | **0.118 / 2.91e+06** | 0.0939 / 7.28e+06 | 0.756 / 3.15e+07 | 0.568 / 1.41e+07 |
| curvature | 0.48 / 4.65e+04 | 0.116 / 1.74e+04 | 8.36 / 9.42e+06 | 0.96 / 1.03e+06 |
| feature_peaks | 0.148 / 9.07e+03 | 0.112 / 1.71e+04 | 0.83 / 8.39e+05 | 0.602 / 3.97e+05 |
| split_merge | 0.139 / 3.55e+06 | **0.0855 / 8.54e+06** | **0.701 / 2.53e+07** | **0.537 / 1.05e+07** |
| greedy | 0.184 / 1.53e+04 | 0.141 / 2.65e+04 | 9.13e+03 / 5.3e+09 | 2.87e+03 / 1.86e+09 |
| bayesopt | 0.154 / 1.02e+04 | 0.13 / 2.09e+04 | 84.8 / 8.29e+07 | 0.8 / 8.16e+05 |
| active_residual | 0.135 / 6.04e+06 | 0.111 / 1.14e+07 | 0.715 / 9.06e+06 | 0.561 / 4.51e+07 |

Mean over 3 seeds; best position RMSE per motion in bold.

**hermite** (n_sites=8, budget=50, `pos_rmse` / `jerk_rmse`)

| knots | staccato | bounce | wobble | chirp |
|---|---|---|---|---|
| uniform | 0.14 / 7.04e+03 | 0.117 / 1.59e+04 | 0.668 / 8.37e+05 | 0.541 / 3.96e+05 |
| chord | **0.0926 / 1.14e+05** | 0.108 / 1.61e+04 | 0.669 / 8.37e+05 | 0.537 / 3.96e+05 |
| curvature | 0.121 / 1.41e+05 | 0.106 / 1.68e+04 | 0.67 / 8.36e+05 | 0.54 / 3.96e+05 |
| feature_peaks | 0.101 / 4.98e+05 | 0.1 / 1.58e+04 | **0.662 / 8.27e+05** | **0.528 / 3.99e+05** |
| split_merge | 0.21 / 1.04e+07 | 0.208 / 1.1e+07 | 0.795 / 5.27e+07 | 0.617 / 2.4e+07 |
| greedy | 0.115 / 3.1e+06 | 0.128 / 1.49e+05 | 0.802 / 3.53e+07 | 0.903 / 6.02e+06 |
| bayesopt | 0.122 / 2.81e+06 | **0.0984 / 2.5e+04** | 9.29 / 1.3e+08 | 0.602 / 7.34e+05 |
| active_residual | 0.145 / 2.43e+07 | 0.155 / 2.41e+07 | 3.59 / 6.44e+08 | 3.85 / 4.1e+08 |

Mean over 3 seeds; best position RMSE per motion in bold.

**pspline** (n_sites=8, budget=50, `pos_rmse` / `jerk_rmse`)

| knots | staccato | bounce | wobble | chirp |
|---|---|---|---|---|
| uniform | 0.139 / 7.19e+03 | 0.118 / 1.6e+04 | 0.725 / 8.37e+05 | 0.577 / 3.96e+05 |
| chord | 0.111 / 2.6e+06 | 0.109 / 9.95e+05 | 0.741 / 1.18e+07 | 0.587 / 9.07e+06 |
| curvature | 0.129 / 2.14e+04 | 0.0946 / 1.56e+04 | 0.713 / 8.38e+05 | 0.57 / 4.05e+05 |
| feature_peaks | **0.105 / 9.2e+03** | 0.0944 / 1.52e+04 | **0.678 / 8.37e+05** | 0.54 / 3.96e+05 |
| split_merge | 0.162 / 3.68e+06 | 0.125 / 3.43e+06 | 0.684 / 1.66e+07 | 0.552 / 9.84e+06 |
| greedy | 0.142 / 3.28e+04 | **0.0849 / 1.55e+04** | 1 / 9.31e+05 | 0.604 / 4.48e+05 |
| bayesopt | 0.151 / 1.58e+05 | 0.091 / 1.55e+04 | 0.751 / 8.33e+05 | 0.629 / 1.37e+06 |
| active_residual | 0.116 / 2.81e+06 | 0.134 / 7.9e+05 | 0.835 / 4.66e+06 | **0.539 / 5.94e+05** |

Mean over 3 seeds; best position RMSE per motion in bold.
<!-- AUTO:6.3 END -->

### 6.4 Robustness (`pos_rmse`)

<!-- AUTO:6.4 START -->
| method \ condition | clean | σ=0.01 | σ=0.03 | 5% outliers | 15% outliers | 10% missing | bias 0.05 |
|---|---|---|---|---|---|---|---|
| hermite + LS | 0.0801 ± 0.0185 | 0.289 ± 0.252 | 0.0859 ± 0.0209 | 0.085 ± 0.014 | 0.12 ± 0.0201 | 0.113 ± 0.0473 | 0.0982 ± 0.0145 |
| hermite + huber | 0.0813 ± 0.0175 | **0.0922 ± 0.0182** | 0.102 ± 0.0303 | 0.0959 ± 0.0158 | 0.137 ± 0.0321 | 0.12 ± 0.0363 | 0.106 ± 0.0155 |
| hermite + ransac | 0.254 ± 0.197 | 0.207 ± 0.252 | 0.139 ± 0.0706 | 0.556 ± 1.01 | 0.413 ± 0.403 | 18.8 ± 41.6 | 0.255 ± 0.241 |
| pspline (diff) | 0.0839 ± 0.00968 | 0.378 ± 0.373 | 0.0907 ± 0.0183 | 0.0952 ± 0.0283 | 0.138 ± 0.0501 | 0.112 ± 0.0454 | 0.1 ± 0.0156 |
| gp_matern52 | **0.0443 ± 0.0341** | 0.144 ± 0.133 | **0.0587 ± 0.0273** | **0.0539 ± 0.0281** | **0.0866 ± 0.0189** | **0.0566 ± 0.0334** | **0.0746 ± 0.0236** |

Budget 50, mean ± sd over 3 seeds × {staccato, bounce}; best per column in bold.
<!-- AUTO:6.4 END -->

### 6.5 Sampling comparison (AULC over budgets)

<!-- AUTO:6.5 START -->
| sampler | bspline5 | gp_rbf | dmp | mlp |
|---|---|---|---|---|
| random | 1.53 | 0.214 | 0.402 | 0.365 |
| lhs | 0.758 | 0.229 | 0.31 | 0.4 |
| sobol | **0.334** | 0.241 | **0.297** | **0.286** |
| jerk_importance | 1.24e+03 | 0.242 | 3.35 | 0.434 |
| derivative_peaks | 214 | 0.246 | 1.11 | 0.43 |
| active_gp | 1.21e+04 | 0.27 | 0.318 | 0.291 |
| active_residual | 804 | 0.25 | 0.809 | 0.442 |
| query_by_committee | 1.75e+03 | **0.208** | 4.34 | 0.38 |

Median AULC over (motion, seed) curves for budgets {10, 20, 50} at fixed `n_sites=8`; lower is better, best per column in bold.
<!-- AUTO:6.5 END -->

### 6.6 Pareto fronts

<!-- AUTO:6.6 START -->
| figure | command | status |
|---|---|---|
| `results/starter_sample_efficiency.png` | `--plots results/starter` | generated |
| `results/starter_pareto.png` | `--plots results/starter` | generated |
| `results/robustness.png` | `plots.plot_robustness` | generated |
<!-- AUTO:6.6 END -->

## 7. Takeaways and analysis

> Populated from the first result pass: quick suites + targeted supplements,
> 2 703 conditions × 3 seeds (`results/*.jsonl`). Effect sizes are means over
> seeds; `*_nrmse` is RMSE divided by the ground-truth derivative RMS.

### 7.1 Executive summary

At 1% observation noise and 50 samples, **GP regression with a Matérn 5/2
kernel is the strongest single representation**: best or near-best position
error on every motion except `chirp` (where GP-RBF wins), best robustness
across all seven noise/outlier/missing/bias columns, and the only model whose
jerk error stays at the ground-truth scale (`jerk_nrmse ≈ 0.9–1.1` on
`bounce`/`wobble`, versus 8.6 for the best penalized spline and >45 for the
rest). It pays for this with `O(N)` stored support points (`n_params_total =
3 + N ≈ 53`) and ~15 ms/fit at `N=50`.

Among closed-form splines the ranking is budget-dependent. At 5–10 samples,
Catmull-Rom/Hermite are competitive and GP-RBF leads `staccato` (median
0.149 vs 0.166); unregularized degree-5 B-splines are unusable there. From
~50 samples, a **cubic B-spline with adaptive interior knots and a difference
penalty** (`split_merge`/`chord` placement, `lam ≈ 1e-3`) closes most of the
gap at 1/50 of the parameters and 1/1000 of the fit time.

Two levers dominate. **(i) Knot placement**: adaptive sites beat uniform at
fixed `n_sites=8` in 16/16 representation × motion cells, by 25–41% position
RMSE for cubic/quintic B-splines and Hermite. **(ii) Regularization**: for an
unpenalized cubic B-spline at σ=0.03 and 10 samples the position error is
~190 (RMSE on a unit-amplitude motion!) and drops to 0.32 with a second-
difference penalty (`lam=1e-3`) — the difference between broken and usable.

The surprise is **sampling: space-filling Sobol wins** for the B-spline, DMP
and MLP, and beats random by 4.6× in AULC for `bspline5`; active-learning and
derivative-peak samplers cluster points near snaps and are 2–5 orders of
magnitude worse for flexible bases. Neural fits need ~100× the parameters and
~1000× the fit time for no low-budget gain, but scale best by 100–200 samples
on `bounce` (0.0078 at B200 vs 0.065 for Catmull-Rom).

### 7.2 Findings by research question

| RQ | Answer (one line) | Best method(s) | Effect vs runner-up | Hypothesis verdict | Evidence |
|---|---|---|---|---|---|
| RQ1 sample efficiency | GP leads once N≥20; piecewise ties at N≤10 | `gp_matern52` (`bounce`/`wobble`), `gp_rbf` (`staccato`/`chirp`) | `bounce` AULC 0.028 vs 0.148 GP-RBF (−81%); `staccato` 0.175 vs 0.203 pspline (−14%) | H1/H4 supported, H3 partial, H7 partial | 6.1, 6.6 |
| RQ2 snappy/high-freq fidelity | GP keeps derivatives at GT scale; splines ring | `gp_matern52` (impacts), `gp_rbf` (chirp), DMP peak timing | `jerk_nrmse` on `bounce`: 0.95 vs 45 (pspline) vs 137 (hermite) | H2 partial, H4 falsified | 6.2 |
| RQ3 knot placement | Adaptive placement always beats uniform; `feature_peaks` is the safe low-jerk choice, `split_merge`/`chord` the low-position choice | `chord`, `split_merge`, `feature_peaks` | bspline5 `staccato` 0.201→0.118 (−41%); bspline5 `bounce` 0.146→0.086 (−41%) | H1 supported | 6.3 |
| RQ4 fitting/regularization | Penalization is essential at low N; robust losses add little; GP most noise-robust | diff penalty `lam≈1e-3`, `gp_matern52` | B-spline σ=0.03 B10: 190→0.315 with penalty; GP best in 7/7 columns | H3 partial, H5 falsified | 6.4 |
| RQ5 sampling strategy | Space-filling wins; derivative-aware/active sampling hurts flexible bases | `sobol` (3/4 reps), `query_by_committee` (GP-RBF, by 3%) | bspline5 AULC 0.334 (sobol) vs 1.53 (random), 1.2e4 (active_gp) | H6 falsified | 6.5 |
| RQ6 cost / Pareto | GP sits on the frontier; Catmull-Rom/PSpline cheapest; neural is dominated below B100 | `gp_rbf`, `catmull_rom`, `pspline` | GP: 53 params, 15 ms, median pos 0.087; Catmull-Rom: 48 params, 0.2 ms, 0.086 | — | 6.6, 7.5 |

### 7.3 Motion-class playbook

| Motion class | Best representation + knots | Best sampler | Degradation mode to watch | Takeaway |
|---|---|---|---|---|
| smooth (`orbit`) | not run in this pass (full suite) | — | — | add `orbit` to the next pass |
| snap / snap-rest (`staccato`, `double_step`) | GP-RBF/Matérn; for splines `feature_peaks`/`chord`; DMP if only peak timing/first derivatives matter | `sobol`; `lhs` similar | high-degree B-spline ringing between clustered knots (bspline5 B10 AULC 8×10³) | regularize before adding knots |
| kinks / impacts (`bounce`, `pulses`) | `gp_matern52`; best spline is a penalized cubic (`pspline`) | `sobol`/`query_by_committee` | unpenalized Hermite tangents amplify impact jerk (`jerk_nrmse` 137 vs 0.95 GP) | Matérn > RBF here |
| bang-bang (`bang_bang`) | GP/`fourier`/Hermite (0.077/0.085/0.086); Chebyshev worst (0.84) | `sobol` | global polynomial bases ring at acceleration jumps | piecewise not required; conditioning is |
| near-Nyquist / chirp (`wobble`, `chirp`, `combo`) | `gp_matern52` (`wobble`), `gp_rbf` (`chirp`); `combo` not run | `sobol` | DMP/MLP need >100 samples; `fourier` aliases (0.767 `wobble`) | bandwidth, not smoothness, is the constraint |

### 7.4 Hypothesis verdicts

| ID | Verdict (supported / falsified / partial) | Evidence | Notes |
|---|---|---|---|
| H1 adaptive placement beats uniform at low budget | **supported** | 6.3 | uniform is the worst or near-worst in all 16 cells; `chord`/`split_merge` cut position RMSE 25–41%; `feature_peaks` also cuts jerk (1–2 orders) |
| H2 piecewise beats global bases on kinks | **partial** | 6.1, 6.2, §7.2 | Chebyshev fails badly (0.84 on `bang_bang`, 2.93 on `wobble`); Fourier is good on `bang_bang` (0.085) but poor on `wobble`; an unregularized cubic B-spline can be as bad as Chebyshev, so conditioning dominates the piecewise/global split |
| H3 P-spline beats plain LS at low budget under noise | **supported at low N, vanishes ≥50** | 6.4, §7.2 | σ=0.03, B10: 190 → 0.315 (lam=1e-3); B20: 1.06 → 0.152; at B50 in 6.1 `pspline` ≈ `bspline3` |
| H4 Matérn GP leads on smooth, over-smooths on kinks | **leads everywhere; over-smoothing falsified** | 6.1, 6.2 | Matérn best AULC on `bounce` (0.028) and `wobble` (0.190) and best/2nd on `pulses`/`bang_bang`, with `jerk_nrmse` 0.56–1.13; RBF only wins on `chirp` |
| H5 robust losses only help with outliers | **falsified** | 6.4 | LS ≥ Huber in nearly every column; Huber only wins at σ=0.01 where LS blows up (0.092 vs 0.289); RANSAC is pathological — it treats snap points as outliers (18.8 with 10% missing) |
| H6 derivative-aware sampling helps most at 5–20 | **falsified** | 6.5 | `sobol` best for B-spline/DMP/MLP; `jerk_importance` adds 0–800× AULC; active samplers 10³–10⁵× worse for `bspline5` |
| H7 neural needs ~10× samples but may win on high-freq later | **partial / directionally supported** | 6.1, 6.6 | MLP worst AULC on `staccato` (0.199 vs 0.175) at 100× params and ~1000× fit time; matches GP on `bounce` (0.150 vs 0.148) and is best at B200 (0.0078 vs 0.065 Catmull-Rom) |

### 7.5 Cross-cutting takeaways

- **Representation vs placement.** At fixed `n_sites=8`, switching placement
  from uniform to the best adaptive scheme buys 25–41% position RMSE
  (bspline3/bspline5/hermite, 6.3); switching representation at the same
  budget and placement buys less once regularized (e.g. `pulses`: GP-Matérn
  0.069 vs Hermite 0.084, −18%). **Placement is the bigger lever per unit of
  code and compute**, but it interacts with the basis: `feature_peaks` keeps
  jerk 1–2 orders lower than `chord`/`split_merge` even when position is
  comparable.
- **Dimensionality of the win.** GP's position lead usually carries to higher
  derivatives: `jerk_nrmse` < 1.2 on `bounce`/`wobble`/`chirp` vs 8.6–282 for
  splines. The exception is DMP, which wins normalized velocity/acceleration
  and peak timing on `double_step` (1.11/1.95, `peak_time_err` 0.072) despite
  mid-pack position error — trajectory-shaping priors help timing laws even
  when the spatial fit is average.
- **Noise robustness.** GP keeps rank order across all noise columns;
  unregularized LS B-splines are unstable at σ=0.01 (mean 0.289 ± 0.252,
  i.e. seed-dependent blow-ups) and are stabilized by either a difference
  penalty (`pspline`) or Huber; RANSAC collapses (18.8 with 10% missing);
  missing observations barely move GP (0.057). Outlier magnitude here is
  comparable to the signal, so averaging-based LS is already hard to beat.
- **Oracle vs realistic.** All Section 6 rows are **realistic**
  (`knots_uses_gt = sampling_uses_gt = False`): placers/samplers see only the
  noisy observations through pilot estimates. Oracle variants are recorded in
  the JSONL flags but were not part of this pass, so the numbers are not
  inflated by ground-truth derivatives.
- **Cost.** Median over `{staccato, bounce, wobble}` at budget 50:

  | method | `n_params_total` | fit (ms) | median `pos_rmse` |
  |---|---:|---:|---:|
  | `catmull_rom` | 48 | 0.2 | 0.086 |
  | `pspline` | 43 | 0.2 | 0.122 |
  | `bspline3` | 40 | 0.2 | 0.143 |
  | `hermite` | 41 | 3.7 | 0.116 |
  | `bspline5` | 49 | 4.3 | 0.163 |
  | `gp_rbf` | 3 + 50 data | 14.7 | **0.087** |
  | `dmp` | 50 | 2.7 | 0.201 |
  | `mlp` | 5 507 | 403 | 0.097 |
  | `bspline_mlp` | 5 561 | 271 | 0.066 |

  GP is the frontier point to beat: near-best error with `O(N)` stored
  parameters; `catmull_rom` is the cheap closed-form fallback; the neural
  hybrids buy ~24% error for ~100× parameters and ~1 000× fit time.
- **Timing law.** Untested in this pass (starter used `linear`). The
  chain-rule/metric plumbing is in place; the `time_param` axis is part of
  the next pass.

### 7.6 Negative and surprising results

- **RANSAC fails on snappy motion by design.** With sparse samples the model
  bias dominates the residual MAD, the consensus threshold balloons, and the
  "outliers" it rejects are the snap points it cannot fit — so it converges
  to a smooth majority curve. Clean `staccato`/`bounce`: 0.254 vs 0.080 for
  LS; 10% missing: 18.8. A threshold tied to an independent noise estimate
  (or a consensus-fraction guard, as now implemented) is required for
  regression splines.
- **Black-box knot search is unstable.** `greedy` and `bayesopt` win some
  cells (bayesopt: hermite `bounce` 0.098 vs uniform 0.117) but lose
  catastrophically in others (bspline5 `greedy` on `wobble`: 9.1×10³ RMSE;
  bayesopt bspline3 `bounce`: 140), because the surrogate/residual landscape
  over knot positions is multi-modal at low N. Hand-designed features
  (`feature_peaks`, `split_merge`, `chord`) are more reliable.
- **High degree without penalty is a trap.** `bspline5` at B10 with
  `curvature` knots reaches AULC ~8×10³ versus 0.20 for cubic; unregularized
  quartic/quintic bases amplify knot-clustering into 10⁷-scale jerk. The same
  basis with a difference penalty is fine — the failure is conditioning, not
  expressiveness.
- **Fourier is not uniformly bad.** It is surprisingly good on `bang_bang`
  (0.085, third best) but aliases on `wobble` (0.767) and loses to Chebyshev
  nowhere — a reminder that "global basis rings" is motion-specific.
- **DMP plateaus early.** On `bounce` it sits at ~0.163 from B20 to B200
  while splines and GP keep improving; the attractor dynamics reject
  high-frequency content independent of the sample count.
- **MLP cost buys nothing at low N.** ~100× parameters and ~1 000× fit time
  for AULC 0.199 (`staccato`) vs 0.175 for GP-RBF; its win only appears at
  B100–200 on high-frequency motion.

Evidence: `results/ANALYSIS.md` (6.1–6.5), raw `results/*.jsonl`, and
`--facts` output in `results/report.log`.

### 7.7 Crossover and operating-point analysis

| Question | Budget where ranked order changes | Before | After | Evidence |
|---|---|---:|---|---|
| lowest budget for ≤5% position error (`staccato`) | B100 | 0.078 (B50, `bspline_mlp`) | 0.043 (B100, `bspline_mlp`) | 6.1 |
| budget where MLP overtakes Catmull-Rom (`bounce`) | between B20 and B50 | 0.179 vs 0.133 (B20) | 0.040 vs 0.076 (B50) | 6.1 |
| budget where P-spline advantage over plain LS vanishes | between B50 and B100 | 190 vs 0.315 (penalty vs LS, B10, σ=0.03) | 0.073 vs 0.054 (`staccato` B100, pspline vs bspline3) | 6.4, 6.1 |
| budget where GP advantage vanishes | never ≤ B200 | GP AULC best on 3/3 motions | still best at B200 (0.0097 vs 0.632 `wobble`, GP-RBF vs Catmull-Rom) | 6.1 |
| derivative-aware sampling vs Sobol (`bspline5`) | never crosses | `jerk_importance` AULC 1.2×10³ vs Sobol 0.33 | same ordering at every budget B10–B50 | 6.5 |

### 7.8 Threats to validity

- **Observation model is benign.** i.i.d. Gaussian noise, uniform outliers
  and missing values; real GS tracks add temporal correlation, per-point
  heteroscedasticity and scale-dependent error, none of which are exercised
  yet. The ~4% position floor at B200 under σ=0.01 is partly this model.
- **Motion coverage.** `orbit`, `contact_drop` and `combo` were not in this
  pass; `wobble`/`chirp` represent the high-frequency end. Normalization to
  unit peak displacement removes absolute-scale effects.
- **Default knot budget is arbitrary.** `n_sites = budget/3` and placements
  return `n_sites` sites regardless of basis degree; this deliberately
  penalizes high-degree splines at low N in 6.1 but confounds "basis" with
  "effective number of parameters". `n_params_total` is the honest axis and
  is recorded everywhere.
- **Harness assumptions matter.** Before the endpoint-site fix, B-spline
  interior knots at 0/1 created 1e-9 spans and 10²⁵ jerk: an implementation
  detail, not a mathematical property. Similar sensitivity exists for any
  placer that clusters sites (see `greedy`/`bayesopt` failures).
- **Storage asymmetry.** GP results store `N` support points
  (`n_data_params = N`); comparing GP to a spline at equal *sample* budget
  understates GP's deployment storage in a per-Gaussian GS setting.
- **Seeds and multiplicity.** 3 seeds per condition and no multiple-testing
  correction; the effect sizes above (especially the 3–18% gaps) should be
  confirmed on the full grid with 5–10 seeds and confidence intervals.
- **Non-oracle by construction, but defaults differ.** Every method uses its
  own initialization and hyperparameter defaults; results speak to
  "reasonable defaults at equal budget", not to per-method after tuning.

### 7.9 Recommended follow-up experiments

Ranked, each as a concrete suite/condition change:

1. **Run the full grids with 5–10 seeds** (`--full`) and add `orbit`,
   `contact_drop`, `combo`; report confidence intervals per cell and
   confirm/falsify the small (<20%) gaps.
2. **Penalty tuning and adaptive knot budgets.** Sweep `reg_lam` over
   decades with GCV/CV selection and replace `n_sites = budget/3` with
   CV/active growth; hypothesize that a tuned penalized cubic B-spline
   matches GP at 5–20 samples with `O(1)` storage.
3. **Time-parameterization study** (linear vs chord/centripetal/accel/jerk at
   fixed placement) and the oracle-vs-realistic comparison for
   `feature_peaks`/`jerk_importance`; this isolates the "separate path from
   timing law" recommendation from the SplineGS study.
4. **Noise realism.** AR(1)/heteroscedastic track noise, outliers
   proportional to track speed, and missing-point bursts; test whether
   Huber/PSpline close the gap to GP under temporally correlated noise.
5. **Hybrid frontier.** Spline + GP residual (or spline + small MLP residual)
   as a closed-form-latency candidate: use the GP posterior residual to
   place knots and then discard the GP at inference.
6. **GS deployment projection.** Multiply `n_params_total` by a Gaussian
   budget and compare storage/latency against the SplineGS / SC-GS numbers
   in the companion study; validate with a trajectory-mode run at equal
   per-Gaussian parameter budget.
7. **Orientation/SO(3) extension** (quaternion B-splines, cumulative
   B-splines, dual quaternions) once translation ordering is settled, since
   the companion study attributes SplineGS's H1 failure to the rotation
   parameterization rather than to splines per se.

## 8. Iteration 2 — stress-testing the Iteration 1 winners

> **Why.** Section 7's conclusions — GP-Matérn is the most sample-efficient
> and noise-robust representation, adaptive knots beat uniform, penalized
> cubic B-splines close the gap, Sobol beats active sampling — were obtained
> under i.i.d. Gaussian noise, translation-only, fixed-duration, fixed
> time-range synthetic motions at 3 seeds. Iteration 2 freezes every
> registry and re-runs the winners under the assumptions that could have
> manufactured those conclusions: structured track noise, non-linear time
> parameterization, extrapolation, adversarial knot layouts and auto-tuned
> knot/penalty counts. It also instruments **every fit** with numerical
> conditioning telemetry and reports **paired bootstrap** confidence
> intervals instead of 3-seed means.

All Iteration 2 records live in `results/iteration2/*.jsonl` (kept separate
from the Iteration 1 files so §6 is unchanged). Runner:

```bash
python scripts/run_iteration2.py        # 4 455 conditions, 5 seeds, ~2 min
python -m splinebench.report --results results \
    --i2-dir results/iteration2 --readme README.md
```

### 8.1 Conditioning telemetry — failures are conditioning failures

Every fit now records the design-matrix condition number and rank, the
effective degrees of freedom, the penalty rank/null space, the GP kernel
condition number, residual autocorrelation and knot-spacing statistics.

<!-- AUTO:I2.1 START -->
| representation | % rank-def | median `design_cond` | median `eff_dof` | median `eff_dof_ratio` | median `gp_kernel_cond` | median `resid_acf1` |
|---|---|---|---|---|---|---|
| bspline3 | 67 | 10.8 | 7 | 0.35 | — | -0.302 |
| catmull_rom | 0 | 3.67 | 7 | 0.32 | — | -0.22 |
| gp_matern52 | — | — | — | — | 1.8e+03 | -0.538 |
| gp_rbf | — | — | — | — | 940 | -0.507 |
| hermite | 0 | 2.89 | 7 | 0.32 | — | -0.213 |
| pspline | 67 | 10.8 | 6.81 | 0.34 | — | -0.283 |

`i2_core`, iid noise, budgets 10–50; `design_cond` infinite/NaN counts as rank-deficient; medians over all motions/seeds.
<!-- AUTO:I2.1 END -->

The headline is **67% of unpenalized `split_merge` cubic fits are
rank-deficient**: the placer puts sites at the observed data extremes, which
leaves the first/last B-spline basis function with no support at any
observation, so the design loses rank and `cond = ∞`. The penalty does not
remove the duplicate knots, but it regularizes the null directions
(`penalty_rank = 8`, `penalty_nullity = 2` for a cubic P-spline), which is why
`pspline` has the same 67% rank deficiency yet does not blow up. GP's kernel
matrix is well conditioned (median `1.8e3` at `N ≤ 50`), and its **negative
residual autocorrelation (`resid_acf1 ≈ -0.54`)** is a direct signature of
noise interpolation — the flip side of its robustness. This supports the
Iteration 2 hypothesis that many "representation failures" are really
conditioning failures.

### 8.2 Realistic track noise — GP's noise robustness survives; RANSAC's does not

Nine observation processes are compared at the same nominal `σ = 0.01`:
i.i.d., AR(1) at ρ=0.90/0.98, speed-scaled heteroscedasticity, speed-weighted
outliers, missing bursts, quantization, per-axis correlation and bias drift.

<!-- AUTO:I2.2 START -->
| method | iid | ar1_0.90 | ar1_0.98 | hetero_speed | speed_outliers | missing_bursts | quantized | axis_corr | bias_drift |
|---|---|---|---|---|---|---|---|---|---|
| gp_matern52 + uniform + least_squares | **0.214** | **0.212** | **0.212** | **0.22** | **0.215** | **0.232** | **0.215** | **0.215** | **0.223** |
| pspline + split_merge + least_squares + diff(0.001) | 0.464 | 0.469 | 0.47 | 0.494 | 0.501 | 0.561 | 0.474 | 0.471 | 0.477 |
| hermite + feature_peaks + huber | 0.346 | 0.349 | 0.349 | 0.344 | 0.35 | 0.44 | 0.346 | 0.348 | 0.353 |
| catmull_rom + uniform + least_squares | 1.19 | 1.19 | 1.19 | 1.22 | 1.2 | 2.41 | 1.19 | 1.19 | 1.25 |
| bspline3 + split_merge + ransac | 1.09 | 0.909 | 0.91 | 0.968 | 0.936 | 3.38 | 0.905 | 473 | 1.66 |

`i2_noise`: mean hold-out `pos_rmse` (lower is better), over {staccato, bounce, wobble} × budgets {20, 50} × 5 seeds; best *absolute* error per column in bold. Compare each cell with its own `iid` column to read off the degradation ratio (e.g. RANSAC is >400× worse under per-axis correlation and 3× worse under missing bursts).
<!-- AUTO:I2.2 END -->

GP-Matérn stays within **8% of its i.i.d. error on every structured noise
process**, reproducing the Iteration 1 robustness ranking. The spline rows
degrade most under **missing bursts** (Catmull-Rom 2.4× its i.i.d. error,
P-spline 1.2×) — i.i.d. missing points are easy because they leave the local
density nearly uniform, whereas a burst removes a whole time neighbourhood.
RANSAC is the cautionary result: it is **>400× worse under per-axis
correlation** and **3× worse under missing bursts**, and it is the only
method that *improves* (relatively) as noise gets structured — because it has
already collapsed to a smooth majority curve. Temporally correlated noise
does **not** break GP, so Iteration 1's "GP is noise-robust" conclusion is
not an artifact of i.i.d. noise; the robust-loss conclusion is.

### 8.3 Time parameterization — warping helps, but *oracle* timing can be worse

The time-parameterization axis was untested in Iteration 1. The spline is fit
in the warped domain `u(t)` and metrics compose derivatives through `u` by
the chain rule; `oracle` mode builds `u(t)` from ground-truth derivatives.

<!-- AUTO:I2.3 START -->
| representation | time param | realistic `pos_rmse` | oracle `pos_rmse` | realistic `peak_time_err` | oracle `peak_time_err` |
|---|---|---|---|---|---|
| gp_matern52 | linear | 0.226 | 0.226 | 0.22 | 0.22 |
| gp_matern52 | chord | 0.273 | 0.274 | 0.182 | 0.156 |
| gp_matern52 | centripetal | 0.29 | 0.273 | 0.243 | 0.206 |
| gp_matern52 | accel | 0.27 | 0.284 | 0.179 | 0.196 |
| gp_matern52 | jerk | 0.278 | 0.292 | 0.192 | 0.209 |
| pspline | linear | 0.464 | 0.464 | 0.452 | 0.452 |
| pspline | chord | 0.316 | 0.321 | 0.258 | 0.261 |
| pspline | centripetal | 0.319 | 0.319 | 0.267 | 0.277 |
| pspline | accel | 0.307 | 0.327 | 0.271 | 0.282 |
| pspline | jerk | 0.32 | 0.33 | 0.255 | 0.282 |
| bspline3 | linear | 0.404 | 0.404 | 0.334 | 0.334 |
| bspline3 | chord | 0.323 | 8.23e+05 | 0.215 | 0.191 |
| bspline3 | centripetal | 0.327 | 8.47e+04 | 0.267 | 0.247 |
| bspline3 | accel | 0.343 | 7.86e+04 | 0.238 | 0.252 |
| bspline3 | jerk | 0.469 | 2.33e+05 | 0.261 | 0.219 |

`i2_timeparam`: mean over {staccato, double_step, wobble} × budgets {20, 50} × 5 seeds; the spline is fit in the warped `u(t)` and metrics compose derivatives through it.
<!-- AUTO:I2.3 END -->

Warped time helps the splines substantially: `pspline` improves from 0.464
(linear) to ~0.31–0.32 (chord/centripetal/accel/jerk) and `bspline3` from
0.404 to ~0.32. But **oracle timing is not automatically better**: on
`double_step` the ground-truth arc-length map has a dwell where `u(t)` is
nearly flat, several observations collapse onto the same `u`, `split_merge`
inserts near-duplicate knots, and the unpenalized `bspline3` reaches
`pos_rmse` of 10⁶–10⁷ (`design_cond` up to 10⁶⁷). The noisy, Savitzky–Golay
smoothed realistic timing law avoids the exact dwell and stays stable. This
sharpens the SplineGS H2 hypothesis: the failure is an **interaction between
the timing law, duplicate knots and conditioning**, not the spline basis per
se — and a difference penalty (or a knot-spacing floor) is the fix, as the
`pspline` oracle rows show (0.32, not 10⁶). GP is unaffected because it has no
knots (0.22–0.29 across all time laws).

### 8.4 Extrapolation — GP generalizes outside the training window, splines do not

Training is restricted to `early` (0–0.5), `late` (0.5–1) and `middle`
(0.3–0.7) windows, or to a disjoint union (0–0.45 ∪ 0.55–1) with a held-out
middle interval (`interp`). Hold-out error is measured on the unseen window.

<!-- AUTO:I2.4 START -->
| representation | none | early | late | middle | interp | holdout/trainwin (mean) |
|---|---|---|---|---|---|---|
| gp_matern52 | 0.169 | 0.306 | 0.308 | 0.314 | 0.248 | 2.87 |
| gp_rbf | 0.166 | 0.305 | 0.306 | 0.299 | 0.246 | 3.08 |
| pspline | 0.363 | 4.16 | 3.71 | 4.54 | 0.317 | 14 |
| bspline3 | 0.393 | 41 | 42.6 | 31 | 0.325 | 88.4 |
| catmull_rom | 1.05 | 9.97 | 8.6 | 6.46e+03 | 0.452 | 44 |
| hermite | 0.249 | 3.23 | 2.5 | 3.03 | 0.273 | 12.7 |

`i2_extrap`: hold-out `pos_rmse`; `early`/`late`/`middle` extrapolate beyond the training window, `interp` holds out a disjoint middle interval (train on 0–0.45 ∪ 0.55–1). Mean over {staccato, bounce, chirp} × budgets {20, 50} × 5 seeds, linear time.
<!-- AUTO:I2.4 END -->

GP-Matérn and GP-RBF keep hold-out error near their interpolation error
(~0.3 versus 0.17), while adaptive-knot splines degrade by 10–100×
(`pspline` 4.2, `bspline3` 41, Hermite 3.2, Catmull-Rom 10–6.5×10³). The
kernel prior is a genuine extrapolation advantage, and the Iteration 2
hypothesis that regularized adaptive-knot splines extrapolate better is
**falsified** with these placers — they anchor their last knot at the edge of
the observed window and then extrapolate a polynomial. Interpolation hold-out
(`interp`) is benign for every method (< 0.5). If extrapolation matters for
deployment, it needs an explicit prior or an edge-aware knot policy, not just
adaptive placement.

### 8.5 Adversarial knot conditioning — the penalty fixes identifiability, not jerk

To separate basis expressiveness from numerical conditioning, knots are
deliberately clustered, spaced just above the duplicate guard, piled at one
end, or given a single huge gap.

<!-- AUTO:I2.5 START -->
| representation | reg | mode | selected sites | median `design_cond` | `pos_rmse` | `jerk_rmse` |
|---|---|---|---|---|---|---|
| bspline3 | diff | cluster | 16 | inf | 0.356 | 2.47e+07 |
| bspline3 | none | cluster | 16 | inf | 0.437 | 3.09e+08 |
| bspline3 | none | default | 16 | inf | 0.75 | 3e+05 |
| bspline3 | none | near_duplicate | 16 | inf | 0.323 | 5.58e+09 |
| bspline3 | none | one_gap | 16 | inf | 0.31 | 1.95e+06 |
| bspline5 | diff | cluster | 16 | inf | 0.331 | 1.03e+07 |
| bspline5 | none | cluster | 16 | inf | 13.5 | 1.33e+10 |
| bspline7 | diff | cluster | 16 | inf | 0.331 | 9.55e+06 |
| bspline7 | none | cluster | 16 | inf | 7.99e+03 | 5.72e+12 |
| hermite | none | cluster | 16 | inf | 0.519 | 1.85e+09 |
| pspline | diff | cluster | 16 | inf | 0.356 | 2.47e+07 |

`i2_adversarial`, budget 50, mean over {staccato, bounce, wobble} × 5 seeds; `uniform` is the reference row. `near_duplicate` sites are 1.1e-4 apart, below any useful resolution.
<!-- AUTO:I2.5 END -->

Clustered and near-duplicate layouts make the design rank-deficient
(`cond = ∞`) for every degree. Unpenalized degree-5/7 B-splines then explode
(`bspline5` 13.5, `bspline7` 8.0×10³ position RMSE; jerk 10¹⁰–10¹²). Adding a
second-difference penalty pulls *position* back to ~0.33 — comparable to the
uniform layout — but leaves **jerk at 10⁷**, two-to-five orders above the
well-spaced case. So regularization restores coefficient identifiability but
does not by itself make clustered knots safe for derivative metrics; a
knot-spacing floor or a derivative penalty is still required. This is the
quantitative version of Iteration 1's "high degree without penalty is a trap."

### 8.6 Knot-count and penalty auto-tuning — naive GCV under-smooths and loses

`n_sites = budget/3` is arbitrary. The `cv` placer selects the *number* of
knots by K-fold cross-validation on the observation track, and `pspline_gcv`
replaces the hand-set `lam` with a GCV scan.

<!-- AUTO:I2.6 START -->
| method | knots | reg | mean sites | B20 `pos_rmse` | B50 `pos_rmse` | `jerk_rmse` |
|---|---|---|---|---|---|---|
| bspline3 | cv | — | 4.05 | 0.381 | 0.3 | 5.44e+05 |
| bspline3 | split_merge | diff | 11.5 | 0.358 | 0.401 | 5.03e+06 |
| bspline3 | split_merge | — | 11.5 | 0.309 | 0.509 | 1.83e+06 |
| pspline | cv | — | 4.7 | 0.383 | 0.295 | 1.67e+06 |
| pspline | split_merge | — | 11.5 | 0.376 | 0.655 | 5.32e+06 |
| pspline_gcv | split_merge | — | 11.5 | 0.297 | 0.732 | 2.31e+06 |

`i2_knotcount`, mean over {staccato, double_step, bounce, wobble} × 5 seeds; `n_sites` is the budget/3 default upper bound, `mean sites` is what the placer actually used.
<!-- AUTO:I2.6 END -->

CV picks **4–5 sites at B50** instead of the default 16 and improves the
unpenalized cubic from 0.509 to 0.300 position RMSE (`pspline` from 0.655 to
0.295) while cutting jerk by ~3×. But the paired bootstrap (§8.7) puts the
55% B50 gap's 95% CI at `[-0.564, 0.025]` (`p = 0.16`): **with 5 seeds and 4
motions the win is not significant**. GCV is the surprise: as the knot count
grows its median `lam` falls (3.5 at B20 → 2.0 at B50) while `eff_dof` rises
(4.6 → 12.8), so it is *worse* than the hand-set `lam = 1e-3` at B50 (0.732
versus 0.655 for the same knots). Tuning is not automatically better than a
sensible default; knot count matters more than penalty strength, which is the
opposite of the Iteration 1 follow-up guess.

### 8.7 Paired statistics — the big claims hold, the small ones do not

Every Iteration 2 condition is run at 5 seeds, and comparisons are paired by
`(motion, budget, seed)` with a 5 000-resample bootstrap CI.

<!-- AUTO:I2.7 START -->
| suite | paired comparison | metric | mean Δ [95% CI] | p | n |
|---|---|---|---|---|---|
| `i2_core` | GP-Matérn − pspline (B20) | `pos_rmse` | -0.164 [-0.218, -0.113] | 0 | 50 |
| `i2_core` | GP-Matérn − pspline (B50) | `pos_rmse` | -0.168 [-0.263, -0.0907] | 0 | 50 |
| `i2_core` | GP-Matérn − Catmull-Rom | `pos_rmse` | -0.434 [-0.838, -0.133] | 0 | 150 |
| `i2_adversarial` | clustered: diff − none | `pos_rmse` | -1.33e+03 [-3.61e+03, -87.2] | 0 | 15 |
| `i2_knotcount` | bspline3: CV − fixed knots (B20) | `pos_rmse` | 0.0721 [-0.0227, 0.243] | 0.637 | 20 |
| `i2_knotcount` | bspline3: CV − fixed knots (B50) | `pos_rmse` | -0.208 [-0.564, 0.0251] | 0.162 | 20 |

Paired bootstrap (5000 resamples) over matched (motion, budget, seed) cells; Δ = first − second, negative favours the first method. p is the two-sided bootstrap sign p-value.
<!-- AUTO:I2.7 END -->

GP-Matérn's lead over `pspline` (−0.164 at B20, −0.168 at B50) and over
Catmull-Rom (−0.434) are significant with CIs excluding zero. The
adversarial-penalty rescue is significant but its CI spans three orders of
magnitude (`[−3.6×10³, −87]`) because it is dominated by the `bspline7`
blow-up. The CV knot-count improvement at B50 is **not** significant. The
Iteration 1 caution therefore stands: 3–18% gaps at 3 seeds should not be
trusted; the Iteration 2 grids give the effect sizes and intervals needed to
tell which conclusions are real.

### 8.8 Iteration 2 verdicts on the Iteration 1 menu

| Iteration 1 claim | Iteration 2 stress | Verdict |
|---|---|---|
| GP-Matérn leads sample efficiency | structured noise, time warps, extrapolation, 5 seeds | **survives** — best or near-best in every suite, significant in paired tests |
| GP is the most noise-robust | AR(1), hetero, speed outliers, bursts, axis corr, drift | **survives** — within 8% of i.i.d.; robust losses do not close the gap |
| Adaptive knots beat uniform | clustered/near-duplicate/one-gap layouts | **partial** — they beat uniform when well-spaced, but rank-deficiency is pervasive (67%) and duplicated knots are dangerous |
| Penalized cubic closes the gap at low N | adversarial knots, GCV tuning, extrapolation | **partial** — penalty rescues position but not jerk, and hurts extrapolation; hand-set `lam=1e-3` beats GCV |
| Active sampling loses to Sobol under i.i.d. | not re-run (sampling unchanged) | **open** — structured noise sampling is the main remaining gap |
| Problem is representation, not conditioning | design-`cond`/rank/DoF telemetry | **falsified** — conditioning explains a large share of the failures |
| Timing law was untested | linear/chord/centripetal/accel/jerk × oracle/realistic | **new** — warping helps splines, but *oracle* timing can be catastrophic via duplicate knots |

**Not covered by Iteration 2** (carried into §9): rotation/`SO(3)` and
`SE(3)` representations, per-Gaussian deployment/storage projection,
online/streaming fits, physical/constraint feasibility, and cost-aware or
active sampling under the new structured noise models. MLP/`bspline_mlp` were
excluded from the main I2 grids to keep them GPU-free and fast.

## 9. Hypotheses and practical tips

Hypotheses this harness is designed to falsify (result verdicts in §7.4):

- **H1.** At equal `n_params`, data-adaptive knot placement
  (`feature_peaks`/`split_merge`/`active_residual`) beats uniform spacing on
  snap scenes in the 5–20 sample regime, where the placement error dominates
  the basis error.
- **H2.** Piecewise (B-spline/Hermite) bases beat global bases (Chebyshev,
  Fourier, polynomial) on `bang_bang`, `bounce` and `wobble` because
  high-order global bases ring at discontinuities.
- **H3.** Difference-penalized B-splines (`pspline`) dominate least-squares
  B-splines at low budget under noise (bias–variance), but the gap closes by
  ~100 samples.
- **H4.** GP with a Matérn kernel is the sample-efficiency leader for smooth
  scenes but over-smooths `bang_bang`/`pulse` scenes relative to piecewise
  polynomials; `gp_rbf` is worse than `gp_matern52` there.
- **H5.** Robust losses (`huber`, `ransac`) only help when
  `outlier_frac > 0`; with clean noise they cost a small bias.
- **H6.** Derivative-aware sampling has the largest effect at budgets 5–20;
  active/uncertainty samplers buy more on `chirp`/`combo` than on `staccato`.
- **H7.** MLP/neural residual methods need order-of-magnitude more samples
  than splines for equal error, but may win on `combo` if the residual is
  genuinely non-spline.

Practical guidance carried over from the SplineGS-empirical study:

- separate the spatial path from the timing law (non-uniform `u(t)`);
- asymmetric time scaling, not symmetric easing, for snap-rest-snap;
- knots at acceleration reversals, impact instants and derivative extrema;
- enforce/penalize jerk and snap explicitly for track-noise robustness;
- regularize heavily at low sample counts, then release;
- compare at 5–20 samples, not only at the final budget — that is where
  sample efficiency lives.

## 10. Repo layout

```
splinebench/
  motions.py         synthetic GT motions, analytic derivatives to snap
  representations.py bases + fitters registry and derivative math
  fitters.py         linear/robust/nonlinear/global optimizers, BO
  knots.py           knot/control-site placement strategies (incl. cv, clustered)
  samplers.py        observation-time sampling strategies (incl. active)
  timeparam.py       monotone t -> u time parameterizations
  metrics.py         trajectory metrics, conditioning, AULC, DTW, Pareto
  experiment.py      Condition tuple, noise oracle (i.i.d. + structured), runner
  suites.py          smoke / starter / ... / full + i2_* condition generators
  catalog.py         machine-readable status + citation for every method
  plots.py           sample-efficiency, Pareto, robustness plots
  report.py          README/ANALYSIS table generation (6.x and I2.x)
  run.py             CLI
scripts/
  run_extra.py       fill Iteration 1 table gaps
  run_iteration2.py  Iteration 2 stress grids -> results/iteration2/
tests/               unit + conformance tests for every implemented axis
references.bib       bibliography (keys used by catalog.py and this README)
thirdparty/          local paper PDFs (git-ignored)
results/             Iteration 1 JSONL runs, ANALYSIS.md and plots (git-ignored)
results/iteration2/  Iteration 2 JSONL runs (git-ignored)
```

## Appendix A — Full method menu (implemented `[x]` / backlog `[ ]`)

### A.1 Representations

Classical piecewise polynomial
`[x]` piecewise linear, cubic Hermite, Catmull-Rom, B-spline deg 2/3/5/7,
P-spline, GCV P-spline (`pspline_gcv`), NURBS, Akima, PCHIP ·
`[ ]` quadratic, quintic, septic, quintic Hermite, Bézier (quadratic, cubic,
higher-order, rational, composite), uniform/non-uniform/clamped/open/periodic/
quasi-uniform B-splines, T-splines [`sederberg2003`], LR B-splines
[`dokken2013`], hierarchical B-splines [`forsey1988`], M-/I-splines
[`ramsay1988`], smoothing/regression splines [`wahba1990`],
Kochanek-Bartels/TCB [`kochanek1984`], Steffen [`steffen1990`],
Fritsch-Carlson monotone, thin-plate/polyharmonic splines [`duchon1977`],
kernel ridge, wavelets, Chebyshev/Legendre/Bernstein/Lagrange/Newton,
exponential/trigonometric/L-/box splines, subdivision curves (Chaikin
[`chaikin1974`], Catmull-Clark [`catmull1978`], Loop [`loop1987`],
Doo-Sabin [`doo1978`]).

Geometric / manifold
`[ ]` quaternion slerp/squad/log-quaternion (SO(3) Bézier/B-spline)
[`shoemake1985`, `shoemake1987`, `park1997`, `kim1995`], SE(3) screw,
dual-quaternion splines [`kavan2007`], Riemannian splines [`park1997`].

Learned / statistical
`[x]` GP regression [`rasmussen2006`], discrete DMP [`ijspeert2013`],
Fourier-feature MLP [`sitzmann2020`], spline + neural residual ·
`[ ]` ProMP [`paraschos2013`], ProDMP, Bayesian interaction primitives
[`benamor2014`], GMM/GMR [`calinon2007`], HMM/HSMM, neural ODE [`chen2018`],
SIREN, implicit neural representations, neural spline flows [`durkan2019`],
VAE, diffusion policies [`chi2023`], transformer policies.

Time-optimal / control-style
`[ ]` minimum jerk [`hogan1984`], minimum snap [`mellinger2011`], bang-bang,
trapezoidal, S-curve, asymmetric S-curve, time-optimal path parameterization
[`pham2014`], MPC, iLQR [`li2004`], DDP [`mayne1966`], direct
collocation/multiple shooting, CHOMP [`zucker2013`], STOMP
[`kalakrishnan2011`], TrajOpt [`schulman2014`], GPMP/GPMP2 [`mukadam2018`].

Hybrids
`[x]` B-spline + neural residual (`bspline_mlp`) ·
`[ ]` B-spline + GP residual, spline path + learned timing law,
contact/event-based hybrid splines, spline + MPC tracking layer.

### A.2 Control-point / knot placement

`[x]` uniform, quantile, chord-length, centripetal, curvature-adaptive,
feature-based (derivative extrema), acceleration reversals, RDP
[`ramer1972`], split-and-merge, recursive bisection, CV knot-count selection
(`cv`), adversarial clustered layouts (`clustered`), greedy forward selection,
orthogonal-matching-pursuit-style greedy [`tropp2007`], k-means clustering,
farthest-point [`elden1997`], Bayesian optimization [`snoek2012`],
residual-based adaptive refinement, active learning (uncertainty) [`cohn1996`].

`[ ]` Foley, universal, corner detection, velocity/acceleration/jerk peaks,
zero crossings, contact/impact events, via-points/task constraints, manual
keyframes, mocap markers, semantic boundaries, Douglas-Peucker, greedy
backward elimination, LARS/LASSO path [`efron2004`], GP-based knot search,
genetic/PSO [`kennedy1995`]/simulated annealing/CMA-ES [`hansen2001`] over
knots, GMM-based placement, dynamic programming over candidate knots, DTW,
PCA, autoencoder latent space, incremental knot insertion,
multi-resolution/coarse-to-fine, cross-validation, AIC/BIC/MDL,
query-by-committee [`seung1992`], expected error reduction/model change
[`cohn1996`, `mackay1992`], D-/A-/E-optimal design [`pukelsheim2006`], Latin
hypercube/Sobol/Halton/Poisson-disk [`mckay1979`, `sobol1967`], importance
sampling, adversarial placement.

### A.3 Fitting / optimization

`[x]` OLS/WLS, ridge, LASSO [`tibshirani1996`], elastic net [`zou2005`], TV
[`rudin1992`], Huber/IRLS [`huber1964`], RANSAC [`fischler1981`],
Levenberg-Marquardt [`levenberg1944`], trust-region-reflective/dogbox
[`branch2000`], Nelder-Mead [`nelder1965`], Powell [`powell1964`],
differential evolution [`storn1997`], Adam [`kingma2015`], Bayesian
optimization [`snoek2012`].

`[ ]` generalized least squares, group LASSO, sparse Bayesian learning,
compressed sensing [`candes2008`], Tukey bisquare, Cauchy, trimmed LS,
Gauss-Newton, Powell hybrid, SGD/momentum/Nesterov/RMSProp/conjugate
gradient/BFGS/L-BFGS/DFP, random/grid search, genetic algorithms, CMA-ES
[`hansen2001`], PSO [`kennedy1995`], ant colony, TPE, Hyperband/BOHB, QP/LP/
SOCP/SDP, SQP, interior point, augmented Lagrangian, ADMM, projected/proximal
gradient, Frank-Wolfe, dynamic programming, A*/D*, branch and bound,
MILP/MINLP, direct shooting/collocation, iLQR [`li2004`], DDP [`mayne1966`],
MPC, CHOMP [`zucker2013`], STOMP [`kalakrishnan2011`], TrajOpt
[`schulman2014`], GPMP2 [`mukadam2018`], IRL [`ziebart2008`], behavior
cloning/DAgger [`ross2011`], PPO [`schulman2017`], SAC [`haarnoja2018`],
TD3 [`fujimoto2018`], DDPG [`lillicrap2016`], A3C [`mnih2016`], evolutionary
strategies, MAML [`finn2017`], Reptile [`nichol2018`], PINN [`raissi2019`],
implicit differentiation, differentiable splines.

### A.4 Time parameterization

`[x]` linear, chord-length, centripetal, acceleration-weighted,
jerk-weighted ·
`[ ]` Foley, universal, quantile-of-feature, asymmetric time scaling, learned
timing law.

### A.5 Sampling strategies

`[x]` random, uniform grid, stratified, Latin hypercube [`mckay1979`],
Sobol [`sobol1967`], Halton, farthest-point [`elden1997`], jerk importance,
derivative peaks (impact/derivative-extrema instants), hybrid,
active GP uncertainty [`cohn1996`], active residual, query by committee
[`seung1992`].

`[ ]` Poisson-disk, expected error reduction/model change
[`cohn1996`, `mackay1992`], Bayesian experimental design [`chaloner1995`],
D-/A-/E-optimal [`pukelsheim2006`], recursive least squares, Kalman filtering
[`kalman1960`], particle filtering [`thrun2005`], incremental knot insertion,
compressed sensing [`candes2008`], adversarial selection, transfer/meta
learning [`finn2017`].

### A.6 Metrics

`[x]` position/velocity/acceleration/jerk/snap RMSE and max, peak-time error,
overshoot, settling time, ISJ and ISJ ratio, constraint-violation
exceedances, train error, AULC, fit/eval time, parameter counts, robustness
under noise/outliers/missing/bias, Pareto fronts, leave-one-demo-out-style
seed aggregation.
`[ ]` extrapolation/generalization splits, integral of squared snap,
curvature/smoothness summaries, torque/collision constraints, per-Gaussian
storage projections, multi-seed confidence intervals in plots.

## Appendix B — Key references

The full list is in [`references.bib`](references.bib). Most load-bearing
entries: [`park2024splinegs`] (the failing baseline), [`deboor1978`],
[`eilers1996`], [`rasmussen2006`], [`ijspeert2013`], [`hogan1984`],
[`snoek2012`], [`cohn1996`], [`huber1964`], [`fischler1981`],
[`tibshirani1996`], [`storn1997`], [`hansen2001`].

## Notes, scope and roadmap

- v1 is **translation-only**; per-axis trajectories are the controlled
  variable. Rotation entries (quaternion B-splines, cumulative B-splines
  [`kim1995`], dual quaternions [`kavan2007`]) slot straight into the
  representation registry and re-use the same metrics after a geodesic
  variant is added.
- Iteration 2 closes the observation-model gap: AR(1), heteroscedastic,
  speed-proportional outliers, missing bursts, timestamp jitter,
  quantization, per-axis correlation and bias drift are implemented (§3.6),
  and the winners were re-tested under them (§8.2). Per-Gaussian
  multi-query budgets, deployment/storage projection and `SO(3)`/`SE(3)`
  remain the next additions.
- Conditioning is now first-class: every record carries design-matrix and
  kernel condition numbers, effective DoF and knot-spacing statistics, and
  §8.1 shows a large share of "representation failures" are rank/conditioning
  failures. Any new method should report `design_cond`, `eff_dof` and
  `knot_near_dupes` before claiming a representation win.
- `full`-suite counts are large (~56.7k conditions) because every tuple is
  stored; use `--limit`/`--suite placement` and resume from JSONL.
- All derivatives used by metrics are analytic (chain rule through the
  fitted `u(t)`), so jerk/snap comparisons are meaningful even at 5 samples.
