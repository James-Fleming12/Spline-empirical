# Spline-empirical

A synthetic, reproducible testbed for answering one question:

> **Which spline parameterization, knot placement, fitting method, time
> parameterization, regularization and sampling strategy give the most
> sample-efficient and robust reconstruction of complex / snappy motion from
> sparse noisy observations?**

This repository is the companion instrument to
`/home/james/Research/Rendering/SplineGS-empirical`, whose README documents
that the per-Gaussian cubic Hermite spline in **SplineGS** (Park et al.,
[`park2024splinegs`]) and its successor WebSpline fail on (H2) velocity
discontinuities, snap-rest-snap and near-Nyquist motion, and that no method in
that study has an *adaptive* temporal budget (MACP only prunes; AD-GS fixes
knots at `frames/3`; OriGS anchors are frame-discrete). That study also found
spline-class methods are the fastest to evaluate and the only ones with
editability, so the open slot is a **sample-efficient, noise-robust,
non-uniform/growth-capable spline** — exactly what this benchmark measures.

> **Status: scaffold.** All registries, the synthetic motion suite, the
> fitting/knot/sampling machinery, the runner and 100 tests are implemented
> and green; the result tables below are placeholders to be filled by the
> experiment runs.

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
`rdp`, `split_merge` (recursive max-residual bisection), `greedy` (forward
selection with a linear/real-representation objective), `kmeans` (weighted on
derivative density), `farthest_point`, `bayesopt` (GP-EI over knot
locations), `active_residual` (iterative residual-driven insertion).

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
Colored (AR(1)) and heteroscedastic track noise are on the roadmap.

### 3.7 Metrics

`pos_rmse`, `pos_mae`, `pos_max`, `vel_rmse`, `acc_rmse`, `jerk_rmse`,
`snap_rmse`, `jerk_max_fit`, `snap_max_fit`, `peak_time_err`, `overshoot`,
`settle_time_err`, `isj_fit`, `isj_ratio`, `cstr_{v,a,j,s}_{mean,max}`,
`train_rmse`, `fit_time_s`, `eval_time_s`, `n_params*`,
plus aggregation metrics `aulc` (area under the error-vs-log-budget curve) and
`pareto_front`.

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

# print the registry / backlog
python -m splinebench.run --list
```

Runs append JSONL and resume automatically (condition keys are hashed);
interrupting is safe. `suites.py` defines `smoke`, `starter`, `placement`,
`fitters`, `sampling`, `robustness`, `full`; `--full` switches to the larger
seed/motion/budget grid.

### Condition counts (unit-tested, not run here)

| Suite | quick | full |
|---|---:|---:|
| smoke | 6 | — |
| starter | 324 | 3 780 |
| placement | 504 | 21 840 |
| fitters | 120 | 7 560 |
| sampling | 240 | 10 920 |
| robustness | 120 | 12 600 |
| full (union) | — | 56 700 |

(Counts are generated by `suites.py`; run
`python -c "from splinebench import suites; print(len(suites.build('starter', quick=True)))"`.)

## 6. Results (placeholders)

> These tables are intentionally empty. Fill them with
> `python -m splinebench.run --report results/<run>.jsonl --summary ...`.

### 6.1 Sample efficiency — starter matrix (pos RMSE, mean ± sd over seeds)

| motion | representation | knots | sampler | 5 | 10 | 20 | 50 | 100 | 200 | AULC ↓ |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| staccato | hermite | feature_peaks | random | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| staccato | bspline3 | chord | random | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| staccato | bspline5 | curvature | random | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| staccato | pspline | split_merge | random | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| staccato | gp_rbf | — | random | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| staccato | dmp | uniform | random | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| staccato | mlp | — | random | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| staccato | bspline_mlp | uniform | random | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| … | | | | | | | | | | |

### 6.2 Peak timing and higher derivatives

| motion | method | `vel_rmse` | `acc_rmse` | `jerk_rmse` | `peak_time_err` | `overshoot` |
|---|---|---:|---:|---:|---:|---:|
| bounce | TBD | TBD | TBD | TBD | TBD | TBD |
| double_step | TBD | TBD | TBD | TBD | TBD | TBD |
| wobble | TBD | TBD | TBD | TBD | TBD | TBD |
| chirp | TBD | TBD | TBD | TBD | TBD | TBD |

### 6.3 Placement ablation (fixed `n_sites`, `pos_rmse` / `jerk_rmse`)

| knots \ motion | staccato | bounce | wobble | chirp |
|---|---|---|---|---|
| uniform | TBD | TBD | TBD | TBD |
| chord | TBD | TBD | TBD | TBD |
| curvature | TBD | TBD | TBD | TBD |
| feature_peaks | TBD | TBD | TBD | TBD |
| split_merge | TBD | TBD | TBD | TBD |
| greedy | TBD | TBD | TBD | TBD |
| bayesopt | TBD | TBD | TBD | TBD |
| active_residual | TBD | TBD | TBD | TBD |

### 6.4 Robustness (`pos_rmse`)

| method \ condition | clean | σ=0.01 | σ=0.03 | 5% outliers | 15% outliers | 10% missing | bias 0.05 |
|---|---|---|---|---|---|---|---|
| hermite + LS | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| hermite + huber | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| hermite + ransac | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| pspline (diff) | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| gp_matern52 | TBD | TBD | TBD | TBD | TBD | TBD | TBD |

### 6.5 Sampling comparison (AULC over budgets)

| sampler | bspline5 | gp_rbf | dmp | mlp |
|---|---:|---:|---:|---:|
| random | TBD | TBD | TBD | TBD |
| lhs | TBD | TBD | TBD | TBD |
| sobol | TBD | TBD | TBD | TBD |
| jerk_importance | TBD | TBD | TBD | TBD |
| derivative_peaks | TBD | TBD | TBD | TBD |
| active_gp | TBD | TBD | TBD | TBD |
| active_residual | TBD | TBD | TBD | TBD |
| query_by_committee | TBD | TBD | TBD | TBD |

### 6.6 Pareto fronts

| figure | command | status |
|---|---|---|
| `results/starter_sample_efficiency.png` | `--plots results/starter` | TBD |
| `results/starter_pareto.png` | `--plots results/starter` | TBD |
| robustness plot | `plots.plot_robustness` | TBD |

## 7. Hypotheses and practical tips

Hypotheses this harness is designed to falsify:

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

## 8. Repo layout

```
splinebench/
  motions.py         synthetic GT motions, analytic derivatives to snap
  representations.py bases + fitters registry and derivative math
  fitters.py         linear/robust/nonlinear/global optimizers, BO
  knots.py           knot/control-site placement strategies
  samplers.py        observation-time sampling strategies (incl. active)
  timeparam.py       monotone t -> u time parameterizations
  metrics.py         trajectory metrics, aggregation, AULC, Pareto
  experiment.py      Condition tuple, noise oracle, runner, JSONL I/O
  suites.py          smoke / starter / placement / fitters / sampling /
                     robustness / full condition generators
  catalog.py         machine-readable status + citation for every method
  plots.py           sample-efficiency, Pareto, robustness plots
  run.py             CLI
tests/               unit + conformance tests for every implemented axis
references.bib       bibliography (keys used by catalog.py and this README)
thirdparty/          local paper PDFs (git-ignored)
results/             JSONL runs and summaries (git-ignored)
```

## Appendix A — Full method menu (implemented `[x]` / backlog `[ ]`)

### A.1 Representations

Classical piecewise polynomial
`[x]` piecewise linear, cubic Hermite, Catmull-Rom, B-spline deg 2/3/5/7,
P-spline, NURBS, Akima, PCHIP ·
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
[`ramer1972`], split-and-merge, recursive bisection, greedy forward selection,
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
- The "GS realism" gap is the observation model, not the math: GS tracks are
  noisy, occasionally missing, and spatially correlated. v1 covers i.i.d.
  noise, outliers, missing and bias; AR(1)/heteroscedastic track noise and
  per-Gaussian multi-query budgets are the next additions.
- `full`-suite counts are large (~56.7k conditions) because every tuple is
  stored; use `--limit`/`--suite placement` and resume from JSONL.
- All derivatives used by metrics are analytic (chain rule through the
  fitted `u(t)`), so jerk/snap comparisons are meaningful even at 5 samples.
