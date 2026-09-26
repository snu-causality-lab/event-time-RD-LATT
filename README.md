# Event-time regression discontinuity

A small Python library and a fully synthetic worked example accompanying
*Event-Time Regression Discontinuity for Biomedical Data: LATT Estimation and
Covariate Distribution Tests*, by Yesong Choe, Yeahoon Kwon, Minjung Kho,
Seunggeun Lee, and Sanghack Lee. Yesong Choe and Yeahoon Kwon contributed equally.

The library accepts user-supplied numerical arrays. The notebook generates all
observations from a stated mathematical model and a fixed random seed. No UK
Biobank participant data, transformed participant records, or empirical reference
results are included. The example illustrates the method; its numerical results
are simulated, not the article's empirical results.

The implemented scope is the article's primary triangular local-linear RD
specification and its window-wide covariate distribution tests. Article-specific
tables, plots, and additional sensitivity specifications are not bundled.

## Install and run

Use Python 3.11 or later. From this directory:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[example]'
python -m unittest discover -s tests -v
python -m notebook synthetic_example.ipynb
```

Run the notebook from top to bottom. To execute it without opening a browser:

```bash
python -m nbconvert --to notebook --execute synthetic_example.ipynb --output synthetic_example_executed.ipynb
```

For library use alone, install with `python -m pip install -e .`.

## Estimate an event-time discontinuity

```python
from event_time_rd import estimate_rd

result = estimate_rd(outcome, event_time, covariates=baseline_covariates)
```

`event_time` is measurement time minus treatment initiation time, in a consistent
unit. Negative values precede the event and positive values follow it. The cutoff
is zero; observations exactly at zero are excluded. The library does not round
times or interpret calendar dates. In an application, a recorded prescription
date need not be the date of actual treatment initiation.

`outcome` and `event_time` are aligned one-dimensional numerical arrays.
`covariates` is an optional aligned numerical vector or matrix. Missing values
are excluded using one common complete-case mask; infinities and invalid shapes
are rejected. Pandas inputs must have matching row indices. Counts of input,
retained, and excluded observations are returned.

Estimation delegates to [rdrobust](https://github.com/rdpackages/rdrobust): local
linear regression, a triangular kernel, quadratic bias correction, and 95%
robust bias-corrected (RBC) inference. With `bandwidth=None`, bandwidths are
selected by `rdrobust`. A positive scalar or left/right pair fixes both the
estimation and bias-correction bandwidths. Small or rank-deficient local samples
cannot support the requested fit and raise an error.
Undefined inference, including nonfinite results or nonpositive standard errors,
also raises an error rather than returning a misleading P value.

The result is a dictionary. In particular:

| Fields | Meaning |
| --- | --- |
| `estimate`, `se`, `ci_low`, `ci_high`, `p_value` | Conventional local-linear estimate and inference |
| `estimate_bias_corrected` | Bias-corrected point estimate |
| `se_robust`, `ci_robust_low`, `ci_robust_high`, `p_value_robust` | RBC inference, with its interval centered on the bias-corrected estimate |
| `bandwidth_left`, `bandwidth_right` | Estimation bandwidths |
| `bias_bandwidth_left`, `bias_bandwidth_right` | Bias-correction bandwidths |
| `n_input`, `n_used`, `n_excluded` | Sample accounting before bandwidth restriction |
| `n_effective_left`, `n_effective_right` | Observations in the fitted local windows |

## Compare covariate distributions

```python
from event_time_rd import covariate_balance

balance = covariate_balance(
    event_time,
    {"baseline": baseline, "group": group_labels},
    categorical=("group",),
)
```

Across the supplied analysis sample (excluding zero), this compares negative and
positive event times. Columns listed in `categorical` receive Pearson's
chi-squared test; other numeric columns receive the two-sample Kolmogorov-Smirnov
test for continuous or ordered variables. Pass categorical column names as a
list or tuple, such as `["group"]`, rather than a bare string.
It retains the article's analysis-code conventions: the limiting Kolmogorov
survival function for K–S P values and Yates correction for 2×2 categorical tables.
Categorical labels may be strings. Missing observations are excluded separately for each variable.
The returned pandas DataFrame includes sample counts, the statistic, the unadjusted P value,
and a status. Empty sides, constant variables, and sparse categorical tables are
reported without a P value rather than treated as evidence of balance.
The optional `bandwidth` restricts the supplied sample strictly inside a chosen
window: a scalar for symmetric limits or a pair for left/right limits. The
default uses every supplied nonzero event time, as in the article's window-wide
distribution comparisons.

These are window-wide distribution comparisons. Smooth covariate drift can
produce a difference even without a discontinuity at zero. Non-rejection does
not establish local comparability or justify choosing adjustment variables from
P values. Use substantively appropriate pretreatment covariates.

## Interpretation and checks

The estimated quantity is an outcome discontinuity. Interpreting it as a local
average treatment effect among the treated (LATT) requires the relevant
identification assumptions: continuity of potential-outcome means near the
cutoff in the observed eventual-user population, consistency, no interference,
correct treatment timing, no anticipation, and no coincident outcome-changing
intervention. Diagnostic tests do not prove these assumptions.

The notebook constructs continuous event time independently of baseline
covariates and errors and imposes a known immediate treatment effect. This makes
the causal interpretation valid within its stated model. Day-rounded event time
with the zero day excluded instead requires extrapolation across the unobserved
interval; the continuous example does not validate that extrapolation.

The tests compare the conventional coefficient with independently computed
triangular weighted least squares and check input handling, undefined inference,
and the article's distribution-test conventions. RBC standard errors are supplied
by `rdrobust`; the coefficient check does not independently validate them.
