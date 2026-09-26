"""Small helpers for an RD at numeric event time zero.

These functions do not establish causal identification. In particular, distribution
comparisons over a time window are not tests of discontinuities at the cutoff.
"""

import numpy as np
import pandas as pd
from rdrobust import rdrobust
from scipy.stats import chi2_contingency, ks_2samp, kstwobign

__all__ = ["estimate_rd", "covariate_balance"]


def _check_alignment(*values):
    """Reject positional fits of differently labeled pandas rows."""
    indices = [v.index for v in values if isinstance(v, (pd.Series, pd.DataFrame))]
    if indices and any(not indices[0].equals(index) for index in indices[1:]):
        raise ValueError("Pandas inputs must have identical indices in identical order.")


def _as_numeric_array(values, name):
    """Convert missing values to NaN; reject infinities and non-real dtypes."""
    array = np.asarray(values)
    if array.dtype.kind in "mMc":
        raise ValueError(f"{name} must contain real numbers, not dates or complex values.")
    try:
        array = np.where(pd.isna(array), np.nan, array).astype(float)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must contain real numbers or missing values.") from error
    if np.isinf(array).any():
        raise ValueError(f"{name} must not contain infinite values.")
    return array


def _parse_bandwidth(value):
    """Return positive left/right widths from a scalar or pair."""
    widths = _as_numeric_array(value, "bandwidth")
    if widths.ndim == 0:
        widths = np.repeat(widths, 2)
    if widths.shape != (2,) or not np.isfinite(widths).all() or (widths <= 0).any():
        raise ValueError("bandwidth must be positive: a scalar or a (left, right) pair.")
    return widths


def _check_outcome(y, x, widths):
    """Reject exact constancy before round-off creates spurious inference."""
    local = y[(x > -widths[0]) & (x < widths[1])]
    if np.all(local == local[0]):
        raise ValueError("Degenerate inference: outcome is constant inside the analysis bandwidths.")


def _check_design(x, widths, degree, covs=None):
    """Check side-specific polynomials and, if present, pooled covariate rank."""
    blocks, selected_covs = [], []
    for side, width in enumerate(widths):
        keep = ((x < 0) if side == 0 else (x > 0)) & (np.abs(x) < width)
        local = x[keep]
        label = "left" if side == 0 else "right"
        if len(local) < 4 or np.unique(local).size < degree + 1:
            raise ValueError(f"Inadequate {label} support for degree {degree} inside bandwidth.")
        scale = width if np.isfinite(width) else np.max(np.abs(local))
        polynomial = np.vander(local / scale, degree + 1, increasing=True)
        if np.linalg.matrix_rank(polynomial) != degree + 1:
            raise ValueError(f"Rank-deficient {label} polynomial design.")
        blocks.append(polynomial)
        if covs is not None:
            selected_covs.append(covs[keep])
    if covs is not None:
        design = np.zeros((sum(len(block) for block in blocks), 2 * (degree + 1)))
        split = len(blocks[0])
        design[:split, :degree + 1] = blocks[0]
        design[split:, degree + 1:] = blocks[1]
        design = np.column_stack((design, np.vstack(selected_covs)))
        scales = np.linalg.norm(design, axis=0)
        design = design / np.where(scales == 0, 1, scales)
        if np.linalg.matrix_rank(design) != design.shape[1]:
            raise ValueError("Covariates are rank-deficient after the side-specific polynomials.")


def estimate_rd(y, event_time, covariates=None, bandwidth=None) -> dict[str, int | float]:
    """Estimate the right-minus-left discontinuity with ``rdrobust``.

    ``y`` and ``event_time`` are equal-length one-dimensional real-valued arrays;
    covariates may be a vector or an n-by-k matrix. Labeled pandas inputs must
    also have identical indices. ``event_time`` must be numeric relative time
    in the desired units; fractional times are retained. Covariates must be
    numeric (encode categories explicitly without redundant columns).

    One complete-case mask excludes rows missing any input and rows at time
    zero. Infinite inputs are rejected before masking. At least 20 rows must
    remain to avoid rdrobust's small-sample bandwidth override. Each local side
    needs four observations and a full-rank polynomial (two distinct times for
    p=1, three for q=2). Covariates must have full rank jointly after removing
    the side-specific linear terms; rdrobust errors propagate. Nonfinite results
    or nonpositive conventional/robust standard errors raise ValueError.

    The cutoff is zero, p=1, q=2, kernel=triangular, level=95, and vce=nn.
    ``bandwidth=None`` uses rdrobust's mserd selector. A positive scalar or
    (left, right) pair fixes both estimation and bias bandwidths, h=b.
    No covariates are selected or silently dropped.

    Returns a plain dict. ``n_used`` counts complete nonzero-time rows before
    bandwidth restriction; ``n_excluded = n_input - n_used``. ``estimate``,
    ``se``, ``ci_low``, ``ci_high``, and ``p_value`` are conventional results.
    ``estimate_bias_corrected``, ``se_robust``, ``ci_robust_low``,
    ``ci_robust_high``, and ``p_value_robust`` report robust bias-corrected
    inference. ``bandwidth_left/right`` are h, ``bias_bandwidth_left/right``
    are b, and ``n_effective_left/right`` are rdrobust's effective h counts.
    """
    _check_alignment(y, event_time, covariates)
    outcome = _as_numeric_array(y, "y")
    x = _as_numeric_array(event_time, "event_time")
    if outcome.ndim != 1 or x.ndim != 1 or len(outcome) != len(x):
        raise ValueError("y and event_time must be one-dimensional with equal length.")
    covs = None
    if covariates is not None:
        covs = _as_numeric_array(covariates, "covariates")
        if covs.ndim == 1:
            covs = covs[:, None]
        if covs.ndim != 2 or covs.shape[0] != len(x) or covs.shape[1] == 0:
            raise ValueError("covariates must have one row per observation and at least one column.")
    n_input = len(x)
    keep = ~np.isnan(outcome) & ~np.isnan(x) & (x != 0)
    if covs is not None:
        keep &= ~np.isnan(covs).any(axis=1)
        covs = covs[keep]
    outcome, x = outcome[keep], x[keep]
    if len(x) < 20:
        raise ValueError("At least 20 complete observations with nonzero event time are required.")
    widths = None if bandwidth is None else _parse_bandwidth(bandwidth)
    support = np.array([np.inf, np.inf]) if widths is None else widths
    _check_design(x, support, 1, covs)
    _check_design(x, support, 2)
    _check_outcome(outcome, x, support)
    fit = rdrobust(
        outcome, x, c=0, p=1, q=2, kernel="triangular",
        covs=covs, covs_drop=False, h=widths, b=widths,
        bwselect="mserd", vce="nn", level=95,
    )
    h = fit.bws.loc["h"].to_numpy()
    b = fit.bws.loc["b"].to_numpy()
    _check_design(x, h, 1, covs)
    _check_design(x, b, 2)
    _check_outcome(outcome, x, np.maximum(h, b))
    result = {
        "n_input": n_input,
        "n_used": len(x),
        "n_excluded": n_input - len(x),
        "estimate": float(fit.coef.loc["Conventional"].iloc[0]),
        "se": float(fit.se.loc["Conventional"].iloc[0]),
        "ci_low": float(fit.ci.loc["Conventional"].iloc[0]),
        "ci_high": float(fit.ci.loc["Conventional"].iloc[1]),
        "p_value": float(fit.pv.loc["Conventional"].iloc[0]),
        "estimate_bias_corrected": float(fit.coef.loc["Bias-Corrected"].iloc[0]),
        "se_robust": float(fit.se.loc["Robust"].iloc[0]),
        "ci_robust_low": float(fit.ci.loc["Robust"].iloc[0]),
        "ci_robust_high": float(fit.ci.loc["Robust"].iloc[1]),
        "p_value_robust": float(fit.pv.loc["Robust"].iloc[0]),
        "bandwidth_left": float(h[0]),
        "bandwidth_right": float(h[1]),
        "bias_bandwidth_left": float(b[0]),
        "bias_bandwidth_right": float(b[1]),
        "n_effective_left": int(fit.N_h[0]),
        "n_effective_right": int(fit.N_h[1]),
    }
    if not np.isfinite(list(result.values())).all() or result["se"] <= 0 or result["se_robust"] <= 0:
        raise ValueError("Undefined or degenerate inference: results must be finite and standard errors positive.")
    return result


def covariate_balance(event_time, covariates, *, bandwidth=None, categorical=()) -> pd.DataFrame:
    """Compare supplied covariate distributions before and after time zero.

    By default, use all supplied nonzero times. An optional positive scalar or
    (left, right) ``bandwidth`` restricts a preselected cohort window to
    ``-h_left < time < h_right``. ``event_time`` is a numeric vector with the
    estimator's same alignment and infinity checks. Fractional times are retained;
    zero and missing times are excluded.
    ``covariates`` is a column-name-to-values dict or a DataFrame. ``categorical``
    is a collection of column names, not a bare string. Columns named
    in ``categorical`` use Pearson chi-square tests with Yates correction for
    2-by-2 tables. Other numeric columns use the two-sided KS statistic D and
    the original limiting calibration ``kstwobign.sf(D * sqrt(n_eff))``, where
    ``n_eff = n_left*n_right/(n_left+n_right)``. Missingness is handled per column;
    counts are reported separately for each side. No p-value is returned when a
    side is missing, the pooled variable is constant, or any expected categorical
    cell count is below five. Status is respectively ``missing_side``,
    ``constant``, ``sparse_table``, or ``ok``. These are unadjusted window-wide
    descriptive comparisons, not cutoff tests or a covariate-selection rule.
    Returns a DataFrame with columns ``covariate``, ``type`` (continuous or
    categorical), ``n_left``, ``n_right``, ``statistic``, ``p_value``, ``status``.
    Unavailable statistics and p-values are NaN.
    """
    if not isinstance(covariates, (dict, pd.DataFrame)):
        raise ValueError("covariates must be a dict or pandas DataFrame.")
    labeled_inputs = covariates.values() if isinstance(covariates, dict) else [covariates]
    _check_alignment(event_time, *labeled_inputs)
    frame = pd.DataFrame(covariates)
    x = _as_numeric_array(event_time, "event_time")
    if x.ndim != 1 or len(frame) != len(x) or not frame.columns.is_unique:
        raise ValueError("Covariate columns must be unique and match event_time length.")
    if isinstance(categorical, str):
        raise ValueError("categorical must be a collection of column names, not a bare string.")
    categories = set(categorical)
    if not categories.issubset(frame.columns):
        raise ValueError("categorical contains an unknown covariate column.")
    window = (x < 0) | (x > 0)
    if bandwidth is not None:
        widths = _parse_bandwidth(bandwidth)
        window &= ((x < 0) & (-x < widths[0])) | ((x > 0) & (x < widths[1]))
    rows = []
    for name in frame:
        is_category = name in categories
        values = frame[name].to_numpy() if is_category else _as_numeric_array(frame[name], str(name))
        if is_category and any(isinstance(v, (float, np.floating)) and np.isinf(v) for v in values):
            raise ValueError(f"{name} must not contain infinite values.")
        valid = window & ~pd.isna(values)
        left = values[valid & (x < 0)]
        right = values[valid & (x > 0)]
        row = {
            "covariate": name,
            "type": "categorical" if is_category else "continuous",
            "n_left": len(left),
            "n_right": len(right),
            "statistic": np.nan,
            "p_value": np.nan,
            "status": "ok",
        }
        if not len(left) or not len(right):
            row["status"] = "missing_side"
        elif pd.unique(np.concatenate((left, right))).size == 1:
            row["status"] = "constant"
        elif is_category:
            codes, labels = pd.factorize(np.concatenate((left, right)))
            table = np.array([np.bincount(codes[:len(left)], minlength=len(labels)),
                              np.bincount(codes[len(left):], minlength=len(labels))])
            statistic, p_value, _, expected = chi2_contingency(table, correction=True)
            if (expected < 5).any():
                row["status"] = "sparse_table"
            else:
                row.update(statistic=float(statistic), p_value=float(p_value))
        else:
            statistic = float(ks_2samp(left, right, method="asymp").statistic)
            n_eff = len(left) * len(right) / (len(left) + len(right))
            row.update(statistic=statistic, p_value=float(kstwobign.sf(statistic * np.sqrt(n_eff))))
        rows.append(row)
    return pd.DataFrame(rows, columns=["covariate", "type", "n_left", "n_right",
                                       "statistic", "p_value", "status"])
