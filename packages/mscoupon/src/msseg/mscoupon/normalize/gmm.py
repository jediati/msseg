"""Two-component Gaussian mixture measurement.

Wraps the compiled ``fit_gmm`` (``lib/mscoupon/gmm.cpp``), which is a
dependency-free port of what ``calculate_2_gaussian_mixture.py`` and
``measure_gmm.py`` used to do with scikit-learn. sklearn is now only a *test*
dependency, used by ``tests/gmm_parity.py`` to hold the port honest.

Components come back sorted by increasing mean, so component 1 is the
low-intensity population (air/void) and component 2 the high one (metal/solid).
"""
from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np

from . import _engine
from .pixels import (_UNSET, estimate_mode, get_valid_pixels, random_downsample,
                     resolve_omit_value, trim_pixels)

# The two presets, matching the scripts they were ported from.
PRESET_TWO_GAUSSIAN = "two_gaussian"   # calculate_2_gaussian_mixture.py
PRESET_MEASURE = "measure"             # measure_gmm.py

# Parameter names carried in the per-image CSVs.
PARAMETERS = ["mu_1", "sigma_1", "mu_2", "sigma_2"]


def fit_two_gaussians(
    image,
    downsample_factor: int = 1,
    omit_value=_UNSET,
    trim_percent: float = 0.0,
    preset: str = PRESET_MEASURE,
    n_components: int = 2,
    n_init: int = 3,
    seed: int = 0,
    compute_hard_stats: bool = True,
    mode_bins: int = 512,
    omit_zeros: Optional[bool] = None,
    **overrides: Any,
) -> Dict[str, Any]:
    """Fit a mixture and return a flat dict of landmarks and diagnostics.

    Keys: ``mu_N``, ``sigma_N``, ``weight_N`` and (with `compute_hard_stats`)
    ``hard_mean_N``, ``median_N``, ``mode_N``, ``n_class_N`` for each 1-based
    component, plus ``n_valid_pixels``, ``n_sampled_pixels``, ``n_fit_pixels``,
    ``trim_percent``, ``trim_lo``, ``trim_hi``, ``converged`` and ``n_iter``.
    """
    params = {
        "preset": preset,
        "n_components": n_components,
        "downsample_factor": int(downsample_factor),
        "omit_value": resolve_omit_value(omit_value, omit_zeros),
        "trim_percent": float(trim_percent),
        "n_init": int(n_init),
        "seed": int(seed),
        "compute_hard_stats": bool(compute_hard_stats),
        "mode_bins": int(mode_bins),
    }
    params.update(overrides)

    raw = _engine.call("fit_gmm", np.asarray(image), {"gmm": params})
    if raw is None:
        raw = _fit_with_numpy(image, params)
    return _flatten(raw, float(trim_percent))


def _flatten(raw: Dict[str, Any], trim_percent: float) -> Dict[str, Any]:
    """Turn the engine's component list into the flat mu_1/mu_2/... shape."""
    out: Dict[str, Any] = {}
    nan = float("nan")
    for i, comp in enumerate(raw["components"], start=1):
        out[f"mu_{i}"] = float(comp["mean"])
        out[f"sigma_{i}"] = float(comp["sigma"])
        out[f"weight_{i}"] = float(comp["weight"])
        # The hard-assignment stats are only computed when asked for; report
        # them as NaN rather than raising when a caller opted out.
        out[f"hard_mean_{i}"] = float(comp.get("hard_mean", nan))
        out[f"median_{i}"] = float(comp.get("median", nan))
        out[f"mode_{i}"] = float(comp.get("mode", nan))
        out[f"n_class_{i}"] = int(comp.get("n_hard", 0))

    out["n_valid_pixels"] = int(raw["n_valid"])
    out["n_sampled_pixels"] = int(raw["n_sampled"])
    out["n_fit_pixels"] = int(raw["n_fit"])
    out["trim_percent"] = trim_percent
    out["trim_lo"] = float(raw["trim_lo"])
    out["trim_hi"] = float(raw["trim_hi"])
    out["log_likelihood"] = float(raw["log_likelihood"])
    out["converged"] = bool(raw["converged"])
    out["n_iter"] = int(raw["n_iter"])
    return out


def _fit_with_numpy(image, params: Dict[str, Any]) -> Dict[str, Any]:
    """Fallback EM used only when the compiled extension is unavailable.

    Plain 1-D EM with k-means-free quantile seeding -- enough to keep the
    analysis scripts working in a checkout with no build, not a replacement for
    the C++ path (which is what the parity test covers).
    """
    k = int(params.get("n_components", 2))
    x = get_valid_pixels(image, omit_value=params.get("omit_value", 0.0))
    n_valid = x.size
    if n_valid < 10:
        raise ValueError(f"Not enough valid nonzero pixels: {n_valid}")

    rng = np.random.default_rng(params.get("seed", 0))
    x = random_downsample(x, params.get("downsample_factor", 1), rng)
    n_sampled = x.size

    x, trim_lo, trim_hi = trim_pixels(x, params.get("trim_percent", 0.0))
    if x.size < 10:
        raise ValueError(f"Not enough pixels after trimming: {x.size}")

    quantiles = np.linspace(100.0 / (k + 1), 100.0 * k / (k + 1), k)
    mu = np.percentile(x, quantiles).astype(np.float64)
    var = np.full(k, max(float(np.var(x)) / k, 1e-12))
    weight = np.full(k, 1.0 / k)
    reg = float(params.get("reg_covar", 1e-12))

    prev = -np.inf
    n_iter = 0
    converged = False
    for n_iter in range(1, int(params.get("max_iter", 500)) + 1):
        # E-step in log space, so tiny variances do not underflow to zero.
        logp = (np.log(np.maximum(weight, 1e-300))[None, :]
                - 0.5 * np.log(2.0 * np.pi * var)[None, :]
                - 0.5 * (x[:, None] - mu[None, :]) ** 2 / var[None, :])
        peak = logp.max(axis=1, keepdims=True)
        resp = np.exp(logp - peak)
        total = resp.sum(axis=1, keepdims=True)
        resp /= total
        ll = float(np.mean(np.log(total[:, 0]) + peak[:, 0]))

        nk = resp.sum(axis=0) + 1e-300
        weight = nk / x.size
        mu = (resp * x[:, None]).sum(axis=0) / nk
        var = (resp * (x[:, None] - mu[None, :]) ** 2).sum(axis=0) / nk + reg

        if abs(ll - prev) < float(params.get("tol", 1e-8)):
            converged = True
            prev = ll
            break
        prev = ll

    order = np.argsort(mu)
    labels = np.argmax(resp, axis=1)
    components = []
    for c in order:
        member = x[labels == c]
        components.append({
            "mean": float(mu[c]),
            "sigma": float(np.sqrt(var[c])),
            "weight": float(weight[c]),
            "n_hard": int(member.size),
            "hard_mean": float(np.mean(member)) if member.size else float("nan"),
            "median": float(np.median(member)) if member.size else float("nan"),
            "mode": estimate_mode(member, params.get("mode_bins", 512))
            if member.size else float("nan"),
        })

    return {
        "components": components,
        "n_valid": n_valid,
        "n_sampled": n_sampled,
        "n_fit": int(x.size),
        "trim_lo": trim_lo,
        "trim_hi": trim_hi,
        "log_likelihood": prev,
        "n_iter": n_iter,
        "converged": converged,
    }


def void_probability_image(image, mu_void, sigma_void, mu_solid, sigma_solid,
                           omit_value=_UNSET, omit_zeros: Optional[bool] = None) -> np.ndarray:
    """Per-pixel posterior probability of belonging to the void population.

    Equal priors, evaluated as a logistic of the log-likelihood ratio so it stays
    stable when the two densities underflow.

    Both components are collapsed onto a common width,
    ``sqrt((sigma_void^2 + sigma_solid^2) / 2)``, which is what
    ``calculate_void_probability.py`` has always done. That is not the exact
    two-Gaussian posterior, and the difference matters: with per-component sigmas
    the exponent is quadratic in intensity, so once the wider component's tail
    takes over, the *brightest* pixels start coming back as probable void. The
    common width makes the ratio linear in intensity, so the map decreases
    monotonically from void to solid -- the behaviour a void map is expected to
    have. Keep it unless you specifically want the exact posterior.

    Masked-out pixels are written as 0.0. The original promised that but only
    delivered it for integer input -- its float branch masked ``isfinite``
    alone, so float zeros silently got a real probability. Here `omit_value`
    governs both dtypes.
    """
    x = np.asarray(image)
    values = x.astype(np.float64, copy=False)

    omit = resolve_omit_value(omit_value, omit_zeros)
    keep = np.ones(x.shape, dtype=bool)
    if np.issubdtype(x.dtype, np.floating):
        keep &= np.isfinite(values)
    if omit is not None:
        keep &= values != float(omit)

    out = np.zeros(x.shape, dtype=np.float32)
    v = values[keep]
    if v.size == 0:
        return out

    var = 0.5 * (float(sigma_void) ** 2 + float(sigma_solid) ** 2)
    var = max(var, 1e-300)
    # log p(void) - log p(solid) at a shared width; the equal priors cancel.
    delta = (float(mu_void) - float(mu_solid)) * (v - 0.5 * (float(mu_void) + float(mu_solid))) / var
    out[keep] = (1.0 / (1.0 + np.exp(-delta))).astype(np.float32)
    return out


def load_reference_csv(path):
    """Load a GMM results CSV, dropping ERROR/NaN/incomplete rows."""
    import pandas as pd

    df = pd.read_csv(path)
    required = ["filename", *PARAMETERS]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Reference CSV is missing columns: {missing}")

    for col in PARAMETERS:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    valid = np.ones(len(df), dtype=bool)
    for col in PARAMETERS:
        valid &= np.isfinite(df[col])
    return df.loc[valid].copy()
