"""Parity check: the C++ 1-D Gaussian mixture vs scikit-learn's GaussianMixture.

The C++ port in ``packages/mscoupon/lib/mscoupon/gmm.{hpp,cpp}`` reproduces the
two analysis scripts under ``src/msseg/mscoupon/``:

  * ``calculate_2_gaussian_mixture.py``  -> preset "two_gaussian"
  * ``measure_gmm.py``                   -> preset "measure"

This script fits the same pixels both ways and compares mu / sigma / weight.

WHAT PARITY MEANS HERE
----------------------
Agreement is *algorithmic*, not bit-for-bit. Two things are deliberately not
reproduced, so the comparison is arranged to take them out of the picture:

  * Random subsampling. ``rng.choice(replace=False)`` uses NumPy's PCG64; the
    C++ uses a seeded partial Fisher-Yates. The two draw different subsets, so
    every case below runs at ``downsample_factor = 1`` and both sides see the
    identical pixel set. (``test_gmm_downsample`` in the C++ suite covers the
    subsampler itself.)
  * k-means++ seeding. sklearn seeds from an MT19937 ``RandomState``, the C++
    from a ``mt19937_64``. Different starts, but on separated populations every
    restart lands in the same EM optimum, which is exactly what this checks.

What must match is everything else: the zero / non-finite mask, the percentile
trim rule, the EM update equations, ``reg_covar``, the convergence test, and the
component ordering.

Run standalone (needs numpy + scikit-learn, and the built extension):

    python packages/mscoupon/tests/gmm_parity.py
    python packages/mscoupon/tests/gmm_parity.py --build-dir <cmake-build-dir>
    python packages/mscoupon/tests/gmm_parity.py --tiff-dir F:\\data\\scan --limit 3

or under pytest (``test_gmm_parity`` is collected automatically; it skips if
scikit-learn or the extension is unavailable).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

# Relative tolerance for mu / sigma / weight. Both sides run the same EM to the
# same optimum in double precision; residual differences come from summation
# order and the (different) k-means start, and land far below this.
RTOL = 1e-4


# ---------------------------------------------------------------------------
# Locating the extension
# ---------------------------------------------------------------------------

def preset_build_dirs(repo_root: Path):
    """Resolve every configurePreset binaryDir in CMakePresets.json."""
    presets = repo_root / "CMakePresets.json"
    if not presets.is_file():
        return []
    try:
        spec = json.loads(presets.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return []
    out = []
    for preset in spec.get("configurePresets", []):
        binary_dir = preset.get("binaryDir")
        if binary_dir:
            out.append(Path(binary_dir.replace("${sourceDir}", str(repo_root))).resolve())
    return out


def load_extension(build_dir: str | None = None):
    """Return the module exposing ``fit_gmm``, or None if it cannot be found.

    Prefers an installed ``msseg.mscoupon``. Falls back to importing the raw
    ``mscoupon_py`` extension straight out of a CMake build tree, so the check
    is runnable right after ``cmake --build`` without a pip install.
    """
    try:
        from msseg import mscoupon

        if getattr(mscoupon, "_HAVE_EXTENSION", False):
            return mscoupon
    except ImportError:
        pass

    roots = []
    if build_dir:
        roots.append(Path(build_dir))
    if os.environ.get("MSSEG_BUILD_DIR"):
        roots.append(Path(os.environ["MSSEG_BUILD_DIR"]))
    repo_root = Path(__file__).resolve().parents[3]
    roots.extend(preset_build_dirs(repo_root))

    for root in roots:
        # The usual location first; only sweep the tree (which contains the
        # FetchContent deps) if the module is not where AddInstance puts it.
        here = root / "packages" / "mscoupon"
        found = list(here.glob("mscoupon_py*")) if here.is_dir() else []
        if not found and root.is_dir():
            found = sorted(root.rglob("mscoupon_py*"))
        for pyd in found:
            if pyd.suffix.lower() not in (".pyd", ".so"):
                continue
            sys.path.insert(0, str(pyd.parent))
            try:
                import mscoupon_py

                return mscoupon_py
            except ImportError:
                sys.path.pop(0)
    return None


# ---------------------------------------------------------------------------
# The sklearn reference, transcribed from the two scripts
# ---------------------------------------------------------------------------

def valid_pixels(image: np.ndarray) -> np.ndarray:
    """Mask exactly as both scripts do: drop zeros, and NaN/Inf for float data."""
    x = np.asarray(image).ravel()
    if np.issubdtype(x.dtype, np.floating):
        keep = np.isfinite(x) & (x != 0)
    else:
        keep = x != 0
    return np.asarray(x[keep], dtype=np.float64)


def sklearn_fit(x: np.ndarray, *, n_components=2, n_init=3, max_iter=300, tol=1e-6,
                reg_covar=1e-6, means_init=None):
    """Fit with sklearn and return components sorted by increasing mean."""
    from sklearn.mixture import GaussianMixture

    gmm = GaussianMixture(
        n_components=n_components,
        covariance_type="full",
        n_init=n_init,
        max_iter=max_iter,
        tol=tol,
        reg_covar=reg_covar,
        means_init=means_init,
        random_state=0,
    )
    gmm.fit(x.reshape(-1, 1))

    means = gmm.means_.ravel()
    sigmas = np.sqrt(gmm.covariances_[:, 0, 0])
    weights = gmm.weights_.ravel()
    order = np.argsort(means)
    return (
        [
            {"mean": float(means[i]), "sigma": float(sigmas[i]), "weight": float(weights[i])}
            for i in order
        ],
        float(gmm.lower_bound_),
        bool(gmm.converged_),
    )


# ---------------------------------------------------------------------------
# Comparison
# ---------------------------------------------------------------------------

class Mismatch(AssertionError):
    pass


def compare(label: str, cpp: dict, ref_components, ref_ll, *, rtol=RTOL) -> float:
    """Print a side-by-side table; raise Mismatch if anything exceeds rtol."""
    got = cpp["components"]
    if len(got) != len(ref_components):
        raise Mismatch(f"{label}: component count {len(got)} != {len(ref_components)}")

    print(f"\n{label}")
    print(f"  {'param':<12}{'c++':>18}{'sklearn':>18}{'rel diff':>12}")
    worst = 0.0
    for i, (a, b) in enumerate(zip(got, ref_components), start=1):
        for field in ("mean", "sigma", "weight"):
            av, bv = a[field], b[field]
            scale = max(abs(bv), 1e-30)
            rel = abs(av - bv) / scale
            worst = max(worst, rel)
            flag = "" if rel <= rtol else "   <-- MISMATCH"
            print(f"  {field + '_' + str(i):<12}{av:>18.10g}{bv:>18.10g}{rel:>12.2e}{flag}")

    ll_rel = abs(cpp["log_likelihood"] - ref_ll) / max(abs(ref_ll), 1e-30)
    print(f"  {'log_lik':<12}{cpp['log_likelihood']:>18.10g}{ref_ll:>18.10g}{ll_rel:>12.2e}")
    print(f"  n_valid={cpp['n_valid']} n_fit={cpp['n_fit']} "
          f"n_iter={cpp['n_iter']} converged={cpp['converged']}")

    if worst > rtol:
        raise Mismatch(f"{label}: worst relative difference {worst:.3e} exceeds rtol {rtol:.1e}")
    return worst


# ---------------------------------------------------------------------------
# Cases
# ---------------------------------------------------------------------------

def check_recovery(label: str, cpp: dict, spec, rtol=0.05, wtol=0.02) -> None:
    """Also assert the fit found the planted populations.

    Agreeing with sklearn is necessary but not sufficient -- both could be
    converging to the same wrong answer. Cases with a known ground truth check
    it, so a silently-degenerate fit cannot pass unnoticed.
    """
    got = cpp["components"]
    want = sorted(spec, key=lambda s: s[1])
    for i, (c, (w, mu, sigma)) in enumerate(zip(got, want), start=1):
        assert abs(c["mean"] - mu) <= rtol * abs(mu), (
            f"{label}: component {i} mean {c['mean']:.6g} != planted {mu:.6g}"
        )
        assert abs(c["sigma"] - sigma) <= rtol * abs(sigma), (
            f"{label}: component {i} sigma {c['sigma']:.6g} != planted {sigma:.6g}"
        )
        assert abs(c["weight"] - w) <= wtol, (
            f"{label}: component {i} weight {c['weight']:.6g} != planted {w:.6g}"
        )


def make_mixture(n, spec, seed):
    """Draw n samples from [(weight, mu, sigma), ...] as float32."""
    rng = np.random.default_rng(seed)
    weights = np.array([s[0] for s in spec], dtype=np.float64)
    which = rng.choice(len(spec), size=n, p=weights / weights.sum())
    mus = np.array([s[1] for s in spec])
    sigmas = np.array([s[2] for s in spec])
    return (rng.normal(mus[which], sigmas[which])).astype(np.float32)


def case_two_gaussian(ext) -> float:
    """calculate_2_gaussian_mixture.py: plain 2-component fit, k-means init."""
    spec = [(0.3, 10.0, 1.0), (0.7, 20.0, 2.0)]
    px = make_mixture(200_000, spec, seed=1)
    cpp = ext.fit_gmm(px, json.dumps({"gmm": {"preset": "two_gaussian"}}))
    ref, ll, converged = sklearn_fit(valid_pixels(px))
    assert converged, "sklearn did not converge"
    worst = compare("two_gaussian preset, clean float32", cpp, ref, ll)
    check_recovery("two_gaussian case", cpp, spec)
    return worst


SMALL_SPEC = [(0.4, 0.0012, 0.0002), (0.6, 0.0035, 0.0004)]


def dirty_slice(seed):
    """Small-magnitude intensities buried in no-data zeros, plus NaN/Inf."""
    clean = make_mixture(120_000, SMALL_SPEC, seed=seed)
    dirty = np.zeros(clean.size * 2, dtype=np.float32)
    dirty[1::2] = clean          # odd lanes carry data, even lanes are background
    dirty[100] = np.nan          # even indices, so no real sample is clobbered
    dirty[300] = np.inf
    dirty[500] = -np.inf
    return dirty.reshape(-1, 400)


def case_masking(ext) -> float:
    """Mask parity on realistic small intensities, using the measure preset.

    Reconstructed slices sit near 1e-3 with component sigmas near 1e-4, which is
    exactly why measure_gmm.py drops reg_covar to 1e-12 -- see
    case_reg_covar_dominates for what the 1e-6 default does to the same data.
    """
    image = dirty_slice(seed=2)
    params = {"gmm": {"preset": "measure", "compute_hard_stats": False}}
    cpp = ext.fit_gmm(image, json.dumps(params))

    x = valid_pixels(image)
    assert cpp["n_valid"] == x.size, (
        f"mask disagrees: c++ kept {cpp['n_valid']}, numpy kept {x.size}"
    )
    q25, q75 = np.percentile(x, [25, 75])
    ref, ll, _ = sklearn_fit(
        x, n_init=3, max_iter=500, tol=1e-8, reg_covar=1e-12,
        means_init=np.array([[q25], [q75]]),
    )
    worst = compare("zero/NaN/Inf mask, intensities near 1e-3 (reg_covar=1e-12)", cpp, ref, ll)
    check_recovery("mask case", cpp, SMALL_SPEC)
    return worst


def case_reg_covar_dominates(ext) -> float:
    """reg_covar must be applied identically, including where it swamps the data.

    With sigmas near 1e-4, the default reg_covar=1e-6 is ~25x the true component
    variance: the likelihood surface flattens and the fit collapses to two
    near-identical wide components. That is not a defect in either
    implementation -- it is what the regulariser does -- but the two must
    degenerate the *same* way, which is a sharp test that the term enters the
    M-step at the same point.
    """
    image = dirty_slice(seed=2)
    cpp = ext.fit_gmm(image, json.dumps({"gmm": {"preset": "two_gaussian"}}))
    ref, ll, _ = sklearn_fit(valid_pixels(image))
    worst = compare("same data with reg_covar=1e-6 (deliberately degenerate)", cpp, ref, ll)
    means = [c["mean"] for c in cpp["components"]]
    assert abs(means[1] - means[0]) < 1e-4, (
        "expected the components to collapse together under an over-large "
        f"reg_covar, but got means {means}"
    )
    return worst


def case_measure_preset(ext) -> float:
    """measure_gmm.py: percentile trim + quantile means_init + reg_covar=1e-12."""
    spec = [(0.35, 10.0, 1.0), (0.65, 20.0, 2.0)]
    px = make_mixture(150_000, spec, seed=3)
    params = {"gmm": {"preset": "measure", "trim_percent": 0.5}}
    cpp = ext.fit_gmm(px, json.dumps(params))

    x = valid_pixels(px)
    lo, hi = np.percentile(x, [0.5, 99.5])
    assert abs(cpp["trim_lo"] - lo) < 1e-9 and abs(cpp["trim_hi"] - hi) < 1e-9, (
        f"trim cut points disagree: c++ ({cpp['trim_lo']}, {cpp['trim_hi']}) "
        f"vs numpy ({lo}, {hi})"
    )
    x_fit = x[(x >= lo) & (x <= hi)]
    assert cpp["n_fit"] == x_fit.size, (
        f"trim kept {cpp['n_fit']} pixels, numpy kept {x_fit.size}"
    )

    q25, q75 = np.percentile(x_fit, [25, 75])
    ref, ll, _ = sklearn_fit(
        x_fit, n_init=3, max_iter=500, tol=1e-8, reg_covar=1e-12,
        means_init=np.array([[q25], [q75]]),
    )
    worst = compare("measure preset (trim 0.5%, quantile init, reg_covar=1e-12)", cpp, ref, ll)
    # Trimming 0.5% off each tail shrinks the fitted sigmas slightly, so the
    # planted-value check is looser here than elsewhere.
    check_recovery("measure case", cpp, spec, rtol=0.08)
    return worst


def case_three_components(ext) -> float:
    """The port generalises past K=2; check a three-population fit."""
    spec = [(0.25, 5.0, 0.8), (0.35, 12.0, 1.2), (0.4, 22.0, 2.0)]
    px = make_mixture(200_000, spec, seed=4)
    params = {"gmm": {"preset": "two_gaussian", "n_components": 3}}
    cpp = ext.fit_gmm(px, json.dumps(params))
    ref, ll, _ = sklearn_fit(valid_pixels(px), n_components=3)
    worst = compare("three components", cpp, ref, ll)
    check_recovery("three-component case", cpp, spec)
    return worst


def case_int16(ext) -> float:
    """Integer input: no finite test, widened to double, same result."""
    spec = [(0.3, 1000.0, 90.0), (0.7, 2200.0, 150.0)]
    px = make_mixture(120_000, spec, seed=5)
    ints = np.rint(px).astype(np.int16)
    cpp = ext.fit_gmm(ints, json.dumps({"gmm": {"preset": "two_gaussian"}}))
    ref, ll, _ = sklearn_fit(valid_pixels(ints))
    worst = compare("int16 input", cpp, ref, ll)
    check_recovery("int16 case", cpp, spec)
    return worst


def case_real_tiffs(ext, folder: Path, limit: int) -> float:
    """Same comparison on real slices -- the case the tolerances actually matter for."""
    import tifffile

    files = sorted(
        (p for p in folder.iterdir()
         if p.is_file() and p.suffix.lower() in (".tif", ".tiff")),
        key=lambda p: p.name,
    )[:limit]
    if not files:
        raise Mismatch(f"no TIFF files in {folder}")

    worst = 0.0
    for path in files:
        image = tifffile.imread(path)
        cpp = ext.fit_gmm(image, json.dumps({"gmm": {"preset": "two_gaussian"}}))
        x = valid_pixels(image)
        assert cpp["n_valid"] == x.size, f"{path.name}: mask disagrees"
        ref, ll, _ = sklearn_fit(x)
        worst = max(worst, compare(f"real slice {path.name}", cpp, ref, ll))
    return worst


# ---------------------------------------------------------------------------
# Entry points
# ---------------------------------------------------------------------------

def run(build_dir=None, tiff_dir=None, limit=3) -> float:
    ext = load_extension(build_dir)
    assert ext is not None, (
        "the mscoupon extension is not importable. Either `pip install "
        "./packages/mscoupon`, or pass --build-dir / set MSSEG_BUILD_DIR to a "
        "CMake build tree configured with -DMSSEG_BUILD_PYTHON=ON."
    )

    worst = max(
        case_two_gaussian(ext),
        case_masking(ext),
        case_reg_covar_dominates(ext),
        case_measure_preset(ext),
        case_three_components(ext),
        case_int16(ext),
    )
    if tiff_dir:
        worst = max(worst, case_real_tiffs(ext, Path(tiff_dir), limit))

    print(f"\ngmm parity OK -- worst relative difference {worst:.3e} (rtol {RTOL:.1e})")
    return worst


def test_gmm_parity() -> None:
    """pytest entry point; skips when sklearn or the extension is unavailable."""
    import pytest

    pytest.importorskip("sklearn", reason="scikit-learn is required for the parity check")
    if load_extension() is None:
        pytest.skip("the mscoupon extension is not built")
    run()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--build-dir", default=None,
                        help="CMake build tree holding mscoupon_py, if not pip-installed.")
    parser.add_argument("--tiff-dir", default=None,
                        help="Optionally also compare on real TIFF slices in this folder.")
    parser.add_argument("--limit", type=int, default=3,
                        help="How many TIFFs to check with --tiff-dir (default 3).")
    args = parser.parse_args()

    try:
        import sklearn  # noqa: F401
    except ImportError:
        print("SKIP: scikit-learn is not installed (pip install scikit-learn)", file=sys.stderr)
        return 2

    try:
        run(args.build_dir, args.tiff_dir, args.limit)
    except AssertionError as exc:
        print(f"\n[FAIL] {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
