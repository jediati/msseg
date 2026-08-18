#include "mscoupon/gmm.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <random>
#include <stdexcept>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace mscoupon {
namespace {

constexpr double kLog2Pi = 1.8378770664093454835606594728112;

// Neumaier compensated summation. NumPy sums pairwise; plain sequential doubles
// drift measurably once a slice runs to 10^7+ samples, and the M-step depends on
// second moments, so the accumulators are compensated throughout.
struct KSum {
  double sum = 0.0;
  double comp = 0.0;

  void add(double v) {
    const double t = sum + v;
    if (std::abs(sum) >= std::abs(v)) {
      comp += (sum - t) + v;
    } else {
      comp += (v - t) + sum;
    }
    sum = t;
  }

  double get() const { return sum + comp; }
};

struct Params {
  std::vector<double> weight;
  std::vector<double> mean;
  std::vector<double> var;
};

// k-means++ seeding: the first center is uniform, each subsequent one is drawn
// with probability proportional to its squared distance from the nearest chosen
// center, taking the best of `2 + log(k)` candidate draws (sklearn's shape).
std::vector<double> kmeans_pp(const std::vector<double>& x, int k, std::mt19937_64& rng) {
  const std::size_t n = x.size();
  std::vector<double> centers;
  centers.reserve(static_cast<std::size_t>(k));

  std::uniform_int_distribution<std::size_t> pick_any(0, n - 1);
  centers.push_back(x[pick_any(rng)]);

  std::vector<double> closest(n);
  for (std::size_t i = 0; i < n; ++i) {
    const double d = x[i] - centers[0];
    closest[i] = d * d;
  }

  const int trials = 2 + static_cast<int>(std::log(static_cast<double>(k)));
  for (int c = 1; c < k; ++c) {
    KSum total;
    for (std::size_t i = 0; i < n; ++i) total.add(closest[i]);
    const double mass = total.get();

    std::size_t chosen = pick_any(rng);
    if (mass > 0.0) {
      double best_potential = std::numeric_limits<double>::infinity();
      std::uniform_real_distribution<double> draw(0.0, mass);
      for (int t = 0; t < trials; ++t) {
        double target = draw(rng);
        std::size_t idx = n - 1;
        for (std::size_t i = 0; i < n; ++i) {
          target -= closest[i];
          if (target <= 0.0) { idx = i; break; }
        }
        KSum potential;
        for (std::size_t i = 0; i < n; ++i) {
          const double d = x[i] - x[idx];
          potential.add(std::min(closest[i], d * d));
        }
        if (potential.get() < best_potential) {
          best_potential = potential.get();
          chosen = idx;
        }
      }
    }

    centers.push_back(x[chosen]);
    for (std::size_t i = 0; i < n; ++i) {
      const double d = x[i] - centers.back();
      closest[i] = std::min(closest[i], d * d);
    }
  }
  return centers;
}

// Lloyd's algorithm; returns the hard assignment and updates `centers` in place.
std::vector<int> lloyd(const std::vector<double>& x, std::vector<double>& centers, int max_iter) {
  const int k = static_cast<int>(centers.size());
  std::vector<int> labels(x.size(), 0);

  for (int it = 0; it < max_iter; ++it) {
    bool changed = false;
    for (std::size_t i = 0; i < x.size(); ++i) {
      int best = 0;
      double best_d = std::numeric_limits<double>::infinity();
      for (int c = 0; c < k; ++c) {
        const double d = std::abs(x[i] - centers[static_cast<std::size_t>(c)]);
        if (d < best_d) { best_d = d; best = c; }
      }
      if (labels[i] != best) { labels[i] = best; changed = true; }
    }

    std::vector<KSum> sums(static_cast<std::size_t>(k));
    std::vector<std::int64_t> counts(static_cast<std::size_t>(k), 0);
    for (std::size_t i = 0; i < x.size(); ++i) {
      const auto c = static_cast<std::size_t>(labels[i]);
      sums[c].add(x[i]);
      counts[c] += 1;
    }
    for (std::size_t c = 0; c < static_cast<std::size_t>(k); ++c) {
      if (counts[c] > 0) centers[c] = sums[c].get() / static_cast<double>(counts[c]);
    }

    if (!changed) break;
  }
  return labels;
}

// One M-step off a hard assignment, which is how sklearn turns the k-means
// labels into the initial weights / means / variances.
Params params_from_labels(const std::vector<double>& x, const std::vector<int>& labels,
                          const std::vector<double>& centers, int k, double reg_covar,
                          double var_floor, double total_var) {
  const double eps10 = 10.0 * std::numeric_limits<double>::epsilon();
  const auto n = static_cast<double>(x.size());

  std::vector<KSum> nk(static_cast<std::size_t>(k)), s1(static_cast<std::size_t>(k)),
      s2(static_cast<std::size_t>(k));
  for (std::size_t i = 0; i < x.size(); ++i) {
    const auto c = static_cast<std::size_t>(labels[i]);
    nk[c].add(1.0);
    s1[c].add(x[i]);
    s2[c].add(x[i] * x[i]);
  }

  Params p;
  p.weight.resize(static_cast<std::size_t>(k));
  p.mean.resize(static_cast<std::size_t>(k));
  p.var.resize(static_cast<std::size_t>(k));
  for (std::size_t c = 0; c < static_cast<std::size_t>(k); ++c) {
    if (nk[c].get() <= 0.0) {
      // Empty cluster (essentially never happens in 1-D after k-means++, but a
      // dead component would poison the EM loop). Park it on its seed center
      // with the global spread and a single pseudo-sample of mass.
      p.weight[c] = 1.0 / n;
      p.mean[c] = centers[c];
      p.var[c] = std::max(total_var, var_floor);
      continue;
    }
    const double nkc = nk[c].get() + eps10;
    const double mu = s1[c].get() / nkc;
    double var = s2[c].get() / nkc - mu * mu + reg_covar;
    if (!(var > var_floor)) var = var_floor;
    p.weight[c] = nkc / n;
    p.mean[c] = mu;
    p.var[c] = var;
  }
  return p;
}

// One fused E+M sweep. Returns the mean per-sample log-likelihood under `p`
// (what sklearn reports as lower_bound_) and writes the updated parameters into
// `next`. Fusing the two steps avoids materialising the n x k responsibility
// matrix, which would be gigabytes at downsample_factor = 1.
double em_sweep(const std::vector<double>& x, const Params& p, double reg_covar, double var_floor,
                Params& next) {
  const std::size_t n = x.size();
  const auto k = static_cast<int>(p.weight.size());
  const auto ku = static_cast<std::size_t>(k);

  std::vector<double> log_w(ku), log_sd(ku), inv_sd(ku);
  for (std::size_t c = 0; c < ku; ++c) {
    const double sd = std::sqrt(p.var[c]);
    log_w[c] = std::log(std::max(p.weight[c], std::numeric_limits<double>::min()));
    log_sd[c] = std::log(sd);
    inv_sd[c] = 1.0 / sd;
  }

  // Per-sample accumulation. The -0.5*log(2*pi) term is constant across
  // components, so it cancels in the softmax and is added once to the
  // log-likelihood instead of k times per sample.
  const auto accumulate = [&](double xi, std::vector<double>& lp, std::vector<KSum>& nk,
                              std::vector<KSum>& s1, std::vector<KSum>& s2, KSum& ll) {
    double top = -std::numeric_limits<double>::infinity();
    for (std::size_t c = 0; c < ku; ++c) {
      const double z = (xi - p.mean[c]) * inv_sd[c];
      lp[c] = log_w[c] - log_sd[c] - 0.5 * z * z;
      if (lp[c] > top) top = lp[c];
    }
    double denom = 0.0;
    for (std::size_t c = 0; c < ku; ++c) {
      lp[c] = std::exp(lp[c] - top);
      denom += lp[c];
    }
    ll.add(top + std::log(denom) - 0.5 * kLog2Pi);
    const double inv_denom = 1.0 / denom;
    for (std::size_t c = 0; c < ku; ++c) {
      const double r = lp[c] * inv_denom;
      nk[c].add(r);
      s1[c].add(r * xi);
      s2[c].add(r * xi * xi);
    }
  };

  std::vector<KSum> nk(ku), s1(ku), s2(ku);
  KSum ll;

#ifdef _OPENMP
#pragma omp parallel
  {
    std::vector<KSum> tnk(ku), ts1(ku), ts2(ku);
    std::vector<double> lp(ku);
    KSum tll;
#pragma omp for schedule(static)
    for (std::ptrdiff_t i = 0; i < static_cast<std::ptrdiff_t>(n); ++i) {
      accumulate(x[static_cast<std::size_t>(i)], lp, tnk, ts1, ts2, tll);
    }
#pragma omp critical
    {
      ll.add(tll.get());
      for (std::size_t c = 0; c < ku; ++c) {
        nk[c].add(tnk[c].get());
        s1[c].add(ts1[c].get());
        s2[c].add(ts2[c].get());
      }
    }
  }
#else
  {
    std::vector<double> lp(ku);
    for (std::size_t i = 0; i < n; ++i) accumulate(x[i], lp, nk, s1, s2, ll);
  }
#endif

  const double eps10 = 10.0 * std::numeric_limits<double>::epsilon();
  next.weight.resize(ku);
  next.mean.resize(ku);
  next.var.resize(ku);
  for (std::size_t c = 0; c < ku; ++c) {
    const double nkc = nk[c].get() + eps10;
    const double mu = s1[c].get() / nkc;
    double var = s2[c].get() / nkc - mu * mu + reg_covar;
    if (!(var > var_floor)) var = var_floor;
    next.weight[c] = nkc / static_cast<double>(n);
    next.mean[c] = mu;
    next.var[c] = var;
  }

  return ll.get() / static_cast<double>(n);
}

// Hard assignment: argmax over the weighted component log-densities. Identical
// ordering to argmax over the responsibilities, without forming them.
int argmax_component(double xi, const Params& p) {
  int best = 0;
  double best_lp = -std::numeric_limits<double>::infinity();
  for (std::size_t c = 0; c < p.weight.size(); ++c) {
    const double sd = std::sqrt(p.var[c]);
    const double z = (xi - p.mean[c]) / sd;
    const double lp = std::log(std::max(p.weight[c], std::numeric_limits<double>::min())) -
                      std::log(sd) - 0.5 * z * z;
    if (lp > best_lp) { best_lp = lp; best = static_cast<int>(c); }
  }
  return best;
}

// Histogram mode with parabolic interpolation around the peak bin, ported from
// estimate_mode() in measure_gmm.py (including its endpoint, zero-denominator
// and offset-clamp bail-outs). Sub-bin localisation, not a Gaussian mode.
double estimate_mode(const std::vector<double>& v, int n_bins) {
  if (v.size() < 10) return std::numeric_limits<double>::quiet_NaN();
  const auto mm = std::minmax_element(v.begin(), v.end());
  const double lo = *mm.first;
  const double hi = *mm.second;
  if (lo == hi) return lo;

  const auto bins = static_cast<std::size_t>(std::max(2, n_bins));
  std::vector<std::int64_t> counts(bins, 0);
  const double width = (hi - lo) / static_cast<double>(bins);
  for (const double x : v) {
    auto b = static_cast<std::size_t>((x - lo) / width);
    if (b >= bins) b = bins - 1;  // the max value lands in the last bin
    counts[b] += 1;
  }

  std::size_t peak = 0;
  for (std::size_t b = 1; b < bins; ++b) {
    if (counts[b] > counts[peak]) peak = b;
  }
  const auto center_of = [&](std::size_t b) {
    return lo + (static_cast<double>(b) + 0.5) * width;
  };
  if (peak == 0 || peak + 1 == bins) return center_of(peak);

  const double offset = parabolic_offset(static_cast<double>(counts[peak - 1]),
                                         static_cast<double>(counts[peak]),
                                         static_cast<double>(counts[peak + 1]));
  return center_of(peak) + offset * width;
}

}  // namespace

GmmResult fit_gmm_1d(std::vector<double>& values, const GmmOptions& opts) {
  const int k = opts.n_components;
  if (k < 1) throw std::runtime_error("gmm: n_components must be >= 1");
  if (opts.trim_percent < 0.0 || opts.trim_percent >= 50.0)
    throw std::runtime_error("gmm: trim_percent must satisfy 0 <= trim_percent < 50");

  GmmResult result;
  result.n_valid = static_cast<std::int64_t>(values.size());
  if (result.n_valid < 10)
    throw std::runtime_error("gmm: not enough valid nonzero pixels (need at least 10)");

  std::mt19937_64 rng(opts.seed);

  // --- Random 1/N subsample, without replacement (partial Fisher-Yates). ---
  subsample_inplace(values, opts.downsample_factor, rng);
  result.n_sampled = static_cast<std::int64_t>(values.size());

  // --- Symmetric percentile trim. ---
  if (opts.trim_percent > 0.0) {
    result.trim_lo = percentile_linear(values, opts.trim_percent);
    result.trim_hi = percentile_linear(values, 100.0 - opts.trim_percent);
    const double lo = result.trim_lo;
    const double hi = result.trim_hi;
    std::erase_if(values, [lo, hi](double v) { return v < lo || v > hi; });
  } else {
    const auto mm = std::minmax_element(values.begin(), values.end());
    result.trim_lo = *mm.first;
    result.trim_hi = *mm.second;
  }
  result.n_fit = static_cast<std::int64_t>(values.size());
  if (result.n_fit < 10) throw std::runtime_error("gmm: not enough pixels after trimming");

  // --- Recentre. Variance is shift-invariant, and reconstructed intensities can
  // sit near 1e-4 with reg_covar=1e-12, where the one-pass E[x^2] - mu^2 update
  // would lose most of its significant digits. Shifting removes that. ---
  const auto n = static_cast<double>(values.size());
  KSum mean_acc;
  for (const double v : values) mean_acc.add(v);
  const double center = mean_acc.get() / n;

  KSum var_acc;
  for (const double v : values) {
    const double d = v - center;
    var_acc.add(d * d);
  }
  const double total_var = var_acc.get() / n;
  if (!(total_var > 0.0))
    throw std::runtime_error("gmm: all fitted values are identical; cannot fit a mixture");

  for (double& v : values) v -= center;
  const double var_floor = std::max(opts.reg_covar, 1e-12 * total_var);

  // Quantile init means, computed once (percentile_linear reorders `values`,
  // which must happen before anything depends on its layout).
  std::vector<double> quantile_means;
  if (opts.init == GmmInit::Quantile) {
    quantile_means.resize(static_cast<std::size_t>(k));
    for (int c = 0; c < k; ++c) {
      quantile_means[static_cast<std::size_t>(c)] =
          percentile_linear(values, 100.0 * (2.0 * c + 1.0) / (2.0 * k));
    }
  }

  // --- n_init restarts; the highest log-likelihood wins. ---
  Params best;
  double best_ll = -std::numeric_limits<double>::infinity();
  int best_iter = 0;
  bool best_converged = false;

  const int n_init = std::max(1, opts.n_init);
  const int max_iter = std::max(1, opts.max_iter);
  for (int run = 0; run < n_init; ++run) {
    std::vector<double> centers = kmeans_pp(values, k, rng);
    const std::vector<int> labels = lloyd(values, centers, 300);
    Params p =
        params_from_labels(values, labels, centers, k, opts.reg_covar, var_floor, total_var);
    if (opts.init == GmmInit::Quantile) p.mean = quantile_means;

    double ll = -std::numeric_limits<double>::infinity();
    int iters = 0;
    bool converged = false;
    Params next;
    for (int it = 1; it <= max_iter; ++it) {
      const double prev = ll;
      ll = em_sweep(values, p, opts.reg_covar, var_floor, next);
      p = next;
      iters = it;
      if (std::isfinite(prev) && std::abs(ll - prev) < opts.tol) {
        converged = true;
        break;
      }
    }

    if (ll > best_ll) {
      best_ll = ll;
      best = p;
      best_iter = iters;
      best_converged = converged;
    }
  }

  // --- Assemble, undoing the recentring on every location statistic. ---
  std::vector<GmmComponent> components(static_cast<std::size_t>(k));
  for (std::size_t c = 0; c < static_cast<std::size_t>(k); ++c) {
    components[c].mean = best.mean[c] + center;
    components[c].sigma = std::sqrt(best.var[c]);
    components[c].weight = best.weight[c];
  }

  if (opts.compute_hard_stats) {
    std::vector<std::vector<double>> members(static_cast<std::size_t>(k));
    for (const double v : values) {
      members[static_cast<std::size_t>(argmax_component(v, best))].push_back(v + center);
    }
    for (std::size_t c = 0; c < static_cast<std::size_t>(k); ++c) {
      auto& m = members[c];
      if (m.empty())
        throw std::runtime_error("gmm: a component has no hard-assigned pixels");

      KSum sum;
      for (const double v : m) sum.add(v);
      components[c].n_hard = static_cast<std::int64_t>(m.size());
      components[c].hard_mean = sum.get() / static_cast<double>(m.size());

      const std::size_t mid = m.size() / 2;
      std::nth_element(m.begin(), m.begin() + static_cast<std::ptrdiff_t>(mid), m.end());
      if (m.size() % 2 == 1) {
        components[c].median = m[mid];
      } else {
        const double upper = m[mid];
        const double lower = *std::max_element(m.begin(), m.begin() + static_cast<std::ptrdiff_t>(mid));
        components[c].median = 0.5 * (lower + upper);
      }
      components[c].mode = estimate_mode(m, opts.mode_bins);
    }
  }

  std::sort(components.begin(), components.end(),
            [](const GmmComponent& a, const GmmComponent& b) { return a.mean < b.mean; });

  result.components = std::move(components);
  result.log_likelihood = best_ll;
  result.n_iter = best_iter;
  result.converged = best_converged;
  return result;
}

GmmResult fit_gmm(const Image2D& image, const GmmOptions& opts) {
  return fit_gmm(image.pixels.data(), image.pixels.size(), opts);
}

GmmOptions gmm_options_two_gaussian() {
  GmmOptions o;
  o.n_components = 2;
  o.init = GmmInit::KMeans;
  o.n_init = 3;
  o.max_iter = 300;
  o.tol = 1e-6;
  o.reg_covar = 1e-6;
  o.trim_percent = 0.0;
  o.compute_hard_stats = false;
  return o;
}

GmmOptions gmm_options_measure() {
  GmmOptions o;
  o.n_components = 2;
  o.init = GmmInit::Quantile;
  o.n_init = 3;
  o.max_iter = 500;
  o.tol = 1e-8;
  o.reg_covar = 1e-12;  // intensities can sit near 1e-4
  o.compute_hard_stats = true;
  o.mode_bins = 512;
  return o;
}

}  // namespace mscoupon
