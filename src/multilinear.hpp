// multilinear.hpp
#pragma once

#include <libff/algebra/curves/alt_bn128/alt_bn128_pp.hpp>
// #include <libfqfft/evaluation_domain/get_evaluation_domain.hpp>
#include <stdexcept>
#include <vector>

namespace wrappers {

using FieldT = libff::Fr<libff::alt_bn128_pp>;

class NaiveMultilinearPolynomial {
public:
  size_t n;                   // arity
  std::vector<FieldT> coeffs; // monomial coefficients (|coeffs| = 2^n)

  explicit NaiveMultilinearPolynomial(const std::vector<FieldT> &c)
      : n(0), coeffs(c) {
    while ((1u << n) < coeffs.size())
      ++n; // derive n
    if (coeffs.size() != (1u << n))
      throw std::invalid_argument("coeff vector length must be 2^n");
  }

  size_t num_variables() const { return n; }

  /* g(x) – slow but simple */
  FieldT evaluate(const std::vector<FieldT> &x) const {
    if (x.size() != n)
      throw std::invalid_argument("bad point length");
    FieldT acc = FieldT::zero();
    for (size_t mask = 0; mask < coeffs.size(); ++mask) {
      FieldT term = coeffs[mask];
      for (size_t i = 0; i < n; ++i)
        if (mask & (1u << i))
          term *= x[i];
      acc += term;
    }
    return acc;
  }

  FieldT sum_over_remaining(size_t k, const std::vector<FieldT> &fixed) const {
    if (fixed.size() != k)
      throw std::invalid_argument("bad k");
    const size_t rem = n - k;
    FieldT acc = FieldT::zero();
    for (size_t mask = 0; mask < (1u << rem); ++mask) {
      std::vector<FieldT> p = fixed;
      p.reserve(n);
      for (size_t i = 0; i < rem; ++i)
        p.emplace_back((mask & (1u << i)) ? FieldT::one() : FieldT::zero());
      acc += evaluate(p);
    }
    return acc;
  }
};

/* ----------------------------------------------------------- *
 * 2.  Fast dense extension built from layer gate *values*     *
 *     Uses the "fold" trick (O(n·2^{n-1}))                    *
 * ----------------------------------------------------------- */
class DenseMultilinearPolynomial {
  size_t n;                   // #variables  (= log_2 |values|)
  std::vector<FieldT> values; // evaluations on the Boolean cube
public:
  explicit DenseMultilinearPolynomial(const std::vector<FieldT> &layer_values)
      : values(layer_values) {
    // derive n from |values|
    n = 0;
    while ((1u << n) < values.size())
      ++n;
    if (values.size() != (1u << n))
      throw std::invalid_argument("layer size must be power of two");
    // FFT accerlaration
    // auto dom = libfqfft::get_evaluation_domain<FieldT>(values.size());
    // (void)dom; // silence "unused" warning for now
  }

  size_t num_variables() const { return n; }

  /*
   * g(r1,...,rn) in O(n·2^{n-1}) time, cache-friendly
   * Evaluate the multilinear extension at an arbitrary field point.
   * After the i-th iteration the vector holds the table of
   * g(r1, ..., ri,  ...)                    // arity = n-i
   * so the length halves each time:  2^n -> 2^{n-1} -> ... -> 1.
   */
  FieldT evaluate(const std::vector<FieldT> &point) const {
    if (point.size() != n)
      throw std::invalid_argument("bad point length");
    std::vector<FieldT> tmp = values;
    for (size_t i = 0; i < n; ++i)
      // for each gate value vi, eliminate xi
      fold_once_inplace(tmp, point[i]);
    return tmp.front(); // single value left
  }

  FieldT sum_over_remaining(size_t k, const std::vector<FieldT> &fixed) const {
    if (fixed.size() != k)
      throw std::invalid_argument("bad k");
    std::vector<FieldT> tmp = values;
    for (size_t i = 0; i < k; ++i)
      fold_once_inplace(tmp, fixed[i]); // fix all first k elements
    FieldT acc = FieldT::zero();
    for (const auto &v : tmp)
      acc += v; // linear combinations of the remaining
    return acc;
  }

  // https://org.weids.dev/agenda/notes/vcmath.html
  static void fold_once_inplace(std::vector<FieldT> &vec, const FieldT &r) {
    // Fold to half
    const FieldT one_minus_r = FieldT::one() - r;
    const size_t half = vec.size() >> 1;

    // For every consecutive pair (x1 = 0, x1 = 1) in the original multilinear
    // polynomial we replace it by the affine interpolation (1-r)*v0 + r*v1.
    // This works because, with all other coordinates fixed, g is linear in x1.
    for (size_t pair = 0; pair < half; ++pair) {
      // pair-by-pair (each value in gates will only being used once)
      FieldT v0 = vec[2 * pair];
      FieldT v1 = vec[2 * pair + 1];
      // linear polynomial determined by two points (0, v0), (1, v1)
      // g(r, ...) = (v1 - v0) r + v0
      // Folding technique obtain g(r) by applying
      // Affine interpolation:  g(r, ...) = (1-r) * v0 + r * v1
      vec[pair] = v0 * one_minus_r + v1 * r; // overwrite
    }

    // After the loop the vector length is halved and now holds the table of the
    // (k-1)-variate polynomial g(r, .).

    vec.resize(half); // drop the now-unused half, make it "in place"
  }

  /* ---------------------------------------------------------------- *
   *  Read‑only access to the full table on {0,1}^n.                   *
   *  Only the specialised prover needs this – no one else mutates it *
   *  so returning a const ref is perfectly safe.                     *
   * ---------------------------------------------------------------- */
  const std::vector<FieldT> &cube() const { return values; }
};

/* -------------------------------------------------------------------- *
 * Quadratic Univariate Polynomial (since individual degree <=2 in GKR) *
 * -------------------------------------------------------------------- */
struct QuadraticPolynomial {
  FieldT c0, c1, c2;
  QuadraticPolynomial(FieldT a0 = FieldT::zero(), FieldT a1 = FieldT::zero(),
                      FieldT a2 = FieldT::zero())
      : c0(a0), c1(a1), c2(a2) {}
  FieldT evaluate(const FieldT &r) const { return c0 + c1 * r + c2 * (r * r); }
  FieldT sum_over_binary() const {
    return evaluate(FieldT::zero()) + evaluate(FieldT::one());
  }
};

/* -------------------------------------------------------------------- *
 * GKR Layer Polynomial (structured f(b,c) for sum-check) *
 * -------------------------------------------------------------------- */
class GKRLayerPoly {
public:
  DenseMultilinearPolynomial add; // MLE of add gates, fixed on a=r_i
  DenseMultilinearPolynomial mul; // MLE of mul gates, fixed on a=r_i
  DenseMultilinearPolynomial w;   // MLE of next layer values W_{i+1}
  GKRLayerPoly(const std::vector<FieldT> &add_vals,
               const std::vector<FieldT> &mul_vals,
               const std::vector<FieldT> &w_vals)
      : add(add_vals), mul(mul_vals), w(w_vals) {
    if (add.num_variables() != mul.num_variables() ||
        add.num_variables() != 2 * w.num_variables()) {
      throw std::invalid_argument("Invalid GKR layer polynomial dimensions");
    }
  }
  size_t num_variables() const { return add.num_variables(); }
  FieldT evaluate(const std::vector<FieldT> &point) const {
    size_t s = w.num_variables();
    if (point.size() != 2 * s) {
      throw std::invalid_argument("Bad point length for GKR layer evaluation");
    }
    std::vector<FieldT> b(point.begin(), point.begin() + s);
    std::vector<FieldT> c(point.begin() + s, point.end());
    FieldT wb = w.evaluate(b);
    FieldT wc = w.evaluate(c);
    FieldT a = add.evaluate(point);
    FieldT m = mul.evaluate(point);
    return a * (wb + wc) + m * (wb * wc);
  }
};

} // namespace wrappers
