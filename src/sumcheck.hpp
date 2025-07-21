// sumcheck.hpp
#pragma once

// #define SUMCHECK_POLY_FFT // comment-out to use the naive class

#include "multilinear.hpp" // brings FieldT + both back-ends
#include <iostream>
#include <stdexcept>
#include <vector>

namespace sumcheck {

#ifdef SUMCHECK_POLY_FFT
using PolyDefault = wrappers::DenseMultilinearPolynomial;
#else
using PolyDefault = wrappers::NaiveMultilinearPolynomial;
#endif

using FieldT = wrappers::FieldT;

/* -------------------------------------------------------------------- *
 *  Helper: Linear polynomial p(x) = c0 + c1·x                      *
 * -------------------------------------------------------------------- */
struct LinearPolynomial {
  FieldT c0, c1;
  FieldT evaluate(const FieldT &x) const { return c0 + c1 * x; }
  FieldT sum_over_binary() const { return c0 + c0 + c1; } // p(0)+p(1)
};

/* -------------------------------------------------------------------- *
 *  Prover – templated over any "multilinear polynomial" class           *
 * -------------------------------------------------------------------- */
template <class Poly = PolyDefault> class Prover {
  const Poly &g;
  const size_t n;

public:
  explicit Prover(const Poly &g_) : g(g_), n(g_.num_variables()) {}

  /* one Sum-Check round – returns linear poly p_{k+1}(x) */
  LinearPolynomial compute_next_linear(const std::vector<FieldT> &fixed) {
    const size_t k = fixed.size();
    if (k >= n)
      throw std::invalid_argument("too many fixed values");

    // p(0)
    std::vector<FieldT> fixed0 = fixed;
    fixed0.emplace_back(FieldT::zero());
    FieldT p0 = g.sum_over_remaining(k + 1, fixed0);

    // p(1)
    std::vector<FieldT> fixed1 = fixed;
    fixed1.emplace_back(FieldT::one());
    FieldT p1 = g.sum_over_remaining(k + 1, fixed1);

    return {p0, p1 - p0}; // c0, c1
  }

  /* public accessor used by the verifier in the final step */
  const Poly &poly() const { return g; }
};

/* -------------------------------------------------------------------- *
 *  Verifier                                                             *
 * -------------------------------------------------------------------- */
template <class Poly = PolyDefault> class Verifier {
  FieldT H; // Claimed result
  const size_t n;

public:
  Verifier(const FieldT &H_, size_t n_) : H(H_), n(n_) {}

  bool verify(Prover<Poly> &prover) {
    std::vector<FieldT> fixed; // r1, ..., rk
    FieldT claim = H;

    for (size_t round = 0; round < n; ++round) {
      LinearPolynomial p = prover.compute_next_linear(fixed);

      if (p.sum_over_binary() != claim) {
        std::cerr << "Round " << round + 1 << " failed\n";
        return false;
      }
      FieldT r = FieldT::random_element(); // Fiat–Shamir in practice
      fixed.push_back(r);
      claim = p.evaluate(r);
    }
    return prover.poly().evaluate(fixed) == claim;
  }
};

} // namespace sumcheck
