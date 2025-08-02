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

/* ===================================================================== *
 *  SPECIALISATION: fast, linear‑time prover for DenseMultilinearPolynomial
 *  Implements the algorithm of Thaler, "Time‑Optimal Interactive Proofs
 *  for Circuit Evaluation" (Thaler 13)).
 *  ---------------------------------------------------------------
 *  •   Total prover time  O(2^n) instead of  O(n*2^n)
 *  •   Memory shrinks with every round: 2^n -> 2^{n-1} -> ... -> 1
 * ===================================================================== */
template <> class Prover<wrappers::DenseMultilinearPolynomial> {
  using Poly = wrappers::DenseMultilinearPolynomial;
  using FieldT = wrappers::FieldT;

  const Poly &g; // immutable reference to full polynomial
  const size_t n;
  size_t processed;          // how many coordinates are already fixed
  // DP-like optimization
  std::vector<FieldT> table; // current :slice", size 2^{n‑processed}

public:
  explicit Prover(const Poly &g_)
      : g(g_), n(g_.num_variables()), processed(0), table(g_.cube()) {}

  /* ------------------------------------------------------------------ *
   *  One Sum‑Check round – O(|table|) = O(2^{n‑processed}) time.       *
   *  1.  Consume any new fixed coordinates the verifier has sent       *
   *      since the last call (at most one, but the loop is robust).    *
   *  2.  Scan current table pair‑wise to obtain C0, C1.               *
   *      (C0 = \SUM_{x=0} g,  C1 = \SUM_{x=1} g − \SUM_{x=0} g)              *
   * ------------------------------------------------------------------ */
  LinearPolynomial compute_next_linear(const std::vector<FieldT> &fixed) {
    /* Step 1: fold away *all* fixed coordinates we have not handled yet */
    while (processed < fixed.size()) {
      g.fold_once_inplace(table, fixed[processed]);
      ++processed;
    }
    if (processed >= n)
      throw std::invalid_argument("too many fixed values");

    /* Step 2: build the degree‑1 polynomial for x_{processed+1}        */
    FieldT c0 = FieldT::zero(); // \SUM_{x=0} g
    FieldT c1 = FieldT::zero(); // \SUM_{x=1} g − \SUM_{x=0} g

    const size_t half = table.size() >> 1;
    for (size_t i = 0; i < half; ++i) {
      const FieldT v0 = table[2 * i];     // partial value @ x = 0
      const FieldT v1 = table[2 * i + 1]; // partial value @ x = 1
      c0 += v0;
      c1 += v1 - v0;
    }
    return {c0, c1};
  }

  /* Needed at the final verifier step */
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
