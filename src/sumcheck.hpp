// sumcheck.hpp
#pragma once

// #define SUMCHECK_POLY_FFT // comment-out to use the naive class

#include "multilinear.hpp" // brings FieldT + both back-ends
#include <iostream>
#include <stdexcept>
#include <vector>

namespace sumcheck {

#ifdef DENSE_POLY
using PolyDefault = wrappers::DenseMultilinearPolynomial;
#else
using PolyDefault = wrappers::NaiveMultilinearPolynomial;
#endif

using FieldT = wrappers::FieldT;
using QuadraticPolynomial = wrappers::QuadraticPolynomial;
using GKRLayerPoly = wrappers::GKRLayerPoly;

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
  size_t processed; // how many coordinates are already fixed
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

/* ===================================================================== *
 * SPECIALISATION: fast, linear-time prover for GKRLayerPoly *
 * Implements efficient computation using DP tables on components. *
 * Total time O(2^{2n}) where n = s, but optimal for dense representation. *
 * ===================================================================== */
template <> class Prover<GKRLayerPoly> {
  using Poly = GKRLayerPoly;
  using FieldT = wrappers::FieldT;
  const Poly &g; // immutable reference to full polynomial
  const size_t n;
  const size_t s;
  size_t processed; // how many coordinates are already fixed
  // DP tables
  std::vector<FieldT> table_add;
  std::vector<FieldT> table_mul;
  std::vector<FieldT> table_wb;
  std::vector<FieldT> table_wc;

  void fold_once_inplace(std::vector<FieldT> &t, const FieldT &r) {
    size_t sz = t.size();
    if (sz % 2 != 0 || sz == 0)
      throw std::invalid_argument("Invalid table size for folding");
    FieldT one_minus_r = FieldT::one() - r;
    for (size_t i = 0; i < sz / 2; ++i) {
      t[i] = one_minus_r * t[2 * i] + r * t[2 * i + 1];
    }
    t.resize(sz / 2);
  }

public:
  explicit Prover(const Poly &g_)
      : g(g_), n(g.num_variables()), s(g.w.num_variables()), processed(0),
        table_add(g.add.cube()), table_mul(g.mul.cube()), table_wb(g.w.cube()),
        table_wc(g.w.cube()) {
    if (n != 2 * s)
      throw std::invalid_argument("GKR layer must have even num_variables");
  }

  QuadraticPolynomial compute_next_quadratic(const std::vector<FieldT> &fixed) {
    while (processed < fixed.size()) {
      FieldT r = fixed[processed];
      fold_once_inplace(table_add, r);
      fold_once_inplace(table_mul, r);
      if (processed < s) {
        fold_once_inplace(table_wb, r);
      } else {
        fold_once_inplace(table_wc, r);
      }
      ++processed;
    }
    if (processed >= n)
      throw std::invalid_argument("too many fixed values");
    FieldT c0 = FieldT::zero();
    FieldT c1 = FieldT::zero();
    FieldT c2 = FieldT::zero();
    bool is_b_var = (processed < s);
    size_t half = table_add.size() >> 1;
    if (is_b_var) {
      // Current variable is in b group
      size_t remaining_b_after = s - processed - 1;
      size_t stride_b = 1ull << remaining_b_after;
      size_t num_c = 1ull << s;
      std::vector<FieldT> partial_add0_plain(stride_b, FieldT::zero());
      std::vector<FieldT> partial_add1_plain(stride_b, FieldT::zero());
      std::vector<FieldT> partial_mul0_plain(stride_b, FieldT::zero());
      std::vector<FieldT> partial_mul1_plain(stride_b, FieldT::zero());
      std::vector<FieldT> partial_add0_wc(stride_b, FieldT::zero());
      std::vector<FieldT> partial_add1_wc(stride_b, FieldT::zero());
      std::vector<FieldT> partial_mul0_wc(stride_b, FieldT::zero());
      std::vector<FieldT> partial_mul1_wc(stride_b, FieldT::zero());
      for (size_t cc = 0; cc < num_c; ++cc) {
        for (size_t bb = 0; bb < stride_b; ++bb) {
          size_t ii = bb + cc * stride_b;
          FieldT a0 = table_add[2 * ii];
          FieldT a1 = table_add[2 * ii + 1];
          FieldT m0 = table_mul[2 * ii];
          FieldT m1 = table_mul[2 * ii + 1];
          partial_add0_plain[bb] += a0;
          partial_add1_plain[bb] += a1;
          partial_mul0_plain[bb] += m0;
          partial_mul1_plain[bb] += m1;
          FieldT wc_val = table_wc[cc];
          partial_add0_wc[bb] += a0 * wc_val;
          partial_add1_wc[bb] += a1 * wc_val;
          partial_mul0_wc[bb] += m0 * wc_val;
          partial_mul1_wc[bb] += m1 * wc_val;
        }
      }
      // add * wb
      FieldT s_add0_w0 = FieldT::zero();
      FieldT s_add0_w1 = FieldT::zero();
      FieldT s_add1_w0 = FieldT::zero();
      FieldT s_add1_w1 = FieldT::zero();
      for (size_t bb = 0; bb < stride_b; ++bb) {
        FieldT w0 = table_wb[2 * bb];
        FieldT w1 = table_wb[2 * bb + 1];
        s_add0_w0 += w0 * partial_add0_plain[bb];
        s_add0_w1 += w1 * partial_add0_plain[bb];
        s_add1_w0 += w0 * partial_add1_plain[bb];
        s_add1_w1 += w1 * partial_add1_plain[bb];
      }
      c0 += s_add0_w0;
      c1 += -FieldT(2) * s_add0_w0;
      c2 += s_add0_w0;
      FieldT two_terms_add = s_add0_w1 + s_add1_w0;
      c1 += two_terms_add;
      c2 += -two_terms_add;
      c2 += s_add1_w1;
      // mul * wb * wc
      FieldT s_mul0_w0 = FieldT::zero();
      FieldT s_mul0_w1 = FieldT::zero();
      FieldT s_mul1_w0 = FieldT::zero();
      FieldT s_mul1_w1 = FieldT::zero();
      for (size_t bb = 0; bb < stride_b; ++bb) {
        FieldT w0 = table_wb[2 * bb];
        FieldT w1 = table_wb[2 * bb + 1];
        s_mul0_w0 += w0 * partial_mul0_wc[bb];
        s_mul0_w1 += w1 * partial_mul0_wc[bb];
        s_mul1_w0 += w0 * partial_mul1_wc[bb];
        s_mul1_w1 += w1 * partial_mul1_wc[bb];
      }
      c0 += s_mul0_w0;
      c1 += -FieldT(2) * s_mul0_w0;
      c2 += s_mul0_w0;
      FieldT two_terms_mul = s_mul0_w1 + s_mul1_w0;
      c1 += two_terms_mul;
      c2 += -two_terms_mul;
      c2 += s_mul1_w1;
      // add * wc
      FieldT V = FieldT::zero();
      FieldT W = FieldT::zero();
      for (size_t bb = 0; bb < stride_b; ++bb) {
        V += partial_add0_wc[bb];
        W += partial_add1_wc[bb];
      }
      c0 += V;
      c1 += -V + W;
    } else {
      // Current variable is in c group
      FieldT wb_fixed = table_wb[0];
      // Compute sums for add0, add1, mul0, mul1
      FieldT sum_add0 = FieldT::zero();
      FieldT sum_add1 = FieldT::zero();
      FieldT sum_mul0 = FieldT::zero();
      FieldT sum_mul1 = FieldT::zero();
      // Compute quadratic sums for add * wc and mul * wc
      FieldT s_add0_w0 = FieldT::zero();
      FieldT s_add0_w1 = FieldT::zero();
      FieldT s_add1_w0 = FieldT::zero();
      FieldT s_add1_w1 = FieldT::zero();
      FieldT s_mul0_w0 = FieldT::zero();
      FieldT s_mul0_w1 = FieldT::zero();
      FieldT s_mul1_w0 = FieldT::zero();
      FieldT s_mul1_w1 = FieldT::zero();
      for (size_t ii = 0; ii < half; ++ii) {
        FieldT a0 = table_add[2 * ii];
        FieldT a1 = table_add[2 * ii + 1];
        FieldT m0 = table_mul[2 * ii];
        FieldT m1 = table_mul[2 * ii + 1];
        FieldT wc0 = table_wc[2 * ii];
        FieldT wc1 = table_wc[2 * ii + 1];
        sum_add0 += a0;
        sum_add1 += a1;
        sum_mul0 += m0;
        sum_mul1 += m1;
        s_add0_w0 += a0 * wc0;
        s_add0_w1 += a0 * wc1;
        s_add1_w0 += a1 * wc0;
        s_add1_w1 += a1 * wc1;
        s_mul0_w0 += m0 * wc0;
        s_mul0_w1 += m0 * wc1;
        s_mul1_w0 += m1 * wc0;
        s_mul1_w1 += m1 * wc1;
      }
      // add * wb_fixed
      FieldT V_add_wb = wb_fixed * sum_add0;
      FieldT W_add_wb = wb_fixed * sum_add1;
      c0 += V_add_wb;
      c1 += -V_add_wb + W_add_wb;
      // add * wc
      c0 += s_add0_w0;
      c1 += -FieldT(2) * s_add0_w0;
      c2 += s_add0_w0;
      FieldT two_terms_add = s_add0_w1 + s_add1_w0;
      c1 += two_terms_add;
      c2 += -two_terms_add;
      c2 += s_add1_w1;
      // mul * wb_fixed * wc
      FieldT s_mul0_w0_wb = wb_fixed * s_mul0_w0;
      FieldT s_mul0_w1_wb = wb_fixed * s_mul0_w1;
      FieldT s_mul1_w0_wb = wb_fixed * s_mul1_w0;
      FieldT s_mul1_w1_wb = wb_fixed * s_mul1_w1;
      c0 += s_mul0_w0_wb;
      c1 += -FieldT(2) * s_mul0_w0_wb;
      c2 += s_mul0_w0_wb;
      FieldT two_terms_mul_wb = s_mul0_w1_wb + s_mul1_w0_wb;
      c1 += two_terms_mul_wb;
      c2 += -two_terms_mul_wb;
      c2 += s_mul1_w1_wb;
    }
    return {c0, c1, c2};
  }
  const Poly &poly() const { return g; }
};

/* -------------------------------------------------------------------- *
 * Verifier for GKR layer (uses QuadraticPolynomial) *
 * -------------------------------------------------------------------- */
template <> class Verifier<GKRLayerPoly> {
  FieldT claim;
  const size_t n;
  std::vector<FieldT> fixed; // Public for access after verify
public:
  Verifier(const FieldT &claim_, size_t n_) : claim(claim_), n(n_) {}
  bool verify(Prover<GKRLayerPoly> &prover) {
    fixed.clear();
    FieldT current_claim = claim;
    for (size_t round = 0; round < n; ++round) {
      QuadraticPolynomial p = prover.compute_next_quadratic(fixed);
      if (p.sum_over_binary() != current_claim) {
        std::cerr << "GKR layer sum-check round " << round + 1 << " failed\n";
        return false;
      }
      FieldT r = FieldT::random_element();
      fixed.push_back(r);
      current_claim = p.evaluate(r);
    }
    // Do NOT perform final evaluation here; GKR replaces it with structured
    // check
    claim = current_claim; // Update claim to final for access
    return true;
  }
  const std::vector<FieldT> &get_fixed() const { return fixed; }
  const FieldT &get_final_claim() const { return claim; }
};

using DefaultProver = Prover<PolyDefault>;
using DefaultVerifier = Verifier<PolyDefault>;

using NaiveProver = Prover<wrappers::NaiveMultilinearPolynomial>;
using NaiveVerifier = Verifier<wrappers::NaiveMultilinearPolynomial>;

using DenseProver = Prover<wrappers::DenseMultilinearPolynomial>;
using DenseVerifier = Verifier<wrappers::DenseMultilinearPolynomial>;

using GKRLayerProver = Prover<GKRLayerPoly>;
using GKRLayerVerifier = Verifier<GKRLayerPoly>;

} // namespace sumcheck
