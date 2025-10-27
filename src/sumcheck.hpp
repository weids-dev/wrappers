// sumcheck.hpp
#pragma once

// #define SUMCHECK_POLY_FFT // comment-out to use the naive class

#include "circuit.hpp"
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
  std::vector<FieldT> table; // current :slice, size 2^{n‑processed}

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
      // mul * wb * wc (any potencial at this point "add")
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

// Sparse GKR P: never materializes add/mul wiring tables of size 2^{2s}.
class SparseGKRLayerProver {
public:
  using FieldT = libff::Fr<libff::alt_bn128_pp>;

  // Build a sparse view for one circuit layer:
  // - gates:         circuit.layers[layer_idx].gates
  // - s_a:           #bits for output addresses at this layer
  // - s_bc:          #bits for each of the two inputs from previous layer
  // - a_r:           verifier's randomness for output addr (pins A,M to (b,c))
  // - w_values:      dense table of previous layer's wire-values (size 2^s_bc)
  SparseGKRLayerProver(const std::vector<Gate> &gates, size_t s_a, size_t s_bc,
                       const std::vector<FieldT> &a_r,
                       const std::vector<FieldT> &w_values)
      : s_(s_bc), last_k_(0), w_full_(w_values), T_w_b_(w_values),
        T_w_c_(w_values) {
    if (w_values.size() != (1ull << s_)) {
      throw std::invalid_argument(
          "SparseGKRLayerProver: w_values size must be 2^s_bc");
    }
    entries_.reserve(gates.size());
    // Precompute alpha_g = chi_out(r_a) for each gate (Kronecker at a=r_a).
    for (const auto &g : gates) {
      FieldT alpha = FieldT::one();
      size_t o = g.o_id;
      for (size_t j = 0; j < s_a; ++j) {
        bool bit = (o >> j) & 1;
        alpha *= bit ? a_r[j] : (FieldT::one() - a_r[j]);
      }
      entries_.push_back(Entry{g.type, g.i_id0, g.i_id1, alpha, alpha});
    }
  }

  wrappers::QuadraticPolynomial
  compute_next_quadratic(const std::vector<FieldT> &fixed) {
    if (fixed.size() > 2 * s_) {
      throw std::invalid_argument(
          "SparseGKRLayerProver: too many fixed coordinates");
    }

    // Handy constants (avoid int*FieldT)
    const FieldT ONE = FieldT::one();
    const FieldT TWO = ONE + ONE;

    // If caller restarted (new layer), reset internal state.
    if (fixed.size() < last_k_) {
      reset_state();
    }

    // Fold w-tables & update per-gate prefix weights for newly fixed coords
    while (last_k_ < fixed.size()) {
      const FieldT &r = fixed[last_k_];
      if (last_k_ < s_) {
        const size_t j = last_k_;
        for (auto &e : entries_) {
          bool bit = (e.left >> j) & 1;
          e.prefix *= bit ? r : (ONE - r);
        }
        wrappers::DenseMultilinearPolynomial::fold_once_inplace(T_w_b_, r);
      } else {
        const size_t j2 = last_k_ - s_;
        for (auto &e : entries_) {
          bool bit = (e.right >> j2) & 1;
          e.prefix *= bit ? r : (ONE - r);
        }
        wrappers::DenseMultilinearPolynomial::fold_once_inplace(T_w_c_, r);
      }
      ++last_k_;
    }

    if (fixed.size() == 2 * s_) {
      throw std::logic_error(
          "SparseGKRLayerProver: all coordinates are already fixed");
    }

    wrappers::QuadraticPolynomial P; // c0,c1,c2 = 0
    const size_t k = fixed.size();

    if (k < s_) {
      // -------- current variable is b_j --------
      const size_t j = k;
      const std::vector<FieldT> &Wc = T_w_c_; // constant across b-rounds

      for (const auto &e : entries_) {
        const bool bbit = (e.left >> j) & 1;
        const FieldT a0 =
            (!bbit && e.gate_type == GateType::ADD) ? e.prefix : FieldT::zero();
        const FieldT a1 =
            (bbit && e.gate_type == GateType::ADD) ? e.prefix : FieldT::zero();
        const FieldT m0 =
            (!bbit && e.gate_type == GateType::MUL) ? e.prefix : FieldT::zero();
        const FieldT m1 =
            (bbit && e.gate_type == GateType::MUL) ? e.prefix : FieldT::zero();

        const size_t suffix_left = (e.left >> (j + 1));
        const size_t base = (suffix_left << 1);
        const FieldT wb0 = T_w_b_[base];
        const FieldT wb1 = T_w_b_[base + 1];

        const FieldT wc = Wc[e.right];

        // A · w(b)
        P.c0 += a0 * wb0;
        P.c1 += (a1 * wb0) + (a0 * wb1) - (TWO * (a0 * wb0));
        P.c2 += (a0 * wb0) - (a1 * wb0) - (a0 * wb1) + (a1 * wb1);

        // A · w(c)  (w(c) constant here)
        P.c0 += a0 * wc;
        P.c1 += (a1 * wc) - (a0 * wc);
        // P.c2 += 0;

        // M · w(b) · w(c)
        const FieldT m0wb0 = m0 * wb0 * wc;
        const FieldT m0wb1 = m0 * wb1 * wc;
        const FieldT m1wb0 = m1 * wb0 * wc;
        P.c0 += m0wb0;
        P.c1 += m1wb0 + m0wb1 - (TWO * m0wb0);
        P.c2 += (m0wb0)-m1wb0 - m0wb1 + (m1 * wb1 * wc);
      }
    } else {
      // -------- current variable is c_j2 --------
      const size_t j2 = k - s_;

      if (T_w_b_.size() != 1) {
        throw std::logic_error(
            "SparseGKRLayerProver: T_w_b_ should be folded to size 1");
      }
      const FieldT wb = T_w_b_.front();

      for (const auto &e : entries_) {
        const bool cbit = (e.right >> j2) & 1;
        const FieldT a0 =
            (!cbit && e.gate_type == GateType::ADD) ? e.prefix : FieldT::zero();
        const FieldT a1 =
            (cbit && e.gate_type == GateType::ADD) ? e.prefix : FieldT::zero();
        const FieldT m0 =
            (!cbit && e.gate_type == GateType::MUL) ? e.prefix : FieldT::zero();
        const FieldT m1 =
            (cbit && e.gate_type == GateType::MUL) ? e.prefix : FieldT::zero();

        const size_t suffix_right = (e.right >> (j2 + 1));
        const size_t base = (suffix_right << 1);
        const FieldT wc0 = T_w_c_[base];
        const FieldT wc1 = T_w_c_[base + 1];

        // A · w(b)   (w(b) constant here)
        P.c0 += a0 * wb;
        P.c1 += (a1 * wb) - (a0 * wb);
        // P.c2 += 0;

        // A · w(c)
        P.c0 += a0 * wc0;
        P.c1 += (a1 * wc0) + (a0 * wc1) - (TWO * (a0 * wc0));
        P.c2 += (a0 * wc0) - (a1 * wc0) - (a0 * wc1) + (a1 * wc1);

        // M · w(b) · w(c)
        const FieldT m0wb = m0 * wb;
        const FieldT m1wb = m1 * wb;
        const FieldT m0wbwc0 = m0wb * wc0;
        const FieldT m0wbwc1 = m0wb * wc1;
        const FieldT m1wbwc0 = m1wb * wc0;
        P.c0 += m0wbwc0;
        P.c1 += m1wbwc0 + m0wbwc1 - (TWO * m0wbwc0);
        P.c2 += (m0wbwc0)-m1wbwc0 - m0wbwc1 + (m1wb * wc1);
      }
    }

    return P;
  }

private:
  struct Entry {
    GateType gate_type;
    uint32_t left;  // i_id0
    uint32_t right; // i_id1
    FieldT alpha;   // chi_out(r_a)
    FieldT
        prefix; // alpha * product of indicator evals for *already fixed* coords
  };

  const size_t s_;             // #bits per side (b and c)
  size_t last_k_;              // #coords already folded into prefix/T_w_*
  std::vector<Entry> entries_; // sparse wiring list with running weights

  // We keep two folded copies of the previous layer’s table:
  const std::vector<FieldT>
      w_full_; // immutable full cube (not used except for sanity)
  std::vector<FieldT> T_w_b_; // folded along b-variables as they are fixed
  std::vector<FieldT> T_w_c_; // folded along c-variables as they are fixed

  void reset_state() {
    last_k_ = 0;
    T_w_b_ = w_full_;
    T_w_c_ = w_full_;
    for (auto &e : entries_) {
      e.prefix = e.alpha; // reset to just chi_out(r_a)
    }
  }
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
