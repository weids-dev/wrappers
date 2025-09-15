#ifndef GKR_HPP
#define GKR_HPP
#include "circuit.hpp"
#include "multilinear.hpp"
#include "sumcheck.hpp"

using DenseMultilinearPolynomial = wrappers::DenseMultilinearPolynomial;
using QuadraticPolynomial = wrappers::QuadraticPolynomial;
using GKRLayerPoly = wrappers::GKRLayerPoly;
using Prover = sumcheck::GKRLayerProver;

/* -------------------------------------------------------------------- *
 * GKR Prover *
 * -------------------------------------------------------------------- */
class GKRProver {
public:
  const Circuit &circuit;
  std::vector<std::vector<FieldT>> layer_vals;

  explicit GKRProver(const Circuit &c, const std::vector<FieldT> &input)
      : circuit(c), layer_vals(c.evaluate(input)) {}

  std::vector<FieldT> get_output() const { return layer_vals.back(); }

  std::vector<FieldT> compute_q_ys(const std::vector<FieldT> &bc,
                                   size_t layer_idx) const {
    size_t s = circuit.layers[layer_idx].nb_vars;
    std::vector<FieldT> b(bc.begin(), bc.begin() + s);
    std::vector<FieldT> c(bc.begin() + s, bc.end());
    DenseMultilinearPolynomial w(layer_vals[layer_idx]);
    std::vector<FieldT> xs(s + 1), ys(s + 1);

    for (size_t tt = 0; tt <= s; ++tt) {
      xs[tt] = FieldT(static_cast<long long>(tt));
      FieldT t = xs[tt];
      std::vector<FieldT> pt(s);
      for (size_t j = 0; j < s; ++j) {
        pt[j] = b[j] * (FieldT::one() - t) + c[j] * t;
      }
      ys[tt] = w.evaluate(pt);
    }
    return ys;
  }

  /* Builds reduced MLEs for add(r_a, b, c) and mul(r_a, b, c) */
  void start_layer(size_t layer_idx, const std::vector<FieldT> &a_r) {
    delete current_poly;
    delete current_prover;
    current_poly = nullptr;
    current_prover = nullptr;

    // output of this layer (a) (# bits)
    size_t s_a = circuit.layers[layer_idx].nb_vars;
    // input to this layer (b, c) (# bits)
    size_t s_bc = circuit.layers[layer_idx - 1].nb_vars;
    // reduced (from size 2^{s_a + 2*s_bc} to 2^{2*s_bc} with output being
    // fixed) Size 2^{2*s_bc}: each entry corresponds to a boolean point
    // (b_0...b_{s-1}, c_0...c_{s-1})
    std::vector<FieldT> reduced_add_vals(1ull << (2 * s_bc), FieldT::zero());
    std::vector<FieldT> reduced_mul_vals(1ull << (2 * s_bc), FieldT::zero());

    for (const auto &gate : circuit.layers[layer_idx].gates) {
      std::vector<FieldT> &reduced =
          (gate.type == ADD) ? reduced_add_vals : reduced_mul_vals;
      FieldT alpha = FieldT::one();
      size_t o_id = gate.o_id; // the index of the gate
      for (size_t j = 0; j < s_a; ++j) {
        bool bit = (o_id >> j) & 1;
        // Lagrange Indicator Polynomial (kronecker delta)
        // chi_a(x) = 1 iff x = b else 0 (when x in {0, 1}^s_a)
        alpha *= bit ? a_r[j] : (FieldT::one() - a_r[j]);
      }
      // flatten the 2D boolean space of inputs into 1D array index
      size_t index = gate.i_id0 + (gate.i_id1 << s_bc);

      // Accumulate alpha at this (b,c) position
      // handles multiple gates with same inputs
      reduced[index] += alpha;
    }

    current_poly = new GKRLayerPoly(reduced_add_vals, reduced_mul_vals,
                                    layer_vals[layer_idx - 1]);
    current_prover = new Prover(*current_poly);
  }

  QuadraticPolynomial get_next_quadratic(const std::vector<FieldT> &fixed) {
    if (!current_prover) {
      throw std::invalid_argument("Layer not started");
    }
    return current_prover->compute_next_quadratic(fixed);
  }

  ~GKRProver() {
    delete current_poly;
    delete current_prover;
  }

private:
  GKRLayerPoly *current_poly = nullptr;
  Prover *current_prover = nullptr;
};

/* -------------------------------------------------------------------- *
 * GKR Verifier *
 * -------------------------------------------------------------------- */
class GKRVerifier {
public:
  const Circuit &circuit;
  std::vector<FieldT> input;

  explicit GKRVerifier(const Circuit &c, const std::vector<FieldT> &in)
      : circuit(c), input(in) {}

  bool verify(GKRProver &prover) {
    // Prover "sends" output (in practice, assume trusted or part of protocol)
    std::vector<FieldT> output = prover.get_output();
    size_t depth = circuit.layers.size() - 1;
    size_t s_out = circuit.layers.back().nb_vars;
    if (output.size() != (1u << s_out)) {
      return false;
    }
    DenseMultilinearPolynomial w_out(output);
    std::vector<FieldT> current_r(s_out);
    for (auto &e : current_r) {
      e = FieldT::random_element();
    }
    FieldT claim = w_out.evaluate(current_r);
    for (int i = depth; i >= 1; --i) {        // Layers from output to input
      size_t s_a = circuit.layers[i].nb_vars; // Output of this layer (a)
      size_t s_bc = circuit.layers[i - 1].nb_vars; // Input to this layer (b,c)
      prover.start_layer(i, current_r);
      std::vector<FieldT> fixed;
      FieldT current_claim = claim;
      for (size_t round = 0; round < 2 * s_bc; ++round) {
        // "Classic" Interactive 2*s_bc round Sum-check Protocol
        QuadraticPolynomial p = prover.get_next_quadratic(fixed);
        if (p.sum_over_binary() != current_claim) {
          return false;
        }
        FieldT r = FieldT::random_element();
        fixed.push_back(r);
        current_claim = p.evaluate(r);
      }
      // Prover "sends" ys for q
      std::vector<FieldT> ys = prover.compute_q_ys(fixed, i - 1);
      // Compute q(0), q(1) using Lagrange
      std::vector<FieldT> xs(s_bc + 1);
      for (size_t tt = 0; tt <= s_bc; ++tt) {
        xs[tt] = FieldT(static_cast<long long>(tt));
      }
      FieldT q0 = lagrange_eval(ys, xs, FieldT::zero());
      FieldT q1 = lagrange_eval(ys, xs, FieldT::one());
      FieldT add_eval = compute_reduced_gate_eval(true, i, current_r, fixed);
      FieldT mul_eval = compute_reduced_gate_eval(false, i, current_r, fixed);
      FieldT eval = add_eval * (q0 + q1) + mul_eval * (q0 * q1);
      if (eval != current_claim) {
        return false;
      }
      // Pick t, compute new r, new claim
      FieldT t = FieldT::random_element();
      FieldT new_m = lagrange_eval(ys, xs, t);
      std::vector<FieldT> new_r(s_bc);
      for (size_t j = 0; j < s_bc; ++j) {
        new_r[j] = fixed[j] * (FieldT::one() - t) + fixed[s_bc + j] * t;
      }
      current_r = std::move(new_r);
      claim = new_m;
    }
    // Final input check
    DenseMultilinearPolynomial input_mle(input);
    FieldT input_eval = input_mle.evaluate(current_r);
    return input_eval == claim;
  }

private:
  static FieldT lagrange_eval(const std::vector<FieldT> &ys,
                              const std::vector<FieldT> &xs, const FieldT &r) {
    size_t deg1 = xs.size();
    FieldT res = FieldT::zero();
    for (size_t i = 0; i < deg1; ++i) {
      FieldT term = ys[i];
      for (size_t j = 0; j < deg1; ++j) {
        if (j == i)
          continue;
        term *= (r - xs[j]) * (xs[i] - xs[j]).inverse();
      }
      res += term;
    }
    return res;
  }

  FieldT compute_reduced_gate_eval(bool is_add, int layer_idx,
                                   const std::vector<FieldT> &a_r,
                                   const std::vector<FieldT> &bc) const {
    size_t s_a = circuit.layers[layer_idx].nb_vars;
    size_t s_bc = circuit.layers[layer_idx - 1].nb_vars;
    std::vector<FieldT> b(bc.begin(), bc.begin() + s_bc);
    std::vector<FieldT> c(bc.begin() + s_bc, bc.end());
    FieldT res = FieldT::zero();
    for (const auto &gate : circuit.layers[layer_idx].gates) {
      if ((gate.type == ADD) != is_add)
        continue;
      FieldT alpha = FieldT::one();
      size_t o_id = gate.o_id;
      for (size_t j = 0; j < s_a; ++j) {
        bool o_bit = (o_id >> j) & 1;
        alpha *= o_bit ? a_r[j] : (FieldT::one() - a_r[j]);
      }
      FieldT chi_b = FieldT::one();
      size_t i0_id = gate.i_id0;
      for (size_t j = 0; j < s_bc; ++j) {
        bool i0_bit = (i0_id >> j) & 1;
        chi_b *= i0_bit ? b[j] : (FieldT::one() - b[j]);
      }
      FieldT chi_c = FieldT::one();
      size_t i1_id = gate.i_id1;
      for (size_t j = 0; j < s_bc; ++j) {
        bool i1_bit = (i1_id >> j) & 1;
        chi_c *= i1_bit ? c[j] : (FieldT::one() - c[j]);
      }
      res += alpha * chi_b * chi_c;
    }
    return res;
  }
};
#endif // GKR_HPP
