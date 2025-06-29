#ifndef SUMCHECK_MULTILINEAR_HPP
#define SUMCHECK_MULTILINEAR_HPP

#include <iostream>
#include <stdexcept>
#include <vector>

#include <libff/algebra/curves/alt_bn128/alt_bn128_pp.hpp>

using FieldT = libff::Fr<libff::alt_bn128_pp>;

/*
 * MultilinearPolynomial stores every coefficient of an
 * n-variable multilinear polynomial in a bit-indexed table and provides
 * exponential-time routines to evaluate it at a point or to sum it over Boolean
 * sub-cube
 */
class MultilinearPolynomial {
public:
  size_t n;                   // number of variables
  std::vector<FieldT> coeffs; // length = 2^n
                              // In C++, class members are initialized in
                              // the order they are declared in the class

  explicit MultilinearPolynomial(size_t n_, const std::vector<FieldT> &c)
      : n(n_), coeffs(c) {
    if (coeffs.size() != (1u << n)) // 1u -> binary 1
      throw std::invalid_argument("Coefficient vector size must be 2^n");
  }

  // Evaluate g(x1, ..., xn) at a given point
  FieldT evaluate(const std::vector<FieldT> &point) const {
    if (point.size() != n)
      throw std::invalid_argument("Point size must match n");

    FieldT acc = FieldT::zero(); // accumulator
    for (size_t mask = 0; mask < (1u << n); ++mask) {
      // 2^n all linear combinations
      FieldT term = coeffs[mask];
      for (size_t i = 0; i < n; ++i)
        if (mask & (1u << i))
          term *= point[i];
      acc += term;
    }
    return acc;
  }

  // Sum g over {0,1} for variables k+1 to n, with first k variables fixed
  FieldT sum_over_remaining(size_t k, const std::vector<FieldT> &fixed) const {
    if (fixed.size() != k)
      throw std::invalid_argument("Fixed values size must match k");

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

// Class representing a linear polynomial p(x) = c0 + c1 * x
class LinearPolynomial {
public:
  FieldT c0, c1;

  LinearPolynomial(const FieldT &c0_, const FieldT &c1_) : c0(c0_), c1(c1_) {}

  FieldT evaluate(const FieldT &x) const { return c0 + c1 * x; }

  FieldT sum_over_binary() const { // p(0) + p(1)
    return evaluate(FieldT::zero()) + evaluate(FieldT::one());
  }
};

class Prover {
public:
  MultilinearPolynomial g;
  size_t n;

  explicit Prover(const MultilinearPolynomial &g_) : g(g_), n(g_.n) {}

  /* returns p_{k+1}(x) for the next round */
  LinearPolynomial compute_next_linear(const std::vector<FieldT> &fixed) {
    const size_t k = fixed.size();
    if (k >= n)
      throw std::invalid_argument("Too many fixed values");

    // p(0)
    std::vector<FieldT> fixed0 = fixed;
    fixed0.emplace_back(FieldT::zero());
    FieldT p0 = g.sum_over_remaining(k + 1, fixed0);

    // p(1)
    std::vector<FieldT> fixed1 = fixed;
    fixed1.emplace_back(FieldT::one());
    FieldT p1 = g.sum_over_remaining(k + 1, fixed1);

    // a line determined by those two points (0, p0), (1, p1)
    return {p0, p1 - p0};
  }
};

class Verifier {
  FieldT H; // Claimed sum
  size_t n;

public:
  Verifier(const FieldT &H_, size_t n_) : H(H_), n(n_) {}

  bool verify(Prover &prover) {
    std::vector<FieldT> fixed;
    FieldT claim = H;

    for (size_t round = 0; round < n; ++round) {
      LinearPolynomial p = prover.compute_next_linear(fixed);

      if (p.sum_over_binary() != claim) {
        std::cerr << "Round " << round + 1 << " failed\n";
        return false;
      }

      FieldT r = FieldT::random_element();
      fixed.push_back(r);
      claim = p.evaluate(r); // new sub-claim
    }

    return (prover.g.evaluate(fixed) == claim);
  }
};

#endif // sumcheck_multilinear.hpp
