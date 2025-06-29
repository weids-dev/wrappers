// tests/test_sumcheck_multilinear.cpp
// ----------------------------------------------
// gtests for the toy multilinear sum-check protocol
// ----------------------------------------------

#include <gtest/gtest.h>

#include <libff/algebra/curves/alt_bn128/alt_bn128_pp.hpp>
using FieldT = libff::Fr<libff::alt_bn128_pp>;

#include "sumcheck_multilinear.hpp"

#include <chrono>
#include <random>
#include <tuple>

// ------------ helpers --------------------------------------------------------

/* Brute-force sum of g over {0,1}^n – used only to cross-check small instances
 */
static FieldT brute_sum_bool_cube(const MultilinearPolynomial &g) {
  const size_t n = g.n;
  FieldT acc = FieldT::zero();
  std::vector<FieldT> pt(n, FieldT::zero());

  for (size_t mask = 0; mask < (1u << n); ++mask) {
    for (size_t i = 0; i < n; ++i)
      pt[i] = (mask & (1u << i)) ? FieldT::one() : FieldT::zero();
    acc += g.evaluate(pt);
  }
  return acc;
}

/* Produce a random multilinear polynomial of arity n. */
static MultilinearPolynomial random_poly(size_t n, std::mt19937_64 &rng) {
  std::uniform_int_distribution<uint64_t> dist; // full field range
  std::vector<FieldT> coeffs(1u << n);
  for (auto &c : coeffs)
    c = FieldT(dist(rng));
  return MultilinearPolynomial{n, coeffs};
}

// ------------ TEST FIXTURE ---------------------------------------------------

class SumcheckFixture : public ::testing::Test {
protected:
  std::mt19937_64 rng;

  void SetUp() override {
    std::random_device rd;
    rng.seed(rd());
  }
};

// ----------------------------------------------------------------------------
// 1. Core algebra sanity (very small, catches blatant issues)
// ----------------------------------------------------------------------------
TEST(AlgebraSanity, FieldTArithmetic) {
  FieldT a(3), b(5);
  ASSERT_EQ(a + b, FieldT(8));
  ASSERT_EQ(a * b, FieldT(15));
  ASSERT_EQ((a - b) + b, a);
  ASSERT_TRUE(FieldT::one() + FieldT(-1) == FieldT::zero());
}

// ----------------------------------------------------------------------------
// 2. Polynomial evaluation & summation correctness (n <= 5, brute-forced)
// ----------------------------------------------------------------------------
TEST_F(SumcheckFixture, PolynomialEvaluateAndSumSmall) {
  for (size_t n = 1; n <= 5; ++n) {
    auto g = random_poly(n, rng);
    // Cross-check sum_over_remaining(k=0) against brute force.
    FieldT lib_sum = g.sum_over_remaining(0, {});
    FieldT brute = brute_sum_bool_cube(g);
    ASSERT_EQ(lib_sum, brute) << "Mismatch for n = " << n;
  }
}

// ----------------------------------------------------------------------------
// 3. Completeness: honest prover should always convince verifier.
//    Routine CI run: n <= 10.
// ----------------------------------------------------------------------------
TEST_F(SumcheckFixture, CompletenessManyParameters) {
  std::vector<size_t> arities = {1, 2, 3, 5, 8, 10};
  for (size_t n : arities) {
    auto g = random_poly(n, rng); // random polynomial
    FieldT H = g.sum_over_remaining(0, {});

    Prover prover(g);
    Verifier verifier(H, n);

    ASSERT_TRUE(verifier.verify(prover)) << "Completeness failed for n = " << n;
  }
}

// ----------------------------------------------------------------------------
// 4. Soundness: if the claimed sum is wrong, verifier should (overwhelmingly)
//    reject. We test **deterministically** by adding 1 to the true sum.
// ----------------------------------------------------------------------------
TEST_F(SumcheckFixture, SoundnessWrongSum) {
  for (size_t n = 1; n <= 10; ++n) {
    auto g = random_poly(n, rng);
    FieldT H = g.sum_over_remaining(0, {});
    FieldT H_fake = H + FieldT::one(); // guaranteed to differ

    Prover prover(g);
    Verifier verifier(H_fake, n);

    ASSERT_FALSE(verifier.verify(prover)) << "Soundness failure for n = " << n;
  }
}

// ----------------------------------------------------------------------------
// 5. Zero polynomial corner-case (all coefficients 0).
// ----------------------------------------------------------------------------
TEST(EdgeCases, ZeroPolynomial) {
  size_t n = 6;
  std::vector<FieldT> zeros(1u << n, FieldT::zero());
  MultilinearPolynomial g(n, zeros);

  FieldT H = FieldT::zero();

  Prover prover(g);
  Verifier verifier(H, n);

  ASSERT_TRUE(verifier.verify(prover));

  // Wrong claim on zero poly should be rejected.
  Verifier bad_verifier(FieldT::one(), n);
  ASSERT_FALSE(bad_verifier.verify(prover));
}

// ----------------------------------------------------------------------------
// 6. Performance scaling (timing is *informative*, not an assertion).
//    Caps at n = 12 so the suite finishes in < 10s on commodity HW.
// ----------------------------------------------------------------------------
TEST_F(SumcheckFixture, PerformanceScaling) {
  std::cout << "\n--- Sum-check scaling ---\n";
  for (size_t n = 4; n <= 12; n += 2) {
    auto g = random_poly(n, rng);
    FieldT H = g.sum_over_remaining(0, {});

    Prover prover(g);
    Verifier verifier(H, n);

    auto start = std::chrono::high_resolution_clock::now();
    ASSERT_TRUE(verifier.verify(prover)); // correctness
    auto end = std::chrono::high_resolution_clock::now();

    double elapsed =
        std::chrono::duration_cast<std::chrono::duration<double>>(end - start)
            .count();
    std::cout << "n = " << n << " -> " << elapsed << " s\n";
  }
  std::cout << "-------------------------\n";
}

// ----------------------------------------------------------------------------
// 7. main()
// ----------------------------------------------------------------------------
int main(int argc, char **argv) {
  // Initialise libff curve parameters once before FieldT operations.
  libff::alt_bn128_pp::init_public_params();

  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}