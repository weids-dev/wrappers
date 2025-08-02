// tests/test_sumcheck_multilinear.cpp
// ----------------------------------------------
#include <gtest/gtest.h>
#include <libff/algebra/curves/alt_bn128/alt_bn128_pp.hpp>

#include "multilinear.hpp"
#include "sumcheck.hpp" // brings Prover / Verifier

#include <chrono> // timings

using FieldT = wrappers::FieldT;

/* ---------------- choose the two implementations ------------------ */
using NaivePoly = wrappers::NaiveMultilinearPolynomial;
using FFTPoly = wrappers::DenseMultilinearPolynomial;

/* ------------ helpers (templated so they work for both) ----------- */
template <class Poly> static FieldT brute_sum_bool_cube(const Poly &g) {
  const size_t n = g.num_variables();
  FieldT acc = FieldT::zero();
  std::vector<FieldT> pt(n, FieldT::zero());

  for (size_t mask = 0; mask < (1u << n); ++mask) {
    for (size_t i = 0; i < n; ++i)
      pt[i] = (mask & (1u << i)) ? FieldT::one() : FieldT::zero();
    acc += g.evaluate(pt);
  }
  return acc;
}

template <class Poly> static Poly random_poly(size_t n, std::mt19937_64 &rng) {
  std::uniform_int_distribution<uint64_t> dist;
  std::vector<FieldT> data(1u << n);
  for (auto &v : data)
    v = FieldT(dist(rng));
  return Poly(data);
}

/* ---------------- typed-fixture over both Poly types -------------- */
template <class Poly> class SumcheckFixture : public ::testing::Test {
protected:
  std::mt19937_64 rng;
  void SetUp() override { rng.seed(std::random_device{}()); }
};

using Implementations = ::testing::Types<NaivePoly, FFTPoly>;

TYPED_TEST_SUITE(SumcheckFixture, Implementations);

/* ------------------------------------------------------------------ *
 * 1. basic field sanity                                              *
 * ------------------------------------------------------------------ */
TEST(AlgebraSanity, FieldTArithmetic) {
  FieldT a(3), b(5);
  ASSERT_EQ(a + b, FieldT(8));
  ASSERT_EQ(a * b, FieldT(15));
  ASSERT_EQ((a - b) + b, a);
  ASSERT_TRUE(FieldT::one() + FieldT(-1) == FieldT::zero());
}

/* ------------------------------------------------------------------ *
 * 2. Polynomial evaluation & summation correctness                   *
 * ------------------------------------------------------------------ */
TYPED_TEST(SumcheckFixture, PolynomialEvaluateAndSumSmall) {
  for (size_t n = 1; n <= 5; ++n) {
    auto g = random_poly<TypeParam>(n, this->rng);
    FieldT lib = g.sum_over_remaining(0, {});
    FieldT brute = brute_sum_bool_cube(g);
    ASSERT_EQ(lib, brute) << "Mismatch for n = " << n;
  }
}

/* ------------------------------------------------------------------ *
 * 3. Completeness                                                    *
 * ------------------------------------------------------------------ */
TYPED_TEST(SumcheckFixture, CompletenessManyParameters) {
  std::vector<size_t> arities = {1, 2, 3, 5, 8, 10};
  for (size_t n : arities) {
    auto g = random_poly<TypeParam>(n, this->rng);
    FieldT H = g.sum_over_remaining(0, {});
    sumcheck::Prover<TypeParam> prover(g);
    sumcheck::Verifier<TypeParam> verifier(H, n);
    ASSERT_TRUE(verifier.verify(prover));
  }
}

/* ------------------------------------------------------------------ *
 * 4. Soundness                                                       *
 * ------------------------------------------------------------------ */
TYPED_TEST(SumcheckFixture, SoundnessWrongSum) {
  for (size_t n = 1; n <= 10; ++n) {
    auto g = random_poly<TypeParam>(n, this->rng);
    FieldT H = g.sum_over_remaining(0, {});
    FieldT H_fake = H + FieldT::one();
    sumcheck::Prover<TypeParam> prover(g);
    sumcheck::Verifier<TypeParam> verifier(H_fake, n);
    ASSERT_FALSE(verifier.verify(prover));
  }
}

/* ------------------------------------------------------------------ *
 * 5. Zero-polynomial corner case                                     *
 * ------------------------------------------------------------------ */
TYPED_TEST(SumcheckFixture, ZeroPolynomial) {
  size_t n = 6;
  std::vector<FieldT> zeros(1u << n, FieldT::zero());
  TypeParam g(zeros);
  FieldT H = FieldT::zero();
  sumcheck::Prover<TypeParam> prover(g);
  sumcheck::Verifier<TypeParam> verifier(H, n);
  ASSERT_TRUE(verifier.verify(prover));
  sumcheck::Verifier<TypeParam> bad_verifier(FieldT::one(), n);
  ASSERT_FALSE(bad_verifier.verify(prover));
}

/* ------------------------------------------------------------------ *
 * 6.  Performance (Naive up to 12; FFT can go higher)                *
 * ------------------------------------------------------------------ */
TYPED_TEST(SumcheckFixture, PerformanceScaling) {
  constexpr bool is_naive = std::is_same<TypeParam, NaivePoly>::value;
  size_t max_n = is_naive ? 12 : 26;

  std::cout << "\n--- " << (is_naive ? "Naive" : "FFT") << " scaling ---\n";
  for (size_t n = 4; n <= max_n; n += 2) {
    auto g = random_poly<TypeParam>(n, this->rng);
    FieldT H = g.sum_over_remaining(0, {});
    sumcheck::Prover<TypeParam> prover(g);
    sumcheck::Verifier<TypeParam> verifier(H, n);
    auto t0 = std::chrono::high_resolution_clock::now();
    ASSERT_TRUE(verifier.verify(prover));
    auto t1 = std::chrono::high_resolution_clock::now();
    double sec = std::chrono::duration<double>(t1 - t0).count();
    std::cout << "n = " << n << "  ->  " << sec << " s\n";
  }
  std::cout << "-------------------------\n";
}

/* ------------------------------------------------------------------ *
 * 7.  main()                                                         *
 * ------------------------------------------------------------------ */
int main(int argc, char **argv) {
  libff::alt_bn128_pp::init_public_params();
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}