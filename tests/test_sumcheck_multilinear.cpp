#include "sumcheck_multilinear.h"
#include <chrono>
#include <gtest/gtest.h>
#include <iostream>

// Test fixture for shared setup (optional, used for protocol tests)
class SumCheckTest : public ::testing::Test {
  protected:
    void SetUp() override {
        // Seed random number generator for reproducibility
        srand(42);
    }
};

// FieldElement Tests
TEST(FieldElementTest, Arithmetic) {
    int P = 7;
    FieldElement a(3, P);
    FieldElement b(5, P);

    // Addition: 3 + 5 = 8 ≡ 1 mod 7
    FieldElement sum = a + b;
    ASSERT_EQ(sum.get_value(), 1);

    // Subtraction: 3 - 5 = -2 ≡ 5 mod 7
    FieldElement diff = a - b;
    ASSERT_EQ(diff.get_value(), 5);

    // Multiplication: 3 * 5 = 15 ≡ 1 mod 7
    FieldElement prod = a * b;
    ASSERT_EQ(prod.get_value(), 1);

    // Edge case: P-1
    FieldElement c(P - 1, P);
    FieldElement d = c + FieldElement(1, P); // (P-1) + 1 = P ≡ 0 mod P
    ASSERT_EQ(d.get_value(), 0);
}

// MultilinearPolynomial Tests
TEST(MultilinearPolynomialTest, ConstructorInvalidSize) {
    int P = 7, n = 2;
    std::vector<int> coeffs = {1, 2, 3}; // Size 3, expected 2^2 = 4
    ASSERT_THROW(MultilinearPolynomial(n, P, coeffs), std::invalid_argument);
}

TEST(MultilinearPolynomialTest, Evaluate) {
    int P = 7, n = 2;
    // g(x1,x2) = 1 + 2*x1 + 3*x2 + 4*x1*x2
    std::vector<int> coeffs = {1, 2, 3, 4};
    MultilinearPolynomial poly(n, P, coeffs);

    std::vector<FieldElement> point00 = {FieldElement(0, P),
                                         FieldElement(0, P)};
    ASSERT_EQ(poly.evaluate(point00).get_value(), 1); // 1

    std::vector<FieldElement> point11 = {FieldElement(1, P),
                                         FieldElement(1, P)};
    ASSERT_EQ(poly.evaluate(point11).get_value(),
              (1 + 2 + 3 + 4) % P); // 10 ≡ 3 mod 7
}

TEST(MultilinearPolynomialTest, SumOverRemaining) {
    int P = 7, n = 2;
    std::vector<int> coeffs = {1, 2, 3, 4};
    MultilinearPolynomial poly(n, P, coeffs);

    // Sum over all variables
    FieldElement sum_all = poly.sum_over_remaining(0, {});
    FieldElement manual_sum(0, P);
    for (int x1 = 0; x1 <= 1; x1++) {
        for (int x2 = 0; x2 <= 1; x2++) {
            std::vector<FieldElement> point = {FieldElement(x1, P),
                                               FieldElement(x2, P)};
            manual_sum = manual_sum + poly.evaluate(point);
        }
    }
    ASSERT_EQ(sum_all.get_value(), manual_sum.get_value());

    // Sum over x2 with x1 = 2
    FieldElement r(2, P);
    FieldElement sum_x2 = poly.sum_over_remaining(1, {r});
    FieldElement manual_sum_x2 = poly.evaluate({r, FieldElement(0, P)}) +
                                 poly.evaluate({r, FieldElement(1, P)});
    ASSERT_EQ(sum_x2.get_value(), manual_sum_x2.get_value());
}

// LinearPolynomial Tests
TEST(LinearPolynomialTest, EvaluateAndSum) {
    int P = 7;
    FieldElement c0(2, P), c1(3, P);
    LinearPolynomial p(c0, c1, P);

    FieldElement x(4, P);
    FieldElement eval = p.evaluate(x); // 2 + 3*4 = 14 ≡ 0 mod 7
    ASSERT_EQ(eval.get_value(), 0);

    FieldElement sum_binary = p.sum_over_binary(); // 2 + (2+3) = 7 ≡ 0 mod 7
    ASSERT_EQ(sum_binary.get_value(), 0);
}

// Prover Tests
TEST(ProverTest, ComputeNextLinear) {
    int P = 7, n = 2;
    std::vector<int> coeffs = {1, 2, 3, 4};
    MultilinearPolynomial g(n, P, coeffs);
    Prover prover(g);

    std::vector<FieldElement> fixed = {FieldElement(2, P)};
    LinearPolynomial p = prover.compute_next_linear(fixed);
    // p(x2) = g(2,x2), check at x2=0 and x2=1
    ASSERT_EQ(p.evaluate(FieldElement(0, P)).get_value(),
              g.evaluate({FieldElement(2, P), FieldElement(0, P)}).get_value());
    ASSERT_EQ(p.evaluate(FieldElement(1, P)).get_value(),
              g.evaluate({FieldElement(2, P), FieldElement(1, P)}).get_value());
}

// Protocol Tests
TEST_F(SumCheckTest, Correctness) {
    int P = 17, n = 3;
    std::vector<int> coeffs = {1, 2, 3, 4, 5, 6, 7, 8};
    MultilinearPolynomial g(n, P, coeffs);
    FieldElement H = g.sum_over_remaining(0, {});

    Prover prover(g);
    Verifier verifier(P, H, n);
    ASSERT_TRUE(verifier.verify(prover));
}

TEST_F(SumCheckTest, Soundness) {
    int P = 17, n = 3;
    std::vector<int> coeffs = {1, 2, 3, 4, 5, 6, 7, 8};
    MultilinearPolynomial g(n, P, coeffs);
    FieldElement H_correct = g.sum_over_remaining(0, {});
    FieldElement H_wrong((H_correct.get_value() + 1) % P, P);

    Prover prover(g);
    Verifier verifier(P, H_wrong, n);

    bool result = verifier.verify(prover);
    ASSERT_FALSE(result); // Should reject in first round
}

// Parameterized Test for Different n and P
TEST(SumCheckParameterizedTest, DifferentParameters) {
    struct TestParam {
        int P;
        int n;
    };
    std::vector<TestParam> params = {{3, 1}, {5, 2}, {17, 3}, {17, 4}, {17, 5}};
    for (auto param : params) {
        int P = param.P, n = param.n;
        std::vector<int> coeffs(1 << n);
        for (int i = 0; i < (1 << n); i++) {
            coeffs[i] = rand() % P; // Random coefficients between 0 and P-1
        }
        MultilinearPolynomial g(n, P, coeffs);
        FieldElement H = g.sum_over_remaining(0, {});
        Prover prover(g);
        Verifier verifier(P, H, n);
        ASSERT_TRUE(verifier.verify(prover))
            << "Failed for P=" << P << ", n=" << n;
    }
}

// Corner Case: Zero Polynomial
TEST_F(SumCheckTest, ZeroPolynomial) {
    int P = 17, n = 2;
    std::vector<int> coeffs(1 << n, 0);
    MultilinearPolynomial g(n, P, coeffs);
    FieldElement H(0, P);

    Prover prover(g);
    Verifier verifier(P, H, n);
    ASSERT_TRUE(verifier.verify(prover));

    FieldElement H_wrong(1, P);
    Verifier verifier_wrong(P, H_wrong, n);
    ASSERT_FALSE(verifier_wrong.verify(prover));
}

TEST_F(SumCheckTest, Performance) {
    int P = 17;
    for (int n = 1; n <= 12; n++) {
        std::vector<int> coeffs(1 << n);
        for (int i = 0; i < (1 << n); i++) {
            coeffs[i] = rand() % P; // Random coefficients
        }
        MultilinearPolynomial g(n, P, coeffs);
        FieldElement H = g.sum_over_remaining(0, {});
        Prover prover(g);
        Verifier verifier(P, H, n);

        auto start = std::chrono::high_resolution_clock::now();
        bool result = verifier.verify(prover);
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed = end - start;

        std::cout << "n = " << n << ", time = " << elapsed.count() << " s\n";
        ASSERT_TRUE(result); // Ensure correctness while measuring performance
    }
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}