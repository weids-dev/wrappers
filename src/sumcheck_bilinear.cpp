#include <ctime>
#include <iostream>

// Field size: F_7
const int P = 7;

// Compute a mod P, handling negative numbers
int mod(int a) { return (a % P + P) % P; }

class BilinearPolynomial {
public:
  int a, b, c, d;
  BilinearPolynomial(int a_, int b_, int c_, int d_)
      : a(mod(a_)), b(mod(b_)), c(mod(c_)), d(mod(d_)) {}

  // g(x1, x2) = mod(a + b*x1 + c*x2 + d*x1*x2)
  int evaluate(int x1, int x2) const {
    return mod(a + mod(b * x1) + mod(c * x2) + mod(d * x1 * x2));
  }

  int compute_sum() const {
    return mod(evaluate(0, 0) + evaluate(0, 1) + evaluate(1, 0) +
               evaluate(1, 1));
  }
};

class LinearPolynomial {
public:
  int c0, c1;
  LinearPolynomial(int c0_, int c1_) : c0(mod(c0_)), c1(mod(c1_)) {}

  int evaluate(int x) const { return mod(c0 + mod(c1 * x)); }

  int sum_over_binary() const { return mod(evaluate(0) + evaluate(1)); }
};

class Prover {
public:
  BilinearPolynomial g;

  Prover(const BilinearPolynomial &g_) : g(g_) {}

  // p1(x1) = g(x1, 0) + g(x1, 1) = a + b*x1 + a + b*x1 + c + d*x1
  //        = (2a+c) + (2b+d)x1
  LinearPolynomial compute_p1() {
    int A = mod(2 * g.a + g.c);
    int B = mod(2 * g.b + g.d);
    return LinearPolynomial(A, B);
  }

  // p2(x2) = g(r1, x2) = a + b*r1 + c*x2 + d*r1*x2
  //        = (a+b*r1) + (c+d*r1)x2
  LinearPolynomial compute_p2(int r1) {
    int C = mod(g.a + mod(g.b * r1));
    int D = mod(g.c + mod(g.d * r1));
    return LinearPolynomial(C, D);
  }

  int compute_g_at(int r1, int r2) { return g.evaluate(r1, r2); }
};

class Verifier {
private:
  int P;
  int H;

public:
  Verifier(int P_, int H_) : P(P_), H(H_) {}

  bool verify(Prover &prover) {
    LinearPolynomial p1 = prover.compute_p1();
    if (p1.sum_over_binary() != H) {
      std::cout << "Round 1 check failed: sum_p1 = " << p1.sum_over_binary()
                << " != H = " << H << "\n";
      return false;
    }

    int r1 = rand() % P;
    std::cout << "Verifier challenge r1 = " << r1 << "\n";

    LinearPolynomial p2 = prover.compute_p2(r1);
    int p1_r1 = p1.evaluate(r1);
    if (p2.sum_over_binary() != p1_r1) {
      std::cout << "Round 2 check failed: sum_p2 = " << p2.sum_over_binary()
                << " != p1(r1) = " << p1_r1 << "\n";
      return false;
    }

    int r2 = rand() % P;
    std::cout << "Verifier challenge r2 = " << r2 << "\n";
    int g_final = prover.g.evaluate(
        r1, r2); // get the original g evaluate at (r1, r2) *final check
    int p2_r2 = p2.evaluate(r2);
    if (g_final != p2_r2) {
      std::cout << "Final check failed: g(r1, r2) = " << g_final
                << " != p2(r2) = " << p2_r2 << "\n";
      return false;
    }
    return true;
  }
};

int main() {
  srand(static_cast<unsigned>(time(nullptr)));

  BilinearPolynomial g(1, 2, 1, 3);
  int H = g.compute_sum(); // H = 3
  std::cout << "Correct H: " << H << "\n";

  Prover prover(g);
  Verifier verifier(7, H);
  std::cout << "Testing with correct H:\n";
  bool result = verifier.verify(prover);
  std::cout << "Result: " << (result ? "Accept" : "Reject") << "\n\n";

  Verifier verifier_wrong(7, 4);
  std::cout << "Testing with incorrect H:\n";
  result = verifier_wrong.verify(prover);
  std::cout << "Result: " << (result ? "Accept" : "Reject") << "\n";

  return 0;
}