#include <ctime>
#include <iostream>
#include <stdexcept>
#include <vector>

// Class representing an element in finite field F_P
class FieldElement {
  private:
    int value;
    int P;

  public:
    FieldElement(int val = 0, int prime = 7) : P(prime) {
        value = (val % P + P) % P; // Ensure 0 <= value < P
    }
    FieldElement operator+(const FieldElement &other) const {
        return FieldElement((value + other.value) % P, P);
    }
    FieldElement operator-(const FieldElement &other) const {
        return FieldElement((value - other.value + P) % P, P);
    }
    FieldElement operator*(const FieldElement &other) const {
        return FieldElement((static_cast<long long>(value) * other.value) % P,
                            P);
    }
    int get_value() const { return value; }
};

// Class representing a multilinear polynomial in n variables
class MultilinearPolynomial {
  public:
    int n; // Number of variables
    int P; // Field size
    std::vector<FieldElement>
        coefficients; // Size 2^n
                      // In C++, class members are initialized in the order they
                      // are declared in the class

    MultilinearPolynomial(int n_, int P_, const std::vector<int> &coeffs)
        : n(n_), P(P_) {
        if (coeffs.size() != (1u << n)) {
            throw std::invalid_argument("Coefficient vector size must be 2^n");
        }
        coefficients.reserve(1 << n);
        for (int i = 0; i < (1 << n); i++) {
            coefficients.push_back(FieldElement(coeffs[i], P));
        }
    }

    // Evaluate g(x1, ..., xn) at a point
    FieldElement evaluate(const std::vector<FieldElement> &point) const {
        if (point.size() != static_cast<size_t>(n)) {
            throw std::invalid_argument(
                "Point size must match number of variables");
        }
        FieldElement result(0, P);
        for (int mask = 0; mask < (1 << n); mask++) {
            FieldElement term = coefficients[mask];
            for (int i = 0; i < n; i++) {
                if (mask & (1 << i)) { // x_{i+1} present
                    term = term * point[i];
                }
            }
            result = result + term;
        }
        return result;
    }

    // Sum g over {0,1} for variables k+1 to n, with first k variables fixed
    FieldElement
    sum_over_remaining(int k,
                       const std::vector<FieldElement> &fixed_values) const {
        if (fixed_values.size() != static_cast<size_t>(k)) {
            throw std::invalid_argument("Fixed values size must match k");
        }
        int remaining = n - k;
        FieldElement sum(0, P);
        for (int mask = 0; mask < (1 << remaining); mask++) {
            std::vector<FieldElement> point = fixed_values;
            for (int i = 0; i < remaining; i++) {
                // Construct the remaining points' combinations
                int val = (mask & (1 << i)) ? 1 : 0;
                point.push_back(FieldElement(val, P));
            }
            sum = sum + evaluate(point); // One combination
        }
        return sum;
    }
};

// Class representing a linear polynomial p(x) = c0 + c1 * x
class LinearPolynomial {
  public:
    FieldElement c0, c1;
    int P; // Add field modulus
    LinearPolynomial(FieldElement c0_, FieldElement c1_, int P_)
        : c0(c0_), c1(c1_), P(P_) {}
    FieldElement evaluate(const FieldElement &x) const {
        // Optional: Check field consistency
        return c0 + c1 * x;
    }
    FieldElement sum_over_binary() const {
        FieldElement zero(0, P);
        FieldElement one(1, P);
        return evaluate(zero) + evaluate(one);
    }
};

class Prover {
  public:
    MultilinearPolynomial g;
    int n;
    int P;
    Prover(const MultilinearPolynomial &g_) : g(g_), n(g_.n), P(g_.P) {}

    // Compute the linear polynomial for the next variable
    LinearPolynomial
    compute_next_linear(const std::vector<FieldElement> &fixed_values) {
        int k = fixed_values.size();
        if (k >= n) {
            throw std::invalid_argument("Too many fixed values");
        }
        FieldElement zero(0, P);
        FieldElement one(1, P);
        // Define p(x) = c0 + c1 * x
        std::vector<FieldElement> fixed_zero = fixed_values;
        fixed_zero.push_back(zero); // [r1, r2, ..., rk, 0]
        // Evaluate p(x) at x = 0, p0 = c0
        FieldElement p0 = g.sum_over_remaining(k + 1, fixed_zero);
        std::vector<FieldElement> fixed_one = fixed_values;
        fixed_one.push_back(one); // [r1, r2, ..., rk, 1]
        // Evaluate p(x) at x = 1, p1 = c0 + c1
        FieldElement p1 = g.sum_over_remaining(k + 1, fixed_one);
        FieldElement c0 = p0;
        FieldElement c1 = p1 - p0;
        return LinearPolynomial(c0, c1, P); // Pass P
    }
};

class Verifier {
  private:
    int P;
    FieldElement H;
    int n;

  public:
    Verifier(int P_, FieldElement H_, int n_) : P(P_), H(H_), n(n_) {}
    bool verify(Prover &prover) {
        std::vector<FieldElement> fixed_values;
        FieldElement current_claim = H;
        for (int round = 0; round < n; round++) {
            LinearPolynomial p = prover.compute_next_linear(fixed_values);
            FieldElement sum_p = p.sum_over_binary();
            if (sum_p.get_value() != current_claim.get_value()) {
                std::cout << "Round " << round + 1
                          << " check failed: sum_p = " << sum_p.get_value()
                          << " != current_claim = " << current_claim.get_value()
                          << "\n";
                return false;
            }
            int r_val = rand() % P;
            FieldElement r(r_val, P);
            fixed_values.push_back(r);
            current_claim = p.evaluate(r);
        }
        FieldElement g_final = prover.g.evaluate(fixed_values);
        if (g_final.get_value() != current_claim.get_value()) {
            std::cout << "Final check failed: g(r1,...,rn) = "
                      << g_final.get_value()
                      << " != pn(rn) = " << current_claim.get_value() << "\n";
            return false;
        }
        return true;
    }
};