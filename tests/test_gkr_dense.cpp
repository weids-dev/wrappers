#include "gkr.hpp"
#include <chrono>
#include <gtest/gtest.h>
#include <iostream>
#include <random>
#include <vector>

using FieldT = wrappers::FieldT;

// Fixture for GKR tests
class GKRFixture : public ::testing::Test {
protected:
  void SetUp() override {
    // Initialize public parameters if needed
    libff::alt_bn128_pp::init_public_params();
  }
};

// Helper function to generate a random circuit with a given number of layers
// (including input layer)
Circuit generate_random_circuit(int num_layers, int nb_vars_per_layer = 2) {
  if (num_layers < 3 || (num_layers % 2 == 0)) {
    throw std::invalid_argument("Number of layers must be odd and at least 3");
  }
  Circuit c;
  int layer_size = 1 << nb_vars_per_layer;
  std::vector<int> input_int(layer_size);
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_int_distribution<> dis(0, 10); // Random small integers for input
  for (auto &e : input_int) {
    e = dis(gen);
  }
  c.set_input_layer(nb_vars_per_layer, input_int);

  // Add (num_layers - 1) layers with random gates
  for (int l = 1; l < num_layers; ++l) {
    std::vector<Gate> gates;
    std::uniform_int_distribution<> type_dis(0, 1); // 0: ADD, 1: MUL
    std::uniform_int_distribution<> id_dis(0, layer_size - 1);
    for (int o = 0; o < layer_size; ++o) {
      GateType type = (type_dis(gen) == 0) ? ADD : MUL;
      int i0 = id_dis(gen);
      int i1 = id_dis(gen);
      gates.emplace_back(type, o, i0, i1);
    }
    c.add_layer(nb_vars_per_layer, gates);
  }
  return c;
}

// BadProver for soundness tests: perturbs a specific layer's values
class BadProver : public GKRProver {
public:
  BadProver(const Circuit &c, const std::vector<FieldT> &input,
            size_t perturb_layer)
      : GKRProver(c, input, true) {
    if (perturb_layer > 0 && perturb_layer < layer_vals.size()) {
      // Perturb the first value in the specified layer
      layer_vals[perturb_layer][0] += FieldT::one();
    }
  }
};

// Basic test: operational check on a toy 3-layer circuit
// (x0,x1) ⟶ (x0+x1) ⟶ (x0+x1)^2 -> y = 49
TEST_F(GKRFixture, BasicTest) {
  std::cout << "=== Basic Test: Toy 3-Layer Circuit ===" << std::endl;
  Circuit c;
  int nb_vars = 1;                     // Size 2 for simplicity
  std::vector<int> input_int = {3, 4}; // x0=3, x1=4
  c.set_input_layer(nb_vars, input_int);

  // Layer 1: add gate o0 = i0 + i1
  std::vector<Gate> gates1 = {Gate(ADD, 0, 0, 1)};
  c.add_layer(nb_vars, gates1);

  // Layer 2: mul gate o0 = i0 * i0 (square)
  std::vector<Gate> gates2 = {Gate(MUL, 0, 0, 0)};
  c.add_layer(nb_vars, gates2);

  std::vector<FieldT> input(2);
  input[0] = FieldT(3);
  input[1] = FieldT(4);

  GKRProver prover(c, input, true);
  GKRVerifier verifier(c, input);

  auto start = std::chrono::high_resolution_clock::now();
  bool result = verifier.verify(prover);
  auto end = std::chrono::high_resolution_clock::now();
  double time_ms =
      std::chrono::duration<double, std::milli>(end - start).count();

  std::vector<FieldT> output = prover.get_output();
  std::cout << "Computed output: " << output[0] << ", " << output[1]
            << std::endl;

  std::cout << "Verification result: " << (result ? "PASS" : "FAIL")
            << std::endl;
  std::cout << "Time taken: " << time_ms << " ms" << std::endl;
  std::cout << std::endl;

  ASSERT_TRUE(result);
}

// Correctness test: aggregate checks for random circuits with odd layer counts
// (3 to 15)
TEST_F(GKRFixture, CorrectnessTest) {
  std::cout
      << "=== Correctness Test: Random Circuits with Varying Layer Counts ==="
      << std::endl;
  for (int num_layers = 3; num_layers <= 21; num_layers += 2) {
    std::cout << "--- Testing " << num_layers << " layers ---" << std::endl;
    Circuit c = generate_random_circuit(num_layers);

    int layer_size = 1 << 2; // nb_vars=2
    std::vector<int> input_int(layer_size);
    // Re-generate input to match the circuit's input
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dis(0, 10);
    for (auto &e : input_int) {
      e = dis(gen);
    }
    // Note: Since set_input_layer uses input_int, but for FieldT, we need to
    // convert
    std::vector<FieldT> input(layer_size);
    for (size_t i = 0; i < input.size(); ++i) {
      input[i] = FieldT(input_int[i]);
    }

    GKRProver prover(c, input, true);
    GKRVerifier verifier(c, input);

    auto start = std::chrono::high_resolution_clock::now();
    bool result = verifier.verify(prover);
    auto end = std::chrono::high_resolution_clock::now();
    double time_ms =
        std::chrono::duration<double, std::milli>(end - start).count();

    // Note: To log per-layer claims, the GKRVerifier would need modification
    // for verbose output. Assuming no changes, we log overall result and time.
    // For per-layer, add verbose as suggested.
    std::cout << "Verification result: " << (result ? "PASS" : "FAIL")
              << std::endl;
    std::cout << "Time taken: " << time_ms << " ms" << std::endl;
    std::cout << std::endl;

    ASSERT_TRUE(result);
  }
}

// Soundness test: aggregate checks for perturbed random circuits (expect
// failure)
TEST_F(GKRFixture, SoundnessTest) {
  std::cout << "=== Soundness Test: Perturbed Random Circuits with Varying "
               "Layer Counts ==="
            << std::endl;
  for (int num_layers = 3; num_layers <= 21; num_layers += 2) {
    std::cout << "--- Testing " << num_layers << " layers (perturbed) ---"
              << std::endl;
    Circuit c = generate_random_circuit(num_layers);

    int layer_size = 1 << 2; // nb_vars=2
    std::vector<int> input_int(layer_size);
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dis(0, 10);
    for (auto &e : input_int) {
      e = dis(gen);
    }
    std::vector<FieldT> input(layer_size);
    for (size_t i = 0; i < input.size(); ++i) {
      input[i] = FieldT(input_int[i]);
    }

    // Perturb a random intermediate layer (1 to num_layers-2)
    std::uniform_int_distribution<> perturb_dis(1, num_layers - 2);
    size_t perturb_layer = perturb_dis(gen);
    std::cout << "Perturbing layer " << perturb_layer << std::endl;

    BadProver bad_prover(c, input, perturb_layer);
    GKRVerifier verifier(c, input);

    auto start = std::chrono::high_resolution_clock::now();
    bool result = verifier.verify(bad_prover);
    auto end = std::chrono::high_resolution_clock::now();
    double time_ms =
        std::chrono::duration<double, std::milli>(end - start).count();

    // The existing std::cerr in Verifier will indicate the sum-check round
    // failure. For "why incorrect", the perturbation causes mismatch at/near
    // the perturbed layer.
    std::cout << "Verification result: "
              << (result ? "PASS (unexpected)" : "FAIL (expected)")
              << std::endl;
    std::cout << "Time taken: " << time_ms << " ms" << std::endl;
    std::cout << std::endl;

    ASSERT_FALSE(result);
  }
}

// Larger workload with increased gates per layer 
TEST_F(GKRFixture, LargeWorkloadTest) {
  int nb_vars = 10;
  std::cout << "=== Large Workload Test: Random Circuits with Wider Layers "
              " === (nb_vars=" << nb_vars << ") "
            << std::endl;
  for (int num_layers = 3; num_layers <= 27; num_layers += 2) {
    std::cout << "--- Testing " << num_layers << " layers ---"
              << std::endl;
    Circuit c = generate_random_circuit(num_layers, nb_vars);
    int layer_size = 1 << nb_vars;
    std::vector<int> input_int(layer_size);
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dis(0, nb_vars); // Small random inputs
    for (auto &e : input_int) {
      e = dis(gen);
    }
    std::vector<FieldT> input(layer_size);
    for (size_t i = 0; i < input.size(); ++i) {
      input[i] = FieldT(input_int[i]);
    }
    GKRProver prover(c, input, true);
    GKRVerifier verifier(c, input);
    auto start = std::chrono::high_resolution_clock::now();
    bool result = verifier.verify(prover);
    auto end = std::chrono::high_resolution_clock::now();
    double time_ms =
        std::chrono::duration<double, std::milli>(end - start).count();
    std::cout << "Verification result: " << (result ? "PASS" : "FAIL")
              << std::endl;
    std::cout << "Time taken: " << time_ms << " ms" << std::endl;
    std::cout << std::endl;
    ASSERT_TRUE(result);
  }
}