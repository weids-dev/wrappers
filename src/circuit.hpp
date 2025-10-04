#ifndef CIRCUIT_HPP
#define CIRCUIT_HPP

#include <cstdint>
#include <libff/algebra/curves/alt_bn128/alt_bn128_pp.hpp>
#include <unordered_set>
#include <vector>

namespace std {
template <> struct hash<libff::Fp_model<4L, libff::alt_bn128_modulus_r>> {
  size_t operator()(const libff::Fp_model<4L, libff::alt_bn128_modulus_r> &elem)
      const noexcept {
    const auto &repr = elem.mont_repr; // Direct access to limbs
    size_t hash_value = 0;
    for (size_t i = 0; i < 4; ++i) {
      hash_value ^= std::hash<mp_limb_t>{}(repr.data[i]) +
                    0x9e3779b97f4a7c15ULL + (hash_value << 6) +
                    (hash_value >> 2);
    }
    return hash_value;
  }
};
} // namespace std

enum GateType { ADD, MUL };

using FieldT = libff::Fr<libff::alt_bn128_pp>;

struct Gate {
  GateType type;  // ADD or MUL
  uint32_t o_id;  // Output index in current layer
  uint32_t i_id0; // First input index from previous layer
  uint32_t i_id1; // Second input index from previous layer
  Gate(GateType t, uint32_t o, uint32_t i0, uint32_t i1)
      : type(t), o_id(o), i_id0(i0), i_id1(i1) {}
};

struct Layer {
  uint32_t nb_vars; // Number of variables to index gates (2^nb_vars >= #gates)
  std::vector<Gate> gates; // Gates computing this layer from the previous layer

  Layer(uint32_t nv) : nb_vars(nv) {}
};

class Circuit {
public:
  std::vector<Layer> layers; // layers[i] computes layer i+1 from layer i

  Circuit() {
    // Input layer (layer 0) has no gates; add placeholder
    layers.emplace_back(Layer(0));
  }

  // Add a layer with specified number of variables and gates
  void add_layer(uint32_t nb_vars, const std::vector<Gate> &gates) {
    Layer layer(nb_vars);
    layer.gates = gates;
    // Ensure o_id values are within 2^nb_vars
    for (const auto &gate : gates) {
      if (gate.o_id >= (1u << nb_vars)) {
        throw std::invalid_argument("Gate o_id exceeds layer size");
      }
    }
    layers.push_back(layer);
  }

  // Set input layer size and values (for V_0)
  void set_input_layer(uint32_t nb_vars, const std::vector<int> &input_values) {
    if (input_values.size() > (1u << nb_vars)) {
      throw std::invalid_argument("Input values exceed 2^nb_vars");
    }
    layers[0].nb_vars = nb_vars;
  }

  // Evaluate the circuit (for testing)
  std::vector<std::vector<FieldT>>
  evaluate(const std::vector<FieldT> &input) const {
    if (input.size() != (1u << layers[0].nb_vars)) {
      throw std::invalid_argument("Input size mismatch");
    }
    std::vector<std::vector<FieldT>> layer_vals(layers.size());
    layer_vals[0] = input; // Input layer
    for (size_t i = 1; i < layers.size(); ++i) {
      uint32_t output_size = 1 << layers[i].nb_vars;
      layer_vals[i].resize(output_size, FieldT(0));
      for (const auto &gate : layers[i].gates) {
        if (gate.i_id0 >= layer_vals[i - 1].size() ||
            gate.i_id1 >= layer_vals[i - 1].size()) {
          throw std::runtime_error("Invalid input index");
        }
        if (gate.o_id >= output_size) {
          throw std::runtime_error("Invalid output index");
        }
        if (gate.type == ADD) {
          layer_vals[i][gate.o_id] =
              layer_vals[i - 1][gate.i_id0] + layer_vals[i - 1][gate.i_id1];
        } else { // MUL
          layer_vals[i][gate.o_id] =
              layer_vals[i - 1][gate.i_id0] * layer_vals[i - 1][gate.i_id1];
        }
      }
    }
    return layer_vals;
  }
};

#endif // CIRCUIT_HPP
