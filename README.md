# Wrappers

**[GKR Explained: From Cubes to Curves](https://org.weids.dev/agenda/notes/gkr-sum-check-tutorial.html)**

- **What's inside:** Sum-check (multilinear), FFT-based dense-MLE eval, folding table, layer-by-layer GKR reduction.
- **Security:** research/education code; not audited; do not use in production.

## How to Build (Ubuntu/Debian)

wrappers is written in C++20, so if you build wrappers yourself, you will need a recent version of a C++ compiler and a C++ standard library.
We recommend GCC 10.2 or Clang 16.0.0 (or later) and libstdc++ 10 or libc++ 7 (or later).

### Install Dependencies

```shell
git clone --recurse-submodules https://github.com/weids-dev/wrappers.git
cd wrappers
git submodule update --init --recursive
sudo apt update
export CXXFLAGS="-DNO_PROCPS"
sudo apt install -y build-essential cmake pkg-config libgtest-dev libgmp-dev libssl-dev

# Try libproc2 first (newer distros), fall back to libprocps (22.04)
sudo apt install -y libproc2-dev || sudo apt install -y libprocps-dev
```

### Compile wrappers

```shell
mkdir build && cd build
cmake ..
make -j$(nproc)
```

Then you may run `./test_gkr_dense` and `./test_sumcheck_multilinear` for unit tests.