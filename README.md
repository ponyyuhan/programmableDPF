# Programmable Distributed Point Functions (PDPF) – C++ Skeleton

This repository contains a C++ implementation of the small-domain programmable DPF construction (Boyle–Gilboa–Ishai–Kolobov, 2023) plus scaffolding for group payloads and Reed–Muller based amplification. It is organized as a library with demos/tests.

## Layout

```
pdpf/
  include/pdpf/...   // public headers
  src/...            // implementations
  tests/             // simple test driver
CMakeLists.txt       // builds libpdpf.a, pdpf_demo, pdpf_tests
```

Key modules:

- `core/`: seed/type definitions, CSPRNG wrapper, group arithmetic.
- `prg/`: length-doubling PRG (`AesCtrPrg`, AES-CTR over 128-bit seed).
- `pprf/`: GGM puncturable PRF with eval/eval_all/puncture/punc_eval.
- `pdpf/`: small-domain binary PDPF, group PDPF wrapper, amplified PDPF.
- `ldc/`: Reed–Muller LDC (Lemma 2 scaffold) for amplification.

## Build

```bash
cmake -S . -B build
cmake --build build
```

Artifacts:

- `build/libpdpf.a` – static library
- `build/pdpf_demo` – main.cpp demo (currently hello-world)
- `build/pdpf_tests` – simple reconstruction demo

Run the demo test:

```bash
./build/pdpf_tests
```

## Module notes and status

### PRG / RNG
- `AesCtrPrg` implements G: {0,1}^128 → {0,1}^256 using AES-128-CTR via CommonCrypto.
- `core::RandomDevice` uses `arc4random_buf`/`arc4random_uniform` for CSPRNG-quality bytes with rejection sampling.

### PPRF (GGM)
- Implements eval/eval_all/puncture/punc_eval over domain [M], outputs modulo N using the PRG to expand seeds.
- Tree depth is ⌈log2 M⌉, 0-based domains.

### Small-domain PDPF (Theorem 4)
- `pdpf_binary.*` implements Gen0/Gen1 and EvalAll0/1 with dummy bucket N (0-based domain, bucket N for β=0).
- `M` chosen heuristically from N, ε via `choose_M`.

### Group PDPF (Theorem 5)
- `pdpf_group.*` wraps multiple binary PDPFs to support finite Abelian groups G=Z_{q1}×…×Z_{qℓ} via bit decomposition of each coordinate.
- Infinite components are not supported; payload bits are inferred from modulus bit-lengths.

### Reed–Muller LDC (Lemma 2 scaffold)
- `reed_muller_ldc.*` builds a prime field with |F| > r·σ+1, sets L=|F|^{w+1}, q=(σ+1)(rσ+1) (if unset), interpolates a degree≤r polynomial through N points (first N monomials/points), encodes C(z)(ρ,x)=ρ·P_z(x) mod p, and samples indices via random degree-σ curves with Lagrange coefficients and ρ-sharing.
- This is structurally faithful but simplified: assumes invertible interpolation matrix, prime fields (no extensions), and does not strictly enforce N≤binom(r+w,r).

### Amplified PDPF (Theorem 6 scaffold)
- `pdpf_amplified.*` uses the LDC to derive q inner PDPFs over Z_p (via PdpfGroup), aggregates EvalAll outputs mod p, and evaluates with an inner product against the LDC codeword.
- Deterministic inner seeds derived from master seed; deltas from LDC sampling.
- Security amplification parameters are not tuned for negligible error; use paper-consistent (p, r, σ, w) for stronger guarantees.

## Caveats vs. the paper

- LDC/amplification: simplified RM code, prime fields only; σ-wise independence and N≤binom(r+w,r) not enforced rigorously; no extension-field support.
- Group decomposition: handles provided cyclic moduli; no prime-power factorization/CRT packing beyond bit layout; infinite components unsupported.
- Domains are 0-based (paper uses 1-based); ensure consistent indexing in applications.
- No side-channel hardening beyond using constant-time crypto libraries.
- `main.cpp` is still the default hello-world; add PDPF demos as needed.

## Suggested next steps

1. **Paper-faithful LDC/amplification:** implement full Reed–Muller code per Lemma 2 (extension field if needed), enforce parameter constraints, and tune r, σ, w, p for negligible error per Theorem 6.
2. **Group decomposition:** add prime-power factorization and CRT-based encoding for arbitrary finite Abelian groups and subsets G′.
3. **Demos/tests:** extend `pdpf_demo` and `pdpf_tests` with PPRF, binary PDPF, group PDPF, and amplified PDPF correctness tests.
4. **Platform crypto:** if not on macOS/CommonCrypto, swap AES-CTR and CSPRNG to an available constant-time library (OpenSSL/libsodium/BoringSSL).

## Quick usage sketch (binary PDPF)

```cpp
#include "pdpf/prg/prg.hpp"
#include "pdpf/pdpf/pdpf_binary.hpp"
using namespace pdpf;

core::Seed master{};
core::RandomDevice rng;
rng.random_seed(master);
auto prg = std::make_shared<prg::AesCtrPrg>(master);

pdpf::PdpfBinary pdpf_bin(prg);
core::SecurityParams sec{128, 16, 0.25};

auto k0 = pdpf_bin.gen_offline(sec);
std::uint64_t alpha = 5; std::uint8_t beta = 1;
auto k1 = pdpf_bin.gen_online(k0, alpha, beta);

std::vector<core::GroupZ::Value> Y0, Y1;
pdpf_bin.eval_all_offline(k0, Y0);
pdpf_bin.eval_all_online(k1, Y1);
// Reconstruct: Y0[x] + Y1[x]
```

