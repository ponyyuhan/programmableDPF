# PDPF & Composite-FSS Library

This repo bundles two related components:

1) **pdpf/** — a C++ implementation of small-domain programmable DPFs (Boyle–Gilboa–Ishai–Kolobov, 2023) plus group payloads and Reed–Muller-based amplification scaffolding.
2) **composite_fss/** — a higher-level MPC/FSS library that builds transformer-style nonlinear gates (GeLU/SiLU, softmax, reciprocal/rsqrt, norm, truncation) on top of PDPF + Beaver triples, with strict “no-open” discipline, packed SUF→PDPF compilation, batching, and optional GPU offload.

The codebase also ships benchmarks and strict harnesses to validate correctness and measure LUT/keygen/online costs.

## Repository layout

```
pdpf/                         # Core PDPF library
  include/pdpf/               # public headers (prg, pprf, pdpf, group, ldc)
  src/                        # implementations
  tests/                      # pdpf_tests demo
composite_fss/                # Composite FSS library
  include/composite_fss/      # gates, SUF IR/packing, PDPF adapters, Beaver
  src/                        # CPU + optional CUDA backend (pdpf_engine)
  tests/                      # strict harness, packing/unit tests, softmax, etc.
  bench/                      # bench_fss and helpers
docs/                         # design notes
bench.md                      # benchmark plan/results (Composite vs SHARK vs SIGMA)
gpu.md                        # GPU offload plan and steps
CompositeFSS.md               # protocol-level write-up
Formalization&protrocol.md    # formal spec for composite gates
COMPARISON_PLAN.md            # comparison methodology
```

Key headers to know:
- `composite_fss/include/composite_fss/gates/*.hpp`: GeLU/SiLU, softmax, recip/rsqrt, norm, trunc, fused_layer.
- `composite_fss/include/composite_fss/suf*.hpp`: SUF IR, channel registry, greedy packing, LUT compiler, unpack helpers.
- `composite_fss/include/composite_fss/pdpf_adapter.hpp`: unified PdpfEngine interface with CPU backend and optional CUDA backend (see GPU section).
- `composite_fss/include/composite_fss/beaver.hpp`: Beaver pools/counters.
- `composite_fss/tests/strict_harness.hpp`: strict “no raw open” harness used by tests.

## Building

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build
```

Main artifacts:
- `build/libpdpf.a`, `build/pdpf_demo`, `build/pdpf_tests`
- `build/composite_fss_tests`, `build/composite_fss_no_raw_open`, `build/composite_fss_strict_spdz`
- `build/bench_fss` (Composite-FSS benchmark harness)

### Optional CUDA backend

Enable CUDA (LUT eval offload) with:
```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DCOMPOSITE_FSS_CUDA=ON
cmake --build build
```
At runtime set `COMPOSITE_FSS_USE_GPU=1` to route PdpfEngine through the CUDA backend (see `gpu.md` for design and steps). CPU remains the default if CUDA is off or the env var is unset.

## Running tests

```bash
# PDPF core
./build/pdpf_tests

# Composite-FSS standard suite (includes GeLU/SiLU, softmax, recip/rsqrt, norm, trunc, packing)
./build/composite_fss_tests

# Strict “no raw open” enforcement
./build/composite_fss_no_raw_open
./build/composite_fss_strict_spdz
```

Softmax/recip tests guarded by env vars:
```bash
COMPOSITE_FSS_RUN_SOFTMAX_TESTS=1 COMPOSITE_FSS_RUN_RECIP_TESTS=1 ./build/composite_fss_tests
```

GPU path (if built with CUDA):
```bash
COMPOSITE_FSS_USE_GPU=1 ./build/composite_fss_tests
```

## Benchmarks

`bench/bench_fss.cpp` emits CSV rows per gate:
```
gate,n_bits,f,dim,key_bytes,lut_bytes,triples,keygen_ms,online_ms
```

Examples:
```bash
./build/bench_fss 1000 4 4          # micro scale
./build/bench_fss 10 128 768        # model-ish scale
COMPOSITE_FSS_USE_GPU=1 ./build/bench_fss 1000 4 4   # GPU if enabled
```
See `bench.md` for the full comparison plan and recorded numbers vs SHARK/SIGMA.

## Feature highlights

- **Packed SUF→PDPF compilation**: all helper bits/indices for gates are packed into single multi-output PDPF programs; decoding via channel IDs (see `suf_packing.hpp`, `suf_unpack.hpp`).
- **Global helper + batching**: `global_helper.hpp` builds a shared helper-bit layout across gates/layers, and `suf_batched.hpp` wraps naive per-instance SUF keys so you can drive whole layers from one descriptor before moving to a shared tree.
- **Fully masked nonlinear gates**: GeLU/SiLU, softmax, reciprocal/rsqrt, norm implemented with SUF LUTs + Beaver + truncation; strict “no open” discipline enforced by separate targets.
- **Fused trunc+activation**: `gates/fused_layer.hpp` exposes a fused TR+Act helper that keeps truncation and activation in one composite step over SPDZ matmul outputs.
- **Batching**: `eval_share_batch` across PdpfEngine backends; gates batch LUT evals for better CPU/GPU throughput.
- **Bench harness**: measures LUT size, Beaver triples, keygen vs online time; CSV-friendly output for downstream analysis.
- **GPU-ready**: optional CUDA backend for PdpfEngine LUT evaluation (see `gpu.md`); CPU/GPU selectable at runtime.

## Quick PDPF usage (binary)

```cpp
#include "pdpf/prg/prg.hpp"
#include "pdpf/pdpf/pdpf_binary.hpp"
using namespace pdpf;

core::Seed master{};
core::RandomDevice rng; rng.random_seed(master);
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

For Composite-FSS gate examples, see `composite_fss/tests/test_composite_fss.cpp` and the strict harness in `tests/strict_harness.hpp`.

## Theoretical backdrop (academic program)

This code tracks the constructions formalized in the accompanying notes:

- **Small-domain PDPF** (Boyle–Gilboa–Ishai–Kolobov, 2023): Gen/Eval for binary PDPF with correctness and privacy over domains [M] (Theorem 4), plus amplification via Reed–Muller LDC (Lemma 2 scaffold → Theorem 6). Group PDPF follows Theorem 5 by decomposing into cyclic factors.
- **SUF IR and compilation**: SUF descriptors encode piecewise polynomial + Boolean expressions with masks `(r_in, r_out)`; compiled to multi-output PDPF programs with packed channels. See `Formalization&protrocol.md` for the SUF semantics and the Composite-FSS definition.
- **Composite-FSS gates**: GeLU/SiLU, softmax, reciprocal/rsqrt, truncation/ARS/DReLU are expressed as SUF programs plus linear/Beaver steps, matching the formal protocol in `CompositeFSS.md` (§2–4). Security follows the standard FSS + SPDZ hybrid: no raw opens in online paths, Beaver triples for multiplications, masks for public indices only.
- **Batching & GPU**: PDPF evaluation is embarrassingly parallel; the batching API and optional CUDA backend implement the same semantics as the CPU PdpfEngine, preserving the security model.

For precise definitions, notation, and proofs, consult:
- `Formalization&protrocol.md` — SUF, PDPF, masking, and Composite gate semantics.
- `CompositeFSS.md` — end-to-end protocol, gate definitions, and security intuition.
- `COMPARISON_PLAN.md` and `bench.md` — how we map theory to empirical comparisons.
