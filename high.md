## Plan 1 – Finish SUF‑based GeLU / SiLU gates (single packed PDPF per gate)

You already have:

* SUF IR + channel/packing + `PackedSufProgram` (+ packing tests)
* ARS using the packed compiler (value + sign channels)

Now we migrate GeLU (and then SiLU) to the **same pattern** described in the formal doc: one main SUF→PDPF for “ReLU + control bits + LUT index”, plus one LUT PDPF.

### 1.1 Code changes: SUF builder for GeLU

**Files to touch**

* `composite_fss/include/composite_fss/gates/gelu.hpp` (or similar name you use now)
* `composite_fss/include/composite_fss/suf.hpp` (if you need helpers)
* `composite_fss/include/composite_fss/suf_packing.hpp` (only if new channel kinds are needed)
* `composite_fss/include/composite_fss/suf_to_lut.hpp` (for LUT program)

**Target SUF layout (per scalar input x)** (matching §4.2 in the formal doc):

We want a SUF

[
F_{\text{GeLU}}(x) = \big( \text{ReLU}(x),; b_{\mathrm{relu}}(x), b_{\mathrm{in}}(x), t(x)\big)
]

where:

* `u = ReLU(x)` – arithmetic output in (\mathbb{Z}_{2^n})
* `b_relu` – 1 bit: (1[x \ge 0])
* `b_in` – 1 bit: “inside clip interval” bit (e.g. x∈[-B,B])
* `t` – 8‑bit LUT index (Sigma’s CPU GeLU uses 8‑bit index)

**Concrete implementation sketch**

In `gelu.hpp` add something like:

```cpp
struct GeLUParams {
    unsigned n;        // ring bits
    unsigned f;        // frac bits
    // maybe m_eff etc if you use effective bitwidth
    std::array<uint64_t, 256> lut;  // delta(t) * 2^f, in Z_{2^n}
};

struct GeLUKeyParty {
    // masks
    Share r_in;   // Z_{2^n}
    Share r_out;  // Z_{2^n}
    // main SUF→PDPF program (ReLU + bits + index)
    PackedSufProgram main_prog;
    // LUT PDPF key
    PdpfProgram lut_prog;
    // shared constant c: the +c_b in the spec
    Share c;     // Z_{2^n}
};
```

Then a helper:

```cpp
SufProgram build_gelu_suf(const GeLUParams& p, const Share& r_in, const Share& r_out);
```

Inside `build_gelu_suf`:

* Create channels:

  ```cpp
  auto ch_val = SufChannelId{"relu_val"};  // kind: Arith, width = n
  auto ch_relu_bit = SufChannelId{"relu_bit"}; // Bool, width=1
  auto ch_in_bit = SufChannelId{"in_bit"};     // Bool, width=1
  auto ch_index = SufChannelId{"lut_index"};   // Uint, width=8
  ```

* Express the same control flow as in Sigma’s CPU GeLU (TR + Clip + Abs + LUT index), but at SUF level:

    * Use your existing TR / Clip / Abs building blocks in SUF IR if you have them, or encode them with piecewise polynomials + comparison predicates as in §2.3.

    * Each branch (interval (I_i)) sets:

        * `P_i(x)[ch_val] = ReLU(x)` (piecewise linear)
        * `B_i(x)[ch_relu_bit] = 1[x ≥ 0]`
        * `B_i(x)[ch_in_bit] = 1[|x|\le B]`
        * `P_i(x)[ch_index] = t(x)` where (t(x)\in {0..255}) is the 8‑bit index of |Clip(y)|’s high bits, as in the GeLU spec.

    * Ensure you incorporate the masks (r_{\text{in}}, r_{\text{out}}) exactly as in §3.1: the IR should output *masked* ReLU and indices consistent with how the compiler expects them.

Finally, call the packed compiler:

```cpp
PackedSufProgram compile_gelu(const GeLUParams& p, const Share& r_in, const Share& r_out) {
    auto suf = build_gelu_suf(p, r_in, r_out);
    return compile_suf_to_packed_pdpf(suf);   // whatever entry point you already use for ARS
}
```

### 1.2 GeLU keygen / eval using the packed compiler

**Keygen**

In `gelu.hpp`, define:

```cpp
std::pair<GeLUKeyParty, GeLUKeyParty>
gelu_keygen(const GeLUParams& params, PdpfBackend& backend);
```

Rough steps:

1. Sample `r_in`, `r_out` shares (reuse your deterministic sharing helpers in `sharing.hpp`).

2. Build & compile SUF:

   ```cpp
   auto main_prog = compile_gelu(params, r_in, r_out);
   ```

3. Build LUT PDPF: use `suf_to_lut.hpp` (or equivalent) to compile a 8‑bit LUT program for `T[t] = delta(t) * 2^f`, using Sigma‑compatible LUT table.

4. Sample / derive per‑party shares of `c` (constant shift in the spec).

5. Pack into `GeLUKeyParty` for each party.

**Eval**

Add something like:

```cpp
Share gelu_eval_party(
    const GeLUKeyParty& Kb,
    uint64_t x_hat,
    PdpfEngine& engine);
```

Algorithm (matches §4.2.3 of the formal doc):

1. Call the main PDPF program:

   ```cpp
   auto out = engine.eval_packed(Kb.main_prog, x_hat);
   auto u_b         = out.get_arith("relu_val");
   auto b_relu_b    = out.get_bool("relu_bit");
   auto b_in_b      = out.get_bool("in_bit");
   auto t_b         = out.get_uint("lut_index"); // 8-bit integer share
   ```

2. Reconstruct `t`:

    * Convert `t_b` shares into an 8‑bit public `t` (auth‑reconstruct or plain reconstruct depending on your mode).
    * This is allowed: `t` is an index, not sensitive, consistent with the spec.

3. Evaluate LUT PDPF:

   ```cpp
   auto delta_b = engine.eval_lut(Kb.lut_prog, t);
   ```

4. Linear layer:

   ```cpp
   Share y_hat_b = u_b - delta_b + Kb.c;
   return y_hat_b;
   ```

Combined across parties, this gives `GeLU(x) + r_out` as in the spec.

### 1.3 Mirroring for SiLU

For SiLU, do the exact same pattern, but:

* Use a larger LUT (e.g. 2¹⁰ = 1024 entries) and possibly a different clip range, as in Sigma’s SiLU.
* The SUF builder is almost identical; only LUT size and piecewise polynomial parameters differ.

I’d define a generic helper:

```cpp
enum class ActivationKind { GeLU, SiLU };

template<ActivationKind AK>
SufProgram build_activation_suf(...);   // shares most of the code
```

and re‑use it in both keygens.

### 1.4 Tests for GeLU/SiLU

**New test file**: `composite_fss/tests/test_gelu_suf.cpp`

Add several groups of tests:

1. **SUF→PDPF correctness for small n**

    * Use tiny config, e.g. `n=8, f=3` (so you can brute force domain).
    * Build `GeLUParams` with a small LUT (e.g. same formula but truncated).
    * For *all* `x ∈ [0, 2^n)`:

        * Compute clear `GeLU_clear(x)` using the same fixed‑point formula/lut as the gate.
        * Run *single‑party* SUF clear eval (you already have `suf_eval.hpp`) to get expected `(u, bits, t)`.
        * Run your packed PDPF pipeline in *local mode* (`pdpf_local.hpp`) to get decoded outputs `(u', bits', t')`.
        * Assert equality for each channel; then reconstruct the final `y_hat` and check `y_hat - r_out == GeLU_clear(x)`.

   This gives you bit‑level equivalence of SUF IR and compiled PDPF for GeLU on a small domain.

2. **Two‑party protocol correctness for realistic n**

    * Use `n=64, f=12`.

    * Sample, say, 1000 random `x ∈ Z_{2^64}`; run:

      ```cpp
      auto [K0, K1] = gelu_keygen(params, backend);
      auto x_hat = x + r_in; // using mask from K0/K1
      auto y0 = gelu_eval_party(K0, x_hat, engine);
      auto y1 = gelu_eval_party(K1, x_hat, engine);
      uint64_t y_hat = (y0 + y1) mod 2^64;
      ```

    * Recompute clear `GeLU_clear(x)` and verify

      ```cpp
      assert(y_hat == GeLU_clear(x) + r_out);
      ```

    * This test should also run under “strict no‑open” mode except for allowed indices.

3. **Regression test: old GeLU vs new SUF GeLU**

    * If you still have the old ad‑hoc GeLU gate implementation, run both pipelines on the same inputs and assert equality of outputs.
    * Once you’re confident, you can delete / gate‑off the old implementation.

Do the same for SiLU with separate tests (`test_silu_suf.cpp`).

---

## Plan 2 – SUF‑ify Reciprocal / rsqrt gates with packed PDPF

Based on §4.3 in the formal doc, RecSqrt/Recip are SUF and can be implemented as **IntervalLookup + TR + LUT**, all under one Composite‑FSS gate.

### 2.1 Code changes: RecSqrt key structure

**Files**

* `composite_fss/include/composite_fss/gates/reciprocal.hpp` or `rsqrt.hpp`
* SUF helpers if you choose to model IntervalLookup as SUF instead of a primitive

**Key structure**

```cpp
struct RecSqrtParams {
    unsigned n;
    unsigned frac;
    // parameters for IntervalLookup, etc.
    std::vector<uint64_t> lut; // 13-bit / custom float LUT
};

struct RecSqrtKeyParty {
    Share r_in;
    Share r_out;
    // PDPF for IntervalLookup: x_hat -> (e, u)
    PdpfProgram interval_prog;
    // PDPF for TR_{n, n-8}: t -> m (8-bit index)
    PdpfProgram tr_prog;
    // PDPF for LUT_{13}: p -> approx(1/sqrt(x))
    PdpfProgram lut_prog;
};
```

You can either:

* treat IntervalLookup itself as a SUF and reuse your SUF compiler; or
* keep it as a dedicated `IntervalLookup` PDPF program (which you already have in your PDPF lib).

### 2.2 Keygen / eval

**Keygen**

Given `RecSqrtParams`:

1. Sample `r_in, r_out` shares.
2. Build IntervalLookup PDPF for mapping `x_hat` to `(e, u)` as in Sigma: `x ≈ 2^e (1 + m/128)`.
3. Build TR PDPF for (\mathrm{TR}_{n,n-8}) (you already have SUF/TR gate; reuse its SUF→PDPF compiler or direct DPF).
4. Build 13‑bit LUT PDPF for the approximate 1/√x function.

**Eval**

```cpp
Share rsqrt_eval_party(const RecSqrtKeyParty& Kb,
                       uint64_t x_hat,
                       PdpfEngine& engine);
```

1. IntervalLookup:

   ```cpp
   auto interval_out = engine.eval(Kb.interval_prog, x_hat);
   auto e_b = interval_out.get_arith("e"); // exponent share
   auto u_b = interval_out.get_arith("u"); // mantissa share
   ```

   Combine across parties only when necessary (or keep shares and let linear layer handle them).

2. Multiply:

   ```cpp
   // t = x_hat * u (in secret sharing, with Beaver triple)
   Share t_b = mul_with_beaver(x_hat_share_b, u_b);
   ```

3. TR:

   ```cpp
   auto m_b = engine.eval(Kb.tr_prog, t_b); // 8-bit index share
   ```

4. Form 13‑bit index `p = extend(m)*2^6 + e` in secret sharing.

5. Evaluate LUT PDPF on `p` to get approx 1/√x:

   ```cpp
   auto y_b = engine.eval(Kb.lut_prog, p_share);
   ```

6. Add `r_out` mask appropriately and return.

### 2.3 Tests

**New test** `tests/test_rsqrtrec_suf.cpp`:

1. **Domain‑reduced exhaustive test**

    * Use smaller n (e.g., `n=12`, `frac=4`).
    * Enumerate a subset of “reasonable” positive x (e.g., 1..4095).
    * For each x:

        * Compute reference `approx_rsqrt(x)` using the same LUT + formula used to fill `RecSqrtParams::lut`.
        * Run the 2‑party protocol and check `y_hat - r_out == approx_rsqrt(x)`.

2. **Random tests @ n=64**

    * Sample random positive `x` with a bias towards ranges you care about in transformer activations.
    * Same pattern as above.

3. **Consistency with Softmax**

    * Once Softmax uses this RecSqrt/Rec, add a test that Softmax computed with this building block equals a “standalone” softmax reference (see Plan 3).

---

## Plan 3 – Make Softmax‑block fully SUF‑based and consistent

You already rewrote softmax to be masked, DReLU+Beaver for max, and no raw opens. Now we want to:

1. Make the *univariate* pieces (nExp and reciprocal) use SUF+packed PDPF.
2. Clean up the Softmax‑block gate so its *only* FSS calls go through SUF‑compiled PDPF programs.

### 3.1 SUF for nExp

From the formal doc: nExp is implemented by clipping + LUT + polynomial approximation on an effective bitwidth (m = n - f + 2).

Define a SUF:

[
F_{\mathrm{nExp}}(x) = (\text{exp_approx}(x), ; b_{\text{clip}}(x), ; t(x))
]

* `exp_approx(x)` is the fixed‑point approximate exp(x).
* `b_clip` indicates whether x is in the “active” interval.
* `t` is LUT index (bitwidth determined by your approximation, e.g., 10 or 11 bits).

Implementation is identical to GeLU plan, but with exp‑specific polynomials and LUT table from Sigma.

### 3.2 Softmax‑block gate structure

In your `gates/softmax.hpp`:

* Refactor into:

  ```cpp
  struct SoftmaxBlockParams {
      unsigned n, f;
      unsigned k;  // block length
      // nExp LUT params
      ...
      // Rec params (re-use from Plan 2)
      ...
  };

  struct SoftmaxBlockKeyParty {
      // SUF keys reused across all elements of the block:
      PdpfProgram max_cmp_prog;     // or reuse an existing comparison SUF→PDPF
      GeLUKeyParty::something?      // not needed; we only need nExp, Rec
      PdpfProgram nexp_prog;
      RecSqrtKeyParty rec_key;      // from Plan 2
      // Masks r_in[i], r_out[i] for each element or a pattern for them.
      std::vector<Share> r_in;
      std::vector<Share> r_out;
  };
  ```

**Eval pipeline (per block)**

Given masked inputs (\hat{\mathbf{x}}) of length k:

1. **Max** – use your DReLU-based max protocol, but ensure its comparisons use SUF‑compiled PDPF (e.g., a shared comparison PDPF program) instead of ad‑hoc DCF.

2. **Shift** – compute (\hat{x_i'} = \hat{x_i} - \widehat{\max} + \text{mask adjust}) in secret sharing.

3. **nExp** – for each element:

   ```cpp
   y_i_b = nexp_eval_party(Kb.nexp_prog, x_i_hat_prime, engine);
   ```

   where `nexp_eval_party` is the SUF→PDPF + LUT evaluation similar to GeLU but using exp LUT.

4. **Sum (linear)** – sum all `y_i` in secret sharing (no FSS).

5. **Reciprocal** – call Rec gate (Plan 2) on the sum.

6. **Normalize** – multiply each `y_i` by `1/sum` via Beaver triples.

This matches the formal softmax decomposition: a few SUF gates + linear operations.

### 3.3 Tests

**New test** `tests/test_softmax_block_suf.cpp`:

1. **Scalar warm‑up** – for k small (e.g. 4), n=16, check that the softmax block output matches a clear softmax with the same nExp + Rec approximations.

2. **Realistic BERT‑style block** – n=64, f=12, k=128 (the config you used earlier):

    * Sample random block `x` in a realistic range (e.g., [-10, 10] scaled).
    * Run the 2‑party softmax block.
    * Run a clear float softmax on high precision, then quantize to your fixed‑point domain; verify that the difference is within the approximation error you design (e.g. ≤ 1 ULP).

3. **Regression** – compare old softmax implementation vs new SUF‑based implementation on random blocks.

---

## Plan 4 – Fused MatMul + Trunc + Nonlinearity gate

This is the “layer fusion” part of `CompositeFSS.md` §4.1:

> implement `(X,W,b) ↦ Nonlin(Trunc(XW + b))` as a *single* composite gate from the FSS standpoint.

You already have:

* SPDZ‑style matmul with Beaver triples
* A standalone TR gate + SUF nonlinearity gates

We can introduce an *internal API* that:

* keeps Z = XW + b *masked*
* passes it directly to a SUF program that knows both truncation f and nonlinearity type.

### 4.1 Code structure

New header:

* `composite_fss/include/composite_fss/gates/fused_linear_nonlin.hpp`

Add:

```cpp
enum class NonlinKind { ReluARS, GeLU, SiLU };

struct FusedLinearNonlinParams {
    unsigned n, f;
    NonlinKind kind;
    // nonlin-specific params like LUT tables
};

struct FusedLinearNonlinKeyParty {
    // 1 SUF→PDPF program that outputs ALL helper bits for TR + chosen Nonlin
    PackedSufProgram main_prog;
    // any extra LUT PDPF (for GeLU/SiLU, etc.)
    PdpfProgram extra_lut;
    // meta (e.g., f, LUT size)
    FusedLinearNonlinParams params;
};
```

### 4.2 SUF for fused gate

For each `NonlinKind`, define a SUF

[
F_{\text{fused}}(z) = \big(\text{Nonlin}(\mathrm{TR}(z)),; \text{all helper bits/indices}\big)
]

Concretely (say `kind = GeLU`):

* SUF outputs:

    * ReLU(TR(z)) (or equivalent)
    * comparisons for TR (w,t, etc.)
    * GeLU clip bit, GeLU LUT index
    * any sign bits `d` you reuse from ReluARS

This is exactly “vertically fused channels” from CompositeFSS.md §2.2.

Implementation:

* Build a SUF IR that encodes:

    * LRS/ARS/TR piecewise formulas
    * ReLU
    * CLIP + index extraction as in GeLU

* Compile *once* into `main_prog`.

Instead of calling separate TR + GeLU SUF→PDPF, you now only call `main_prog` for each output neuron.

### 4.3 Evaluation protocol

In fused linear‑nonlin eval:

1. Do `Z = XW + b` with Beaver triples as usual.
2. Keep `\hat{Z} = Z + r_in` inside the fused layer.
3. For each neuron coordinate `z_i`:

   ```cpp
   auto out = engine.eval_packed(Kb.main_prog, z_hat_i);
   // decode as Nonlin + helper bits
   auto y_i_b = reconstruct_nonlin_from_suf_output(out, ...);
   ```

This yields `Nonlin(TR(z_i)) + r_out_i`, matching the “one big gate” formal definition.

### 4.4 Tests

**New test** `tests/test_fused_linear_nonlin.cpp`:

1. **Equivalence to unfused pipeline (small sizes)**

    * Choose a tiny linear layer (e.g., input dim = 4, output dim = 3).
    * For many random X,W,b:

        * Evaluate `FusedLinearNonlin` gate.
        * Evaluate “matmul → TR → Nonlin” using the existing separate gates.
        * Check that outputs match (up to masks) for all coordinates.

2. **Random regression @ BERT head size**

    * For a BERT‑like layer (dim 768 → 768), run both fused and unfused versions on a handful of random batches; compare outputs.
    * Measure total AES calls + key sizes (simple counters) to empirically confirm the expected savings.

---

## Plan 5 – PDPF backend: towards batched / multipoint evaluation

Right now `pdpf_full_impl.hpp` wraps `pdpf::pdpf::PdpfLut` with caching and a simple loop. You eventually want something closer to the multi‑index PDPF described in CompositeFSS.md §2.1 (domain `(i,x_i)`).

Given the effort, I’d stage it:

### 5.1 Instrumentation first

Before changing algorithms, add counters:

* In `pdpf_group.hpp` or `prg.hpp`, add a global or per‑engine counter `aes_calls`.
* Increment this in the PRG/AES core.
* Expose a debug API:

  ```cpp
  struct PdpfStats {
      uint64_t aes_calls;
      size_t keys_bytes;
  };

  PdpfStats PdpfEngine::get_stats() const;
  void PdpfEngine::reset_stats();
  ```

Use it in tests/benchmarks to *verify* the theoretical AES counts from the formal doc.

### 5.2 Simple batched eval (same key, many inputs)

Add:

```cpp
std::vector<PdpfOutput> PdpfEngine::eval_many(
    const PdpfProgram& prog,
    gsl::span<const uint64_t> xs);
```

Implementation initially: just loop and call `eval` per x (for correctness). Then later:

* Optimize by sorting xs by path in the DPF tree, reusing intermediate seeds (depth‑first style, like Sigma GPU EvalAll).

This already helps Softmax and GeLU, where a whole layer uses the *same* SUF key.

### 5.3 Tests

* `pdpf/tests/test_pdpf_batched.cpp`:

    * For random `prog` and random vector of `x[i]`, assert `eval_many(prog, xs)[i] == eval(prog, xs[i])` for all i.
    * Also assert `aes_calls(eval_many)` ≤ `sum aes_calls(eval)` (when you implement the optimization).

---

## Plan 6 – Formal alignment tests: SUF→PDPF correctness harness

This is the one that really nails “is SUF → PDPF correct?” without relying on gate implementation.

Add a *generic* property test harness in `tests/test_suf_compiler_correctness.cpp`:

### 6.1 API sketch

Define a templated property test entry:

```cpp
struct SufCompilerTestCase {
    std::string name;
    std::function<SufProgram(unsigned n)> build_suf;  // builds SUF for given n
    std::function<uint64_t(uint64_t x)> eval_clear;   // reference F(x) (arith only or struct)
};

void run_suf_compiler_test(const SufCompilerTestCase& tc, unsigned n);
```

For `n` small (e.g. 6 or 7 bits):

1. Build SUF, compile to `PackedSufProgram`.
2. For all `x ∈ [0, 2^n)`:

    * Evaluate SUF IR directly with `suf_eval.hpp` – get `(arith_out, bool_out)`.
    * Evaluate PDPF locally with `pdpf_local.hpp` – decode channels via layout – get `(arith_out', bool_out')`.
    * Assert equality for all components.

Instantiate this test for:

* ReluARS SUF (already done for ARS, extend to ReluARS)
* GeLU SUF (from Plan 1)
* SiLU SUF
* nExp SUF
* Rec/rsqrt SUF (on smaller domain)

This gives you an extremely strong correctness argument directly at the SUF→PDPF level.

---

## Plan 7 – Gate regression + benchmarking

Finally, extend your tests & bench harness to match the checklist in `CompositeFSS.md` §6.

### 7.1 Gate regression tests

In `tests/test_composite_fss.cpp`:

* For each gate (ReluARS, GeLU, SiLU, Rec, rsqrt, Softmax‑block), add a test case:

    * Random inputs at `n=64,f=12`.
    * 2‑party evaluation pipeline.
    * Clear reference computed using the same fixed‑point formulas.
    * Assert masked correctness.

Wire these tests to run in CI.

### 7.2 Bench harness extension

In `bench/bench_composite_fss.cpp`:

* Add microbench functions:

    * `bench_reluars_layer(size_t m)`
    * `bench_gelu_layer(size_t m)`
    * `bench_softmax_block(size_t k)`

* For each:

    * Measure total runtime and `aes_calls`.
    * Print effective AES per gate, key size per gate, etc., so you can directly populate the “Table 1 / 2”‑style comparisons vs SHARK and Sigma.

