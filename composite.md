1. Make softmax normalization fully oblivious again (no clear reconstruction), but still compatible with the 16‑bit adapter.
2. SUF‑ify reciprocal / rsqrt for Norm (reusing the SHARK construction instead of ad‑hoc SPDZ code).
3. Turn your strict SPDZ test into a reusable harness.
4. Sketch the fused MatMul+Trunc+Nonlin + PDPF batching plan so you know what to aim for next.

---

## 1. Restore fully oblivious softmax normalization

Right now in `softmax_eval_pair` you:

* Compute max via DReLU + Beaver (share‑only ✅).
* Compute exp via nExp (share‑only ✅).
* Reconstruct `exp_hat` and `denom_hat` in the clear and normalize to a public share on party 0 ❌.

Goal: replace the clear path with:

[
y_i = \frac{\exp_i}{\sum_j \exp_j}
]

using **only**:

* the `Inv` gate on `denom`,
* Beaver multiplies,
* a fixed truncation / right‑shift.

### 1.1 Decide the fixed-point layout for inverse and product

You’re on a 16‑bit ring in the adapter test (`N_BITS = 16`, `f = 8`).

Let:

* `exp_i` be Q_f fixed point: `exp_fp ≈ exp(real) * 2^f`.
* `denom = Σ exp_i` also Q_f.
* You choose `Inv` to output Q_{f_inv}: `inv_fp ≈ (1/denom_real) * 2^{f_inv}`.

Then:

* Product scale is `exp * inv` → Q_{f + f_inv}.
* To get probabilities in Q_f, you right‑shift by:

[
\text{shift}*\text{norm} = f*\text{exp} + f_\text{inv} - f_\text{target}
]

If you set `f_target = f_exp = f`, this simplifies to:

[
\text{shift}*\text{norm} = f*\text{inv}
]

**Overflow constraint:**

We need `exp_fp * inv_fp` to fit in the 16‑bit ring:

* `exp_fp ≤ 2^f` (since `exp(-z) ≤ 1` after max‑trick).
* `denom_real ∈ [1, k]` with `k = vec_len`. So `1/denom_real ∈ [1/k, 1]`.
* `inv_fp ∈ [2^{f_inv}/k, 2^{f_inv}]`.
* Worst‑case product ≤ `2^f * 2^{f_inv} = 2^{f + f_inv}`.

To avoid wraparound, enforce:

[
f + f_\text{inv} ≤ N_BITS - 1
]

With `N_BITS=16` and `f=8`, this means `f_inv ≤ 7`. To give yourself some margin, choose e.g.:

```text
f_inv = 6
shift_norm = f_inv = 6
```

Then:

* Product scale is ≤ 2^(8+6) = 2^14 < 2^16.
* Softmax outputs are Q8 after truncation (we’ll right‑shift by 6 bits).

### 1.2 Change SoftmaxKey / keygen to carry correct inverse precision

Modify `SoftmaxKey` to store the actual `inv_f` rather than “extra” bits:

```cpp
struct SoftmaxKey {
    SoftmaxParams params;
    unsigned inv_f = 0;  // fractional bits of the inverse output
    ...
};
```

In `softmax_keygen`:

```cpp
inline SoftmaxKeyPair softmax_keygen(const SoftmaxParams &params,
                                     PdpfEngine &engine,
                                     std::mt19937_64 &rng) {
    SoftmaxKeyPair kp;
    kp.k0.params = kp.k1.params = params;

    // Choose inverse precision so that exp*inv stays below 2^{n_bits}.
    // Require params.f + inv_f <= params.n_bits - 1.
    unsigned max_inv_f =
        (params.n_bits > params.f + 1)
          ? (params.n_bits - 1 - params.f)
          : 0;
    unsigned inv_f = std::min<unsigned>(params.f, max_inv_f);
    // For N_BITS=16, f=8 this gives inv_f=7; you can cap to 6 if you want extra safety.
    inv_f = std::min<unsigned>(inv_f, 6);
    kp.k0.inv_f = kp.k1.inv_f = inv_f;

    ...
    InvGateParams ip{
        params.n_bits,
        inv_f,
        static_cast<unsigned>(
            params.vec_len * (1u << (params.f + 5))) // denom bound
    };
    auto ikeys = gen_inv_gate(ip, engine, rng);
    kp.k0.inv_key = ikeys.k0;
    kp.k1.inv_key = ikeys.k1;
    return kp;
}
```

(You can still keep `inv_extra_f` internally if other code expects it, but conceptually you want “absolute inverse fractional bits,” not “f + delta”.)

### 1.3 Re‑implement normalization with Beaver + truncation

You already have:

* `exp0[i], exp1[i]` as shares of `exp_i` (Q_f).
* `denom0, denom1` as sum of exps.

Do this in `softmax_eval_pair` *instead* of the clear reconstruction block:

```cpp
    // denom = sum exp_i (shares already in exp0/exp1)
    Share denom0 = exp0[0];
    Share denom1 = exp1[0];
    for (std::size_t i = 1; i < k; ++i) {
        denom0 = add(cfg, denom0, exp0[i]);
        denom1 = add(cfg, denom1, exp1[i]);
    }

    // Protect against denom == 0 in the rare all-underflow case.
    // Add a tiny epsilon = 1 ulp in Q_f.
    const std::uint64_t eps = 1ull << k0.params.f;
    denom0 = add_const(cfg, denom0, eps);

    // Inverse: Q_{inv_f}
    auto inv_pair =
        invgate_eval_from_share_pair(cfg, k0.inv_key, k1.inv_key,
                                     denom0, denom1, engine);
    Share inv0 = inv_pair.first;
    Share inv1 = inv_pair.second;

    // Normalize: y_i = exp_i * inv / 2^{shift_norm}
    unsigned shift_norm = k0.inv_f;
    for (std::size_t i = 0; i < k; ++i) {
        // product shares via Beaver; you can use real beaver_mul if you have per-party pools
        auto prod_pair =
            beaver_mul_sim_pair(cfg, exp0[i], exp1[i], inv0, inv1);
        Share prod0 = prod_pair.first;
        Share prod1 = prod_pair.second;

        // Truncation: use your existing LRS / ARS gate rather than local >>.
        // Conceptually: y = floor(prod / 2^{shift_norm}) in Q_f.

        // (a) Keygen side: add a TruncKey/LRSKey for 'shift_norm' to SoftmaxKey.
        // (b) Here: call trunc_eval_from_share_pair(cfg, trunc_key0, trunc_key1,
        //                                          prod0, prod1, pool0, pool1)
        // and store the result into out0.y[i], out1.y[i].
    }
```

**Important:** *do not* simply shift the two shares locally with `>>` – that’s wrong in general (truncation is not linear). You really want to pass `(prod0, prod1)` through the same truncation primitive you already have in `trunc.hpp` (your LRS/ARS IFSS gate):

* Add a `TruncParams` member to `SoftmaxParams` or compute it in `softmax_keygen`.

* Generate a single truncation key pair for the product precision (`n_bits`, shift amount = `shift_norm`).

* Store those keys in `SoftmaxKey`.

* Add a small helper in `trunc.hpp` analogous to `nexpgate_eval_from_share_pair`:

  ```cpp
  std::pair<Share,Share> lrs_eval_from_share_pair(const RingConfig &cfg,
                                                  const LrsKey &k0,
                                                  const LrsKey &k1,
                                                  const Share &x0,
                                                  const Share &x1,
                                                  PdpfEngine &engine,
                                                  BeaverPool &pool0,
                                                  BeaverPool &pool1);
  ```

* Call that from softmax normalization loop.

That gives you a fully share‑only path:

1. exp, denom by additive operations and nExp.
2. inv by Inv gate.
3. per‑coordinate normalization as Beaver product + LRS.

### 1.4 Sanity checks in the strict test

Keep `test_strict_spdz.cpp` but:

* Switch the softmax test to the oblivious normalization path (i.e., remove the clear fallback branch from `softmax_eval_pair`).
* Add extra asserts while debugging:

    * After computing `denom0,denom1`, reconstruct `denom_hat` and assert `denom_hat != 0` and `denom_hat < (1u << 15)` in the 16‑bit adapter.
    * After computing each product, reconstruct `prod_hat` and assert `(prod_hat & (1u << 15)) == 0` (no sign bit set).
    * Check that resulting `y` still matches the reference softmax within tolerance `~1e-2` or `0.1` as before.

Once this passes with `COMPOSITE_FSS_INTERNAL=0`, you can remove the clear simulation path from `softmax.hpp` entirely (or hide it behind a dedicated `#ifdef COMPOSITE_FSS_ENABLE_CLEAR_DEBUG`).

---

## 2. SUF‑ify reciprocal / rsqrt for Norm

Right now you have a separate `inv` gate plus (presumably) some Norm‑specific clear or SPDZ helper for `1/(σ + ε)` or `1/√(σ^2 + ε)`.

The SHARK paper gives you a concrete actively secure reciprocal construction built from:

* GEZ (sign bit),
* LRS (fixed‑point truncation),
* a simple (m,d) spline (piecewise constant in their reciprocal),
* and a small LUT over a mantissa.

You want to fold that into CompositeFSS as SUF gates so Norm doesn’t need any clear arithmetic or manual SPDZ code.

### 2.1 Define SUF–style reciprocal gate

Add something like `recip.hpp`:

```cpp
struct RecipParams {
    unsigned n_bits;
    unsigned f;          // input fractional bits
    unsigned f_int;      // extra internal precision (SHARK's f_int)
    double   clip_min;   // minimal |x| treated as non-zero
    double   clip_max;   // optional upper bound if you want
};

struct RecipKey {
    RecipParams params;
    // underlying sub-gates
    DreluKey  sign_key;     // for sign(x)
    LrsKey    abs_trunc_key;
    SplineKey spline_key;   // for mantissa part if you follow SHARK
    LutKey    mant_lut_key; // T[m] table
};

struct RecipKeyPair {
    RecipKey k0, k1;
};
```

Keygen side:

* Fix `n_bits=16`, `f=8` for Norm; pick a small `ℓ_m` (e.g., 7) and `f_int` (e.g., 20) as in SHARK.
* Precompute LUT `T[m] = ⌊2^{f_int} / (2^{ℓ_m} + m)⌋` for `m ∈ [0, 2^{ℓ_m})` and compile it to a PDPF LUT program.
* Build a one‑dimensional degree‑0 spline SUF (or a low‑degree polynomial, if you want better accuracy) that encodes the powers‑of‑two factor `2^{2f - e + ℓ_m}` based on the exponent `e` (see Alg. “Reciprocal” in the appendix of SHARK).
* Pack all sub‑keys into `RecipKey`.

Evaluation:

* Given shares of `x` (Norm’s variance or stddev), compute:

    1. `d = GEZ(x)` and `a = |x| = x - 2·select(d,x)` (via DReLU and Beaver as they do).
    2. Clip `a` to `(clip_min, clip_max]` in fixed‑point by SUF or simple polynomial.
    3. Convert to “floating” `(m,e)` as SHARK does: use LRS to extract exponent, then a small spline to compute `t1 = 2^{2f - e + ℓ_m}`; multiply by `a` and LRS again to get the mantissa `m`.
    4. Evaluate LUT at `m` to get `t2 ≈ 2^{f_int} / (2^{ℓ_m} + m)`.
    5. Multiply `h = t1 * t2` and LRS by `f_int` bits to get `z ≈ 1/a` in your target fixed‑point.
    6. Apply sign: `sign = 1 - 2·d`, `result = sign · z`.

All of those substeps are either:

* existing gates (GEZ, LRS, LUT) that you already have in SUF/PDPF form, or
* straightforward SUF splines with tiny `m` and `d`.

This gives you a **Recip gate** with exactly the same structure as SHARK’s gate but expressed in CompositeFSS idioms instead of their SPDZ macros.

### 2.2 Rsqrt gate for Norm

For layernorm you usually want `1/√(var + eps)`. You can get this either by:

1. **Reuse Recip**: approximate `rsqrt(x) = 1/√x` via spline directly (build a separate SUF for `1/√x`), or
2. **First approximate `1/x`, then take a polynomial for `√·` (less nice).

Simplest: just copy the Recip pattern but change the LUT table and spline:

* Replace `1/x` with `1/√x` when generating the spline coefficients / LUT entries.
* Domain: `[clip_min, clip_max]` where `clip_min ≈ eps` (e.g., `1e-3` in Q8) and `clip_max` is some upper bound for typical variances (e.g., `8` or `16`).
* Use a small `(m,d)` spline, say `m = 16, d=2`, over that domain and bake it into `rsqrt_suf.hpp`.

Then Norm becomes:

```cpp
// Given variance 'v' in Q_f
auto [inv_sqrt0, inv_sqrt1] = rsqrt_eval_from_share_pair(cfg, k0.rsqrt_key,
                                                         k1.rsqrt_key, v0, v1,
                                                         engine, pool0, pool1);
// Normalize: y = (x - mean) * inv_sqrt
auto prod_pair = beaver_mul_sim_pair(cfg, num0, num1, inv_sqrt0, inv_sqrt1);
// then a truncation to bring back to Q_f
```

### 2.3 Swap Norm to SUF gates

Once `recip` / `rsqrt` exist:

* Replace any Norm‑specific SPDZ / clear code with calls into `recip_eval_from_share_pair` or `rsqrt_eval_from_share_pair`.
* Add Norm to your strict SPDZ test binary with a similar pattern to GeLU/SiLU:

    * sample small inputs,
    * compute reference `y_ref` in double,
    * run Norm gate with PdpfAdapter + Beaver,
    * check per‑coordinate error < 0.05–0.1.

---

## 3. Factor your strict SPDZ test into a reusable harness

`test_strict_spdz.cpp` is already a good prototype. I’d turn it into a tiny “test DSL” so adding more gates is trivial.

### 3.1 Create a generic gate harness template

In `tests/strict_harness.hpp`:

```cpp
struct StrictContext {
    static constexpr unsigned N_BITS = 16;
    static constexpr unsigned F = 8;

    Ring64 ring{N_BITS};
    RingConfig cfg = make_ring_config(N_BITS);
    PdpfEngineAdapter engine{N_BITS};
    MPCContext dealer_ctx{N_BITS, 0x12345678};
    std::mt19937_64 rng{0xC0FFEEu};
};

template<typename Params, typename KeyPair,
         typename KeygenFn, typename EvalPairFn, typename RefFn, typename SampleFn>
bool run_strict_gate_test(const char *name,
                          StrictContext &ctx,
                          const Params &params,
                          KeygenFn keygen,
                          EvalPairFn eval_pair,
                          RefFn ref,
                          SampleFn sample,
                          double tolerance) {
    auto keys = keygen(params, ctx.engine, ctx.dealer_ctx);
    BeaverPool pool0(ctx.cfg, 0xA1A2u, 0);
    BeaverPool pool1(ctx.cfg, 0xA1A2u, 1);

    for (int i = 0; i < 50; ++i) {
        double x_real = sample(ctx.rng);
        // encode x_real -> fixed-point, mask, etc.
        ...
        auto [y0, y1] = eval_pair(keys, x_hat, ctx.engine, pool0, pool1);
        double y_fp = decode(ctx, y0, y1, params.f);
        double y_ref = ref(x_real);
        if (std::fabs(y_fp - y_ref) > tolerance) {
            std::cerr << name << " mismatch ...\n";
            return false;
        }
    }
    return true;
}
```

Then in `test_strict_spdz.cpp`:

* Replace the hand‑rolled GeLU/SiLU tests with calls to `run_strict_gate_test`.
* Add one for softmax (`SampleFn` samples a small vector, `RefFn` computes softmax).
* Later, add Norm, reciprocal, rsqrt with new gates.

Once this is in place, every time you tweak a gate you can add a 5‑line block to get a strict regression for it.

---

## 4. Sketch for fused MatMul+Trunc+Nonlin and PDPF batching

This is for later, but it’s helpful to have the shape in mind while you design the gates.

### 4.1 Fused MatMul + Trunc + Nonlin

Define a new composite gate that encapsulates:

```text
Y = φ( Trunc( XW + b ) )
```

for φ ∈ {ReLU, GeLU, SiLU, softmax}.

Keygen:

1. Generate Beaver matrix triples for `(X, W)` as in SHARK’s matrix FSS (they already show a 2·(d0 d1 + d1 d2 + d0 d2)(n+s)-bit triple scheme).
2. Generate a truncation key for shifting by `f_weight` bits.
3. Generate activation keys (GeLU/SilU/softmax) for the appropriate vector length.

Evaluation:

1. Use Beaver matrix triples to compute `Z = XW + b` in shares.
2. Apply truncation IFSS to each coordinate (or blockwise).
3. Feed the resulting masked wires directly into the activation gate (`gelu_eval_main`/`softmax_eval_pair` etc.) without returning control to the caller.

In code, this is mostly about building a clean API:

```cpp
FusedLinearKeyPair fused_linear_gen(const FusedLinearParams &p, ...);

std::pair<std::vector<Share>, std::vector<Share>>
fused_linear_eval_pair(const FusedLinearKeyPair &keys,
                       const std::vector<MaskedWire> &x0,
                       const std::vector<MaskedWire> &x1,
                       PdpfEngine &engine,
                       BeaverPool &pool0,
                       BeaverPool &pool1);
```

The “fusion” is in the fact that you *don’t* expose intermediate `Z` shares to the caller; they only get final activations.

### 4.2 PDPF batching

Once softmax, GeLU, SiLU, and (later) Norm/recip/rsqrt are stable:

* Exploit the fact that many gates at a layer share the **same** SUF or LUT program and differ only in:

    * the party index (0 vs 1),
    * the masked input point `x_hat`,
    * and sometimes the output channel index.

* Extend `PdpfEngine` with a batched API:

  ```cpp
  void eval_share_batch(PdpfProgramId prog,
                        int party,
                        span<const std::uint64_t> inputs,
                        span<std::uint64_t> outputs);
  ```

* In your composite gates, accumulate all masked inputs to the same SUF program across a layer and call `eval_share_batch` once, instead of per‑element `eval_share`.

This is orthogonal to the cryptographic correctness — it’s just a performance pass — so it’s safe to do after all strict tests pass.

