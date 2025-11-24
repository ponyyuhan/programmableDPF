## 1. Step 1 – Harden the “no open” discipline

**Goal:** enforce “no cleartext except in test/debug”, so you can later claim semi‑honest security cleanly.

### 1.1 Move all opens behind a controlled API

**Concrete tasks:**

1. In `arith.hpp`:

    * Make `debug_open` clearly marked as “TEST ONLY”.
    * Add a non‑debug `open` function that:

        * is only compiled in a test build (e.g., behind `#ifdef COMPOSITE_FSS_ENABLE_OPEN`), or
        * is kept in a dedicated `testing` namespace.
2. Grep the codebase for any direct `.value`/`.v` access on `Share` or `MaskedWire` **outside**:

    * `arith.hpp`
    * `beaver.hpp`
    * `wire.hpp`
    * tests
3. For each such access:

    * Replace with a ring op (`add`, `sub`, `mul_const`, etc.) **or**
    * Replace with an explicit call to `debug_open` *inside the tests only*.
4. Optionally, add a debug assertion/macro:

   ```cpp
   #ifndef COMPOSITE_FSS_INTERNAL
   #define COMPOSITE_FSS_INTERNAL
   #endif

   struct Share {
     int party;
   private:
     u64 v_;
   public:
     u64 value_internal() const {
       static_assert(/* only allowed in internal code paths */);
       return v_;
     }
     // No public getter
   };
   ```

   And only allow `value_internal()` where you define `COMPOSITE_FSS_INTERNAL`.

**Tests:**

* Add a `test_no_raw_open` that:

    * includes a gate header *without* defining `COMPOSITE_FSS_INTERNAL`,
    * tries to access `.value` and confirms it doesn’t compile (or is blocked via static_assert).

This gives you a clean separation of “MPC code” vs. “testing/simulation shortcuts”.

---

## 2. Step 2 – Align SUF IR & compiler with the *paper* definition

Right now your SUF IR/compilers are “naive” and only fully used for unary LUT‑like gates. You want them to match the formal SUF in `Formalization&protrocol.md` and be used for **all** 1D non‑linear gates.

### 2.1 Extend SUF IR to match the spec

From the MD spec, SUF descriptor is:
[
\mathsf{desc}(F) = ({\alpha_i}, {P_i}, {B_i}, r_\text{in}, r_\text{out})
]

**Concrete code tasks (in `suf.hpp`):**

1. Ensure `SufDesc` (or equivalent) contains:

    * `std::vector<u64> alpha;` // sorted boundaries
    * `std::vector<PolyVec> polys;` // `P_i` coefficients, each a vector of `u64` for each output coord
    * `std::vector<BoolExprVec> bools;` // per‑interval boolean expressions for output bits
    * `u64 r_in; u64 r_out;` // masks (eventually may move to `MaskedWire` keys; but keep here conceptually)
2. Extend `BoolExpr` to allow:

    * predicates of the forms:

        * `x < beta`
        * `x mod 2^f < gamma`
        * `MSB(x)` and `MSB(x+c)`
        * constants 0/1
    * plus AND/OR/XOR/complement combinations.
3. Add helper constructors:

    * `make_relu_suf(n, r_in, r_out, ...)`
    * `make_reluaRS_suf(n,f,...)`
    * `make_drelu_suf(...)`
    * etc., matching the formal pseudo‑code you already wrote.

### 2.2 Upgrade SUF→PDPF compiler to “single PDPF per gate”

In `suf.cpp` (or wherever you compile):

1. Implement the mapping from SUF to PDPF programs as in §3.2 of the MD: one PDPF for all comparison bits, one for interval coefficients, one for LUT if needed.

    * For now, you can keep them as separate programs but:

        * **use the new multi‑output PdpfEngine**, so you at least pack all comparison bits into one Pdpf program ID, and all polynomial coefficients for a gate into one program.
2. Introduce `SufCompiled`:

   ```cpp
   struct SufCompiled {
       PdpfProgramId cmp_prog;
       PdpfProgramId poly_prog;
       PdpfProgramId lut_prog;  // optional
       // metadata: how to interpret outputs
   };
   ```
3. Implement:

   ```cpp
   SufCompiled compile_suf_to_pdpf(const SufDesc& desc, PdpfEngine& engine);
   ```

   which:

    * computes all thresholds used in BoolExprs,
    * builds the packed comparison vector,
    * builds the multi‑point payload for coefficients/LUTs,
    * calls `engine.make_lut_program` / `make_cmp_program` appropriately.

### 2.3 Move more gates onto SUF

You already migrated ReLU, GeLU, nExp, Inv, RecSqrt. Next:

* Refactor:

    * GEZ
    * DReLU
    * LRS / ARS / ReluARS
* to:

    1. build a `SufDesc` (using your formal algebra),
    2. call `compile_suf_to_pdpf`,
    3. store `SufCompiled` in the gate key,
    4. in `eval`, call the Pdpf programs once per input and stitch outputs via linear operations + Beaver.

This brings the *implementation* in line with the *formal Composite‑FSS definition*, which will matter a lot for the paper.

---

## 3. Step 3 – Make softmax fully oblivious & SUF‑based

Softmax is currently the “ugly” piece: still has opens for max/denom and some heuristic flows. The MD spec explicitly wants Softmax‑block as a Composite‑FSS gate built out of Max + nExp + Recip + multiplies.

### 3.1 Implement a proper 2‑input Max gate on shares

**Goal:** `max(x,y)` where `x,y` are secret‑shared, using DReLU and Beaver only.

**Protocol sketch:**

* Given shares `[x], [y]`:

    1. Compute `[d] = [x - y]` using share arithmetic.
    2. Run **DReLU** on `[d]` to get `[b] = 1[x≥y]`.
    3. Compute:
       [
       [\text{max}] = [b] \cdot [x] + (1-[b])\cdot [y]
       ]
       using Beaver `mul` and `mul_const`.

**Concrete tasks:**

* In `gates/drelu.hpp`:

    * Ensure DReLU takes a `MaskedWire` input and returns a *share* `[b]` with correct masking semantics (or returns a `MaskedWire` with a Boolean payload).
    * Back DReLU by SUF→PDPF if possible, rather than ad‑hoc LUT.
* In `gates/softmax.hpp` (or a new `max.hpp`):

    * Implement `Share max(const Share& x, const Share& y, BeaverPool& beaver, PdpfEngine& eng, const DreluKey& drelu_key)` using the above protocol.
* Replace any “open diff, compare in clear” pattern with this.

### 3.2 Make nExp fully masked

You already have SUF‑based nExp, but you currently evaluate it on **opened** inputs in softmax. Fix that:

* `gates/nexp.hpp`:

    * Ensure it accepts a `MaskedWire` and returns a `MaskedWire` for `exp(x)`, with consistent masks:

        * Input wire carries `(hat_x, r_in)`,
        * SUF compiled for nExp should embed `r_in, r_out` as per your SUF spec,
        * Output wire carries `(hat_y = exp(x) + r_out, r_out)`.
* In softmax:

    * When you have `[x_i]` as masked wires, **do not open**; call `nexp_eval(wire_i, nexpk, eng, beaver)` and get masked exponentials.

### 3.3 Masked denominator & reciprocal

Softmax denominator is `Z = Σ exp(x_j)`.

**Tasks:**

1. Represent `exp(x_j)` outputs as `MaskedWire`s.
2. Sum them using share arithmetic:

    * If you keep a common output mask `r_exp` for all nExp outputs, you can sum shares directly without additional masks.
3. Pass the masked `Z` into your SUF‑based Reciprocal gate (also on `MaskedWire`):

    * Implement `reciprocal_eval(MaskedWire z, RecKey key, PdpfEngine&, BeaverPool&) -> MaskedWire`.
4. Multiply each `exp(x_i)` by `1/Z` using Beaver triples (already partially done).
5. Remove all opens in the max/exp/sum/denom path:

    * The only place you may still “open” is in tests, to check correctness.

### 3.4 Update tests

* In `test_composite_fss.cpp`:

    * For softmax tests, do:

        1. Start from clear inputs.
        2. Secret share & mask them into `MaskedWire`s.
        3. Run softmax on shares only.
        4. At the end, `debug_open` the outputs and compare to a clear softmax (maybe approximate within a tolerance).
* Add assertions that softmax does **not** call the generic `open` function except in tests (this can be done indirectly by searching for it).

After this step, you’ll have an actually SPDZ‑style, SUF+PDPF‑backed softmax that matches the spec in your MD notes.

---

## 4. Step 4 – Improve the PDPF backend from “sum-of-points” to “structured”

Right now `PdpfEngineFullImpl` builds a `pdpf_lut` per output word as a sum of point‑DPFs. It’s correct, but not scalable.

Given that `/pdpf` doesn’t have a native multipoint interface, you can still improve constant factors and structure.

### 4.1 Clarify PdpfLut’s internal representation and cache

**Tasks in `/pdpf` (pdpf_lut.hpp/cpp):**

1. Document and clean up:

    * how `pdpf_lut` stores per‑point payloads;
    * how it composes them into a single DPF key.
2. Add a small cache layer:

    * many gates (e.g., all ReLUs for a layer) will share the **same** PDPF program for helper bits; only masks differ.
    * Introduce a `ProgramCacheKey` struct:

      ```cpp
      struct ProgramCacheKey {
        size_t domain_bits;
        size_t out_words;
        // maybe a hash of the SUF descriptor, except masks
      };
      ```
    * Cache compiled PdpfGroup keys keyed by `(domain_bits, out_words, desc_hash_without_masks)`.
    * In `PdpfEngineFullImpl::make_lut_program`, check the cache and reuse compiled keys if possible, only re‑randomizing masks at the MPC layer.

This doesn’t change asymptotics, but it’s a concrete, impactful optimization and aligns with your “composite key reuse” story.

### 4.2 Batch eval more aggressively

Your `eval_batch` probably just loops over `eval`. You can do better:

1. In `PdpfEngineFullImpl::eval_batch`:

    * inspect whether the backend DPF supports vectorized evaluation (e.g., reuse intermediate node states across closely related inputs);
    * if not, at least hoist per‑program setup out of the inner loop.
2. Keep statistics in instrumentation:

    * count AES calls per `eval_batch` (approximate, even if not exact),
    * log the number of inputs per batch.

This gets you closer to the “batched PDPF” story in your CompositeFSS.md

---

## 5. Step 5 – Introduce a “Composite gate key” abstraction in code

Your MD spec talks about Composite‑FSS keys (`K_b`) that package masks, PDPF keys, and constants for one gate type.

You’re half‑way there (each gate struct holds its own PdpfProgramIds + masks), but it’s still a bit ad‑hoc.

### 5.1 Define a generic CompositeGateKey template

In `composite_fss/include/composite_fss/`, add a header `composite_gate.hpp`:

```cpp
struct PdpfProgramBundle {
    PdpfProgramId cmp_prog;
    PdpfProgramId poly_prog;
    PdpfProgramId lut_prog;
    // ...
};

struct CompositeGateKey {
    int party;
    u64 r_in;
    u64 r_out;
    PdpfProgramBundle programs;
    // small constants (LUT adjustments, etc.)
};
```

Then for each gate type τ (ReluARS, GeLU, nExp, etc.) define:

```cpp
struct ReluARSKey {
    CompositeGateKey base;
    // τ-specific metadata (e.g., f, gap, etc.)
};
```

### 5.2 Refactor gate Gen/Eval to use this

* Gen:

    * Build SUF descriptor;
    * Compile to `PdpfProgramBundle`;
    * Fill `CompositeGateKey` + τ‑specific metadata.
* Eval:

    * Take `MaskedWire` (or `Share` plus mask),
    * Call `PdpfEngine::eval`/`eval_batch` using `base.programs`,
    * Perform gate‑specific linear/Beaver arithmetic.

This makes your implementation look almost identical to the formal definitions in `Formalization&protrocol.md`, which is great for writing the paper and for reasoning about security.

---

## 6. Step 6 – Add microbench harness & collect numbers

Once steps 1–5 are done, you will have:

* fully masked SUF+PDPF gates (ReLU, ReluARS, GeLU, nExp, Inv, RecSqrt, softmax);
* hardened Share/Beaver discipline;
* a somewhat optimized PDPF backend.

Now you need **numbers** to support “2–4× keysize / AES savings vs SHARK/Sigma”, as you’ve already derived analytically.

### 6.1 Add a “bench_composite_fss” binary

In `composite_fss/`:

* Create `bench/bench_composite_fss.cpp`:

    * takes arguments: `gate_type`, `n`, `f`, maybe `k` for softmax length;
    * builds random inputs;
    * runs:

        * keygen,
        * online eval for a large batch (e.g. 10^5 gates),
    * prints:

        * PdpfEngine stats (programs, output words, eval calls),
        * Beaver triple count,
        * wall‑clock time.

### 6.2 Use instrumentation to dump per‑gate costs

* Expose functions:

  ```cpp
  PdpfStats PdpfEngine::stats() const;
  BeaverStats BeaverPool::stats() const;
  ```
* From the bench binary, dump them in a CSV‑friendly format.

Then you can later post‑process and compare to the theoretical SHARK/Sigma formulas you already derived.

---

## 7. In what order should you actually do this?

If you want a practical sequence that “moves the needle” quickly:

1. **Harden opens** (Step 1) – this cleans up security story.
2. **Align SUF/ReluARS/DReLU** (Step 2, and migrate trunc/ReluARS/DReLU to SUF).
3. **Make softmax fully oblivious** (Step 3) – big conceptual milestone.
4. **Refine PDPF backend & caching** (Step 4) – improves performance and supports the “batched composite” narrative.
5. **Introduce CompositeGateKey** (Step 5) – structural cleanup, paper‑friendly.
6. **Add benchmarks** (Step 6) – start collecting numbers.

Each of these steps can be given to Codex as a self‑contained task (“In file X, implement Y, refactor Z, add these tests…”). If you want, next time you can paste one specific gate (e.g., current `gates/softmax.hpp` or `gates/trunc.hpp`) and I can write an even more concrete “change these functions like this” diff‑style plan.
