## 1. Lock down the invariant & API

Right now you effectively have:

* deterministic “shares” (both parties can recompute the clear value),
* softmax doing cleartext max/exp/sum,
* LUT gates returning correct values but not in a proper 2PC abstraction.

**Goal invariant for the next iteration:**

* Every *wire* is represented as `Share<T>` with:

    * `Share::party` ∈ {0,1}
    * `Share::value` ∈ Z_{2^n}
* At *all* times during Eval:

    * Each party only sees its share.
    * The only operation that reconstructs is an explicit `open()` API (used only in tests / debugging, not inside gates).

So first step: in `arith.hpp` / `beaver.hpp`, document this invariant and make sure every public function either:

* preserves it (pure local ops or Beaver‑based MPC), or
* is clearly marked as `debug_open()` and only used in tests.

---

## 2. Replace deterministic arithmetic with real Beaver arithmetic

You already have `arith.hpp` and `beaver.hpp`. Next step is to make them “real”:

### 2.1. A minimal SPDZ‑style core

In `beaver.hpp`:

* Define a reusable triple pool:

```cpp
struct BeaverTriple {
    uint64_t a, b, c; // c = a * b mod 2^n
};

class BeaverPool {
public:
    explicit BeaverPool(size_t nbits, size_t seed);
    BeaverTriple get_triple();   // offline: random a,b; c=a*b mod 2^n
private:
    // PRG keyed by seed so both parties can derive consistent triples
};
```

* For now, implement a *trusted‑dealer / PRG* triple generator:

    * Both parties seeded with the same key ⇒ they independently generate identical `(a,b,c)` sequences (this matches the preprocessing model used in Sigma).

In `arith.hpp`:

* Implement:

```cpp
Share add(const Share& x, const Share& y);
Share sub(const Share& x, const Share& y);
Share mul(const Share& x, const Share& y, BeaverPool& triples);

// utility
Share constant(uint64_t v, int party);  // P0: v, P1: 0; or vice versa
Share negate(const Share& x);
```

* **mul via Beaver triple** (standard SPDZ):

For `x,y` (each party holds `x_b, y_b`):

1. Get triple `(a,b,c)` from pool.
2. Locally compute `d_b = x_b - a`, `e_b = y_b - b`.
3. Open `d = d_0 + d_1` and `e = e_0 + e_1` via explicit `open()` API.
4. Each party sets:

   ```cpp
   z_b = c 
         + d * b    // public * secret
         + e * a
         + (party == 0 ? d*e : 0);  // one party adds d*e
   ```

Return `Share{z_b}`.

> Important constraint: **no ad‑hoc “reconstruct and re‑share” anywhere.** All multiplications must go through this path.

### 2.2 Refactor existing gates to use `Share`

Now go through each file:

* `gates/gez.hpp`, `relu.hpp`, `lrs.hpp`, `ars.hpp`, `relu_ars.hpp`, `gelu.hpp`, `nexp.hpp`, `inv.hpp`, `recsqrt.hpp`:

    * Change signatures from “deterministic shares” or raw integers to `Share`.
    * Any multiplication inside them must call `mul()` with a `BeaverPool`.
    * Any places where you reconstruct (`x0 + x1` in code) should be removed or pushed to an explicit `open_for_test()` only used in tests.

At this point you should be able to run `./build/composite_fss_tests` and see **no cleartext reconstruction** for the “normal” path.

---

## 3. Make softmax fully oblivious (still semi‑honest)

Right now softmax:

> reconstructs inputs for max/exp in clear and skips secure comparison/DReLU; masks are zeroed for simplicity.

Next step: implement a real 2PC softmax using your existing PDPF gates and the new arithmetic layer.

Let’s say you have a vector of shares `std::vector<Share> x` of length `k`.

### 3.1 Max via DReLU + select

Use your PDPF‑based DReLU gate (or GEZ) to build a secure max:

* Implement:

```cpp
Share drelu(const Share& x, PdpfEngineAdapter& pdpf);
Share select(const Share& bit, const Share& a, const Share& b, BeaverPool& triples);
// returns (bit ? a : b)
```

* Max over k elements:

```cpp
Share max_share = x[0];
for (size_t i = 1; i < k; ++i) {
    Share diff = sub(x[i], max_share);
    Share bit  = drelu(diff, pdpf);           // 1 if x[i] >= max_share
    max_share  = select(bit, x[i], max_share, triples);
}
```

No reconstruction; just gate sequence + Beaver.

### 3.2 Negative exponent via nExp LUT gate

You already have `nexp.hpp` as a unary PDPF LUT gate. Wire it to shares:

* Inputs: `x[i] - max_share` for each i.
* Call `nexp_eval(Share)` which internally:

    * uses PDPF LUT programs to get `exp(-x)` shares, as you’ve already done for scalar `nExp` tests.

Finally you get `z[i] = nExp(x[i] - max)` as shares.

### 3.3 Sum and inverse

* Sum: `sum = Σ z[i]` via `add()`.
* Inverse: use your `inv.hpp` gate on `sum` ⇒ `inv_sum`.

### 3.4 Normalize

* For each i: `y[i] = GapARS( mul(z[i], inv_sum) )` (as in Sigma’s softmax).

All of this must be implemented entirely on shares, using Beaver and PDPF LUTs; **no** open except maybe in tests.

Once done, re‑run tests and add a dedicated “softmax correctness” test:

* Reconstruct answers and compare to plain fixed‑point softmax within epsilon.

---

## 4. Clean abstraction over PDPF: from adapter → scalable engine

Your `PdpfEngineAdapter` is currently:

> a table‑based small‑domain helper rather than a scalable backend.

After step 2 & 3 are stable, the next concrete task is to make the PDPF integration match your paper abstraction:

1. Introduce an engine interface:

```cpp
class PdpfEngine {
public:
    virtual PdpfKey proggen(const FunctionDesc& desc, int party) = 0;
    virtual std::vector<uint64_t> eval(const PdpfKey&, uint64_t x) = 0;
    virtual ~PdpfEngine() = default;
};
```

2. Make `PdpfEngineAdapter` implement this, but add a second implementation that wraps the **actual** `pdpf` library with full domain support.

3. In composite gates, depend only on `PdpfEngine&` (not on the adapter), so you can later swap the engine from “toy adapter / 16‑bit LUT” to “tree‑based PDPF”.

---

## 5. Start on performance / compositional optimizations

Once you have:

* real SPDZ‑style arithmetic, and
* fully oblivious softmax,

you’re finally at a “cryptographically correct” composite layer. Then you can start doing the fun, research‑y optimizations that differentiate you from SHARK/Sigma.

Concretely:

### 5.1 Multi‑table nExp / Inv / RecSqrt

Copy Sigma’s tricks into your PDPF world:

* `nExp`: 16‑bit clipped input → 2 × 8‑bit LUTs (`T0`, `T1`) and one Beaver multiply + ARS; reduces AES calls and keysize.
* `Inv`: truncate input to `q` bits depending on softmax length, then a `q`‑bit LUT.
* `RecSqrt`: 13‑bit custom “floating” encoding (`mantissa|exponent`) → 13‑bit LUT.

You already have `nexp.hpp`, `inv.hpp`, `recsqrt.hpp` as single‑LUT gates; refactor them to:

* precompute smaller LUTs in Gen,
* orchestrate 2 LUT calls + 1 multiply (for nExp),
* expose the same Composite‑FSS API.

### 5.2 Helper‑bit pipelines / unified PDPF per gate

For gates like ReluARS / GeLU:

* Identify the `(w,t,d,...)` bit patterns (as we did in the theory).
* Write a multi‑output PDPF function that returns all helper bits at once in a product group (e.g., Z₂³ × Z_{2ⁿ}⁸),
* Evaluate it once and feed its outputs into the arithmetic layer.

This is exactly how you’ll get keysize/computation improvements over Sigma’s per‑function keys.

---

## 6. Strengthen tests as you go

At each step, extend `composite_fss_tests`:

1. **No‑open invariant tests**

    * Instrument `open()` to increment a global counter.
    * For “production” tests (not explicit debug tests), assert that the counter remains 0.

2. **Gate‑level functional tests**

    * For each gate (ReLU, LRS, ARS, ReluARS, GeLU, nExp, Inv, RecSqrt, Softmax):

        * Sample random clear `x` (respecting domain assumptions).
        * Run `Gen_τ` + `Eval_τ` to get masked outputs.
        * Reconstruct and compare with a plain C++ reference implementation (e.g., from `arith_ref.hpp`).

3. **Micro‑benchmarks**

    * Add simple timing tests for each gate type to track AES calls & communication.
    * That will later let you quantify X× savings vs SHARK/Sigma analytically *and* empirically.

---

### TL;DR concrete next steps for you *right now*

If you want a strict order:

1. **Refactor `arith.hpp` / `beaver.hpp` to real Beaver‑triple mult + additive shares; remove all “reconstruct and re‑share” from gates.**
2. **Re‑implement `softmax.hpp` as fully oblivious: max via DReLU + select, nExp/Inv via existing LUT gates, normalize via Beaver mult + ARS.**
3. Once tests pass and no hidden opens remain, move to PDPF engine abstraction and nExp/Inv/RecSqrt multi‑LUT optimizations.

If you want, next message we can drill into one specific file (e.g., `beaver.hpp` or `softmax.hpp`) and sketch the exact C++ signatures and call graph you should implement.
