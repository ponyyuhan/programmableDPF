## 1. Target invariant: “fully oblivious softmax”

When `softmax_eval()` finishes, we want:

* Inputs: a vector of **masked wires**
  `x[i]` with public `x_hat[i]` and secret mask shares `r_in[i]`.
* Outputs: a vector of **shares** (or masked wires) of `y[i] = softmax(x)[i]`.
* Inside the gate:

    * Use **DReLU + select + Beaver** for max.
    * Use **SUF → PDPF unary gates** (nExp, Inv) on **masked arguments**, not on opened plain values.
    * **No `open_debug()` / no raw `Share` opens.**
      Only “FSS opens” of the form `open(z + r)` where `r` is a fresh, preprocessing mask known to FSS.

In code terms:

* No direct calls to `arith::open_debug(Share)` in `gates/softmax.hpp`.
* Any place you must open something, you first add a one‑time random mask and treat the opened value as `hat` for a PDPF gate.

---

## 2. API shape you should end up with

At the end, the softmax gate should look roughly like:

```cpp
// gates/softmax.hpp
struct SoftmaxKey {
    // per element secrets (could be shared across vectors):
    std::vector<DreluProgramId> drelu_prog;   // for pairwise comparisons
    std::vector<Share>          drelu_mask;   // masks r_diff[j] for comparisons

    std::vector<NExpProgramId>  nexp_prog;    // one per input or shared
    std::vector<Share>          nexp_mask_in; // r_in for nExp
    std::vector<Share>          nexp_mask_out;// r_out for nExp

    InvProgramId                inv_prog;
    Share                       inv_mask_in;  // r_in for Inv
    Share                       inv_mask_out; // r_out for Inv

    // truncation params, bitwidth info, k, n, f, etc.
    std::size_t k;
    unsigned    n_bits;
    unsigned    frac_bits;
};

SoftmaxKey gen_softmax_key(
    std::size_t k,
    unsigned n_bits,
    unsigned frac_bits,
    PdpfEngine& pdpf,
    BeaverPool& beaver,
    Party party);

std::vector<Share> eval_softmax(
    const SoftmaxKey& key,
    const std::vector<MaskedWire>& x,
    PdpfEngine& pdpf,
    BeaverPool& beaver,
    Party party);
```

You can later swap `std::vector<Share>` for `std::vector<MaskedWire>` if you want the output masked; the protocol is the same.

---

## 3. Step‑by‑step algorithm inside `eval_softmax`

### 3.1 Convert MaskedWire → Share for logits

Current `MaskedWire` (from your description) looks like:

```cpp
struct MaskedWire {
    u64   hat;   // public x + r_in
    Share mask;  // share of r_in
};
```

Add a helper in `wire.hpp` or `arith.hpp`:

```cpp
// Converts MaskedWire (hat, mask share) to a Share of the true value x.
inline Share wire_to_share(const MaskedWire& w, Party party) {
    // public_share(v) is: value = (party == P0 ? v : 0).
    Share hat_share = public_share(w.hat, party);
    return sub(hat_share, w.mask);  // x = hat - r_in
}
```

In `eval_softmax`:

```cpp
std::vector<Share> x_shares(k);
for (std::size_t i = 0; i < k; ++i) {
    x_shares[i] = wire_to_share(x[i], party);
}
```

Now `x_shares[i]` is the SPDZ‑style additive share of the logit.

---

### 3.2 Secure max via DReLU + Beaver select

We want a **max over secret x_shares** using your existing DReLU gate, without opening any comparisons.

#### 3.2.1 Preprocessing (Gen)

In `gen_softmax_key`:

* Precompute `k-1` DReLU masks and PDPF programs for differences:

```cpp
SoftmaxKey key;
key.k        = k;
key.n_bits   = n_bits;
key.frac_bits= frac_bits;

key.drelu_prog.resize(k - 1);
key.drelu_mask.resize(k - 1);

// For each comparison j: (we compare candidate vs x[j+1])
for (std::size_t j = 0; j < k - 1; ++j) {
    // Sample random mask r_diff[j] as a Share
    u64 r_plain = random_u64_mod_2n(n_bits);
    key.drelu_mask[j] = secret_share(r_plain, party);  // store share

    // Build a DReLU SUF for bitwidth m = effective_bitwidth(...) if you have it
    SufProgram suf = make_drelu_suf(/*n_bits or m, r_in = r_plain, maybe r_out*/);
    key.drelu_prog[j] = pdpf.make_suf_program(suf, party);
}
```

The important thing: each DReLU for `x_b - x_a` needs its own mask `r_diff` whose plaintext value is stored only in SUF/PDPF metadata, and the share lives in `SoftmaxKey`.

#### 3.2.2 Online max loop (Eval)

Work entirely in the `Share` domain and call DReLU via masked opens.

```cpp
Share x_max = x_shares[0];

for (std::size_t j = 1; j < k; ++j) {
    // diff = x[j] - x_max
    Share diff = sub(x_shares[j], x_max);  // secret

    // Mask diff: t = diff + r_diff
    Share masked_diff = add(diff, key.drelu_mask[j - 1]);

    // Open masked_diff to a public hat_diff.
    // This must be a "masked open": every caller does open(masked_diff),
    // not open(diff). Implement it with your normal share-open primitive (not debug).
    u64 hat_diff = open_share(masked_diff, party);  // SPDZ-like open

    // Build a one-use MaskedWire for diff, using r_diff mask
    MaskedWire diff_wire{hat_diff, key.drelu_mask[j - 1]};

    // Call your existing DReLU gate, which returns a Share bit b s.t. b = [diff >= 0].
    Share bit_ge = eval_drelu(key.drelu_prog[j - 1], diff_wire, pdpf, party);

    // Select new max: x_max = (1 - b)*x_max + b * x[j]
    Share delta      = sub(x_shares[j], x_max);
    Share delta_mask = beaver_mul(bit_ge, delta, beaver, party);
    x_max            = add(x_max, delta_mask);
}
```

Notes:

* `open_share()` here is semantically secure: it opens `diff + r_diff` where `r_diff` is random and never reused.
* All comparisons use DReLU via PDPF. No plaintext comparisons on `x` or `diff`.

---

### 3.3 Compute z[i] = x_max − x[i] (the input to nExp)

Use Sigma’s convention: nExp takes non‑negative `z` and computes `e^{-z}`.

```cpp
std::vector<Share> z_shares(k);
for (std::size_t i = 0; i < k; ++i) {
    z_shares[i] = sub(x_max, x_shares[i]);  // z_i = x_max - x_i >= 0
}
```

All of this is linear; no FSS or Beaver needed.

---

### 3.4 Masked nExp using SUF → PDPF

You already have:

* SUF IR (`suf.hpp`).
* A naive SUF→PDPF compiler.
* Unary gates `nExp` implemented via SUF+PdpfEngine, currently on **masked** inputs.

The missing piece is a **bridge**: given a `Share z`, feed it into the SUF/PDPF nExp gate without opening `z` itself.

#### 3.4.1 Preprocessing (Gen)

In `gen_softmax_key`:

```cpp
key.nexp_prog.resize(k);
key.nexp_mask_in.resize(k);
key.nexp_mask_out.resize(k);

for (std::size_t i = 0; i < k; ++i) {
    // Sample input mask r_in_nexp[i]
    u64 r_in_plain  = random_u64_mod_2n(n_bits);
    u64 r_out_plain = random_u64_mod_2n(n_bits);

    key.nexp_mask_in[i]  = secret_share(r_in_plain,  party);
    key.nexp_mask_out[i] = secret_share(r_out_plain, party);

    // Build SUF description for nExp with these masks.
    SufProgram suf = make_nexp_suf(n_bits, frac_bits,
                                   /*r_in*/  r_in_plain,
                                   /*r_out*/ r_out_plain);

    key.nexp_prog[i] = pdpf.make_suf_program(suf, party);
}
```

You can also share one `nExp` program with different masks if your SUF allows factoring out masks; but start with per‑element keys: it’s simpler.

#### 3.4.2 Bridge function Share→MaskedWire→SUF

Add a helper in `gates/nexp.hpp`:

```cpp
Share eval_nexp_from_share(
    const Share& z,
    const Share& r_in_share,
    const Share& r_out_share,
    PdpfProgramId prog,
    PdpfEngine& pdpf,
    BeaverPool& beaver,
    Party party)
{
    // Mask the input: z_hat_share = z + r_in
    Share z_hat_share = add(z, r_in_share);

    // Open masked input z_hat = z + r_in (public)
    u64 z_hat = open_share(z_hat_share, party);

    // Build masked wire for existing nExp gate
    MaskedWire z_wire{z_hat, r_in_share};

    // Call your existing nExp gate that consumes a MaskedWire
    Share y_hat_share = eval_nexp_masked(prog, z_wire, pdpf, party);

    // Remove the output mask: y = (y_hat - r_out)
    return sub(y_hat_share, r_out_share);
}
```

Then in `eval_softmax`:

```cpp
std::vector<Share> exp_shares(k);
for (std::size_t i = 0; i < k; ++i) {
    exp_shares[i] = eval_nexp_from_share(
        z_shares[i],
        key.nexp_mask_in[i],
        key.nexp_mask_out[i],
        key.nexp_prog[i],
        pdpf,
        beaver,
        party);
}
```

Now `exp_shares[i]` is a share of `nExp(z_i) ≈ exp(x[i] - x_max)`.

---

### 3.5 Sum denominator in the share domain

Compute `denom = Σ exp_shares[i]` purely linearly:

```cpp
Share denom = zero_share(party);
for (std::size_t i = 0; i < k; ++i) {
    denom = add(denom, exp_shares[i]);
}
```

This gives a secret share of the denominator, exactly as in Sigma (their `z` in the paper).

---

### 3.6 Masked inverse using SUF → PDPF

Same pattern as nExp.

#### 3.6.1 Preprocessing (Gen)

In `gen_softmax_key`:

```cpp
u64 r_inv_in_plain  = random_u64_mod_2n(n_bits);
u64 r_inv_out_plain = random_u64_mod_2n(n_bits);

key.inv_mask_in  = secret_share(r_inv_in_plain,  party);
key.inv_mask_out = secret_share(r_inv_out_plain, party);

// SUF for inverse on domain [1, k] with reduced bitwidth as in Sigma
SufProgram suf_inv = make_inv_suf(
    n_bits, frac_bits, /*k*/ k,
    /*r_in*/  r_inv_in_plain,
    /*r_out*/ r_inv_out_plain);

key.inv_prog = pdpf.make_suf_program(suf_inv, party);
```

#### 3.6.2 Online: eval inverse from share

Helper in `gates/inv.hpp`:

```cpp
Share eval_inv_from_share(
    const Share& denom,
    const Share& r_in_share,
    const Share& r_out_share,
    PdpfProgramId prog,
    PdpfEngine& pdpf,
    BeaverPool& beaver,
    Party party)
{
    Share denom_hat_share = add(denom, r_in_share);
    u64 denom_hat = open_share(denom_hat_share, party);

    MaskedWire denom_wire{denom_hat, r_in_share};

    Share inv_hat_share = eval_inv_masked(prog, denom_wire, pdpf, party);
    return sub(inv_hat_share, r_out_share);
}
```

In `eval_softmax`:

```cpp
Share inv_denom = eval_inv_from_share(
    denom,
    key.inv_mask_in,
    key.inv_mask_out,
    key.inv_prog,
    pdpf,
    beaver,
    party);
```

`inv_denom` is a share of approximately `1 / Σ_j exp(x[j] - x_max)`.

---

### 3.7 Normalize with Beaver + truncation

Now compute:

```cpp
// y_untr[i] = inv_denom * exp_shares[i] (fixed-point)
std::vector<Share> y_untr(k);
for (std::size_t i = 0; i < k; ++i) {
    y_untr[i] = beaver_mul(inv_denom, exp_shares[i], beaver, party);
}
```

Then apply your truncation / GapARS gate (you already have ReluARS/GapARS in `trunc.hpp`) to bring the scale from `2^{2f}` back to `2^f`:

```cpp
// Suppose you have a gate: Share trunc_gap(const Share&, ...).
std::vector<Share> y(k);
for (std::size_t i = 0; i < k; ++i) {
    y[i] = eval_gap_trunc(y_untr[i], /*params n_bits, frac_bits, etc.*/,
                          pdpf, beaver, party);
}
```

`y[i]` is now a share of softmax(x)[i] with scale `2^f`.

Return `y`.

---

## 4. How to test correctness

You want tests that check:

1. **Numerical correctness**: softmax(x) matches a cleartext reference with the same fixed‑point scaling and approximations.
2. **No raw opens**: the gate never calls `open_debug()` or reads `Share::value_internal` in the soft path.

### 4.1 Numerical tests

In `composite_fss/tests/test_composite_fss.cpp`:

1. **Test fixture** for softmax:

```cpp
TEST(Softmax, RandomVectors) {
    const unsigned n_bits   = 64;
    const unsigned f_bits   = 12;
    const std::size_t k     = 8;
    const int num_iters     = 100;

    PdpfEngineFullImpl pdpf0, pdpf1;
    BeaverPool beaver0(/*seed*/0), beaver1(/*seed*/1);
    Party p0 = Party::P0;
    Party p1 = Party::P1;

    SoftmaxKey key0 = gen_softmax_key(k, n_bits, f_bits, pdpf0, beaver0, p0);
    SoftmaxKey key1 = gen_softmax_key(k, n_bits, f_bits, pdpf1, beaver1, p1);

    for (int iter = 0; iter < num_iters; ++iter) {
        // 1. Sample random plaintext logits x[i] in a reasonable range
        std::vector<double> x_real(k);
        for (std::size_t i = 0; i < k; ++i) {
            x_real[i] = random_double_in_range(-5.0, 5.0);
        }

        // 2. Encode to fixed-point and build masked wires for each party
        std::vector<MaskedWire> wires0(k), wires1(k);
        for (std::size_t i = 0; i < k; ++i) {
            u64 x_fp = encode_fixed(x_real[i], f_bits);

            u64 r_plain = random_u64_mod_2n(n_bits);
            u64 hat     = x_fp + r_plain; // mod 2^n

            Share r0 = secret_share(r_plain, p0);
            Share r1 = secret_share(r_plain, p1);

            wires0[i] = MaskedWire{hat, r0};
            wires1[i] = MaskedWire{hat, r1};
        }

        // 3. Run softmax on both parties
        auto y0 = eval_softmax(key0, wires0, pdpf0, beaver0, p0);
        auto y1 = eval_softmax(key1, wires1, pdpf1, beaver1, p1);

        // 4. Open outputs and decode
        std::vector<double> y_sec(k);
        for (std::size_t i = 0; i < k; ++i) {
            u64 y_fp = open_share(add(y0[i], y1[i]), p0 /*or both*/);
            y_sec[i] = decode_fixed(y_fp, f_bits);
        }

        // 5. Compute reference softmax in double
        // Shift by max for numeric stability
        double max_x = *std::max_element(x_real.begin(), x_real.end());
        double sum_exp = 0.0;
        std::vector<double> y_ref(k);
        for (std::size_t i = 0; i < k; ++i) {
            double val = std::exp(x_real[i] - max_x);
            y_ref[i] = val;
            sum_exp += val;
        }
        for (std::size_t i = 0; i < k; ++i) {
            y_ref[i] /= sum_exp;
        }

        // 6. Compare: allow small approximation error (e.g. 1e-3)
        for (std::size_t i = 0; i < k; ++i) {
            EXPECT_NEAR(y_sec[i], y_ref[i], 1e-3);
        }
    }
}
```

2. Add tests for edge cases:

* Very negative logits (should give nearly one‑hot).
* All logits equal (output ~ uniform).
* Inputs at the clipping boundary for nExp.

### 4.2 “No raw open” tests

You don’t have an automatic “no‑open” checker, but you can get close:

1. **Macro guard**: make sure `arith::open_debug()` and `Share::value_internal` are only available under `COMPOSITE_FSS_INTERNAL`. In softmax code, don’t include that macro; use a separate `open_share()` function that always adds a random mask internally (or use SPDZ open semantics).

2. **Build with strict config**:

* Add a configuration where `COMPOSITE_FSS_INTERNAL` is forced to `0`.
* In that build, make sure `open_debug` and `value_internal` are either undefined or `static_assert(false)` so they cannot compile.
* Compile `composite_fss_tests` in that mode and ensure **softmax** (and its helpers) compile cleanly. If compilation fails, you know some path is still using an internal “raw open”.

This enforces that:

* DReLU in softmax only opens **masked diffs** (`diff + r_diff`).
* nExp/Inv only open **masked inputs** (`z + r_in`, `denom + r_in`).
* Denominator, max, etc. are never opened unmasked.

---

## 5. Recap: what you actually need to change

Concretely, to finish “fully oblivious softmax”:

1. **Refactor `eval_softmax`** in `gates/softmax.hpp` to:

    * Convert inputs `MaskedWire → Share`.
    * Compute max using the DReLU + Beaver pattern above.
    * Compute `z[i] = x_max − x[i]` in the share domain.
    * Call `eval_nexp_from_share` for each `z[i]`.
    * Sum `exp_shares` into `denom` as a share.
    * Call `eval_inv_from_share` for `denom`.
    * Normalize with Beaver multiplication and truncation gate.

2. **Extend `SoftmaxKey`** and `gen_softmax_key` to:

    * Include per‑comparison DReLU masks + programs.
    * Include per‑element nExp masks + programs.
    * Include Inv masks + program.
    * (Reuse your existing SUF IR for nExp/Inv.)

3. **Add Share→masked bridge helpers**:

    * `eval_nexp_from_share` in `gates/nexp.hpp`.
    * `eval_inv_from_share` in `gates/inv.hpp`.

4. **Update tests**:

    * Replace old semi‑open softmax test with the SPDZ‑style test above.
    * Add 1–2 additional stress tests for edge cases.

If you want, next time we can do the same level of detail for **SUF compiler packing/multipoint** (second bullet) and explain exactly how to redesign the SUF→PDPF compiler and PdpfEngine to emit packed helper‑bit vectors instead of scalar LUTs.
