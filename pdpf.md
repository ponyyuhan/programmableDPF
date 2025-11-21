I’ll treat what you have now as the “final skeleton” and give you a **concrete implementation plan for every remaining TODO**, so that, together with Codex, you can finish the PDPF (small‑domain, group payloads, and amplification) in one pass.

I’ll assume the file layout you described (and that I suggested earlier) is still roughly correct.

All references to the construction (Theorems, Figures, Lemmas) are to the PDPF paper you attached.

---

## 0. Global conventions to fix *before* coding

To avoid subtle bugs, fix these conventions and then implement everything consistently:

### 0.1 Indexing conventions

Use **0‑based indexing in C++**:

* DPF domain: paper’s ([N] = {1,\dots,N})
  → code: indices `x ∈ {0,…,N-1}` and vectors `Y.size() == N`.

* PPRF input domain: ([M]) → code: `ℓ ∈ {0,…,M-1}`.

* PPRF output for the PDPF construction:

    * Paper uses ([N+1] = {1,…,N+1}).
    * Code: use `range_size = N + 1`, values `y ∈ {0,…,N}`.
    * Index `0..N-1` are “real buckets” (point function coordinates).
    * Index `N` is the dummy bucket (paper’s `N+1`).

So:

```text
paper bucket: 1   2   ... N   N+1
code bucket:  0   1   ... N-1  N   (dummy)
```

Point α in the paper:
→ code: `alpha` as a `uint64_t` with `0 ≤ alpha < N`.

### 0.2 Seed → integer mapping

Define **one single mapping** `Seed -> uint64_t` and use it everywhere:

* `pprf::Pprf::seed_to_uint64`
* `pdpf::PdpfBinary::seed_to_uint64`
* any KDF/KDF-like use where you need a non‑uniform integer.

Recommended mapping:

```cpp
uint64_t seed_to_uint64(const core::Seed &s) {
    uint64_t v = 0;
    // little-endian: first 8 bytes
    for (int i = 0; i < 8; ++i) {
        v |= static_cast<uint64_t>(s[i]) << (8 * i);
    }
    return v;
}
```

For *all* modulo operations, always do `seed_to_uint64(seed) % modulus`.

### 0.3 Types & ranges

* `core::GroupZ::Value` = `int64_t` is fine:
  it must hold counts up to `M`, and `M` will be ≤ polynomial in N.
* For group moduli (`GroupDescriptor.moduli`), use `uint64_t`, but do modular reduction using `int64_t` intermediate and then fix sign.

---

## 1. PRG: AES‑CTR (or ChaCha) length‑doubling

**File:** `src/pdpf/prg/prg.cpp`
**Class:** `AesCtrPrg`

### 1.1 Design

We want a length‑doubling PRG:

[
G : {0,1}^{128} \rightarrow {0,1}^{256}
]

Implementation idea:

* Treat `master_key_` as an AES‑128 key.
* For input `seed` (16 bytes) treat it as the **IV/nonce**.
* Run AES‑CTR with:

    * counter = 0 → first 16 bytes (call this `block0`)
    * counter = 1 → next 16 bytes (`block1`)
* Output `block0` as `left`, `block1` as `right`.

### 1.2 Implementation plan

1. Add a **thin AES abstraction**:

    * Either:

        * integrate OpenSSL: `EVP_CIPHER_CTX` with `EVP_aes_128_ctr`, or
        * integrate libsodium: `crypto_stream_aes128ctr`.
    * Or, if you want a pure C++ fallback for research only, you can use a small constant‑time AES implementation (e.g., a known public-domain AES), but in practice you’ll use a library.

2. In `AesCtrPrg::AesCtrPrg`:

    * Initialize the AES key schedule or library context from `master_key_`.

3. In `AesCtrPrg::expand`:

    * Prepare a 16‑byte IV = `seed`.
    * Prepare two 16‑byte counters: 0 and 1 (put in last 4 or 8 bytes).
    * Encrypt `IV || counter0` → `left`.
    * Encrypt `IV || counter1` → `right`.
    * Ensure **no branches on secret key bits** inside AES (library should handle that).

For now, you can let Codex fill actual OpenSSL calls once you write the signatures and comments clearly.

---

## 2. PPRF: GGM eval / puncture / punc_eval

**File:** `src/pdpf/pprf/pprf.cpp`
**Class:** `pdpf::pprf::Pprf`

The construction follows Theorem 3 (GGM PPRF).

### 2.1 `seed_to_uint64`

Implement as per global mapping (section 0.2). Expose it as a private method.

### 2.2 `eval(const PprfKey &k, uint64_t x)`

Goal: implement GGM PRF:

[
y = \text{Eval}(k, M, N, x) \in [N]
]

where `M = k.params.M`, `N = k.params.N`.

Algorithm:

1. Assert `x < M`.

2. `d = tree_depth(M)` (already implemented).

3. `Seed node = k.root_seed;`

4. For level `lvl = 0 .. d-1`:

    * `expand(node, left, right)` via `seed_to_children`.
    * Bit index (MSB-first):

      ```cpp
      uint32_t shift = d - 1 - lvl;
      uint8_t bit = (x >> shift) & 1;
      node = (bit == 0) ? left : right;
      ```

5. Convert `node` to integer:

   ```cpp
   uint64_t raw = seed_to_uint64(node);
   uint64_t y = raw % k.params.N;
   return y;
   ```

This is O(log M) PRG calls per `eval`.

**Note about M not being a power of 2**: using `d = ceil(log2 M)` and interpreting `x` as a `d`‑bit string is standard and fine—the unused leaves correspond to inputs `x ≥ M`, which we never call.

### 2.3 `eval_all(const PprfKey &k, std::vector<uint64_t> &out)`

Simplest correct implementation (not optimal but OK):

```cpp
out.resize(k.params.M);
for (uint64_t x = 0; x < k.params.M; ++x) {
    out[x] = eval(k, x);
}
```

This is O(M log M) instead of O(M), which is fine for a first complete implementation; if you later care about matching the PRG call counts in Theorem 3 exactly, you can switch to building the entire GGM tree in O(M).

### 2.4 `puncture(const PprfKey &k, uint64_t xp)`

We use the simplest GGM puncturing scheme:

Key idea:

* Let `d = tree_depth(M)`.
* At each level `lvl`, along path to `xp`, we store the **sibling seed**.

Algorithm:

1. Assert `xp < k.params.M`.
2. `PprfPuncturedKey kp; kp.params = k.params; kp.xp = xp;`
3. `Seed node = k.root_seed;`
4. For `lvl = 0..d-1`:

    * `Seed left,right; seed_to_children(node,left,right);`
    * `shift = d - 1 - lvl; bit = (xp >> shift) & 1;`
    * If `bit == 0`:

        * `kp.co_path_seeds.push_back(right);`
        * `node = left;`
    * Else:

        * `kp.co_path_seeds.push_back(left);`
        * `node = right;`
5. Return `kp`.

So `kp.co_path_seeds[lvl]` is the seed of the sibling node at depth `lvl`.

### 2.5 `punc_eval(const PprfPuncturedKey &kp, uint64_t x)`

We reconstruct from the **earliest divergence** between `x` and `xp`.

Algorithm:

1. Assert `x < kp.params.M`.

2. If `x == kp.xp`, return `PUNCTURED_SENTINEL`.

3. `d = tree_depth(kp.params.M)`.

4. Compute the **first differing bit index** `i` in `[0, d-1]`:

   ```cpp
   uint64_t diff = x ^ kp.xp;
   if (diff == 0) return PUNCTURED_SENTINEL; // already handled, but be safe
   // find most significant differing bit (MSB first)
   uint32_t i = d; // sentinel
   for (uint32_t lvl = 0; lvl < d; ++lvl) {
       uint32_t shift = d - 1 - lvl;
       uint8_t bit_x = (x  >> shift) & 1;
       uint8_t bit_p = (kp.xp >> shift) & 1;
       if (bit_x != bit_p) { i = lvl; break; }
   }
   // i must be < d
   ```

5. Starting seed: `Seed node = kp.co_path_seeds[i];`
   This seed belongs to the node with prefix equal to `x`’s prefix of length `i+1`:

    * Because at level `i` we take the sibling of xp’s path, which is exactly x’s path at the divergence.

6. For subsequent levels `lvl = i+1..d-1`:

   ```cpp
   for (uint32_t lvl2 = i + 1; lvl2 < d; ++lvl2) {
       Seed left, right;
       seed_to_children(node, left, right);
       uint32_t shift = d - 1 - lvl2;
       uint8_t bit = (x >> shift) & 1;
       node = (bit == 0) ? left : right;
   }
   ```

7. Finally:

   ```cpp
   uint64_t raw = seed_to_uint64(node);
   uint64_t y = raw % kp.params.N;
   return y;
   ```

### 2.6 `punc_eval_all`

Like `eval_all`:

```cpp
out.resize(kp.params.M);
for (uint64_t x = 0; x < kp.params.M; ++x) {
    out[x] = punc_eval(kp, x);
}
```

---

## 3. Binary PDPF (Theorem 4)

**File:** `src/pdpf/pdpf_binary.cpp`
**Class:** `pdpf::pdpf::PdpfBinary`

### 3.1 `seed_to_uint64`

Implement identically to `Pprf::seed_to_uint64` (copy‑paste the logic or factor out a common helper).

### 3.2 Consistent target / bucket logic in `gen_online`

Recall our conventions:

* PPRF output domain size: `range = N + 1`.
* Values: `0..N` where:

    * `0..N-1` = actual domain points.
    * `N` = dummy bucket.

We choose:

* For payload `β = 1`: we want to *remove one ball from bucket α*.
* For `β = 0`: we want to *remove one ball from dummy bucket N*.

So in `gen_online`:

1. Parameters:

   ```cpp
   uint64_t N = k0.params.sec.domain_size_N;
   uint64_t M = k0.params.M;
   ```

2. Expand `k_star`:

   ```cpp
   core::Seed s_seed{}, k_pprf_seed{};
   prg_->expand(k0.k_star, s_seed, k_pprf_seed);
   ```

3. Build PPRF key:

   ```cpp
   pprf::PprfParams pp{M, N + 1};
   pprf::PprfKey pkey{k_pprf_seed, pp};
   pprf::Pprf pprf(prg_);
   ```

4. Compute shift:

   ```cpp
   uint64_t s_val = seed_to_uint64(s_seed) % (N + 1);
   ```

5. Find candidate set:

   ```cpp
   std::vector<uint64_t> candidates;
   candidates.reserve(M / (N + 1) + 1);

   uint64_t target = (beta == 1) ? alpha : N;  // α in [0..N-1], dummy = N

   for (uint64_t ell = 0; ell < M; ++ell) {
       uint64_t val = pprf.eval(pkey, ell);  // 0..N
       uint64_t shifted = (val + s_val) % (N + 1);
       if (shifted == target) {
           candidates.push_back(ell);
       }
   }
   ```

6. Robustness on empty `candidates`:

    * Theorem 4 allows treating this as a negligible failure event.

    * For a *first implementation*, **throw** with a clear error:

      ```cpp
      if (candidates.empty()) {
          throw std::runtime_error(
              "PdpfBinary::gen_online: candidate set L is empty; "
              "choose larger M or treat this event as failure.");
      }
      ```

    * If you want to implement “lazy Gen” as in Proposition 4, define a compile‑time switch:

        * Instead of scanning all ℓ, repeatedly sample random ℓ and test until success or `T` tries.
        * Fail with probability ≤ 2^{-(N+1)}.

7. Pick random ℓ from `candidates` using `RandomDevice::random_u64`.

8. Puncture PPRF:

   ```cpp
   auto kp = pprf.puncture(pkey, ell_star);
   ```

9. Fill `OnlineKey`:

   ```cpp
   OnlineKey k1;
   k1.kp = std::move(kp);
   k1.s  = s_seed;
   k1.params = k0.params;
   return k1;
   ```

### 3.3 `eval_all_offline`

Re-derive `s` and `k_pprf_seed` from `k_star`:

```cpp
core::Seed s_seed{}, k_pprf_seed{};
prg_->expand(k0.k_star, s_seed, k_pprf_seed);
uint64_t s_val = seed_to_uint64(s_seed) % (N + 1);
```

Build PPRF key & object as above.

Initialize `Y`:

```cpp
Y.assign(N, 0);  // indices 0..N-1
```

Loop:

```cpp
for (uint64_t ell = 0; ell < M; ++ell) {
    uint64_t val = pprf.eval(pkey, ell);    // 0..N
    uint64_t shifted = (val + s_val) % (N + 1);

    if (shifted < N) { // real bucket
        Y[shifted] += 1;
    }
    // shifted == N => dummy bucket → ignored
}
```

### 3.4 `eval_all_online`

Very similar, but using punctured PPRF and negative counts.

```cpp
Y.assign(N, 0);
pprf::Pprf pprf(prg_);
uint64_t s_val = seed_to_uint64(k1.s) % (N + 1);

for (uint64_t ell = 0; ell < M; ++ell) {
    uint64_t val = pprf.punc_eval(k1.kp, ell);
    if (val == pprf::Pprf::PUNCTURED_SENTINEL) {
        continue; // the removed ball
    }
    uint64_t shifted = (val + s_val) % (N + 1);
    if (shifted < N) {
        Y[shifted] -= 1;
    }
}
```

### 3.5 Correctness sanity check

For each `x`:

* Offline share: `Y0[x]` = number of ℓ with shifted = x.
* Online share: `Y1[x]` = − number of ℓ ≠ ℓ* with shifted = x.

If `β=1` and ℓ* was chosen with shifted(ℓ*) = α:

* At `x = α`: offline sees 1 extra ball; online does not → `Y0[α] + Y1[α] = 1`.
* At `x ≠ α`: both counts match → sum 0.

If `β=0` and ℓ* in dummy bucket:

* For all `x < N`, the histograms are identical → `Y0[x] + Y1[x] = 0`.

So the reconstruction `f(x) = Y0[x] + Y1[x]` is the desired point function.

---

## 4. Group PDPF (Theorem 5)

**File:** `src/pdpf/pdpf_group.cpp`
**Class:** `pdpf::pdpf::PdpfGroup`

We implement the bit‑decomposition approach:

* Encode β ∈ G′ as `payload_bits` bits using the structure of `GroupDescriptor`.
* Run one `PdpfBinary` per bit.
* Decode each server’s share as a linear combination of bit shares.

### 4.1 Bit layout

Given `GroupDescriptor group`:

* `group.moduli = [q0, q1, ..., q_{ℓ-1}]`, all finite (no 0 for now).
* For component i, we allocate `b_i = ceil(log2(q_i))` bits.
* `payload_bits = sum_i b_i`.

This should match your `infer_payload_bits`.

You need a function (local helper) that, given `group` and `payload_bits`, builds a **layout**:

```cpp
struct BitPosition {
    std::size_t component;   // i
    std::size_t bit_index;   // j in that component (0 = LSB)
};

std::vector<BitPosition> build_layout(const GroupDescriptor &group,
                                      std::size_t payload_bits);
```

Algorithm:

```cpp
std::vector<BitPosition> layout;
layout.reserve(payload_bits);
for (size_t i = 0; i < group.moduli.size(); ++i) {
    uint64_t q = group.moduli[i];
    if (q == 0) throw; // for now, we only support finite components
    size_t b = 0;
    uint64_t val = q - 1;
    while (val > 0) { val >>= 1; ++b; }
    for (size_t j = 0; j < b; ++j) {
        layout.push_back({i, j});
    }
}
```

Check `layout.size() == payload_bits`.

### 4.2 `encode_payload`

Signature:

```cpp
std::vector<uint8_t> PdpfGroup::encode_payload(
    const GroupDescriptor &group,
    const GroupElement &beta,
    std::size_t payload_bits) const;
```

Algorithm:

1. Assert `beta.size() == group.arity()`.

2. Compute layout as above.

3. Initialize `bits(payload_bits)`.

4. For each `k = 0..payload_bits-1`:

    * Let `(i,j) = layout[k]`.

    * `uint64_t q = group.moduli[i];`

    * Normalize `beta[i]` to `[0..q-1]`:

      ```cpp
      int64_t v = beta[i];
      if (q > 0) {
          int64_t mod = static_cast<int64_t>(q);
          v %= mod;
          if (v < 0) v += mod;
      }
      ```

    * Then:

      ```cpp
      bits[k] = static_cast<uint8_t>((v >> j) & 1);
      ```

5. Return `bits`.

### 4.3 `decode_payload`

Signature:

```cpp
GroupElement PdpfGroup::decode_payload(
    const GroupDescriptor &group,
    const std::vector<int64_t> &bit_values,
    std::size_t payload_bits) const;
```

Interpretation: `bit_values` is *this server’s* integer share for each bit index; decoding must be linear so that sum of both servers’ `GroupElement`s equals the true β.

Algorithm:

1. Assert `bit_values.size() == payload_bits`.

2. Build layout.

3. Initialize `GroupElement result = group_zero(group);`

4. For each `k = 0..payload_bits-1`:

    * `(i, j) = layout[k]`.
    * Contribution for this bit: `bit_values[k] * 2^j` in component `i`.

      ```cpp
      int64_t contrib = bit_values[k] * (static_cast<int64_t>(1) << j);
      if (group.moduli[i] == 0) {
          result[i] += contrib;  // Z component
      } else {
          int64_t mod = static_cast<int64_t>(group.moduli[i]);
          int64_t tmp = result[i] + contrib;
          tmp %= mod;
          if (tmp < 0) tmp += mod;
          result[i] = tmp;
      }
      ```

5. Return `result`.

Because group addition is linear, we get:

[
\text{decode}(Y0^{(bit)}) + \text{decode}(Y1^{(bit)})
= \text{decode}(Y0^{(bit)} + Y1^{(bit)})
= \beta.
]

### 4.4 `eval_all_offline` / `eval_all_online`

You already compute `Y_bits[i]` for each bit.

Let:

* `payload_bits = k0.bit_offline_keys.size();`
* `N = k0.sec.domain_size_N;`

Algorithm for offline:

```cpp
Y.assign(N, core::group_zero(k0.group_desc));

std::size_t payload_bits = k0.bit_offline_keys.size();
std::vector<int64_t> bit_vals(payload_bits);

for (uint64_t x = 0; x < N; ++x) {
    // collect this server's per-bit shares at position x
    for (size_t b = 0; b < payload_bits; ++b) {
        bit_vals[b] = Y_bits[b][x];
    }
    Y[x] = decode_payload(k0.group_desc, bit_vals, payload_bits);
}
```

Online version is identical, but uses `k1.bit_online_keys` and `base_pdpf_.eval_all_online`.

### 4.5 Testing Group PDPF

* Use a small finite group, e.g. `G = Z_16` or `G = Z_7 × Z_11`.
* For many random `(alpha, beta)`:

    * `k0_group = gen_offline(sec, group, 0 /*let infer_payload_bits*/);`
    * `k1_group = gen_online(k0_group, alpha, beta);`
    * `Y0_group, Y1_group` via eval_all_*.
    * For each `x`, compute `GroupElement g = group_add(Y0_group[x], Y1_group[x])` and check:

        * `g == 0` for `x != alpha`.
        * `g == beta` for `x == alpha`.

---

## 5. LDC: encoding & decoder sampling (Lemma 2)

**File:** `src/ldc/reed_muller_ldc.cpp`
**Class:** `pdpf::ldc::ReedMullerLdc`

The paper uses a Reed–Muller‑style LDC with:

* message length N,
* codeword length L,
* q queries,
* σ‑wise independence.

Implementing the **full generality** is a lot of work. For a first working PDPF, you can:

* restrict to **univariate case** (`w=1`) and field (F = \mathbb{Z}_p),
* implement encoding only for **unit vectors** `e_x` (which is all the amplification construction uses),
* implement `sample_indices(α)` exactly as in Lemma 2 (random σ‑degree curve degenerates to random σ‑degree polynomial in one variable).

### 5.1 Field representation

* `p` is a (small) prime, e.g. 2^31-1 or something like 2^61-1 for 64‑bit safety.
* Represent field element as `int64_t` with value in `[0..p-1]`.

Helpers:

```cpp
inline int64_t add_mod(int64_t a, int64_t b, int64_t p);
inline int64_t sub_mod(int64_t a, int64_t b, int64_t p);
inline int64_t mul_mod(int64_t a, int64_t b, int64_t p);
int64_t inv_mod(int64_t a, int64_t p); // extended Euclid
```

### 5.2 Points & indexing

For simplicity:

* Let `w = 1` (univariate).
* Choose evaluation points `x_α = α` for `α ∈ {0,..,N-1}`.
* Choose codeword positions `j ∈ {0..L-1}` corresponding to points `(ρ_j, x_j)` in `F × F`:

    * `x_j = j % p`
    * `ρ_j = j / p`
    * So `L ≤ p^2`.

You can set `L = p^2` and ignore some positions if `L` in params is smaller.

Map:

```cpp
inline std::pair<int64_t,int64_t> index_to_pair(uint64_t idx) {
    int64_t x = idx % p;
    int64_t rho = (idx / p) % p;
    return {rho, x};
}
```

And inverse.

### 5.3 `encode_unit(x)`

We must compute `C(e_x)[(ρ, y)] = ρ * P_x(y)`, where `P_x` is the polynomial of degree ≤ N-1 with:

* `P_x(x) = 1`
* `P_x(x') = 0` for all other message points `x'`.

For univariate case, `P_x(t)` is the Lagrange basis polynomial:

[
P_x(t) = \prod_{i \ne x} \frac{t - i}{x - i}.
]

Algorithm:

1. Precompute denominators `den_x` for each `x` once in constructor:

   ```cpp
   den[x] = ∏_{i != x} (x - i) (mod p);
   den_inv[x] = inv_mod(den[x], p);
   ```

2. For each codeword index `idx`:

    * Get `(rho,y) = index_to_pair(idx)`.

    * If `y` equals any of the message points, you can use a shortcut:

        * If `y == x`:
          `P_x(y) = 1` by definition.
        * If `y == i != x`:
          `P_x(y) = 0`.

    * Else, compute:

      ```cpp
      int64_t num = 1;
      for (int64_t i = 0; i < N; ++i) {
          if (i == x) continue;
          num = mul_mod(num, sub_mod(y, i, p), p);
      }
      int64_t Px_y = mul_mod(num, den_inv[x], p);
      ```

    * Then:

      ```cpp
      codeword[idx] = mul_mod(rho, Px_y, p);
      ```

Complexity is high (O(N L)), but for small N it’s acceptable. You can also precompute `P_x(y)` only for needed `y` if you wish.

**encode(z)** (for arbitrary z) can optionally be implemented by linearity:

* `C(z) = Σ_{x} z_x * C(e_x)`.

You can let Codex generate that if you want; but for amplification you really only need `encode_unit`.

### 5.4 `sample_indices(alpha)`

We need `q` indices in `[L]` such that:

* Their codeword positions correspond to lines on a random σ‑degree curve through `x_α`.
* For univariate case, the “curve” is just a random σ‑degree polynomial `γ(s)` with `γ(0) = α`.

Simpler, still following the lemma’s structure:

1. Choose random σ‑degree polynomial `γ(s)`:

    * Pick random coefficients `a_1..a_σ ∈ F`, and set:

      ```cpp
      γ(s) = α + Σ_{k=1..σ} a_k * s^k
      ```

2. Pick `rσ + 1` distinct non‑zero `s_ℓ ∈ F` (for ℓ=0..rσ) uniformly at random.

3. Compute Lagrange coefficients `c_ℓ` such that for any polynomial `Q(s)` of degree ≤ rσ:

    * `Q(0) = Σ_{ℓ} c_ℓ * Q(s_ℓ)`.

   That is:

   ```cpp
   c_ℓ = ∏_{m≠ℓ} (-s_m) / (s_ℓ - s_m)  (mod p)
   ```

4. For each ℓ, choose random segment `u_ℓ^1..u_ℓ^{σ+1} ∈ F` such that:

    * `Σ_j u_ℓ^j = c_ℓ` (mod p).

   Example: choose `u_ℓ^1..u_ℓ^σ` uniform and set `u_ℓ^{σ+1} = c_ℓ - Σ_{j≤σ} u_ℓ^j`.

5. Now define indices (q = (σ+1)(rσ+1) total):

   For each ℓ and each j in `1..σ+1`:

    * `rho = u_ℓ^j`,
    * `x = γ(s_ℓ)`,
    * map `(rho,x)` to index `idx` via the inverse of `index_to_pair`.

6. Output the list `Δ = [idx_1, ..., idx_q]`.

This matches the construction in Lemma 2 (restricted to w=1).

---

## 6. Amplified PDPF (Theorem 6)

**File:** `src/pdpf/pdpf_amplified.cpp`
**Class:** `pdpf::pdpf::PdpfAmplified`

Goal: from a 1/poly‑secure PDPF over domain `[L]` and group `Z_p`, build a negligible‑error PDPF over domain `[N]`.

### 6.1 Inner seed derivation: `derive_inner_seed`

We want a PRF keyed by `master_seed` on inputs `index ∈ [0..q-1]`.

Simplest: **AES‑ECB**:

```cpp
core::Seed PdpfAmplified::derive_inner_seed(const core::Seed &master,
                                            uint64_t index) const {
    // Represent index as 16-byte block
    core::Seed block{};
    for (int i = 0; i < 8; ++i) {
        block[15 - i] = static_cast<uint8_t>(index >> (8 * i)); // big-endian
    }
    // Use master as AES-128 key; encrypt block
    core::Seed out{};
    aes_ecb_encrypt(master, block, out); // implement using OpenSSL or your AES wrapper
    return out;
}
```

If you don’t want to introduce AES‑ECB, you can also construct a small PRF using `AesCtrPrg`: e.g., apply `expand(master)` several times in a counter‑style fashion and treat `index` as the counter.

### 6.2 Modular arithmetic: `mod_p`

Implementation is straightforward:

```cpp
int64_t PdpfAmplified::mod_p(int64_t x) const {
    int64_t p = static_cast<int64_t>(prime_p_);
    int64_t r = x % p;
    if (r < 0) r += p;
    return r;
}
```

Use this in all `Z_p` additions and multiplications (you may define helpers for multiply + add).

### 6.3 `gen_offline`

Already mostly done: just sample `master_seed` with `RandomDevice`.

### 6.4 `gen_online`

Match Figure 2.

Parameters:

* `N = k0.sec.domain_size_N;`
* `L = k0.ldc_params.L;`
* `q = k0.ldc_params.q;`

Algorithm:

1. Use `ldc_.sample_indices(alpha)` to get `Δ` (size `q`), each `Δ_ℓ ∈ [0..L-1]`.

2. For each `ℓ = 0..q-1`:

    * Derive inner seed: `inner_seed = derive_inner_seed(k0.master_seed, ℓ)`.

    * Construct `OfflineKey` for **inner base PDPF**:

      ```cpp
      core::SecurityParams inner_sec = k0.sec;
      inner_sec.domain_size_N = L;
 
      OfflineKey inner_off;
      inner_off.k_star = inner_seed;
      inner_off.params.sec = inner_sec;
      inner_off.params.M   = /* choose_M(inner_sec) */; // you may call PdpfBinary::choose_M
      ```

    * Now run binary PDPF generation for point `Δ_ℓ` and same payload `β` (here scalar in `Z_p`; for simplicity treat `β ∈ {0,1}` or embed as in Theorem 5):

      For the basic construction in Theorem 6 they use `Z_p` outputs and may require more care; but if your base PDPF is binary, you can first finish that path (1‑bit amplification). For full `Z_p`, you can combine with `PdpfGroup`.

    * Assuming binary payload for now (`β ∈ {0,1}`):

      ```cpp
      OnlineKey inner_on = base_pdpf_.gen_online(inner_off, Δ[ℓ], static_cast<uint8_t>(beta & 1));
      k1.inner_keys.push_back(std::move(inner_on));
      ```

3. Fill `AmplifiedOnlineKey` with all inner keys and copy `sec`, `ldc_params`, `prime_p`.

**Remark:** The theorem uses base PDPF over `Z_p`, not just bits. In your implementation, you can:

* first complete amplification for **binary** outputs to check code paths, and
* then replace `PdpfBinary` with a `PdpfGroup` instance over `Z_p` (so each inner PDPF shares β in `Z_p`).

### 6.5 Evaluation: high‑level structure

Figure 2 (Eval0/Eval1) can be implemented as:

For any `x ∈ [N]`:

1. Compute `C(e_x) ∈ Z_p^L` via `ldc_.encode_unit(x, codeword)`.

2. For each ℓ, compute `Y0^ℓ = EvalAll0(kℓ0)` and `Y1^ℓ = EvalAll1(kℓ1)` (each ∈ `Z^L` or `Z_p^L`).

3. Let `S0 = Σ_ℓ Y0^ℓ` and `S1 = Σ_ℓ Y1^ℓ` (component-wise, with `mod_p`).

4. Offline share:

   ```cpp
   y0(x) = ⟨ codeword, S0 ⟩ mod p;
   ```

5. Online share:

   ```cpp
   y1(x) = ⟨ codeword, S1 ⟩ mod p;
   ```

6. Reconstruction: `y0(x) + y1(x) mod p = f_α,β(x)`.

### 6.6 Implementing `eval_all_offline` / `eval_all_online`

Since `Y0^ℓ, Y1^ℓ` **do not depend on x**, you should precompute them once, then reuse.

Pseudo‑implementation:

```cpp
void PdpfAmplified::eval_all_offline(const AmplifiedOfflineKey &k0,
                                     const AmplifiedOnlineKey &k1,
                                     std::vector<int64_t> &Y0) const {
    uint64_t N = k0.sec.domain_size_N;
    uint64_t L = k0.ldc_params.L;
    uint64_t q = k0.ldc_params.q;

    // 1. Precompute S0[j] = sum_ℓ Y0^ℓ[j]
    std::vector<int64_t> S0(L, 0);
    for (uint64_t ell = 0; ell < q; ++ell) {
        // derive inner OfflineKey (same as in gen_online)
        OfflineKey inner_off = derive_inner_offline_key(k0, ell);
        std::vector<core::GroupZ::Value> Y0_inner;
        base_pdpf_.eval_all_offline(inner_off, Y0_inner); // Z^L

        for (uint64_t j = 0; j < L; ++j) {
            S0[j] = mod_p(S0[j] + Y0_inner[j]);
        }
    }

    // 2. For each x, compute inner product with C(e_x)
    Y0.assign(N, 0);
    std::vector<int64_t> codeword(L);
    for (uint64_t x = 0; x < N; ++x) {
        ldc_.encode_unit(x, codeword);
        int64_t acc = 0;
        for (uint64_t j = 0; j < L; ++j) {
            acc = mod_p(acc + codeword[j] * S0[j]);
        }
        Y0[x] = acc;
    }
}
```

Online side analogous, using `k1.inner_keys[ell]` and `base_pdpf_.eval_all_online`.

You can then implement `eval_offline(k0,k1,x)` as:

```cpp
std::int64_t PdpfAmplified::eval_offline(..., uint64_t x) const {
    std::vector<int64_t> Y0;
    eval_all_offline(k0, k1, Y0);
    return Y0[x];
}
```

Same for `eval_online`.

---

## 7. RNG & constant‑time

### 7.1 Replace `RandomDevice`

Current `RandomDevice` uses `std::random_device` directly and probably isn’t constant‑time nor well‑seeded.

You should:

* Either:

    * Depend on **libsodium** and use `randombytes_buf` for seeds and random indices.
    * Or depend on **OpenSSL** and use `RAND_bytes`.
* Wrap it in `RandomDevice`:

```cpp
void RandomDevice::random_seed(core::Seed &seed) {
    // e.g. using libsodium
    randombytes_buf(seed.data(), seed.size());
}

uint64_t RandomDevice::random_u64(uint64_t bound) {
    if (bound == 0) throw;
    uint64_t r;
    do {
        randombytes_buf(&r, sizeof(r));
        r %= bound;
    } while (r >= bound); // rejection to avoid bias (can simplify)
    return r;
}
```

### 7.2 Constant‑time concerns

* **PRG / AES / ChaCha**: should come from a constant‑time implementation (library).
* **PPRF**:

    * The input `x` is often *public* (a domain index), so branches on its bits are acceptable.
    * If you ever use PPRF as PRF on *secret* inputs, you must remove data‑dependent branches and implement a branchless selection (`bit` selects left/right with bit‑masks).
* **PDPF**:

    * All branching on `alpha`, `beta`, and counters is done on public values in typical applications, but be mindful if you use PDPF in settings where α could be secret (e.g., inside a larger protocol).

---

## 8. Test plan (high level)

Finally, add tests in `tests/test_pdpf.cpp` (or separate files) to validate each module:

### 8.1 PRG tests

* Determinism: same input seed → same `(left,right)` bytes.
* Basic diffusion: flip one bit in seed changes many output bits (sanity).
* Use known AES test‑vectors if possible.

### 8.2 PPRF tests

For small `M=8`, `N=5`:

* Generate random `root_seed`.
* For many random `x` and `xp`:

    * `y = eval(k,x)`.
    * `kp = puncture(k,xp)`.
    * If `x != xp`, check `punc_eval(kp,x) == y`.
    * If `x == xp`, `punc_eval(kp,x) == PUNCTURED_SENTINEL`.
* Check `eval_all` equals repeated `eval`.

### 8.3 Binary PDPF tests

For random `N` (e.g. 16), and many random `(alpha, beta ∈ {0,1})`:

* `k0 = gen_offline(sec)`; `k1 = gen_online(k0,alpha,beta)`.
* `Y0,Y1` via eval_all_*.
* For each `x` compute `f[x] = Y0[x] + Y1[x]`.
* Assert:

    * `f[alpha] == beta`.
    * `f[x] == 0` for `x != alpha`.

### 8.4 Group PDPF tests

* G = Z_7, Z_11, and Z_7×Z_11.
* Random `(alpha, beta)`; same pattern as above, but using group element equality.

### 8.5 Amplified PDPF tests (small parameters)

* Choose **very small** parameters to make it easy:

    * N=8, p=13, w=1, r=2, σ=2, etc.
* Construct LDC with those parameters.
* Use base PDPF with artificially weak security (tiny M).
* Check that the **amplified** PDPF still has correct reconstruction:

    * `f[alpha] = beta`, `f[x≠alpha] = 0` in `Z_p`.

(You won’t “see” the security amplification experimentally; correctness is what we test.)

---

If you follow this plan and let Codex fill in the straightforward loops and library calls, you should be able to **fully complete**:

* PRG (AES‑CTR).
* GGM PPRF (Eval/Punc).
* 1/poly‑secure binary PDPF (Theorem 4).
* Group‑valued PDPF over general finite abelian groups (Theorem 5).
* Reed–Muller‑style LDC (at least in a univariate special case).
* Security amplification PDPF (Theorem 6).

Once you have a first working end‑to‑end implementation, we can then iterate on **performance tuning** (tree‑based eval_all, lazy Gen, multi‑threading) and on making LDC more efficient and closer to the parameter choices in the paper.
