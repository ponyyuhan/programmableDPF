Below is a concrete, C++‑oriented development plan for implementing the programmable DPF (PDPF) scheme from the “Programmable Distributed Point Functions” paper you uploaded.

I’ll assume the primary goal is:

* Implement the *small‑domain PDPF* (Theorem 4, Figure 1) over output group (G = \mathbb{Z}) with payloads in ({0,1}),
* Optionally extend to general payload groups and the security amplification construction (Theorem 5, Theorem 6),
* Do everything in C++, with an architecture that Codex/Copilot can help fill in.

---

## 1. Target functionality and scope

### 1.1 Cryptographic functionality

Implement:

1. **PRG**

    * Length‑doubling PRG (G: {0,1}^\lambda \to {0,1}^{2\lambda}), instantiated via AES‑CTR or ChaCha20 (pluggable).

2. **Puncturable PRF (PPRF)** via GGM tree

    * `Eval(k, M, N, x)` – PRF from domain ([M]) to ([N]).
    * `Punc(k, M, N, xp)` – punctured key (k_p).
    * `PuncEval(kp, x)` – evaluation for all (x \neq x_p).

3. **Small‑domain PDPF over (G=\mathbb{Z}), payload set ({0,1})** (Figure 1).

    * `Gen0(λ, N)` – offline key (k_0).
    * `Gen1(k0, α, β)` – online key (k_1).
    * `EvalAll0(k0)` – full‑domain evaluation share (Y^{(0)}[1..N]).
    * `EvalAll1(k1)` – full‑domain evaluation share (Y^{(1)}[1..N]).
    * Reconstruction: (f_{\alpha,\beta}(x) = Y^{(0)}[x] + Y^{(1)}[x]).

4. **Optional extensions**

    * **General payload groups** (G) and allowed subset (G') as in Theorem 5 (bit decomposition & group factorization).
    * **Security amplification** via Reed–Muller‑style LDC as in Figure 2 (Theorem 6).
    * **Big‑payload optimization** for (\beta \in \mathbb{Z}_{2^\ell}) (Figure 3, Theorem 9).

For AsiaCCS, you don’t need all extensions at once; starting with the 1/poly‑secure small‑domain PDPF and then adding amplification is already a solid artifact.

---

## 2. High‑level architecture in C++

Use a modular namespace structure:

```text
pdpf/
  core/      – basic types, parameters, randomness, group abstractions
  prg/       – PRG interface + implementations (AES-CTR, ChaCha20)
  pprf/      – GGM-based puncturable PRF
  pdpf/      – programmable DPF (small-domain)
  ldc/       – codes for amplification (Reed–Muller style)
  test/      – unit tests & benchmarks
```

### 2.1 Core types & parameters

Create a header `core/types.hpp` with:

```cpp
namespace pdpf::core {

using uint128 = unsigned __int128;   // if available; otherwise custom type
using Seed    = std::array<uint8_t, 16>; // λ = 128 bits by default

struct SecurityParams {
    uint32_t lambda_bits = 128;
    uint64_t domain_size_N;    // DPF/PDPF domain [N]
    double   epsilon;          // target security error ϵ
};

struct GroupZ {
    // For G = Z, payloads in {0,1}. Represent with int64_t.
    using Value = int64_t;
};

// Simple RAII wrapper for zeroizing sensitive memory on destruction
template<class T>
class SecureBuffer {
public:
    SecureBuffer() = default;
    explicit SecureBuffer(size_t n) : buf_(n) {}
    ~SecureBuffer() { secure_zeroize(); }
    // ... as needed: operator[], data(), size()
private:
    std::vector<T> buf_;
    void secure_zeroize();
};

} // namespace pdpf::core
```

Implement `secure_zeroize()` using `std::fill` plus compiler barriers or `std::memset_s` if available.

---

## 3. PRG module

### 3.1 Interface design

`prg/prg.hpp`:

```cpp
namespace pdpf::prg {

class IPrg {
public:
    virtual ~IPrg() = default;

    // Expand a λ-bit seed into 2λ bits.
    virtual void expand(
        const core::Seed &seed,
        core::Seed &left,
        core::Seed &right
    ) const = 0;
};

} // namespace pdpf::prg
```

Concrete implementations:

* `AesCtrPrg` using AES‑128 in counter mode from a crypto library (OpenSSL, libsodium, BoringSSL, etc).
* `ChaChaPrg` as alternative.

Codex can auto‑generate low‑level AES/ChaCha wrappers once you define this clean interface and some tests.

### 3.2 Design decisions

* Keep PRG pure and stateless; all state is in the seed.
* Respect constant‑time behavior: use constant‑time AES implementations from a vetted library.
* Add a small benchmark driver to measure PRG calls/sec (match the paper’s 1.8×10⁸ AES calls/sec reference if possible).

---

## 4. GGM‑based puncturable PRF (PPRF)

You need: `Eval`, `Punc`, `PuncEval`, plus efficient `EvalAll` for all inputs (used by PDPF).

### 4.1 Key & parameters

`pprf/pprf.hpp`:

```cpp
namespace pdpf::pprf {

struct PprfParams {
    uint64_t M;  // input domain size [M]
    uint64_t N;  // output domain size [N]
};

struct PprfKey {
    core::Seed root_seed;   // seed at root of GGM tree
    PprfParams params;
};

struct PprfPuncturedKey {
    // For GGM, store co-path seeds for the punctured point.
    std::vector<core::Seed> co_path_seeds; // length ≈ λ log M
    uint64_t xp;       // punctured input index
    PprfParams params;
};

class Pprf {
public:
    explicit Pprf(std::shared_ptr<prg::IPrg> prg);

    uint64_t eval(const PprfKey &k, uint64_t x) const;
    uint64_t eval_all(const PprfKey &k, std::vector<uint64_t> &out) const;

    PprfPuncturedKey puncture(const PprfKey &k, uint64_t xp) const;

    uint64_t punc_eval(const PprfPuncturedKey &kp, uint64_t x) const;
    uint64_t punc_eval_all(const PprfPuncturedKey &kp, std::vector<uint64_t> &out) const;

private:
    std::shared_ptr<prg::IPrg> prg_;
};

} // namespace pdpf::pprf
```

### 4.2 Implementation details

1. **GGM tree layout**

    * Depth (d = \lceil \log_2 M \rceil).
    * For `eval(k, x)`:

        * Interpret `x` as a `d`‑bit binary string.
        * Start from `seed = root_seed`.
        * For each bit (b), call `expand(seed, left, right)` and choose `seed = (b ? right : left)`.
        * Finally, map the last seed to an integer in `[N]` with a reduction (`uint64_t val = seed_to_uint64(seed) % N`), *ensuring any bias is negligible* (e.g., use rejection sampling if N is not a power of two).

2. **Puncture** (Theorem 3)

    * For input `xp`, compute the path from root to leaf `xp`, collecting *siblings* on the path (co‑path seeds).
    * Punctured key stores these co‑path seeds.
    * `punc_eval(kp, x)` reconstructs needed seeds from co‑path seeds except at `xp`, where it returns a default or ⊥.

3. **EvalAll**:

    * Straightforward but heavy: build the full tree (2M−1 nodes) and then read off leaves as outputs.
    * Implement both:

        * `eval_all` for full tree from a master key.
        * `punc_eval_all` for punctured key: reuse co‑path seeds and skip the punctured leaf.

4. **Testing**:

    * Check a small M=8, N=5:

        * `eval_all` outputs match individual `eval` calls.
        * `punc_eval` equals `eval` for all `x ≠ xp`.
    * Codex can auto‑generate these tests once signatures & small harness are in place.

---

## 5. Small‑domain PDPF (binary payload, Theorem 4 & Figure 1)

We implement the construction with error (\epsilon \approx \sqrt{(N+1)/M} + \text{negl}(\lambda)).

### 5.1 Parameter mapping

Given:

* Security parameter: (\lambda = 128).
* Domain size: (N).
* Target error: (\epsilon).

Choose:

* (M \approx c \cdot (N+1)/\epsilon^2).
  The paper uses analysis via balls‑and‑bins and recommends values near (M \approx 0.318 \cdot (N+1)/\epsilon^2) empirically.

Store this inside a `PdpfParams` structure.

### 5.2 API and key structures

`pdpf/pdpf.hpp`:

```cpp
namespace pdpf::pdpf {

struct PdpfParams {
    core::SecurityParams sec;
    uint64_t M;         // PPRF input domain size
};

struct OfflineKey {
    core::Seed k_star;   // k*
    PdpfParams params;
};

struct OnlineKey {
    pprf::PprfPuncturedKey kp; // punctured PPRF key
    core::Seed s;              // shift value s from G(k*)
    PdpfParams params;
};

class PdpfBinary {
public:
    PdpfBinary(std::shared_ptr<prg::IPrg> prg);

    // Gen0: offline
    OfflineKey gen_offline(const core::SecurityParams &sec);

    // Gen1: online key for point function f_{α,β}
    // β ∈ {0,1}
    OnlineKey gen_online(const OfflineKey &k0, uint64_t alpha, uint8_t beta);

    // EvalAll0: full evaluation share for offline key
    void eval_all_offline(const OfflineKey &k0,
                          std::vector<core::GroupZ::Value> &Y) const;

    // EvalAll1: full evaluation share for online key
    void eval_all_online(const OnlineKey &k1,
                         std::vector<core::GroupZ::Value> &Y) const;

private:
    std::shared_ptr<prg::IPrg> prg_;
};

} // namespace pdpf::pdpf
```

### 5.3 Implementing Gen0 / Gen1 (Figure 1)

**Gen0** (offline):

1. Sample `k_star ← U_λ`.
2. Compute `M` from `sec.domain_size_N` and `sec.epsilon`.
3. Return `OfflineKey {k_star, PdpfParams{sec, M}}`.

**Gen1** (online):

Paper Figure 1 does:

* Compute `(s, k_PPRF) = G(k*)`, where both values are λ‑bit strings.
* Define PPRF over domain `[M]`, output domain `[N+1]`.
* For β=1:

    * (L = {\ell \in [M] : \text{PPRF.Eval}(k_\text{PPRF}, \ell) + s = \alpha}).
* For β=0:

    * (L = {\ell \in [M] : \text{PPRF.Eval}(k_\text{PPRF}, \ell) + s = N+1}).
* Pick random (\ell \in L); compute punctured key `kp = PPRF.Punc(k_PPRF, ℓ)`; output `(kp, s)` as online key.

C++ steps:

```cpp
OnlineKey PdpfBinary::gen_online(const OfflineKey &k0,
                                 uint64_t alpha,
                                 uint8_t beta) {
    // 1. Expand k* -> (s, k_PPRF_seed)
    core::Seed left, right;
    prg_->expand(k0.k_star, left, right);
    core::Seed s = left;
    core::Seed k_pprf_seed = right;

    uint64_t N = k0.params.sec.domain_size_N;
    uint64_t M = k0.params.M;

    // 2. Construct PPRF key
    pprf::PprfParams pp{M, N + 1};
    pprf::PprfKey pkey{ k_pprf_seed, pp };

    // 3. Find all ℓ with Eval(k_PPRF, ℓ) + s == α (or N+1)
    std::vector<uint64_t> candidates;
    candidates.reserve( /* expect about M/(N+1) */ );

    uint64_t target = (beta == 1) ? alpha : (N + 1);
    pdpf::pprf::Pprf pprf(prg_);

    for (uint64_t ell = 0; ell < M; ++ell) {
        uint64_t val = pprf.eval(pkey, ell);      // in [N+1]
        uint64_t shifted = (val + seed_to_uint64(s)) % (N + 1);
        if (shifted == target) {
            candidates.push_back(ell);
        }
    }

    // In practice: if L empty, either retry or treat as negligible failure (as in the paper).
    if (candidates.empty()) {
        // handle failure: either throw or use the "fail" variant in the paper
    }

    // 4. Pick a random candidate
    uint64_t idx = secure_random_index(candidates.size());
    uint64_t ell_star = candidates[idx];

    // 5. Puncture
    auto kp = pprf.puncture(pkey, ell_star);

    OnlineKey ok;
    ok.kp = std::move(kp);
    ok.s  = s;
    ok.params = k0.params;
    return ok;
}
```

Enhancement: use the “lazy Gen” optimization (Proposition 4) to avoid scanning all M values—try random indices until one matches, trading negligible correctness/privacy error.

### 5.4 EvalAll0 / EvalAll1 (Figure 1)

The idea: evaluate the PPRF (or punctured PPRF) over all inputs, count occurrences of each output bucket (after shifting by `s`), and use those counts as integer shares.

**EvalAll0**:

```cpp
void PdpfBinary::eval_all_offline(const OfflineKey &k0,
                                  std::vector<core::GroupZ::Value> &Y) const
{
    uint64_t N = k0.params.sec.domain_size_N;
    uint64_t M = k0.params.M;

    Y.assign(N, 0);

    core::Seed left, right;
    prg_->expand(k0.k_star, left, right);
    core::Seed s = left;
    core::Seed k_pprf_seed = right;

    pprf::PprfParams pp{M, N + 1};
    pprf::PprfKey pkey{ k_pprf_seed, pp };
    pprf::Pprf pprf(prg_);

    for (uint64_t ell = 0; ell < M; ++ell) {
        uint64_t val = pprf.eval(pkey, ell);         // in [N+1]
        uint64_t shifted = (val + seed_to_uint64(s)) % (N + 1);
        if (shifted >= 1 && shifted <= N) {
            Y[shifted - 1] += 1;
        }
        // if shifted == N+1, it's the dummy bucket; ignore
    }
}
```

**EvalAll1**:

```cpp
void PdpfBinary::eval_all_online(const OnlineKey &k1,
                                 std::vector<core::GroupZ::Value> &Y) const
{
    uint64_t N = k1.params.sec.domain_size_N;
    uint64_t M = k1.params.M;

    Y.assign(N, 0);

    pprf::Pprf pprf(prg_);

    for (uint64_t ell = 0; ell < M; ++ell) {
        uint64_t val = pprf.punc_eval(k1.kp, ell);   // in [N+1] or ⊥ for punctured index
        if (val == PUNCTURED_SENTINEL) continue;
        uint64_t shifted = (val + seed_to_uint64(k1.s)) % (N + 1);
        if (shifted >= 1 && shifted <= N) {
            Y[shifted - 1] -= 1; // note the minus sign
        }
    }
}
```

Correctness: for all (x \neq \alpha), counts match → sum 0; for (x = \alpha) and β=1, offline share has one extra “ball” compared to online share → sum is 1; for β=0, construction uses dummy bucket (N+1) so that sums stay 0.

---

## 6. Extensions

### 6.1 General payload groups (Theorem 5)

Plan:

1. **Bit‑decomposition** of payload β in subgroup (G' \subseteq G):

    * If (G' \subseteq \mathbb{Z}_q), represent β in binary; run one binary PDPF instance per bit; sum appropriately.
    * Encapsulate in `PdpfGroup` class with:

        * `encode_payload(β) -> bit_vector`
        * `decode_payload(bit_shares) -> β`.

2. **Group decomposition**:

    * For general finite Abelian (G), decompose (G \cong \mathbb{Z}*{q_1} \times \dots \times \mathbb{Z}*{q_\ell}).
    * Implement `GroupDescriptor` storing `q_i` and convert operations in G to operations in each component.

Design:

```cpp
struct GroupDescriptor {
    std::vector<uint64_t> moduli; // q_i
};
```

Implement `PdpfGroup` that internally holds multiple `PdpfBinary` instances and orchestrates them.

---

### 6.2 Security amplification via LDC (Theorem 6, Figure 2)

This is more involved but doable.

Components to implement:

1. **Reed–Muller‑style code C and randomized decoder d** (Lemma 2).

    * Encode (z \in \mathbb{Z}_p^N) as evaluations of a low‑degree polynomial (P_z) on (L) points in (\mathbb{F}_p^{w+1}).
    * Decoder `d(α)` outputs indices (\Delta_1,\dots,\Delta_q \in [L]) which are σ‑wise independent and satisfy (\sum C(z)*{\Delta_i} = z*\alpha).

2. **Amplified PDPF construction** (Figure 2):

    * For each query index (\Delta_\ell), run a binary PDPF over domain size L, payload β, with independent offline seeds derived from PRF on (k^*).
    * `Eval0`/`Eval1` for the amplified PDPF:

        * Offline/online call `EvalAll` on each of the q underlying PDPFs.
        * Multiply vector `C(e_x)` (codeword corresponding to unit vector at x) with sum of these `EvalAll` outputs.

Code skeleton:

```cpp
namespace pdpf::ldc {

struct LdcParams {
    uint64_t N;    // original domain
    uint64_t p;    // prime modulus (e.g. small prime)
    uint32_t sigma;
    uint32_t r;
    uint32_t w;
    uint64_t L;
    uint64_t q;
    // Precomputed field tables, points x_α, etc.
};

class ReedMullerLdc {
public:
    explicit ReedMullerLdc(const LdcParams &params);

    // C(z): encode vector z of length N to length L
    void encode(const std::vector<int64_t> &z,
                std::vector<int64_t> &codeword) const;

    // d(α): sample q indices Δ ∈ [L]^q
    std::vector<uint64_t> sample_indices(uint64_t alpha) const;

    // encode unit vector e_x
    void encode_unit(uint64_t x,
                     std::vector<int64_t> &codeword) const;
};

} // namespace pdpf::ldc
```

Then create `PdpfAmplified` that:

* Stores a master offline key (k^*).
* Uses a PRF (e.g., GGM again) to derive q independent `OfflineKey` instances for underlying PDPFs.
* On `gen_online`, for each ℓ:

    * Derive offline key seed via PRF, construct `OfflineKey`.
    * Run base `PdpfBinary::gen_online` for point `∆_ℓ` and β.
* On evaluation, encode `e_x`, then aggregate q base `EvalAll` outputs as per Figure 2.

For an AsiaCCS paper, you can first implement and measure the 1/poly PDPF, then the amplified PDPF, and compare key sizes and running times as suggested in Corollary 2.

---

## 7. Testing & verification plan

### 7.1 Correctness tests

1. **Unit tests for PRG**

    * Deterministic: same seed → same `(left,right)`.
    * Different seeds → different outputs (basic sanity).

2. **PPRF tests**

    * For small M (e.g., 8) and N (up to 16), generate many random keys k, xp:

        * Check that `punc_eval(kp, x) == eval(k, x)` for all `x ≠ xp`.
        * Check that `punc_eval_all` matches `eval_all` except at xp where punctured.

3. **PDPF binary tests**
   For many random `(α,β)`:

    * `k0 ← gen_offline`, `k1 ← gen_online(k0, α, β)`.
    * `Y0 = eval_all_offline(k0)`, `Y1 = eval_all_online(k1)`.
    * Reconstruct:

      ```cpp
      std::vector<int64_t> f(N);
      for (size_t x = 0; x < N; ++x) f[x] = Y0[x] + Y1[x];
      ```
    * Verify `f[α] == β` and `f[x] == 0` for all `x ≠ α`.

4. **Compare with naive DPF**

    * Implement trivial DPF that uses additive sharing of the full truth table.
    * Check equality of reconstructed outputs for random α,β.

### 7.2 Security sanity tests (statistical)

* For fixed (λ,N,M), sample many `(α,β)` and collect distributions of `k1` (serialized).
* Check that basic statistics (bit frequencies, etc.) don’t depend on α.
* For PDPF vs naive PDPF, run chi‑square tests on aggregated histograms of candidate indices; this is mostly for debugging obvious mistakes.

---

## 8. Performance & engineering

### 8.1 Benchmarks

Implement:

* `benchmark_prg()`: measure PRG calls/sec.
* `benchmark_pprf(M,N)`: measure `eval`, `eval_all`, `puncture`.
* `benchmark_pdpf(N, epsilon)`: measure `gen_online`, `eval_all_offline`, `eval_all_online`, compare with naive PDPF as in Table 1 of the paper.

These benchmarks will directly feed into an AsiaCCS experimental section.

### 8.2 Constant‑time & safety

* Use constant‑time AES/ChaCha; avoid branches on secret bits in PRG, PPRF, and PDPF.
* Use `SecureBuffer` for seeds and keys.
* Use a cryptographically secure RNG:

    * `std::random_device` + OS CSPRNG, or library API.
* Avoid logging secrets; have debug modes that only print sizes/indices.

---

## 9. Using Codex effectively

To get Codex to help without breaking cryptographic logic:

1. **Design first, ask later**

    * You hand‑write all *interfaces*, data structures, and high‑level loops (as in this plan).
    * Let Codex fill in repetitive boilerplate (e.g., `SecureBuffer` methods, serialization, small helper functions).

2. **Lock down critical algorithms**

    * For `Pprf::puncture`, `Pprf::punc_eval`, `PdpfBinary::gen_online`, and eval functions, copy the steps *directly* from the paper and this plan, and then use Codex only for low‑level details (e.g., bit manipulations).

3. **Write tests before asking for optimizations**

    * Once correctness tests pass, you can ask Codex for micro‑optimizations (e.g., multi‑threaded `eval_all`, unrolling loops) while preserving external behavior.

---

If you’d like, next step I can sketch concrete C++ class definitions for 1–2 modules (e.g., a fully fleshed PPRF or PDPFBinary header + .cpp skeleton) that you can paste into your project and then iterate on with Codex.
