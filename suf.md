## 1. Target: what “SUF packing/multipoint” should achieve

Right now:

* SUF IR exists (`suf.hpp`) and you have a **naive SUF→PDPF compiler** that:

    * Materializes LUTs over the domain (small `n`, e.g. ≤ 16 bits).
    * Uses `PdpfEngine::make_lut_program` to get a PDPF program with **one output word** (or a few fixed ones).
* Some gates (ReLU, GeLU, nExp, Inv, RecSqrt) are already expressed as SUF programs, but **each helper bit/output** is compiled to its own LUT/program (or split poly/cmp).

The goal of this item:

1. **Multi-output SUF → single multi-output PDPF program per gate**:

    * For a gate like ReluARS, we want one program `P(hat_x)` that returns **all helper bits** `(w, t, d, …)` in a packed vector.
    * For GeLU/SiLU etc., single program returns all control bits & LUT indices you need.

2. **Packed payload layout**:

    * Pack multiple bits into 64-bit (or `u64`) “output words”.
    * Potentially pack small integer outputs (e.g. 8-bit LUT index) next to bits.

3. **Gate-side decoding**:

    * Gate Eval only calls `engine.eval(program_id, hat_x)` once.
    * It then decodes returned words into structured helper bits/indices.

4. (Optional but nice) **Cross–gate packing**:

    * Same SUF program can serve as helper-bit generator for multiple gates that share the same masked input wire; but this can be a later optimization, not required in the first pass.

We’re *not* yet changing the pdpf backend to a fancy tree/multipoint structure (that’s the next item). Here we just make the SUF layer **produce fewer PDPF programs with packed outputs**.

---

## 2. Concrete implementation plan

### 2.1 Extend the SUF IR to support multi-output channels

In `suf.hpp` you likely have something like:

```cpp
struct SufProgram {
    // current: describes a function f : {0,1}^n -> Z_2^w
    unsigned input_bits;
    unsigned output_bits;  // total bits in a single scalar, or similar
    // nodes, intervals, polynomials...
};
```

We want to move to a representation like:

```cpp
enum class SufChannelKind {
    Bit,        // boolean output
    Unsigned,   // unsigned small integer (e.g. LUT index)
    Signed      // maybe, for future
};

struct SufChannel {
    SufChannelKind kind;
    unsigned bits;        // number of bits for this channel (1 for bit)
    std::string name;     // "w", "t", "d", "lut_idx", etc. (for debugging)
    // maybe an index into some node/expression tree
    SufNodeId expr;       // identifier of the SUF expression that computes it
};

struct SufProgram {
    unsigned input_bits;              // n
    std::vector<SufChannel> outputs;  // list of logical channels
    // existing expression / piecewise structure for each expr
    std::vector<SufExpr> exprs;
};
```

Key idea:

* A **channel** is a logical scalar output.
* Each channel is a SUF expression (`SufExpr`) mapping `{0,1}^n -> {0,1}^bits`.

Most unary gates will use:

* A few **bit channels** (e.g. `w`, `t`, `d`).
* Maybe a small **index channel** (e.g. 8-bit LUT index).

### 2.2 Add a “packing layout” that maps channels → output words

We now need a `SufPackingLayout` that packs these channels into `u64` words (the unit the PdpfEngine currently uses):

```cpp
struct SufPackedField {
    // which channel and where inside an output word it lives
    size_t channel_index;    // index in SufProgram::outputs
    unsigned bit_offset;     // offset within the word
    unsigned bit_width;      // 1 for bit, >1 for small ints
    unsigned word_index;     // which output word this field belongs to
};

struct SufPackingLayout {
    unsigned word_count;                 // how many u64 words in output
    std::vector<SufPackedField> fields;  // all the packed fields
};
```

Then implement a simple greedy packer:

```cpp
SufPackingLayout pack_channels(const SufProgram& prog) {
    SufPackingLayout layout;
    layout.word_count = 0;
    uint64_t remaining_bits = 0;
    uint64_t used_bits = 0;

    auto open_word = [&]() {
        layout.word_count++;
        remaining_bits = 64;
        used_bits = 0;
    };

    open_word();
    for (size_t i = 0; i < prog.outputs.size(); ++i) {
        unsigned width = prog.outputs[i].bits;
        if (width > remaining_bits) {
            open_word();
        }
        SufPackedField f;
        f.channel_index = i;
        f.bit_offset = static_cast<unsigned>(used_bits);
        f.bit_width = width;
        f.word_index = layout.word_count - 1;
        layout.fields.push_back(f);
        used_bits += width;
        remaining_bits -= width;
    }
    return layout;
}
```

This is “good enough” for now:

* Minimizes word_count.
* Keeps encoding simple.

### 2.3 Implement SUF → LUT compiler with packing

Right now your SUF compiler probably has a function along the lines of:

```cpp
PdpfProgramId compile_suf_to_pdpf_lut(
    const SufProgram& suf,
    PdpfEngine& engine,
    unsigned output_bits);
```

You want a new signature:

```cpp
struct CompiledSufProgram {
    PdpfProgramId program_id;
    SufProgram suf;
    SufPackingLayout layout;
};

CompiledSufProgram compile_suf_to_pdpf_packed(
    const SufProgram& suf,
    PdpfEngine& engine)
{
    // 1. Build packing layout
    SufPackingLayout layout = pack_channels(suf);

    // 2. Build LUT table: for each x in domain, compute output words
    const uint64_t size = 1ull << suf.input_bits;
    std::vector<std::array<uint64_t, MAX_WORDS>> table(size);
    // (you can replace std::array with std::vector<uint64_t> per entry)

    for (uint64_t x = 0; x < size; ++x) {
        // evaluate SUF expressions into channel values
        std::vector<uint64_t> channel_values(suf.outputs.size());
        for (size_t ch = 0; ch < suf.outputs.size(); ++ch) {
            channel_values[ch] = eval_suf_expression(suf, suf.outputs[ch].expr, x);
            // ensure it's masked to suf.outputs[ch].bits
            uint64_t mask = (suf.outputs[ch].bits == 64)
                ? ~0ull
                : ((1ull << suf.outputs[ch].bits) - 1);
            channel_values[ch] &= mask;
        }

        // pack into words
        std::vector<uint64_t> words(layout.word_count, 0);
        for (const auto& f : layout.fields) {
            uint64_t v = channel_values[f.channel_index];
            words[f.word_index] |= (v << f.bit_offset);
        }

        // store in table[x][*]
        for (unsigned w = 0; w < layout.word_count; ++w) {
            table[x][w] = words[w];
        }
    }

    // 3. Call new multi-output PdpfEngine API
    // Assume engine.make_lut_program_multi(bits, word_count, table_flat, desc)
    PdpfProgramId pid = engine.make_lut_program_multi(
        suf.input_bits,
        layout.word_count,
        table       // or flatten into a single vector<uint64_t>
    );

    return {pid, suf, layout};
}
```

Notes:

* You already have an LUT-based PdpfEngine backend (`PdpfEngineFullImpl`):

    * It can easily support `word_count > 1` by treating each word as an independent group PDPF, or using the pdpf_lut layer you wrote.
* For small domains (≤ 16 bits) this enumeration is fine; you’re already doing it.

### 2.4 Gate-side decoding helper

Gates need an easy way to go from `std::vector<uint64_t>` to typed helper bits. Add a small helper in e.g. `suf.hpp` or `wire.hpp`:

```cpp
struct PackedEvalResult {
    std::vector<uint64_t> words;           // from PdpfEngine eval
    const SufProgram* suf;
    const SufPackingLayout* layout;
};

inline uint64_t get_channel_value(
    const PackedEvalResult& res,
    size_t channel_index)
{
    // find field for this channel (you can pre-index in a map if needed)
    for (const auto& f : res.layout->fields) {
        if (f.channel_index == channel_index) {
            uint64_t word = res.words[f.word_index];
            uint64_t mask = (f.bit_width == 64)
                ? ~0ull
                : ((1ull << f.bit_width) - 1);
            return (word >> f.bit_offset) & mask;
        }
    }
    // shouldn't happen
    return 0;
}
```

Usage in gate Eval:

```cpp
auto words = engine.eval_share(program_id, hat_x, party);  // returns vector<u64>
PackedEvalResult res{words, &compiled_suf.suf, &compiled_suf.layout};

uint64_t w_bit = get_channel_value(res, channel_index_of_w);
uint64_t t_bit = get_channel_value(res, channel_index_of_t);
// etc...
```

You can store `channel_index` constants in the gate key (`CompositeGateKey`) so you don’t linear-scan `fields` each time; but that’s a micro-optimization you can do later.

### 2.5 Wiring: example for ReluARS / LRS

Take one gate you care about (say **ReluARS**), and make it the first user of the new compiler:

1. **Keygen** (`ReluArsKey::gen`):

    * Instead of calling `engine.make_cmp_program` multiple times, build a `SufProgram`:

      ```cpp
      SufProgram suf;
      suf.input_bits = n;  // domain bitwidth for hat_x
 
      auto w_expr = build_w_expr(...);  // 1{hat_x < alpha...} etc.
      auto t_expr = build_t_expr(...);
      auto d_expr = build_d_expr(...);
 
      size_t w_idx = suf.outputs.size();
      suf.outputs.push_back({SufChannelKind::Bit, 1, "w", w_expr});
 
      size_t t_idx = suf.outputs.size();
      suf.outputs.push_back({SufChannelKind::Bit, 1, "t", t_expr});
 
      size_t d_idx = suf.outputs.size();
      suf.outputs.push_back({SufChannelKind::Bit, 1, "d", d_expr});
 
      auto compiled = compile_suf_to_pdpf_packed(suf, engine);
      ```

    * Store in the gate key:

      ```cpp
      struct ReluArsKey {
          PdpfProgramId pid;
          SufPackingLayout layout;
          // indices so we can call get_channel_value
          size_t w_index;
          size_t t_index;
          size_t d_index;
          // other masks, constants, etc.
      };
      ```

2. **Eval**:

   ```cpp
   std::vector<uint64_t> words =
       engine.eval_share(key.pid, hat_x, party);

   PackedEvalResult res{words, &suf_program_from_key, &key.layout};

   uint64_t w = get_channel_value(res, key.w_index);
   uint64_t t = get_channel_value(res, key.t_index);
   uint64_t d = get_channel_value(res, key.d_index);

   // Then feed these into Beaver / arith layer to compute final share.
   ```

Once this works for one gate, you can:

* Convert LRS/ARS/ReluARS helper bits to use the **same** packed program.
* Later, extend to GeLU/SiLU softmax helper bits.

---

## 3. How to test correctness

You want tests at three levels: packing, compiler, and gate integration.

### 3.1 Packing layout + encode/decode unit tests

Add tests under `composite_fss/tests/test_suf_packing.cpp`:

1. **Simple 3‑bit example**:

    * Build a synthetic SUF program:

        * input_bits = 4 (domain 0..15).
        * outputs:

            * channel0: `x & 1`
            * channel1: `(x >> 1) & 1`
            * channel2: `(x >> 2) & 1`

    * Build layout via `pack_channels`.

    * Build table manually (no Pdpf engine): for each `x`, compute channel values, pack into words, then decode with `get_channel_value` and assert:

      ```cpp
      ASSERT_EQ(get_channel_value(res, 0), (x & 1));
      ASSERT_EQ(get_channel_value(res, 1), (x >> 1) & 1);
      ASSERT_EQ(get_channel_value(res, 2), (x >> 2) & 1);
      ```

   This tests pack/unpack logic without any cryptography in the way.

2. **Multi-width example**:

    * Channel0: 1 bit.
    * Channel1: 3 bits (value `x & 0x7`).
    * Channel2: 8 bits (value `x * 3`, truncated to 8 bits).

   Check that packing+unpacking reconstructs exactly the values.

### 3.2 SUF→PDPF compiler tests

Add tests `test_suf_to_pdpf.cpp`:

1. **Exhaustive small-domain check**:

    * Build simple SUF program:

        * `input_bits = 6`.
        * Channel0: bit `x < 7`.
        * Channel1: bit `x < 32`.
        * Channel2: 3-bit `x % 8`.

    * Compile via `compile_suf_to_pdpf_packed`.

    * Use `PdpfEngine` with a **local** backend (adapter) so eval is deterministic and no randomness:

        * For each `x` in `[0, 2^6)`:

          ```cpp
          auto words0 = engine.eval_share(pid, x, /*party=*/0);
          auto words1 = engine.eval_share(pid, x, /*party=*/1);
   
          // recombine shares:
          std::vector<uint64_t> words(words0.size());
          for (size_t i = 0; i < words.size(); ++i)
              words[i] = (words0[i] + words1[i]) & ((1ull << 64) - 1);
   
          PackedEvalResult res{words, &suf, &layout};
          // Check each channel against direct SUF eval
          EXPECT_EQ(get_channel_value(res, 0), (x < 7 ? 1 : 0));
          EXPECT_EQ(get_channel_value(res, 1), (x < 32 ? 1 : 0));
          EXPECT_EQ(get_channel_value(res, 2), (x % 8));
          ```

   If this passes, your SUF compiler + PdpfEngine multi-output path is correct on a non‑trivial example.

2. **Randomized larger‑domain check**:

    * For `input_bits = 12`, generate a random SUF program with a few simple channels (e.g. random linear functions, simple comparisons).
    * Sample 1000 random `x`, evaluate SUF in clear and through PdpfEngine as above; assert equality.

### 3.3 Gate integration tests (ReluARS example)

In `test_composite_fss.cpp`:

1. **Existing tests still pass**:

    * Replace ReluARS gate keygen/eval to use the new `CompiledSufProgram`.
    * Re-run existing tests for ReluARS, LRS, ARS, softmax, etc. They should still pass.

2. **Explicit helper-bit test**:

    * Add a test that directly exercises helper bits from the SUF program:

      ```cpp
      TEST(ReluArsSufHelpers, PackedHelperBitsMatchReference) {
          auto engine = make_pdpf_engine_full();
          auto key = ReluArsKey::gen(...);
 
          for (uint64_t x = 0; x < (1ull << 12); ++x) {
              uint64_t hat_x = (x + mask) & ((1ull << n) - 1);
 
              auto w0 = engine.eval_share(key.pid, hat_x, 0);
              auto w1 = engine.eval_share(key.pid, hat_x, 1);
              std::vector<uint64_t> words(w0.size());
              for (size_t i = 0; i < words.size(); ++i)
                  words[i] = (w0[i] + w1[i]) & ((1ull << 64) - 1);
 
              PackedEvalResult res{words, &key.suf, &key.layout};
              uint64_t w = get_channel_value(res, key.w_index);
              uint64_t t = get_channel_value(res, key.t_index);
              uint64_t d = get_channel_value(res, key.d_index);
 
              // compare to the original "reference" helper bits implemented in clear
              auto [w_ref, t_ref, d_ref] = reluars_reference_bits(x, params);
 
              EXPECT_EQ(w, w_ref);
              EXPECT_EQ(t, t_ref);
              EXPECT_EQ(d, d_ref);
          }
      }
      ```

   This gives you very precise assurance that:

    * SUF expressions are correct.
    * Packing & Pdpf multi-output eval are correct.
    * Gate uses correct channel indices.

3. **Stats check (optional)**:

    * Use PdpfEngine’s instrumentation: ensure that for ReluARS you now see:

        * 1 program generated, `output_words = 1` (or small).
        * vs. the old version which generated multiple compare/LUT programs.

   You can add assertions in tests or at least log these numbers to catch regressions.

---

If you implement these steps, you’ll have:

* A **real multi-output SUF→PDPF compiler** that packs all helper bits/indices for a gate into a single PDPF program.
* Clean gate-side decoding via a small helper.
* A solid test suite verifying correctness both at the packing level and at the gate level.

When you’re ready, we can move on to the next unfinished item (PDPF backend structure / tree-based multipoint) and design that in the same level of detail.
