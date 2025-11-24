
## 0. Goal / mental model

Right now:

* You have **SufShape** and a basic “packed SUF → PDPF” compiler.
* ARS demonstrates packing by putting “shifted value + sign bit” into a single multi‑output PDPF program.
* But the packing is ad‑hoc: no named channels, no general layout builder, no reusable decode helpers.

The goal of this step is:

1. Introduce a **channel abstraction**: each SUF output component is part of a *named channel* with a known bitwidth and type.
2. Implement a **greedy bit‑packing layout** that maps channels to PDPF output words (`u64`s or `R` elements) and bit‑ranges.
3. Provide **decode helpers** so gates can say “give me the value of channel X” without thinking about where it lives in the packed payload.
4. Keep everything **offline/compile‑time**: gates only use channel names & indices, not bit offsets.

After this, migrating gates to “single packed SUF per gate” becomes much easier.

---

## 1. Define the channel/type abstraction (suf.hpp)

First, extend your SUF IR with explicit channels.

### 1.1 Add enums/types for field kinds

In `suf.hpp`, introduce something like:

```cpp
namespace composite_fss {

enum class SufFieldKind {
  Ring,      // element in Z_{2^n}
  Bool,      // single bit
  Index,     // small integer (e.g. 8-bit LUT index)
};

struct SufChannelId {
  // Simple opaque id, comparable
  uint32_t id;
};

struct SufChannelDesc {
  SufChannelId id;
  std::string name;    // "relu_bit", "trunc_val", ...
  SufFieldKind kind;
  uint32_t    width_bits;   // for Ring/Index; Bool always 1.
  uint32_t    count;        // number of elements in this channel
  // (optional): is_signed, or “effective bitwidth” if you want
};

} // namespace composite_fss
```

You’ll later use `name` only for debugging/logging; gate code should keep the `SufChannelId` or just a small enum.

### 1.2 Extend SufShape to expose channels

`SufShape` currently describes the multi‑output shape (number of ring outputs, bool outputs, etc.). Extend it to attach a list of channels:

```cpp
struct SufShape {
  uint32_t input_bitwidth;  // domain bits
  // old fields: e.g. num_ring_outputs, num_bool_outputs, ...
  // ...

  std::vector<SufChannelDesc> channels;

  // convenience:
  SufChannelId add_channel(const std::string& name,
                           SufFieldKind kind,
                           uint32_t width_bits,
                           uint32_t count);
};
```

**How gates use this:**

* During gate keygen, when they construct a SUF, they call `shape.add_channel(...)` for:

    * “main arith output(s)” (e.g., shifted value, ReLU(x), etc.).
    * control bits (`w, t, d`, sign bit, clip bits, etc.).
    * LUT indices `(t(x))`, exponents, etc.

The SUF compiler will pack all these channels.

---

## 2. Greedy packing layout (SufPackedLayout)

Next, build a **layout descriptor** that maps channels to packed output words.

### 2.1 Layout structure

In `suf.hpp` (or a new `suf_packing.hpp`), introduce:

```cpp
struct SufChannelField {
  SufChannelId channel;
  uint32_t element_index;   // which element in the channel
  uint32_t width_bits;      // e.g. 1 for Bool, 8 for LUT index, n for ring
};

struct SufPackedField {
  SufChannelField logical;
  uint32_t word_index;      // which PDPF output word
  uint32_t bit_offset;      // within that word
};

struct SufPackedLayout {
  uint32_t word_bitwidth;      // usually 64 or n
  uint32_t num_words;
  std::vector<SufPackedField> fields;

  // Helpers:
  const SufPackedField* find_field(SufChannelId ch, uint32_t elem_idx) const;
};
```

### 2.2 Greedy packing algorithm

Implement a function (e.g., in `suf_packing.cpp`):

```cpp
SufPackedLayout make_greedy_packed_layout(const SufShape& shape,
                                          uint32_t word_bitwidth);
```

Algorithm (simple and deterministic):

1. Create a list of all logical fields:

   ```cpp
   std::vector<SufChannelField> logical_fields;
   for each channel in shape.channels:
     for i in [0 .. channel.count-1]:
       logical_fields.push_back({channel.id, i, channel.width_bits});
   ```

2. (Optional but good) **sort** logical fields by decreasing `width_bits` to pack wide values first; or sort by channel id to keep channels grouped.

3. Greedy packing:

   ```cpp
   uint32_t word_index = 0;
   uint32_t bit_used   = 0;
   std::vector<SufPackedField> packed;

   for (const auto& f : logical_fields) {
     if (bit_used + f.width_bits > word_bitwidth) {
       // move to next word
       ++word_index;
       bit_used = 0;
     }
     packed.push_back({
       /* logical = */ f,
       /* word_index = */ word_index,
       /* bit_offset = */ bit_used
     });
     bit_used += f.width_bits;
   }
   ```

4. Set:

   ```cpp
   SufPackedLayout layout;
   layout.word_bitwidth = word_bitwidth;
   layout.num_words = word_index + 1;
   layout.fields = std::move(packed);
   ```

5. Implement `find_field` as a linear scan or hash‑map for convenience:

   ```cpp
   const SufPackedField* SufPackedLayout::find_field(SufChannelId ch,
                                                     uint32_t idx) const {
     for (const auto& f : fields) {
       if (f.logical.channel.id == ch.id &&
           f.logical.element_index == idx) return &f;
     }
     return nullptr;
   }
   ```

You can optimize lookup later with a map; correctness is more important now.

---

## 3. Wire the packing into the SUF compiler (compile_suf_to_pdpf_packed)

You already have `compile_suf_to_pdpf_packed` which produces a multi‑output PDPF program. Now:

### 3.1 Extend the compiler signature

In `suf.hpp` / `suf_to_lut.hpp`:

```cpp
struct PackedSufProgram {
  PdpfProgramId program_id;
  SufShape      shape;
  SufPackedLayout layout;
};

PackedSufProgram compile_suf_to_pdpf_packed(
    const SufProgram& suf,
    PdpfEngine& engine,
    uint32_t word_bitwidth = 64);
```

So the compiler returns both:

* The `PdpfProgramId` (for eval), and
* The `layout` and `shape` (for decode).

### 3.2 Generation phase: map logical outputs → words

Where you currently build a LUT/program per output word, you now:

1. Call `make_greedy_packed_layout(shape, word_bitwidth)`.
2. For each `word_index in [0..layout.num_words-1]`, build a **LUT description** or group PDPF describing that word as a function `f_word(x)`:

    * For each SUF input value `x`, clear logic computes:

        * All channel outputs (polys & bits).
        * Uses `layout` to pack them into `word_bitwidth` bits.

    * In terms of your existing `PdpfEngineFullImpl` with `pdpf_lut`, “each word” is just a **multi-point LUT** where the value is the packed word.

Implementation sketch (`suf_to_lut.cpp`):

```cpp
for (uint32_t w = 0; w < layout.num_words; ++w) {
  // Build a table or SUF LUT for this word:
  LutDescriptor word_lut;
  // For each domain point alpha_i (or per-interval rep), compute packed_value:
  //   packed_value[bit] = ...
  // Then call engine.make_lut_program(...) to get a PdpfProgramId for this word.
}
```

But you already have a SUF→LUT builder; you just need to:

* Instead of returning multiple scalar outputs, merge them into word(s) according to `layout`.

3. Store the `PdpfProgramId` for each word; if your `PdpfEngine` now has multi‑output support, you can treat them as **one program with N words** (which you already have). If not, you can stash them in a small struct and simulate multi‑output by repeated eval; but you *do* have a multi‑output API now, so use that.

---

## 4. Decode helpers for gates

Now gates shouldn’t need to know bit offsets. They only know:

* Which **channel** they want,
* and which element index in that channel.

Introduce a helper (e.g., in `suf_eval.hpp` or `wire.hpp`):

```cpp
// Given packed PDPF outputs (per word, per party),
// extract the reconstructed logical value for (channel, idx).

uint64_t suf_unpack_channel_u64(const SufPackedLayout& layout,
                                SufChannelId ch,
                                uint32_t element_idx,
                                const std::vector<uint64_t>& packed_words);
```

If you want to keep it share‑aware:

```cpp
Share unpack_channel_as_share(const SufPackedLayout& layout,
                              SufChannelId ch,
                              uint32_t element_idx,
                              const std::vector<Share>& packed_word_shares);
```

### 4.1 Implement unpacking

For reconstructed (clear) words (used in tests):

```cpp
uint64_t suf_unpack_channel_u64(..., const std::vector<uint64_t>& words) {
  const auto* field = layout.find_field(ch, element_idx);
  assert(field != nullptr);
  uint32_t w = field->word_index;
  uint32_t off = field->bit_offset;
  uint32_t bits = field->logical.width_bits;

  uint64_t mask = (bits == 64) ? ~0ULL : ((1ULL << bits) - 1);
  uint64_t v = (words[w] >> off) & mask;
  return v;
}
```

For **shares**, do the same per party:

```cpp
Share unpack_channel_as_share(..., const std::vector<Share>& word_shares) {
  const auto* field = layout.find_field(ch, element_idx);
  assert(field != nullptr);
  const Share& w_share = word_shares[field->word_index];

  uint64_t mask = (field->logical.width_bits == 64)
                    ? ~0ULL
                    : ((1ULL << field->logical.width_bits) - 1);

  // assuming Share has .v() accessor or getter; use ring helpers
  uint64_t v = (w_share.value_internal() >> field->bit_offset) & mask;
  return Share::from_value(w_share.party(), v);
}
```

You can add convenience wrappers for:

* `bool suf_unpack_bool(...)`,
* `Share unpack_bool_share(...)` (with width_bits=1),
* or `Share unpack_ring_share(...)` for ring outputs.

Gates then look like:

```cpp
auto ch_trunc = suf_shape.lookup_channel("trunc_val");
auto ch_sign  = suf_shape.lookup_channel("ars_sign");

Share trunc_val = unpack_channel_as_share(layout, ch_trunc, 0, packed_word_shares);
Share sign_bit  = unpack_channel_as_share(layout, ch_sign, 0, packed_word_shares);
```

No direct bit arithmetic in gates.

---

## 5. How to test correctness (for this feature alone)

You already have a “stacking SUF” test. Extend/add tests specifically for channel/field packing:

### 5.1 Unit test: layout & invertibility

Create `test_suf_packing.cpp` with tests like:

1. **Deterministic layout**:

    * Build a `SufShape` with a mix of channels:

      ```cpp
      SufShape shape;
      auto ch_a = shape.add_channel("a", SufFieldKind::Ring, 16, 2);
      auto ch_b = shape.add_channel("b", SufFieldKind::Bool, 1, 5);
      auto ch_c = shape.add_channel("idx", SufFieldKind::Index, 8, 1);
      ```

    * Call `make_greedy_packed_layout(shape, 32)` and assert:

        * `layout.num_words` is plausible (e.g., 2 or 3).
        * No field overlaps: for any two fields with same `word_index`, their [offset, offset+width) ranges are disjoint.
        * All fields fit within `word_bitwidth`.

2. **Pack → unpack is identity (synthetic)**:

    * Construct fake “logical values” for each channel/element: e.g., `a[0]=0xA, a[1]=0xB, b[i]=i&1, idx=42`.
    * Manually pack into `words` using the **same** algorithm as in `suf_to_lut.cpp` (or just call a local pack function).
    * Then call `suf_unpack_channel_u64` for each logical field and assert you get the original value.

This tests the layout + decode helpers without PDPF yet.

### 5.2 Integration test: SUF→PDPF→eval→decode

Extend your existing SUF stacking test:

* Construct a small SUF with 2–3 channels (ring + bool + index) with known clear semantics.
* Compile via `compile_suf_to_pdpf_packed` using `PdpfEngineAdapter` (small domain).
* For all `x` in the domain:

    1. Evaluate the SUF in the clear (`suf_eval_clear`).
    2. Call `PdpfEngine::eval_share` for both parties to get packed word shares.
    3. Reconstruct the packed words for each party (just `.value_internal()` since adapter uses deterministic sharing).
    4. Use `suf_unpack_channel_u64` / `unpack_channel_as_share` to recover each channel element.
    5. Compare to the clear SUF outputs.

Key idea: **this test guarantees that your packing layout + compiler + decode match the SUF semantics exactly**.

### 5.3 Regression: ARS gate still correct

Finally, update the ARS gate test to assert:

* After migrating ARS to use the new **channel‑based** layout (instead of ad‑hoc packing), the gate still matches the clear ARS implementation for random inputs.

Test:

1. Generate random `x` values in `Z_{2^n}`.
2. Build a masked wire, keygen ARS (which internally builds a SUF with channels and compiles to PDPF).
3. Run `Eval` via the existing gate API.
4. Open result (debug path) and compare to `clear_ars(x)`.

If all tests pass (layout unit tests + SUF packing test + ARS regression), you’ve verified the channel/field packing abstraction is **correct and stable**.

---

## 6. After this step

Once you have:

* `SufChannelDesc` / `SufPackedLayout`,
* a greedy packer,
* decode helpers,
* and tests that prove pack↔unpack correctness and SUF→PDPF equivalence,

then the *next* natural steps are:

1. Migrate ReluARS helper bits, GeLU, etc. to use these named channels and the packed compiler.
2. Add more packing‑focused tests (especially edge cases: narrow/wide mixes, many bools).
3. Start cross‑gate packing (reusing a global helper‑bit SUF program).

But first, I’d make sure the channel/field packing abstraction is 100% in place and tested as above.

If you’d like, next time we can go bullet‑by‑bullet through **“Migrating remaining gates (ReluARS helper bits, GeLU, etc.) to packed SUFs with single eval per gate”** and I’ll spell out exactly how to change each gate to use channels + packed PdpfProgramIds.
