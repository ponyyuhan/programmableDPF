#pragma once

#include "suf.hpp"
#include <algorithm>
#include <cstdint>
#include <optional>
#include <stdexcept>
#include <unordered_map>
#include <vector>

namespace cfss {

// A global schema that fixes bit positions for all helper kinds the model uses.
enum class HelperKind {
    Gez,
    Neg,
    Ars_w,
    Ars_t,
    Ars_d,
    Clip_lo,
    Clip_hi,
    Is_small,
    Spline_interval_0,
    Softmax_cutoff,
    Norm_block_bit,
};

struct HelperSlot {
    HelperKind kind = HelperKind::Gez;
    std::uint32_t width_bits = 1;
    std::uint32_t offset = 0;
};

struct GlobalHelperSchema {
    std::uint32_t total_bits = 0;
    std::vector<HelperSlot> slots;
    std::unordered_map<int, std::size_t> slot_index;

    // Registers a helper kind and reserves width_bits consecutive bits.
    std::uint32_t add_kind(HelperKind kind, std::uint32_t width_bits = 1) {
        if (width_bits == 0) throw std::runtime_error("GlobalHelperSchema: width_bits must be > 0");
        int key = static_cast<int>(kind);
        if (slot_index.count(key)) return slots[slot_index[key]].offset;
        HelperSlot slot;
        slot.kind = kind;
        slot.width_bits = width_bits;
        slot.offset = total_bits;
        std::size_t idx = slots.size();
        slots.push_back(slot);
        slot_index[key] = idx;
        total_bits += width_bits;
        return slot.offset;
    }

    std::optional<std::uint32_t> lookup_offset(HelperKind kind) const {
        int key = static_cast<int>(kind);
        auto it = slot_index.find(key);
        if (it == slot_index.end()) return std::nullopt;
        return slots[it->second].offset;
    }

    std::uint32_t num_words() const {
        return (total_bits + 63) / 64;
    }
};

struct GateHelperInput {
    std::size_t gate_index = 0;
    SufDesc bool_suf; // boolean-only SUF for the gate's helper bits (masked input semantics baked in)
    // Mapping from local bit index -> HelperKind to embed into global layout.
    std::vector<std::pair<std::uint32_t, HelperKind>> bit_mapping;
};

struct GlobalHelperGateKey {
    bool valid = false;
    PdpfProgramId program = 0;
    unsigned domain_bits = 0;
    unsigned output_words = 0;
};

struct GlobalHelperKey {
    GlobalHelperSchema schema;
    std::vector<GlobalHelperGateKey> gates;
};

inline BoolExpr make_const_false() {
    BoolExpr e;
    e.kind = BoolExpr::CONST;
    e.const_value = false;
    return e;
}

// Expand a gate-local helper SUF into the global layout (all other bits are 0).
inline SufDesc embed_helper_bits(const GlobalHelperSchema &schema,
                                 const SufDesc &local,
                                 const std::vector<std::pair<std::uint32_t, HelperKind>> &mapping) {
    if (schema.total_bits == 0) throw std::runtime_error("GlobalHelperSchema: no helper bits registered");
    if (local.l_outputs == 0) throw std::runtime_error("GlobalHelper: local helper SUF must have boolean outputs");
    SufDesc desc = local;
    desc.r_outputs = 0; // only helper bits are exported
    desc.l_outputs = schema.total_bits;

    // Ensure at least one interval boundary exists.
    if (desc.alpha.empty()) {
        desc.alpha.push_back(0);
        desc.alpha.push_back(1ULL << desc.shape.domain_bits);
    }
    std::size_t intervals = std::max<std::size_t>(desc.bools.size(), 1);
    desc.bools.assign(intervals, std::vector<BoolExpr>(desc.l_outputs, make_const_false()));
    // Carry over interval-specific predicates for mapped bits.
    for (std::size_t iv = 0; iv < intervals && iv < local.bools.size(); ++iv) {
        for (const auto &pair : mapping) {
            std::uint32_t local_idx = pair.first;
            auto target = schema.lookup_offset(pair.second);
            if (!target.has_value()) continue;
            if (local_idx >= local.bools[iv].size()) {
                throw std::runtime_error("GlobalHelper: local bit mapping out of range");
            }
            desc.bools[iv][*target] = local.bools[iv][local_idx];
        }
    }
    // Maintain interval count for polys even though there are no arithmetic outputs.
    if (desc.polys.size() < intervals) desc.polys.resize(intervals);
    desc.shape.num_words = std::max<std::uint32_t>(desc.shape.num_words, schema.num_words());
    return desc;
}

inline GlobalHelperKey make_global_helper_key(const GlobalHelperSchema &schema,
                                              const std::vector<GateHelperInput> &inputs,
                                              PdpfEngine &engine) {
    GlobalHelperKey key;
    key.schema = schema;
    std::size_t max_gate = 0;
    for (const auto &in : inputs) {
        max_gate = std::max(max_gate, in.gate_index);
    }
    key.gates.assign(max_gate + 1, GlobalHelperGateKey{});

    for (const auto &in : inputs) {
        SufDesc embedded = embed_helper_bits(schema, in.bool_suf, in.bit_mapping);
        auto compiled = compile_suf_to_pdpf(embedded, engine);
        GlobalHelperGateKey gk;
        gk.valid = true;
        gk.program = compiled.pdpf_program;
        gk.domain_bits = compiled.domain_bits;
        gk.output_words = compiled.output_words ? compiled.output_words : schema.num_words();
        key.gates[in.gate_index] = gk;
    }
    return key;
}

inline void global_helper_eval(const GlobalHelperKey &key,
                               int party,
                               std::size_t gate_index,
                               std::uint64_t masked_x,
                               PdpfEngine &engine,
                               std::vector<std::uint64_t> &out_words) {
    if (gate_index >= key.gates.size() || !key.gates[gate_index].valid) {
        out_words.clear();
        return;
    }
    const auto &g = key.gates[gate_index];
    std::uint32_t words = g.output_words ? g.output_words : key.schema.num_words();
    out_words.assign(words, 0);
    engine.eval_share(g.program, party, masked_x, out_words);
}

inline void global_helper_eval_batch(const GlobalHelperKey &key,
                                     int party,
                                     std::size_t gate_index,
                                     const std::vector<std::uint64_t> &masked_xs,
                                     PdpfEngine &engine,
                                     std::vector<std::uint64_t> &flat_out) {
    if (gate_index >= key.gates.size() || !key.gates[gate_index].valid || masked_xs.empty()) {
        flat_out.clear();
        return;
    }
    const auto &g = key.gates[gate_index];
    std::uint32_t words = g.output_words ? g.output_words : key.schema.num_words();
    flat_out.assign(masked_xs.size() * words, 0);
    engine.eval_share_batch(g.program, party, masked_xs.data(), masked_xs.size(), flat_out.data());
}

} // namespace cfss
