#pragma once

#include "suf_eval.hpp"
#include "pdpf.hpp"
#include "suf_packing.hpp"
#include <unordered_map>

namespace cfss {

struct MultiLut {
    LutProgramDesc desc;
    std::vector<std::uint64_t> table_flat; // row-major: idx * num_words + j
};

// Compile stacked scalar SUFs into a single multi-output LUT table.
inline MultiLut compile_suf_to_lut_multi(const StackedSuf &stacked) {
    if (stacked.shape.domain_bits == 0) {
        throw std::runtime_error("compile_suf_to_lut_multi: empty shape");
    }
    std::size_t domain_size = 1ULL << stacked.shape.domain_bits;
    MultiLut out;
    out.desc.domain_bits = stacked.shape.domain_bits;
    out.desc.output_words = stacked.shape.num_words;
    out.table_flat.resize(domain_size * stacked.shape.num_words);
    for (std::size_t x = 0; x < domain_size; ++x) {
        auto vals = eval_suf_vector(stacked, static_cast<std::uint64_t>(x));
        for (std::size_t j = 0; j < vals.size(); ++j) {
            out.table_flat[x * stacked.shape.num_words + j] = vals[j];
        }
    }
    return out;
}

struct PackedSufProgram {
    SufCompiled compiled;
    SufPackedLayout layout;
    std::vector<SufChannelDesc> channels;
};

inline PackedSufProgram compile_suf_desc_packed(const SufDesc &desc,
                                                PdpfEngine &engine,
                                                std::optional<SufPackedLayout> layout_hint = std::nullopt,
                                                uint32_t word_bitwidth = 64) {
    // Build channels if not provided: arith outputs followed by bool bits.
    std::vector<SufChannelDesc> channels = desc.shape.channels;
    if (channels.empty()) {
        for (unsigned i = 0; i < desc.r_outputs; ++i) {
            channels.push_back(SufChannelDesc{SufChannelId{static_cast<uint32_t>(i)}, "arith_" + std::to_string(i),
                                              SufFieldKind::Ring, word_bitwidth, 1});
        }
        for (unsigned i = 0; i < desc.l_outputs; ++i) {
            channels.push_back(SufChannelDesc{SufChannelId{static_cast<uint32_t>(desc.r_outputs + i)},
                                              "bool_" + std::to_string(i),
                                              SufFieldKind::Bool, 1, 1});
        }
    }
    SufChannelRegistry reg;
    reg.channels = channels;
    reg.next_id = static_cast<uint32_t>(channels.size());
#if COMPOSITE_FSS_INTERNAL
    SufPackedLayout layout = layout_hint.has_value() ? *layout_hint : make_greedy_packed_layout(reg, word_bitwidth);
#else
    // Strict build: disable bit packing; each channel occupies its own word.
    SufPackedLayout layout;
    layout.word_bitwidth = word_bitwidth;
    for (const auto &ch : reg.channels) {
        for (uint32_t i = 0; i < ch.count; ++i) {
            SufChannelField logical{ch.channel_id, i, ch.width_bits};
            SufPackedField pf{logical, static_cast<uint32_t>(layout.fields.size()), 0};
            layout.fields.push_back(pf);
        }
    }
    layout.num_words = static_cast<uint32_t>(layout.fields.size());
#endif
    layout.build_index();

    if (desc.shape.domain_bits == 0) throw std::runtime_error("SufDesc: n_bits must be > 0");
    std::size_t domain_size = 1ULL << desc.shape.domain_bits;
    std::vector<std::uint64_t> table_flat(domain_size * layout.num_words, 0);
    std::uint64_t mask = (desc.shape.domain_bits == 64) ? ~0ULL : ((1ULL << desc.shape.domain_bits) - 1ULL);
    RingConfig cfg = make_ring_config(desc.shape.domain_bits);

    // Map channel id -> logical index
    std::unordered_map<uint32_t, uint32_t> channel_index;
    for (std::size_t i = 0; i < reg.channels.size(); ++i) {
        channel_index[reg.channels[i].channel_id.id] = static_cast<uint32_t>(i);
    }

    for (std::size_t x = 0; x < domain_size; ++x) {
        std::uint64_t unmasked = 0;
        if (desc.shape.domain_bits == 64) {
            unmasked = static_cast<std::uint64_t>(x) - desc.r_in;
        } else {
            std::uint64_t modulus = 1ULL << desc.shape.domain_bits;
            unmasked = (static_cast<std::uint64_t>(x) + modulus - (desc.r_in & mask)) & mask;
        }
        std::size_t interval_idx = find_interval(desc, unmasked);
        if (interval_idx >= desc.polys.size()) continue;
        const auto &pvec = desc.polys[interval_idx].polys;
        std::vector<std::uint64_t> arith_vals(desc.r_outputs, 0);
        for (unsigned r = 0; r < desc.r_outputs && r < pvec.size(); ++r) {
            std::uint64_t val = eval_poly_mod(pvec[r], unmasked);
            val = ring_add(cfg, val, desc.r_out & mask);
            arith_vals[r] = val;
        }
        std::uint64_t bool_bits = 0;
        if (desc.l_outputs > 0 && interval_idx < desc.bools.size()) {
            for (unsigned b = 0; b < desc.l_outputs && b < desc.bools[interval_idx].size(); ++b) {
                bool bit = eval_bool_expr(desc.bools[interval_idx][b], unmasked);
                bool_bits |= (static_cast<std::uint64_t>(bit) << b);
            }
        }

        std::vector<std::uint64_t> words(layout.num_words, 0);
        for (const auto &f : layout.fields) {
            auto it = channel_index.find(f.logical.channel.id);
            if (it == channel_index.end()) continue;
            std::uint32_t ch_idx = it->second;
            std::uint64_t value = 0;
            if (ch_idx < arith_vals.size()) {
                value = arith_vals[ch_idx];
            } else {
                std::uint32_t bidx = ch_idx - static_cast<std::uint32_t>(arith_vals.size());
                value = (bool_bits >> bidx) & 1ULL;
            }
            words[f.word_index] |= (value & ((f.logical.width_bits == 64) ? ~0ULL : ((1ULL << f.logical.width_bits) - 1ULL)))
                                   << f.bit_offset;
        }
        for (std::size_t w = 0; w < words.size(); ++w) {
            table_flat[x * layout.num_words + w] = words[w];
        }
    }

    LutProgramDesc lut_desc;
    lut_desc.domain_bits = desc.shape.domain_bits;
    lut_desc.output_words = layout.num_words;
    PdpfProgramId pid = engine.make_lut_program(lut_desc, table_flat);

    SufCompiled compiled;
    compiled.pdpf_program = pid;
    compiled.poly_prog = pid;
    compiled.lut_prog = pid;
    compiled.domain_bits = desc.shape.domain_bits;
    compiled.num_arith_outputs = desc.r_outputs;
    compiled.num_bool_outputs = desc.l_outputs;
    compiled.output_words = layout.num_words;
    compiled.shape = desc.shape;

    PackedSufProgram out;
    out.compiled = compiled;
    out.layout = layout;
    out.channels = reg.channels;
    return out;
}

} // namespace cfss
