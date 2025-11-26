#pragma once

#include "suf.hpp"
#include <string>
#include <vector>
#include <optional>
#include <unordered_map>

namespace cfss {

struct SufChannelField {
    SufChannelId channel;
    uint32_t element_index = 0;
    uint32_t width_bits = 0;
};

struct SufPackedField {
    SufChannelField logical;
    uint32_t word_index = 0;
    uint32_t bit_offset = 0;
};

struct SufPackedLayout {
    uint32_t word_bitwidth = 64;
    uint32_t num_words = 0;
    std::vector<SufPackedField> fields;
    std::unordered_map<std::uint64_t, std::size_t> field_index;

    void build_index() {
        field_index.clear();
        for (std::size_t i = 0; i < fields.size(); ++i) {
            const auto &f = fields[i];
            std::uint64_t key = (static_cast<std::uint64_t>(f.logical.channel.id) << 32) | f.logical.element_index;
            field_index[key] = i;
        }
    }

    const SufPackedField *find_field(SufChannelId ch, uint32_t elem_idx) const {
        std::uint64_t key = (static_cast<std::uint64_t>(ch.id) << 32) | elem_idx;
        auto it = field_index.find(key);
        return it == field_index.end() ? nullptr : &fields[it->second];
    }
};

// Simple channel registry attached to a shape.
struct SufChannelRegistry {
    std::vector<SufChannelDesc> channels;
    uint32_t next_id = 0;

    SufChannelId add_channel(const std::string &name,
                             SufFieldKind kind,
                             uint32_t width_bits,
                             uint32_t count) {
        SufChannelId id{next_id++};
        channels.push_back(SufChannelDesc{id, name, kind, width_bits, count});
        return id;
    }
};

inline SufPackedLayout make_greedy_packed_layout(const SufChannelRegistry &reg,
                                                 uint32_t word_bitwidth) {
    std::vector<SufChannelField> logical;
    for (const auto &ch : reg.channels) {
        for (uint32_t i = 0; i < ch.count; ++i) {
            logical.push_back(SufChannelField{ch.channel_id, i, ch.width_bits});
        }
    }
    // Greedy pack, widest-first for stability.
    std::sort(logical.begin(), logical.end(),
              [](const SufChannelField &a, const SufChannelField &b) {
                  return a.width_bits > b.width_bits;
              });

    SufPackedLayout layout;
    layout.word_bitwidth = word_bitwidth;
    uint32_t word_idx = 0;
    uint32_t bit_used = 0;
    for (const auto &f : logical) {
        if (bit_used + f.width_bits > word_bitwidth) {
            ++word_idx;
            bit_used = 0;
        }
        layout.fields.push_back(SufPackedField{f, word_idx, bit_used});
        bit_used += f.width_bits;
    }
    layout.num_words = word_idx + (layout.fields.empty() ? 0 : 1);
    layout.build_index();
    return layout;
}

} // namespace cfss
