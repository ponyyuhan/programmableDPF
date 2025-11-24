#pragma once

#include "suf_packing.hpp"
#include "arith.hpp"

namespace cfss {

inline std::uint64_t suf_unpack_channel_u64(const SufPackedLayout &layout,
                                            SufChannelId ch,
                                            uint32_t elem_idx,
                                            const std::vector<std::uint64_t> &words) {
    const auto *field = layout.find_field(ch, elem_idx);
    if (!field) return 0;
    if (field->word_index >= words.size()) return 0;
    std::uint64_t word = words[field->word_index];
    std::uint64_t mask = (field->logical.width_bits == 64)
                             ? ~0ULL
                             : ((1ULL << field->logical.width_bits) - 1ULL);
    return (word >> field->bit_offset) & mask;
}

inline Share suf_unpack_channel_share(const RingConfig &cfg,
                                      const SufPackedLayout &layout,
                                      SufChannelId ch,
                                      uint32_t elem_idx,
                                      const std::vector<Share> &word_shares) {
    const auto *field = layout.find_field(ch, elem_idx);
    if (!field) return Share{};
    if (field->word_index >= word_shares.size()) return Share{};
    const Share &w = word_shares[field->word_index];
    std::uint64_t mask = (field->logical.width_bits == 64)
                             ? ~0ULL
                             : ((1ULL << field->logical.width_bits) - 1ULL);
    std::uint64_t v = (w.value_internal() >> field->bit_offset) & mask;
    return Share{w.party(), v & cfg.modulus_mask};
}

} // namespace cfss
