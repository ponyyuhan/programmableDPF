#include "../include/composite_fss/suf_packing.hpp"
#include "../include/composite_fss/suf_unpack.hpp"
#include <cassert>
#include <iostream>

using namespace cfss;

int main() {
    // Layout determinism / non-overlap
    SufChannelRegistry reg;
    auto ch_a = reg.add_channel("a", SufFieldKind::Ring, 16, 2);
    auto ch_b = reg.add_channel("b", SufFieldKind::Bool, 1, 5);
    auto ch_c = reg.add_channel("idx", SufFieldKind::Index, 8, 1);
    auto layout = make_greedy_packed_layout(reg, 32);
    assert(layout.num_words >= 1);
    for (std::size_t i = 0; i < layout.fields.size(); ++i) {
        for (std::size_t j = i + 1; j < layout.fields.size(); ++j) {
            if (layout.fields[i].word_index != layout.fields[j].word_index) continue;
            auto s1 = layout.fields[i].bit_offset;
            auto e1 = s1 + layout.fields[i].logical.width_bits;
            auto s2 = layout.fields[j].bit_offset;
            auto e2 = s2 + layout.fields[j].logical.width_bits;
            assert(!(s1 < e2 && s2 < e1)); // no overlap
        }
    }

    // Pack -> unpack identity
    std::vector<std::uint64_t> words(layout.num_words, 0);
    auto set_field = [&](const SufPackedField &f, std::uint64_t v) {
        std::uint64_t mask = (f.logical.width_bits == 64) ? ~0ULL : ((1ULL << f.logical.width_bits) - 1ULL);
        words[f.word_index] |= (v & mask) << f.bit_offset;
    };
    set_field(*layout.find_field(ch_a, 0), 0xA);
    set_field(*layout.find_field(ch_a, 1), 0xB);
    for (uint32_t i = 0; i < 5; ++i) {
        set_field(*layout.find_field(ch_b, i), i & 1);
    }
    set_field(*layout.find_field(ch_c, 0), 42);

    assert(suf_unpack_channel_u64(layout, ch_a, 0, words) == 0xA);
    assert(suf_unpack_channel_u64(layout, ch_a, 1, words) == 0xB);
    for (uint32_t i = 0; i < 5; ++i) {
        assert(suf_unpack_channel_u64(layout, ch_b, i, words) == (i & 1));
    }
    assert(suf_unpack_channel_u64(layout, ch_c, 0, words) == 42);

    std::cout << "test_suf_packing: ok" << std::endl;
    return 0;
}
