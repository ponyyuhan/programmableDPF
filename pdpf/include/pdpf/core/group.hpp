// ================================================================
// File: include/pdpf/core/group.hpp
// ================================================================

#pragma once

#include <cstdint>
#include <vector>
#include <stdexcept>

namespace pdpf::core {

/**
 * GroupDescriptor represents finite abelian groups as in the PDPF paper:
 *   G ≅ Z_{q1} × ... × Z_{qℓ}, where qi > 1 (finite) or qi = 0 for Z (infinite). :contentReference[oaicite:4]{index=4}
 *
 * For G = Z, use moduli = {0}.
 */
struct GroupDescriptor {
    std::vector<std::uint64_t> moduli; // 0 means infinite Z in that component.

    bool empty() const noexcept { return moduli.empty(); }
    std::size_t arity() const noexcept { return moduli.size(); }

    bool is_infinite(std::size_t i) const {
        if (i >= moduli.size()) throw std::out_of_range("GroupDescriptor index");
        return moduli[i] == 0;
    }
};

/**
 * GroupElement = vector of signed 64-bit coordinates.
 * Interpretation:
 *  - If moduli[i] > 0, coordinate is taken modulo moduli[i].
 *  - If moduli[i] == 0, coordinate is in Z with no reduction.
 */
using GroupElement = std::vector<std::int64_t>;

/**
 * Convenience type for G = Z with payloads in {0,1} or small integers.
 */
struct GroupZ {
    using Value = std::int64_t;
};

/**
 * Add two group elements: out = a + b in group described by desc.
 */
void group_add(const GroupDescriptor &desc,
               const GroupElement &a,
               const GroupElement &b,
               GroupElement &out);

/**
 * Negate group element: out = -a.
 */
void group_negate(const GroupDescriptor &desc,
                  const GroupElement &a,
                  GroupElement &out);

/**
 * Add integer scalar (for G = Z or Z_q) into coordinate 0.
 * This is mainly used when payload set G' is a subset of Z or Z_q. :contentReference[oaicite:5]{index=5}
 */
void group_add_scalar(const GroupDescriptor &desc,
                      std::int64_t scalar,
                      GroupElement &out);

/**
 * Initialize a zero element for given group descriptor.
 */
GroupElement group_zero(const GroupDescriptor &desc);

} // namespace pdpf::core
