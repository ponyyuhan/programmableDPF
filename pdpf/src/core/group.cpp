// ================================================================
// File: src/core/group.cpp
// ================================================================

#include "pdpf/core/group.hpp"
#include <limits>

namespace pdpf::core {

static std::int64_t mod_reduce(std::int64_t x, std::uint64_t m) {
    if (m == 0) return x; // infinite Z case.
    std::int64_t mod = static_cast<std::int64_t>(m);
    std::int64_t r = static_cast<std::int64_t>(x % mod);
    if (r < 0) r += static_cast<std::int64_t>(m);
    return r;
}

void group_add(const GroupDescriptor &desc,
               const GroupElement &a,
               const GroupElement &b,
               GroupElement &out) {
    if (a.size() != desc.arity() || b.size() != desc.arity()) {
        throw std::invalid_argument("group_add: mismatched dimensions");
    }
    out.resize(desc.arity());
    for (std::size_t i = 0; i < desc.arity(); ++i) {
        std::int64_t v = a[i] + b[i];
        out[i] = mod_reduce(v, desc.moduli[i]);
    }
}

void group_negate(const GroupDescriptor &desc,
                  const GroupElement &a,
                  GroupElement &out) {
    if (a.size() != desc.arity()) {
        throw std::invalid_argument("group_negate: mismatched dimensions");
    }
    out.resize(desc.arity());
    for (std::size_t i = 0; i < desc.arity(); ++i) {
        if (desc.moduli[i] == 0) {
            out[i] = -a[i];
        } else {
            // Negation modulo m: -a ≡ m - a (mod m).
            std::uint64_t m = desc.moduli[i];
            std::int64_t v = static_cast<std::int64_t>(m) - a[i];
            out[i] = mod_reduce(v, m);
        }
    }
}

void group_add_scalar(const GroupDescriptor &desc,
                      std::int64_t scalar,
                      GroupElement &out) {
    if (desc.arity() == 0) {
        throw std::invalid_argument("group_add_scalar: empty group");
    }
    if (out.empty()) {
        out = group_zero(desc);
    }
    if (out.size() != desc.arity()) {
        throw std::invalid_argument("group_add_scalar: mismatched dimensions");
    }
    out[0] = mod_reduce(out[0] + scalar, desc.moduli[0]);
}

GroupElement group_zero(const GroupDescriptor &desc) {
    GroupElement e(desc.arity(), 0);
    return e;
}

} // namespace pdpf::core
