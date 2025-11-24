#pragma once

#include "arith.hpp"

namespace cfss {

// MaskedWire models the Composite-FSS masked wire: public masked value (hat),
// and additive shares of the true value and its input mask.
struct MaskedWire {
    std::uint64_t hat = 0;
    Share x;
    Share r_in;
};

// Public share: party 0 holds v, party 1 holds 0.
inline Share public_share(const RingConfig &cfg, std::uint64_t v, int party) {
    return Share{party, (party == 0) ? (v & cfg.modulus_mask) : 0};
}

// Recover a Share of the true value from a masked wire.
inline Share wire_to_share(const RingConfig &cfg, const MaskedWire &w, int party) {
    Share hat_share = public_share(cfg, w.hat, party);
    return sub(cfg, hat_share, w.r_in);
}

// Construct a masked wire from additive shares of x and r_in.
inline MaskedWire make_masked_wire(const RingConfig &cfg,
                                   const Share &x0,
                                   const Share &x1,
                                   const Share &r0,
                                   const Share &r1) {
    std::uint64_t r_sum = ring_add(cfg, r0.raw_value_unsafe(), r1.raw_value_unsafe());
    std::uint64_t x_sum = ring_add(cfg, x0.raw_value_unsafe(), x1.raw_value_unsafe());
    MaskedWire w;
    w.hat = ring_add(cfg, x_sum, r_sum);
    w.x = x0;
    w.r_in = r0;
    return w;
}

// Derive a new masked wire with the same x shares but a public offset c added.
inline MaskedWire add_const(const RingConfig &cfg, const MaskedWire &w, std::uint64_t c) {
    MaskedWire out = w;
    out.hat = ring_add(cfg, out.hat, c);
    return out;
}

} // namespace cfss
