#pragma once

#include "pdpf.hpp"
#include "arith.hpp"
#include <cstdint>
#include <vector>
#include <stdexcept>
#include <string>
#include "sharing.hpp"

namespace cfss {

struct Poly {
    // c0 + c1 x + ... + cd x^d (coefficients in fixed-point ring)
    std::vector<std::int64_t> coeffs;
};

enum class SufFieldKind {
    Ring,
    Bool,
    Index,
};

struct SufChannelId {
    uint32_t id = 0;
    bool operator==(const SufChannelId &o) const { return id == o.id; }
};

struct SufChannelDesc {
    SufChannelId channel_id;
    std::string name;
    SufFieldKind kind = SufFieldKind::Ring;
    uint32_t width_bits = 64;
    uint32_t count = 1;
};

struct SufShape {
    uint32_t domain_bits = 0;
    uint32_t word_bits = 64;   // default ring word bits
    uint32_t num_words = 1;    // number of scalar outputs
    // Optional logical channels; if empty, channels are inferred from outputs.
    std::vector<struct SufChannelDesc> channels;
    uint32_t next_channel_id = 0;

    SufChannelId add_channel(const std::string &name,
                             SufFieldKind kind,
                             uint32_t width_bits,
                             uint32_t count) {
        SufChannelId id{next_channel_id++};
        channels.push_back(SufChannelDesc{id, name, kind, width_bits, count});
        return id;
    }
};

struct BoolExpr {
    enum Kind {
        CONST,
        LT_CONST,
        LT_MOD,
        MSB,
        MSB_SHIFT,
        ATOM_ID,
        NOT,
        AND,
        OR,
        XOR,
    };
    Kind kind = CONST;
    bool const_value = false;
    std::uint64_t param = 0;
    unsigned f = 0; // for LT_MOD
    unsigned atom_idx = 0; // for ATOM_ID (structured backend)
    // Children are used for NOT (size 1) and AND/OR/XOR (size 2).
    std::vector<BoolExpr> children;
};

struct PolyVec {
    std::vector<Poly> polys;
};

struct SufDesc {
    // SufDesc semantics:
    // - Domain: x ∈ Z_{2^{shape.domain_bits}}, treated as unsigned.
    // - alpha: sorted boundaries define intervals [alpha_i, alpha_{i+1}).
    // - polys[i]: arithmetic outputs P_i(x) on interval i.
    // - bools[i]: boolean outputs B_i(x) on interval i expressed via BoolExpr.
    // - r_in, r_out: masks; SUF is defined on unmasked x, we evaluate on
    //   \hat{x} = x + r_in (mod 2^n) and add r_out to arithmetic outputs.
    SufShape shape;
    unsigned r_outputs = 0; // arithmetic outputs
    unsigned l_outputs = 0; // boolean outputs
    unsigned degree = 0;
    std::vector<std::uint64_t> alpha;               // sorted boundaries
    std::vector<PolyVec> polys;                     // per-interval arithmetic polys
    std::vector<std::vector<BoolExpr>> bools;       // per-interval boolean exprs
    std::uint64_t r_in = 0;
    std::uint64_t r_out = 0;
};

struct SufCompiled {
    enum class Backend { LutFull, Structured };
    Backend backend = Backend::LutFull;
    PdpfProgramId pdpf_program = 0; // legacy single-program view
    PdpfProgramId cmp_prog = 0;
    PdpfProgramId poly_prog = 0;
    PdpfProgramId lut_prog = 0;
    unsigned domain_bits = 0;
    unsigned num_arith_outputs = 0;
    unsigned num_bool_outputs = 0;
    unsigned output_words = 0; // for pdpf_program
    unsigned arith_words = 0;  // packed arithmetic words (LutFull)
    unsigned bool_words = 0;   // packed boolean words (LutFull/Structured)
    SufShape shape;
    std::vector<BoolExpr> structured_bools; // packed bool expressions (when backend==Structured)
    std::vector<PredicateSpec> predicate_specs; // primitive predicates (Structured)
    std::uint64_t r_in = 0;
    std::uint64_t r_out = 0;
};

inline std::uint64_t eval_poly_mod(const Poly &p, std::uint64_t x) {
    std::uint64_t acc = 0;
    std::uint64_t pow = 1;
    for (std::size_t i = 0; i < p.coeffs.size(); ++i) {
        std::uint64_t coeff = static_cast<std::uint64_t>(p.coeffs[i]);
        acc = acc + coeff * pow;
        pow = pow * x;
    }
    return acc;
}

inline bool eval_bool_expr(const BoolExpr &expr, std::uint64_t x, unsigned n_bits) {
    switch (expr.kind) {
        case BoolExpr::CONST:
            return expr.const_value;
        case BoolExpr::LT_CONST:
            return x < expr.param;
        case BoolExpr::LT_MOD: {
            std::uint64_t mask = (expr.f >= 64) ? ~0ULL : ((1ULL << expr.f) - 1ULL);
            return (x & mask) < expr.param;
        }
        case BoolExpr::MSB:
            return (x >> ((n_bits == 0) ? 0 : (n_bits - 1))) & 1ULL;
        case BoolExpr::MSB_SHIFT:
            return ((x + expr.param) >> ((n_bits == 0) ? 0 : (n_bits - 1))) & 1ULL;
        case BoolExpr::ATOM_ID:
            return false;
        case BoolExpr::NOT:
            return !expr.children.empty() && !eval_bool_expr(expr.children[0], x, n_bits);
        case BoolExpr::AND:
            return expr.children.size() == 2
                   && eval_bool_expr(expr.children[0], x, n_bits)
                   && eval_bool_expr(expr.children[1], x, n_bits);
        case BoolExpr::OR:
            return expr.children.size() == 2
                   && (eval_bool_expr(expr.children[0], x, n_bits) || eval_bool_expr(expr.children[1], x, n_bits));
        case BoolExpr::XOR:
            return expr.children.size() == 2
                   && (eval_bool_expr(expr.children[0], x, n_bits) != eval_bool_expr(expr.children[1], x, n_bits));
        default:
            return false;
    }
}

inline std::size_t find_interval(const SufDesc &desc, std::uint64_t x) {
    if (desc.alpha.empty()) return 0;
    for (std::size_t i = 0; i + 1 < desc.alpha.size(); ++i) {
        if (x >= desc.alpha[i] && x < desc.alpha[i + 1]) {
            return i;
        }
    }
    return desc.alpha.size() - 1;
}

// Compile a SUF description into packed PDPF programs (currently LUT-backed).
inline SufCompiled compile_suf_to_pdpf(const SufDesc &desc, PdpfEngine &engine) {
    if (desc.shape.domain_bits == 0) throw std::runtime_error("SufDesc: n_bits must be > 0");
    if (desc.shape.domain_bits >= 63) throw std::runtime_error("SufDesc: LUT backend supports domain_bits < 63");
    std::size_t domain_size = 1ULL << desc.shape.domain_bits;
    unsigned arith_words = desc.r_outputs;
    unsigned bool_words = (desc.l_outputs + 63) / 64;
    if (arith_words + bool_words == 0) throw std::runtime_error("SufDesc: no outputs");

    std::vector<std::uint64_t> arith_flat(domain_size * arith_words, 0);
    std::vector<std::uint64_t> bool_flat(domain_size * bool_words, 0);
    std::uint64_t mask = (desc.shape.domain_bits == 64) ? ~0ULL : ((1ULL << desc.shape.domain_bits) - 1ULL);
    std::uint64_t modulus = (desc.shape.domain_bits == 64) ? 0ULL : (1ULL << desc.shape.domain_bits);
    RingConfig cfg = make_ring_config(desc.shape.domain_bits);

    for (std::size_t x = 0; x < domain_size; ++x) {
        std::uint64_t unmasked = 0;
        if (desc.shape.domain_bits == 64) {
            unmasked = static_cast<std::uint64_t>(x) - desc.r_in;
        } else {
            unmasked = (static_cast<std::uint64_t>(x) + modulus - (desc.r_in & mask)) & mask;
        }
        // We evaluate on x = \hat{x} - r_in (mod 2^n); predicates and polys use unmasked x.
        std::size_t interval_idx = find_interval(desc, unmasked);
        if (interval_idx >= desc.polys.size()) continue;
        const auto &pvec = desc.polys[interval_idx].polys;
        std::size_t base_arith = x * arith_words;
        for (unsigned r = 0; r < desc.r_outputs && r < pvec.size(); ++r) {
            std::uint64_t val = eval_poly_mod(pvec[r], unmasked);
            val = ring_add(cfg, val, desc.r_out & mask);
            arith_flat[base_arith + r] = val;
        }
        if (desc.l_outputs > 0 && interval_idx < desc.bools.size()) {
            for (unsigned b = 0; b < desc.l_outputs && b < desc.bools[interval_idx].size(); ++b) {
                bool bit = eval_bool_expr(desc.bools[interval_idx][b], unmasked, desc.shape.domain_bits);
                std::size_t word_idx = b / 64;
                std::size_t bit_off = b % 64;
                std::size_t base = x * bool_words;
                bool_flat[base + word_idx] |= (static_cast<std::uint64_t>(bit) << bit_off);
            }
        }
    }

    PdpfProgramId poly_pid = 0;
    PdpfProgramId cmp_pid = 0;
    if (arith_words > 0) {
        LutProgramDesc lut_desc;
        lut_desc.domain_bits = desc.shape.domain_bits;
        lut_desc.output_words = arith_words;
        poly_pid = engine.make_lut_program(lut_desc, arith_flat);
    }
    if (bool_words > 0) {
        if (bool_words == 1) {
            CmpProgramDesc cmp_desc{desc.shape.domain_bits};
            cmp_pid = engine.make_cmp_program(cmp_desc, bool_flat);
        } else {
            LutProgramDesc lut_desc{desc.shape.domain_bits, bool_words};
            cmp_pid = engine.make_lut_program(lut_desc, bool_flat);
        }
    }

    SufCompiled compiled;
    compiled.pdpf_program = (poly_pid != 0) ? poly_pid : cmp_pid;
    compiled.cmp_prog = cmp_pid;
    compiled.poly_prog = poly_pid;
    compiled.lut_prog = poly_pid;
    compiled.backend = SufCompiled::Backend::LutFull;
    compiled.domain_bits = desc.shape.domain_bits;
    compiled.num_arith_outputs = desc.r_outputs;
    compiled.num_bool_outputs = desc.l_outputs;
    compiled.output_words = (arith_words > 0) ? arith_words : bool_words;
    compiled.arith_words = arith_words;
    compiled.bool_words = bool_words;
    compiled.shape = desc.shape;
    compiled.r_in = desc.r_in;
    compiled.r_out = desc.r_out;
    return compiled;
}

// Compile SUF into a single LUT program that packs arithmetic + boolean outputs
// into one multi-output Pdpf program.
inline SufCompiled compile_suf_to_pdpf_packed(const SufDesc &desc, PdpfEngine &engine) {
    if (desc.shape.domain_bits == 0) throw std::runtime_error("SufDesc: n_bits must be > 0");
    if (desc.shape.domain_bits >= 63) throw std::runtime_error("SufDesc: LUT backend supports domain_bits < 63");
    std::size_t domain_size = 1ULL << desc.shape.domain_bits;
    unsigned arith_words = desc.r_outputs;
    unsigned bool_words = (desc.l_outputs + 63) / 64;
    unsigned output_words = arith_words + bool_words;
    if (output_words == 0) throw std::runtime_error("SufDesc: no outputs");

    std::vector<std::uint64_t> table_flat(domain_size * output_words, 0);
    std::uint64_t mask = (desc.shape.domain_bits == 64) ? ~0ULL : ((1ULL << desc.shape.domain_bits) - 1ULL);
    std::uint64_t modulus = (desc.shape.domain_bits == 64) ? 0ULL : (1ULL << desc.shape.domain_bits);
    RingConfig cfg = make_ring_config(desc.shape.domain_bits);

    for (std::size_t x = 0; x < domain_size; ++x) {
        std::uint64_t unmasked = 0;
        if (desc.shape.domain_bits == 64) {
            unmasked = static_cast<std::uint64_t>(x) - desc.r_in;
        } else {
            unmasked = (static_cast<std::uint64_t>(x) + modulus - (desc.r_in & mask)) & mask;
        }
        // Evaluate predicates and polys on unmasked x = \hat{x} - r_in (mod 2^n).
        std::size_t interval_idx = find_interval(desc, unmasked);
        if (interval_idx >= desc.polys.size()) continue;
        const auto &pvec = desc.polys[interval_idx].polys;
        std::size_t base = x * output_words;
        for (unsigned r = 0; r < desc.r_outputs && r < pvec.size(); ++r) {
            std::uint64_t val = eval_poly_mod(pvec[r], unmasked);
            val = ring_add(cfg, val, desc.r_out & mask);
            table_flat[base + r] = val;
        }
        if (desc.l_outputs > 0 && interval_idx < desc.bools.size()) {
            for (unsigned b = 0; b < desc.l_outputs && b < desc.bools[interval_idx].size(); ++b) {
                bool bit = eval_bool_expr(desc.bools[interval_idx][b], unmasked, desc.shape.domain_bits);
                std::size_t word_idx = b / 64;
                std::size_t bit_off = b % 64;
                table_flat[base + arith_words + word_idx] |= (static_cast<std::uint64_t>(bit) << bit_off);
            }
        }
    }

    LutProgramDesc lut_desc;
    lut_desc.domain_bits = desc.shape.domain_bits;
    lut_desc.output_words = output_words;
    PdpfProgramId pid = engine.make_lut_program(lut_desc, table_flat);

    SufCompiled compiled;
    compiled.pdpf_program = pid;
    compiled.poly_prog = pid;
    compiled.lut_prog = pid;
    compiled.backend = SufCompiled::Backend::LutFull;
    compiled.domain_bits = desc.shape.domain_bits;
    compiled.num_arith_outputs = desc.r_outputs;
    compiled.num_bool_outputs = desc.l_outputs;
    compiled.output_words = output_words;
    compiled.arith_words = arith_words;
    compiled.bool_words = bool_words;
    compiled.shape = desc.shape;
    compiled.r_in = desc.r_in;
    compiled.r_out = desc.r_out;
    return compiled;
}

// Helper: build a SUF that encodes a flat LUT table (domain 2^n_bits, output_words words).
inline SufDesc table_to_suf(unsigned n_bits,
                            unsigned output_words,
                            const std::vector<std::uint64_t> &table_flat) {
    if (n_bits >= 63) {
        throw std::runtime_error("table_to_suf: n_bits must be < 63 for LUT backend");
    }
    std::size_t domain_size = 1ULL << n_bits;
    if (table_flat.size() != domain_size * output_words) {
        throw std::runtime_error("table_to_suf: size mismatch");
    }
    SufDesc suf;
    suf.shape.domain_bits = n_bits;
    suf.shape.num_words = output_words;
    suf.shape.channels.clear();
    for (unsigned w = 0; w < output_words; ++w) {
        SufChannelDesc ch;
        ch.channel_id = SufChannelId{w};
        ch.name = "out_" + std::to_string(w);
        ch.kind = SufFieldKind::Ring;
        ch.width_bits = n_bits;
        ch.count = 1;
        suf.shape.channels.push_back(ch);
    }
    suf.r_outputs = output_words;
    suf.l_outputs = 0;
    suf.degree = 0;
    suf.alpha.reserve(domain_size);
    suf.polys.reserve(domain_size);
    suf.bools.reserve(domain_size);
    for (std::size_t x = 0; x < domain_size; ++x) {
        suf.alpha.push_back(x);
        PolyVec pv;
        pv.polys.resize(output_words);
        for (unsigned w = 0; w < output_words; ++w) {
            pv.polys[w].coeffs = {static_cast<std::int64_t>(table_flat[x * output_words + w])};
        }
        suf.polys.push_back(std::move(pv));
        suf.bools.push_back({});
    }
    return suf;
}

// --- Legacy interval-based SUF for backward compatibility ---
struct BoolPredicate {
    enum Kind { LT_CONST, LT_MOD, MSB, MSB_SHIFT };
    Kind kind = LT_CONST;
    std::uint64_t param = 0;
    unsigned f = 0; // for LT_MOD
};

struct SufInterval {
    std::uint64_t alpha_start = 0;
    std::uint64_t alpha_end = 0; // exclusive
    std::vector<Poly> polys; // one per arithmetic output
    std::vector<BoolPredicate> bool_preds; // one per boolean output
};

struct SufFunction {
    unsigned n_bits = 0;
    unsigned r_outputs = 0; // arithmetic outputs
    unsigned l_outputs = 0; // boolean outputs
    unsigned degree = 0;
    std::vector<SufInterval> intervals;
};

inline BoolExpr pred_to_expr(const BoolPredicate &pred) {
    BoolExpr e;
    switch (pred.kind) {
        case BoolPredicate::LT_CONST:
            e.kind = BoolExpr::LT_CONST;
            e.param = pred.param;
            break;
        case BoolPredicate::LT_MOD:
            e.kind = BoolExpr::LT_MOD;
            e.param = pred.param;
            e.f = pred.f;
            break;
        case BoolPredicate::MSB:
            e.kind = BoolExpr::MSB;
            break;
        case BoolPredicate::MSB_SHIFT:
            e.kind = BoolExpr::MSB_SHIFT;
            e.param = pred.param;
            break;
        default:
            e.kind = BoolExpr::CONST;
            e.const_value = false;
            break;
    }
    return e;
}

inline SufDesc legacy_to_desc(const SufFunction &fn) {
    SufDesc desc;
    desc.shape.domain_bits = fn.n_bits;
    desc.shape.num_words = fn.r_outputs + ((fn.l_outputs > 0) ? 1u : 0u);
    desc.r_outputs = fn.r_outputs;
    desc.l_outputs = fn.l_outputs;
    desc.degree = fn.degree;
    desc.alpha.reserve(fn.intervals.size());
    desc.polys.reserve(fn.intervals.size());
    desc.bools.reserve(fn.intervals.size());
    for (const auto &iv : fn.intervals) {
        desc.alpha.push_back(iv.alpha_start);
        PolyVec pv;
        pv.polys = iv.polys;
        desc.polys.push_back(std::move(pv));
        std::vector<BoolExpr> exprs;
        exprs.reserve(iv.bool_preds.size());
        for (const auto &bp : iv.bool_preds) {
            exprs.push_back(pred_to_expr(bp));
        }
        desc.bools.push_back(std::move(exprs));
    }
    return desc;
}

inline SufCompiled compile_suf_to_pdpf(const SufFunction &fn, PdpfEngine &engine) {
    return compile_suf_to_pdpf(legacy_to_desc(fn), engine);
}

// Structured backend: predicate extraction + local boolean eval on predicate shares.
struct PrimitiveRegistry {
    std::vector<PredicateSpec> preds;

    unsigned intern(const PredicateSpec &spec) {
        for (unsigned i = 0; i < preds.size(); ++i) {
            const auto &p = preds[i];
            if (p.kind == spec.kind && p.param == spec.param && p.f == spec.f) {
                return i;
            }
        }
        preds.push_back(spec);
        return static_cast<unsigned>(preds.size() - 1);
    }

    unsigned intern_lt_const(std::uint64_t beta) {
        PredicateSpec spec;
        spec.kind = PredicateSpec::Kind::LT_CONST;
        spec.param = beta;
        return intern(spec);
    }

    unsigned intern_lt_mod(std::uint64_t gamma, unsigned f) {
        PredicateSpec spec;
        spec.kind = PredicateSpec::Kind::LT_MOD;
        spec.param = gamma;
        spec.f = f;
        return intern(spec);
    }

    unsigned intern_msb() {
        PredicateSpec spec;
        spec.kind = PredicateSpec::Kind::MSB;
        return intern(spec);
    }

    unsigned intern_msb_shift(std::uint64_t c) {
        PredicateSpec spec;
        spec.kind = PredicateSpec::Kind::MSB_SHIFT;
        spec.param = c;
        return intern(spec);
    }
};

inline void rewrite_bool_expr_to_atoms(BoolExpr &expr, PrimitiveRegistry &reg) {
    switch (expr.kind) {
        case BoolExpr::LT_CONST:
            expr.atom_idx = reg.intern_lt_const(expr.param);
            expr.kind = BoolExpr::ATOM_ID;
            expr.children.clear();
            break;
        case BoolExpr::LT_MOD:
            expr.atom_idx = reg.intern_lt_mod(expr.param, expr.f);
            expr.kind = BoolExpr::ATOM_ID;
            expr.children.clear();
            break;
        case BoolExpr::MSB:
            expr.atom_idx = reg.intern_msb();
            expr.kind = BoolExpr::ATOM_ID;
            expr.children.clear();
            break;
        case BoolExpr::MSB_SHIFT:
            expr.atom_idx = reg.intern_msb_shift(expr.param);
            expr.kind = BoolExpr::ATOM_ID;
            expr.children.clear();
            break;
        case BoolExpr::NOT:
        case BoolExpr::AND:
        case BoolExpr::OR:
        case BoolExpr::XOR:
            for (auto &c : expr.children) {
                rewrite_bool_expr_to_atoms(c, reg);
            }
            break;
        default:
            break;
    }
}

inline bool eval_bool_expr_on_bits(const BoolExpr &expr,
                                   const std::vector<std::uint64_t> &words) {
    switch (expr.kind) {
        case BoolExpr::CONST:
            return expr.const_value;
        case BoolExpr::ATOM_ID: {
            unsigned idx = expr.atom_idx;
            std::size_t word_idx = idx / 64;
            std::size_t bit_off = idx % 64;
            if (word_idx >= words.size()) return false;
            return (words[word_idx] >> bit_off) & 1ULL;
        }
        case BoolExpr::NOT:
            return !expr.children.empty() && !eval_bool_expr_on_bits(expr.children[0], words);
        case BoolExpr::AND:
            return expr.children.size() == 2
                   && eval_bool_expr_on_bits(expr.children[0], words)
                   && eval_bool_expr_on_bits(expr.children[1], words);
        case BoolExpr::OR:
            return expr.children.size() == 2
                   && (eval_bool_expr_on_bits(expr.children[0], words) || eval_bool_expr_on_bits(expr.children[1], words));
        case BoolExpr::XOR:
            return expr.children.size() == 2
                   && (eval_bool_expr_on_bits(expr.children[0], words) != eval_bool_expr_on_bits(expr.children[1], words));
        default:
            return false;
    }
}

inline std::vector<std::uint64_t> eval_bool_exprs_on_bits(const SufCompiled &compiled,
                                                          const std::vector<std::uint64_t> &pred_words) {
    std::vector<std::uint64_t> out((compiled.num_bool_outputs + 63) / 64, 0);
    if (compiled.backend != SufCompiled::Backend::Structured || compiled.structured_bools.empty()) {
        return out;
    }
    for (unsigned b = 0; b < compiled.num_bool_outputs && b < compiled.structured_bools.size(); ++b) {
        bool bit = eval_bool_expr_on_bits(compiled.structured_bools[b], pred_words);
        std::size_t word_idx = b / 64;
        std::size_t bit_off = b % 64;
        out[word_idx] |= (static_cast<std::uint64_t>(bit) << bit_off);
    }
    return out;
}

inline std::vector<std::uint64_t> eval_predicates_local(const SufCompiled &compiled,
                                                        std::uint64_t masked_x) {
    unsigned num_preds = static_cast<unsigned>(compiled.predicate_specs.size());
    unsigned bool_words = (num_preds + 63) / 64;
    std::vector<std::uint64_t> pred_words(bool_words, 0);
    if (compiled.backend != SufCompiled::Backend::Structured || num_preds == 0) {
        return pred_words;
    }
    for (unsigned i = 0; i < num_preds; ++i) {
        bool bit = eval_primitive_pred(compiled.predicate_specs[i],
                                       masked_x,
                                       compiled.domain_bits,
                                       compiled.r_in);
        if (!bit) continue;
        std::size_t word_idx = i / 64;
        std::size_t bit_off = i % 64;
        pred_words[word_idx] |= (1ULL << bit_off);
    }
    return pred_words;
}

// Structured compiler: extract primitive predicates, compile via engine, keep rewritten BoolExprs.
inline SufCompiled compile_suf_to_pdpf_structured(const SufDesc &desc, PdpfEngine &engine) {
    if (desc.shape.domain_bits == 0) throw std::runtime_error("SufDesc: n_bits must be > 0");
    if (desc.l_outputs == 0) {
        throw std::runtime_error("structured SUF: no boolean outputs");
    }

    PrimitiveRegistry reg;
    std::vector<std::vector<BoolExpr>> bools_rewritten = desc.bools;
    for (auto &interval_bools : bools_rewritten) {
        for (auto &b : interval_bools) {
            rewrite_bool_expr_to_atoms(b, reg);
        }
    }
    unsigned num_preds = static_cast<unsigned>(reg.preds.size());
    if (num_preds == 0) {
        throw std::runtime_error("structured SUF: no primitive predicates collected");
    }

    SufCompiled compiled;
    compiled.backend = SufCompiled::Backend::Structured;
    bool use_predicate_prog = desc.shape.domain_bits < 31;
    PdpfProgramId pid = 0;
    if (use_predicate_prog) {
        PredProgramDesc pdesc;
        pdesc.domain_bits = desc.shape.domain_bits;
        pdesc.num_preds = num_preds;
        pdesc.r_in = desc.r_in;
        pid = engine.make_predicate_program(pdesc, reg.preds);
    }
    compiled.pdpf_program = pid;
    compiled.cmp_prog = pid;
    compiled.domain_bits = desc.shape.domain_bits;
    compiled.num_arith_outputs = desc.r_outputs;
    compiled.num_bool_outputs = desc.l_outputs;
    compiled.bool_words = (num_preds + 63) / 64;
    compiled.output_words = compiled.bool_words;
    compiled.arith_words = 0;
    compiled.shape = desc.shape;
    compiled.r_in = desc.r_in;
    compiled.r_out = desc.r_out;
    compiled.predicate_specs = reg.preds;
    compiled.structured_bools = bools_rewritten.empty() ? std::vector<BoolExpr>{} : bools_rewritten.front();
    return compiled;
}

inline std::vector<std::uint64_t> eval_structured_bool(const SufCompiled &compiled, std::uint64_t masked_x) {
    auto pred_words = eval_predicates_local(compiled, masked_x);
    return eval_bool_exprs_on_bits(compiled, pred_words);
}

// Deterministically split a structured bool result into additive shares of 0/1.
inline Share structured_bool_share(int party,
                                   const SufCompiled &compiled,
                                   std::uint64_t masked_x,
                                   unsigned bit_idx) {
    auto pred_words = eval_predicates_local(compiled, masked_x);
    bool bit = false;
    if (bit_idx < compiled.structured_bools.size()) {
        bit = eval_bool_expr_on_bits(compiled.structured_bools[bit_idx], pred_words);
    }
    Ring64 ring(compiled.domain_bits);
    // Use masked_x and bit_idx to derive a nonce.
    std::uint64_t nonce = masked_x ^ (static_cast<std::uint64_t>(bit_idx) << 48) ^ 0xBADC0FFEEULL;
    return deterministic_share(party, ring, bit ? 1ULL : 0ULL, nonce);
}

} // namespace cfss
