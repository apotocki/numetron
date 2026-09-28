// Numetron — Compile-time and runtime arbitrary-precision arithmetic
// (c) Alexander Pototskiy
// Licensed under the MIT License. See LICENSE file for details.

// The multiplication chain above the basecase: the choice among Karatsuba, the Toom plans and the
// FFT (umul_dispatch, usqr_dispatch) and everything they run. This is the heavy part of the
// headers (the Toom engine, every plan, the FFT); umul.hpp includes it only without
// NUMETRON_COMPILED, which moves it into the numetron library (src/umul_large.cpp).

#pragma once

#include "umul_basecase.hpp"
#include "toom/engine.hpp"
#include "toom/thresholds.hpp"
#include "numetron/detail/stack_allocator.hpp"
#include "numetron/config/implementation.hpp" // NUMETRON_KARATSUBA_IMPL, NUMETRON_TOOM3_IMPL

#include "umul_karatsuba.hpp"
#include "umul_karatsuba_fused.hpp"
#if NUMETRON_KARATSUBA_IMPL == NUMETRON_KARATSUBA_IMPL_ASM
#   include "umul_karatsuba_asm.hpp"
#endif

// Always included: besides the hand-written Toom-3 it defines detail::toom3_split_fits(), which
// also gates the engine's balanced Toom-3 plan. NUMETRON_TOOM3_IMPL only picks which of the two
// runs for balanced operands.
#include "umul_toom3.hpp"

#include "umul_fft.hpp"

namespace numetron::limb_arithmetic {

// The FFT takes any shape (its length follows un + vn). A long u (un >= 2vn, cut into pieces that
// fill a length: fft_slice_length) from vn >= fft_threshold(); below that one transform, where
// its coefficient count reaches fft_threshold() and the length is well filled
// (detail::fft_product_fills). 64-bit limbs only.
template <std::unsigned_integral LimbT>
inline bool is_fft_applicable([[maybe_unused]] size_t un, size_t vn) noexcept
{
    assert(un >= vn);
    if constexpr (sizeof(LimbT) == 8) {
        if (un >= 2 * vn) return vn >= fft_threshold();
        return detail::fft_product_fills(un, vn);
    }
    else return false;
}

// u at least twice as long as v: u is cut into vn-limb pieces, each multiplied by v as a
// balanced product (detail::umul_sliced). From the Karatsuba threshold up the pieces get
// Karatsuba/Toom; from slicing_threshold() up (below it when Karatsuba starts late) even
// basecase pieces pay off -- one long basecase runs rows of un limbs, and the basecase is slower
// per limb on long rows than on vn-limb ones. Below both, one basecase keeps any un.
inline bool is_slicing_applicable(size_t un, size_t vn) noexcept
{
    assert(un >= vn);
    return vn >= (std::min)(karatsuba_threshold(), slicing_threshold()) && un >= 2 * vn;
}

// Toom-3/2 for 1.25 <= un/vn < 1.75 (GMP's window between toom22 and toom42); below that
// Karatsuba, above it Karatsuba up to 2vn and slicing from there.
inline bool is_toom32_applicable(size_t un, size_t vn) noexcept
{
    assert(un >= vn);
    return vn >= toom32_threshold() && 4 * un >= 5 * vn && 4 * un < 7 * vn && detail::toom32_split_fits(un, vn);
}

// Toom-6.5 7 x 6 for un/vn < 1.4 in the Toom-6.5 range: after the balanced Toom-6.5 (which
// takes up to ~1.2), before Toom-4 and toom32, which would take these shapes otherwise.
inline bool is_toom76_applicable(size_t un, size_t vn) noexcept
{
    assert(un >= vn);
    return vn >= toom76_threshold() && 5 * un < 7 * vn && detail::toom76_split_fits(un, vn);
}

// Toom-8.5 N x (17 - N) (toom_8h_half.hpp), each in its un/vn window from its threshold up: 9 x 8
// for 1.08 <= un/vn < 1.2, 10 x 7 for 1.32 <= un/vn < 5/3, 11 x 6 for 5/3 <= un/vn < 2. Checked
// before the balanced Toom-8.5 / Toom-6.5, toom76 and the smaller unbalanced plans, which would
// take these shapes otherwise (outside the windows those are as fast or faster).
inline bool is_toom98_applicable(size_t un, size_t vn) noexcept
{
    assert(un >= vn);
    return vn >= toom98_threshold() && 25 * un >= 27 * vn && 5 * un < 6 * vn && detail::toom8h_half_split_fits<9>(un, vn);
}

inline bool is_toom107_applicable(size_t un, size_t vn) noexcept
{
    assert(un >= vn);
    return vn >= toom107_threshold() && 25 * un >= 33 * vn && detail::toom8h_half_split_fits<10>(un, vn);
}

// Near un/vn = 2 v's top piece is only about half a chunk while Toom-6/3 splits u into six full
// ones, so from un >= 1.95 vn (slicing's 2vn - 1 pieces) 11 x 6 takes over only from 7/4 of its
// threshold (2vn - 1 x vn: even at ~900 on GCC, ~800-960 on MSVC, with the thresholds at 500 / 531).
inline bool is_toom116_applicable(size_t un, size_t vn) noexcept
{
    assert(un >= vn);
    const size_t t = toom116_threshold();
    return vn >= t && un < 2 * vn && (20 * un < 39 * vn || vn - t >= t / 4 * 3) && detail::toom8h_half_split_fits<11>(un, vn);
}

// Toom-5/4 (toom_5x4.hpp) for 1.2 <= un/vn < 1.45 and Toom-5/3 (toom_5x3.hpp) for
// 1.55 <= un/vn < 1.85, each from its threshold up to the Toom-6.5 one: below that they beat
// what takes these shapes otherwise (toom32, toom42, toom63, toom76, the Toom-8.5 halves; 0.85-0.98
// at vn = 192..512, both compilers), above it the plans of the Toom-6.5 / 8.5 family, splitting
// into ~vn/6 pieces against their ~vn/3..vn/4, do. Checked first after slicing.
inline bool is_toom54_applicable(size_t un, size_t vn) noexcept
{
    assert(un >= vn);
    return vn >= toom54_threshold() && vn < toom6h_threshold() && 5 * un >= 6 * vn && 20 * un < 29 * vn
        && detail::toom54_split_fits(un, vn);
}

// Toom-4/3 (toom_4x3.hpp) for 1.3 <= un/vn < 1.45, the same way, checked before toom54: there
// its six products of ~vn/3 beat toom54's eight of ~vn/4 (0.92-0.98 at vn = 192..512).
inline bool is_toom43_applicable(size_t un, size_t vn) noexcept
{
    assert(un >= vn);
    return vn >= toom43_threshold() && vn < toom6h_threshold() && 10 * un >= 13 * vn && 20 * un < 29 * vn
        && detail::toom43_split_fits(un, vn);
}

inline bool is_toom53_applicable(size_t un, size_t vn) noexcept
{
    assert(un >= vn);
    return vn >= toom53_threshold() && vn < toom6h_threshold() && 20 * un >= 31 * vn && 20 * un < 37 * vn
        && detail::toom53_split_fits(un, vn);
}

// Toom-6/3 for 1.75 <= un/vn < 2 from its threshold up: the same window as toom42, checked first
// (eight products of ~vn/3 against toom42's five of ~vn/2).
inline bool is_toom63_applicable(size_t un, size_t vn) noexcept
{
    assert(un >= vn);
    return vn >= toom63_threshold() && 4 * un >= 7 * vn && un < 2 * vn && detail::toom63_split_fits(un, vn);
}

// Toom-4/2 for 1.75 <= un/vn < 2 (toom32's window ends at 1.75, slicing starts at 2).
inline bool is_toom42_applicable(size_t un, size_t vn) noexcept
{
    assert(un >= vn);
    return vn >= toom42_threshold() && 4 * un >= 7 * vn && un < 2 * vn && detail::toom42_split_fits(un, vn);
}

inline bool is_karatsuba_applicable(size_t un, size_t vn) noexcept
{
    assert(un >= vn);
    // Condition is expressed in terms of vn so that Toom-k dispatch is uniform:
    // each algorithm checks vn >= its threshold and un < k*vn (u fits in k pieces).
    // For Karatsuba (k=2): vn >= threshold and un < 2*vn.
    return vn >= karatsuba_threshold() && 2 * vn > un;
}

inline bool is_toom3_applicable(size_t un, size_t vn) noexcept
{
    assert(un >= vn);
    return vn >= toom3_threshold() && 3 * vn > un;
}

// Balanced Toom-4 only: more unbalanced products fall through to Toom-3 / Karatsuba.
inline bool is_toom4_applicable(size_t un, size_t vn) noexcept
{
    assert(un >= vn);
    return vn >= toom4_threshold() && detail::toom4_split_fits(un, vn);
}

// Balanced Toom-6.5 only, like Toom-4.
inline bool is_toom6h_applicable(size_t un, size_t vn) noexcept
{
    assert(un >= vn);
    return vn >= toom6h_threshold() && detail::toom6h_split_fits(un, vn);
}

// Balanced Toom-8.5 only, like Toom-4.
inline bool is_toom8h_applicable(size_t un, size_t vn) noexcept
{
    assert(un >= vn);
    return vn >= toom8h_threshold() && detail::toom8h_split_fits(un, vn);
}

namespace detail {

// rb[0..un+vn) = u * v for un >= vn (is_slicing_applicable): u is cut into pieces of pl limbs
// (vn, or 2vn - 1, see below) u_0, u_1, ... (the last one shorter), and u*v = sum u_k*v*B^(k*pl).
// Each piece product goes straight into rb at k*pl; its low vn limbs overlap the high vn limbs of
// the previous one, which are saved to a vn-limb scratch first and added back. Every piece product
// goes through umul_dispatch, so the balanced ones get the whole Karatsuba/Toom/FFT chain and
// the short last piece is itself sliced again if it is short enough. pl = 0 picks the piece
// length below; the FFT range passes its own (fft_slice_length()), with full_piece(dst, a) for
// the full pl-limb pieces (dst[0..pl+vn) = a * v; umul_fft_sliced()).
// Returns rb + un + vn.
template <std::unsigned_integral LimbT, typename AllocatorT, typename FullPieceF = std::nullptr_t>
LimbT* umul_sliced(const LimbT* u, size_t un, const LimbT* v, size_t vn, LimbT* rb, AllocatorT alloc, size_t pl = 0,
    FullPieceF&& full_piece = nullptr)
{
    NUMETRON_ASSERT(vn > 0 && un >= vn);
    LimbT* const re = rb + un + vn;

    // dst[0..an+vn) = a * v, zero-padding what umul_dispatch leaves above the significant limbs
    // (it strips leading zero limbs, and a piece from the middle of u may have them).
    auto mul_piece = [&](LimbT* dst, const LimbT* a, size_t an) {
        if constexpr (!std::is_same_v<std::remove_cvref_t<FullPieceF>, std::nullptr_t>) {
            if (an == pl) {
                full_piece(dst, a);
                return;
            }
        }
        LimbT* e = umul_dispatch(a, an, v, vn, dst, alloc);
        std::memset(e, 0, static_cast<size_t>(dst + an + vn - e) * sizeof(LimbT));
    };

    // Piece length: 2vn - 1 from the Toom-4/2 threshold up, so that every piece is one toom42 or
    // toom63 product (five of ~vn/2 / eight of ~vn/3 limbs) instead of two balanced vn x vn ones,
    // as GMP's mpn_mul does: 2-8% faster at vn = 128..512 with toom42, 3-8% at 1024..2048 with
    // toom63. Where the balanced pieces would take Toom-6.5 / 8.5, only toom63 beats them (toom42
    // was up to +5% at vn = 1024), so there it needs the Toom-6/3 threshold.
    if (!pl) pl = (vn >= toom42_threshold() && (vn < toom6h_threshold() || vn >= toom63_threshold())) ? 2 * vn - 1 : vn;
    NUMETRON_ASSERT(pl >= vn);
    mul_piece(rb, u, (std::min)(pl, un));
    if (un <= pl) return re;

    LimbT* saved = std::allocator_traits<AllocatorT>::allocate(alloc, vn);
    NUMETRON_SCOPE_EXIT([&] { std::allocator_traits<AllocatorT>::deallocate(alloc, saved, vn); });

    for (size_t off = pl; off < un; off += pl) {
        const size_t an = (std::min)(pl, un - off);
        LimbT* dst = rb + off;
        std::memcpy(saved, dst, vn * sizeof(LimbT)); // the previous piece's high half
        mul_piece(dst, u + off, an);
        // The partial sum u[0..off+an) * v fits in off + an + vn limbs, so the carry stops there.
        if (LimbT c = uadd_inplace(dst, saved, saved + vn)) {
            [[maybe_unused]] LimbT out = uadd_limb(dst + vn, dst + an + vn, c);
            NUMETRON_ASSERT(!out);
        }
    }
    return re;
}

// umul_sliced in the FFT range: pl from fft_slice_length(), v transformed once for all the full
// pieces (fft_fixed_v; created first, so its scratch is released last from the stack allocator).
template <std::unsigned_integral LimbT, typename AllocatorT>
requires(sizeof(LimbT) == 8)
LimbT* umul_fft_sliced(const LimbT* u, size_t un, const LimbT* v, size_t vn, LimbT* rb, AllocatorT alloc, size_t pl)
{
    fft_fixed_v<LimbT, AllocatorT> fixed_v{ v, vn, pl, alloc };
    return umul_sliced(u, un, v, vn, rb, alloc, pl, [&](LimbT* dst, const LimbT* a) { fixed_v.mul(dst, a); });
}

}

template <std::unsigned_integral LimbT, typename AllocatorT>
inline LimbT* umul_dispatch(
    const LimbT* u, size_t un,
    const LimbT* v, size_t vn,
    LimbT* rb,
    AllocatorT alloc)
{
    while (un > 0 && u[un - 1] == 0) --un;
    while (vn > 0 && v[vn - 1] == 0) --vn;
    if (un < vn) {
        std::swap(u, v);
        std::swap(un, vn);
    }

    // A square (the same limbs on both sides): its own chain.
    if (u == v && un == vn) {
        return usqr_dispatch(u, un, rb, std::move(alloc));
    }

    // Below every algorithm's threshold: the basecase, without asking each of them.
    if (vn < basecase_limit()) {
        return vn ? umul_basecase<LimbT>(u, un, v, vn, rb) : rb;
    }

    if constexpr (sizeof(LimbT) == 8) {
        if (is_fft_applicable<LimbT>(un, vn)) {
            // a long u in pieces that fill a transform length (umul_fft.hpp)
            if (const size_t pl = detail::fft_slice_length(un, vn)) return detail::umul_fft_sliced(u, un, v, vn, rb, std::move(alloc), pl);
            return detail::umul_fft_impl(u, un, v, vn, rb, std::move(alloc));
        }
    }

    if (is_slicing_applicable(un, vn)) {
        return detail::umul_sliced(u, un, v, vn, rb, std::move(alloc));
    }

    if (is_toom43_applicable(un, vn)) {
        if (detail::toom43_split_by_u(un, vn)) return toom43_u_engine::umul(u, un, v, vn, rb, std::move(alloc));
        return toom43_v_engine::umul(u, un, v, vn, rb, std::move(alloc));
    }

    if (is_toom54_applicable(un, vn)) {
        if (detail::toom54_split_by_u(un, vn)) return toom54_u_engine::umul(u, un, v, vn, rb, std::move(alloc));
        return toom54_v_engine::umul(u, un, v, vn, rb, std::move(alloc));
    }

    if (is_toom53_applicable(un, vn)) {
        if (detail::toom53_split_by_u(un, vn)) return toom53_u_engine::umul(u, un, v, vn, rb, std::move(alloc));
        return toom53_v_engine::umul(u, un, v, vn, rb, std::move(alloc));
    }

    if (is_toom98_applicable(un, vn)) {
        if (detail::toom8h_half_split_by_u<9>(un, vn)) return toom8h_half_u_engine<9>::umul(u, un, v, vn, rb, std::move(alloc));
        return toom8h_half_v_engine<9>::umul(u, un, v, vn, rb, std::move(alloc));
    }

    if (is_toom107_applicable(un, vn)) {
        if (detail::toom8h_half_split_by_u<10>(un, vn)) return toom8h_half_u_engine<10>::umul(u, un, v, vn, rb, std::move(alloc));
        return toom8h_half_v_engine<10>::umul(u, un, v, vn, rb, std::move(alloc));
    }

    if (is_toom116_applicable(un, vn)) {
        if (detail::toom8h_half_split_by_u<11>(un, vn)) return toom8h_half_u_engine<11>::umul(u, un, v, vn, rb, std::move(alloc));
        return toom8h_half_v_engine<11>::umul(u, un, v, vn, rb, std::move(alloc));
    }

    if (is_toom8h_applicable(un, vn)) {
        return toom8h_balanced_engine::umul(u, un, v, vn, rb, std::move(alloc));
    }

    if (is_toom6h_applicable(un, vn)) {
        return toom6h_balanced_engine::umul(u, un, v, vn, rb, std::move(alloc));
    }

    if (is_toom76_applicable(un, vn)) {
        if (detail::toom76_split_by_u(un, vn)) return toom76_u_engine::umul(u, un, v, vn, rb, std::move(alloc));
        return toom76_v_engine::umul(u, un, v, vn, rb, std::move(alloc));
    }

    if (is_toom4_applicable(un, vn)) {
        return toom4_balanced_engine::umul(u, un, v, vn, rb, std::move(alloc));
    }

    // Toom-3 for balanced operands only. The rest below 2vn (the Toom-3 range's un/vn up to ~1.5)
    // takes Karatsuba, whose uneven halves recurse into sliced / balanced products; the old
    // generic toom_engine<3,3> plan used to take it and was slower than either.
    if (is_toom3_applicable(un, vn) && detail::toom3_split_fits(un, vn)) {
#if NUMETRON_TOOM3_IMPL == NUMETRON_TOOM3_IMPL_CXX
        return detail::umul_toom3_impl(std::span{u, un}, std::span{v, vn}, rb, alloc);
#else
        return toom3_balanced_engine::umul(u, un, v, vn, rb, std::move(alloc));
#endif
    }

    if (is_toom32_applicable(un, vn)) {
        if (detail::toom32_split_by_u(un, vn)) return toom32_u_engine::umul(u, un, v, vn, rb, std::move(alloc));
        return toom32_v_engine::umul(u, un, v, vn, rb, std::move(alloc));
    }

    if (is_toom63_applicable(un, vn)) {
        if (detail::toom63_split_by_u(un, vn)) return toom63_u_engine::umul(u, un, v, vn, rb, std::move(alloc));
        return toom63_v_engine::umul(u, un, v, vn, rb, std::move(alloc));
    }

    if (is_toom42_applicable(un, vn)) {
        if (detail::toom42_split_by_u(un, vn)) return toom42_u_engine::umul(u, un, v, vn, rb, std::move(alloc));
        return toom42_v_engine::umul(u, un, v, vn, rb, std::move(alloc));
    }

    if (is_karatsuba_applicable(un, vn)) {
#if NUMETRON_KARATSUBA_IMPL == NUMETRON_KARATSUBA_IMPL_ENGINE
        return toom_engine<2, 2>::umul(u, un, v, vn, rb, std::move(alloc));
#elif NUMETRON_KARATSUBA_IMPL == NUMETRON_KARATSUBA_IMPL_ASM
        // The asm recursion stays in Karatsuba all the way down. From the Toom-3 threshold up
        // (unbalanced operands only get here there) one C++ level instead: its children go back
        // through this dispatch -- the balanced halves to Toom, the uneven one to slicing.
        if (vn >= toom3_threshold()) {
            return detail::umul_karatsuba_impl(std::span{u, un}, std::span{v, vn}, rb, alloc);
        }
        return detail::umul_karatsuba_asm_impl(std::span{u, un}, std::span{v, vn}, rb, alloc);
#elif NUMETRON_KARATSUBA_IMPL == NUMETRON_KARATSUBA_IMPL_FUSED
        return detail::umul_karatsuba_fused_impl(std::span{u, un}, std::span{v, vn}, rb, alloc);
#else
        return detail::umul_karatsuba_impl(std::span{u, un}, std::span{v, vn}, rb, alloc);
#endif
    }
    if (vn) {
        return umul_basecase<LimbT>(u, un, v, vn, rb);
    }
    return rb;
}

// rb[0..2n) = u^2, the square counterpart of umul_dispatch (u's leading zero limbs dropped
// first; returns one past the last limb written). The FFT (from sqr_fft_threshold(), where the
// transform length is well filled: detail::fft_square_fills) squares by
// itself; below it the squaring variants of the balanced Toom plans (toom/square.hpp: A's
// evaluation only, the pointwise products squares -- back through umul_dispatch, which sends
// them here), from the sqr_toom*_threshold()s; then Karatsuba squaring (three squares) from
// sqr_karatsuba_threshold() and the squaring basecase.
template <std::unsigned_integral LimbT, typename AllocatorT>
inline LimbT* usqr_dispatch(const LimbT* u, size_t n, LimbT* rb, AllocatorT alloc)
{
    while (n > 0 && u[n - 1] == 0) --n;
    if (!n) return rb;
    if constexpr (sizeof(LimbT) == 8) {
        if (n >= sqr_fft_threshold() && detail::fft_square_fills(n)) return detail::umul_fft_impl(u, n, u, n, rb, std::move(alloc));
    }
    if (n >= sqr_toom8h_threshold() && detail::toom8h_split_fits(n, n)) return toom8h_square_engine::umul(u, n, u, n, rb, std::move(alloc));
    if (n >= sqr_toom6h_threshold() && detail::toom6h_split_fits(n, n)) return toom6h_square_engine::umul(u, n, u, n, rb, std::move(alloc));
    if (n >= sqr_toom4_threshold() && detail::toom4_split_fits(n, n)) return toom4_square_engine::umul(u, n, u, n, rb, std::move(alloc));
    if (n >= sqr_toom3_threshold() && detail::toom3_split_fits(n, n)) {
        return toom3_square_engine::umul(u, n, u, n, rb, std::move(alloc));
    }
    if (n >= sqr_karatsuba_threshold()) {
#if NUMETRON_KARATSUBA_IMPL == NUMETRON_KARATSUBA_IMPL_ASM
        if constexpr (sizeof(LimbT) == 8) return detail::usqr_karatsuba_asm_impl(u, n, rb, std::move(alloc));
        else
#endif
        return detail::usqr_karatsuba_impl(u, n, rb, std::move(alloc));
    }
    return usqr_basecase<LimbT>(u, n, rb);
}

namespace detail {

// rb[0..un+vn) = u * v for un, vn > 0 (either longer, leading zero limbs allowed): umul_dispatch
// with all the scratch from the thread-local stack allocator (served LIFO; its blocks are kept
// for reuse instead of going to the heap per recursion node). The one allocator type the whole
// chain is instantiated with. Returns one past the last limb written (at most rb + un + vn).
template <std::unsigned_integral LimbT>
LimbT* umul_large_impl(const LimbT* u, size_t un, const LimbT* v, size_t vn, LimbT* rb)
{
    numetron::detail::stack_allocator<LimbT> scratch_alloc;
    return umul_dispatch(u, un, v, vn, rb, scratch_alloc);
}

}

}
