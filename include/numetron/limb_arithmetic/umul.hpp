// Numetron — Compile-time and runtime arbitrary-precision arithmetic
// (c) Alexander Pototskiy
// Licensed under the MIT License. See LICENSE file for details.

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

// The FFT takes any shape (its length follows un + vn); only the size of the smaller operand
// decides. 64-bit limbs only.
template <std::unsigned_integral LimbT>
inline bool is_fft_applicable([[maybe_unused]] size_t un, size_t vn) noexcept
{
    assert(un >= vn);
    if constexpr (sizeof(LimbT) == 8) return vn >= fft_threshold();
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
// the short last piece is itself sliced again if it is short enough.
// Returns rb + un + vn.
template <std::unsigned_integral LimbT, typename AllocatorT>
LimbT* umul_sliced(const LimbT* u, size_t un, const LimbT* v, size_t vn, LimbT* rb, AllocatorT alloc)
{
    NUMETRON_ASSERT(vn > 0 && un >= vn);
    LimbT* const re = rb + un + vn;

    // dst[0..an+vn) = a * v, zero-padding what umul_dispatch leaves above the significant limbs
    // (it strips leading zero limbs, and a piece from the middle of u may have them).
    auto mul_piece = [&](LimbT* dst, const LimbT* a, size_t an) {
        LimbT* e = umul_dispatch(a, an, v, vn, dst, alloc);
        std::memset(e, 0, static_cast<size_t>(dst + an + vn - e) * sizeof(LimbT));
    };

    // Piece length: 2vn - 1 from the Toom-4/2 threshold up, so that every piece is one toom42 or
    // toom63 product (five of ~vn/2 / eight of ~vn/3 limbs) instead of two balanced vn x vn ones,
    // as GMP's mpn_mul does: 2-8% faster at vn = 128..512 with toom42, 3-8% at 1024..2048 with
    // toom63. Where the balanced pieces would take Toom-6.5 / 8.5, only toom63 beats them (toom42
    // was up to +5% at vn = 1024), so there it needs the Toom-6/3 threshold.
    const size_t pl = (vn >= toom42_threshold() && (vn < toom6h_threshold() || vn >= toom63_threshold())) ? 2 * vn - 1 : vn;
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

    if constexpr (sizeof(LimbT) == 8) {
        if (is_fft_applicable<LimbT>(un, vn)) {
            return detail::umul_fft_impl(u, un, v, vn, rb, std::move(alloc));
        }
    }

    if (is_slicing_applicable(un, vn)) {
        return detail::umul_sliced(u, un, v, vn, rb, std::move(alloc));
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

template <std::unsigned_integral LimbT, typename AllocatorT>
requires(std::is_same_v<LimbT, typename std::allocator_traits<AllocatorT>::value_type>)
inline std::tuple<LimbT*, size_t, size_t> umul(std::span<const LimbT> u, std::span<const LimbT> v, AllocatorT alloc)
{
    if (v.empty()) [[unlikely]] {
        return { nullptr, 0, 0 };
    }

    if constexpr (sizeof(LimbT) == 8) {
        if (is_fft_applicable<LimbT>(u.size(), v.size())) {
            numetron::detail::stack_allocator<LimbT> scratch_alloc;
            return umul_fft(u, v, std::move(alloc), scratch_alloc);
        }
    }

    if (is_slicing_applicable(u.size(), v.size())) {
        numetron::detail::stack_allocator<LimbT> scratch_alloc; // see below
        const size_t rsz = u.size() + v.size();
        LimbT* r = std::allocator_traits<AllocatorT>::allocate(alloc, rsz);
        try {
            LimbT* re = detail::umul_sliced(u.data(), u.size(), v.data(), v.size(), r, scratch_alloc);
            while (re != r && *(re - 1) == 0) --re;
            return { r, static_cast<size_t>(re - r), rsz };
        }
        catch (...) {
            std::allocator_traits<AllocatorT>::deallocate(alloc, r, rsz);
            throw;
        }
    }

    //if (v.size() >= NUMETRON_KARATSUBA_THRESHOLD) {
        const bool toom8h = is_toom8h_applicable(u.size(), v.size());
        const bool toom6h = !toom8h && is_toom6h_applicable(u.size(), v.size());
        const bool toom76 = !toom8h && !toom6h && is_toom76_applicable(u.size(), v.size());
        const bool toom4 = !toom8h && !toom6h && !toom76 && is_toom4_applicable(u.size(), v.size());
        // Balanced Toom-3 only; see umul_dispatch.
        const bool toom3 = !toom8h && !toom6h && !toom76 && !toom4 && is_toom3_applicable(u.size(), v.size())
            && detail::toom3_split_fits(u.size(), v.size());
        const bool toom32 = !toom8h && !toom6h && !toom76 && !toom4 && !toom3 && is_toom32_applicable(u.size(), v.size());
        const bool toom63 = !toom8h && !toom6h && !toom76 && !toom4 && !toom3 && !toom32 && is_toom63_applicable(u.size(), v.size());
        const bool toom42 = !toom8h && !toom6h && !toom76 && !toom4 && !toom3 && !toom32 && !toom63 && is_toom42_applicable(u.size(), v.size());
        if (toom8h || toom6h || toom76 || toom4 || toom3 || toom32 || toom63 || toom42 || is_karatsuba_applicable(u.size(), v.size())) {
            // The one place scratch memory is chosen for a whole recursive multiplication: only
            // the result comes from the caller's allocator (it outlives this call), everything
            // below -- Toom slabs, Karatsuba temporaries, all nested levels -- from the
            // thread-local stack allocator, which is served LIFO and keeps its blocks for reuse
            // instead of going to the heap per recursion node. Created here rather than up front
            // so the basecase path doesn't pay for the thread_local lookup.
            numetron::detail::stack_allocator<LimbT> scratch_alloc;
            if (toom8h) {
                return toom8h_balanced_engine::umul(u, v, std::move(alloc), scratch_alloc);
            }
            if (toom6h) {
                return toom6h_balanced_engine::umul(u, v, std::move(alloc), scratch_alloc);
            }
            if (toom76) {
                if (detail::toom76_split_by_u(u.size(), v.size())) return toom76_u_engine::umul(u, v, std::move(alloc), scratch_alloc);
                return toom76_v_engine::umul(u, v, std::move(alloc), scratch_alloc);
            }
            if (toom4) {
                return toom4_balanced_engine::umul(u, v, std::move(alloc), scratch_alloc);
            }
            if (toom3) {
    #if NUMETRON_TOOM3_IMPL == NUMETRON_TOOM3_IMPL_CXX
                return umul_toom3(u, v, std::move(alloc), scratch_alloc);
    #else
                return toom3_balanced_engine::umul(u, v, std::move(alloc), scratch_alloc);
    #endif
            }
            if (toom32) {
                if (detail::toom32_split_by_u(u.size(), v.size())) return toom32_u_engine::umul(u, v, std::move(alloc), scratch_alloc);
                return toom32_v_engine::umul(u, v, std::move(alloc), scratch_alloc);
            }
            if (toom63) {
                if (detail::toom63_split_by_u(u.size(), v.size())) return toom63_u_engine::umul(u, v, std::move(alloc), scratch_alloc);
                return toom63_v_engine::umul(u, v, std::move(alloc), scratch_alloc);
            }
            if (toom42) {
                if (detail::toom42_split_by_u(u.size(), v.size())) return toom42_u_engine::umul(u, v, std::move(alloc), scratch_alloc);
                return toom42_v_engine::umul(u, v, std::move(alloc), scratch_alloc);
            }
    #if NUMETRON_KARATSUBA_IMPL == NUMETRON_KARATSUBA_IMPL_ENGINE
            return toom_engine<2, 2>::umul(u, v, std::move(alloc), scratch_alloc);
    #elif NUMETRON_KARATSUBA_IMPL == NUMETRON_KARATSUBA_IMPL_ASM
            if (v.size() >= toom3_threshold()) { // see umul_dispatch
                return umul_karatsuba(u, v, std::move(alloc), scratch_alloc);
            }
            return umul_karatsuba_asm(u, v, std::move(alloc), scratch_alloc);
    #elif NUMETRON_KARATSUBA_IMPL == NUMETRON_KARATSUBA_IMPL_FUSED
            return umul_karatsuba_fused(u, v, std::move(alloc), scratch_alloc);
    #else
            return umul_karatsuba(u, v, std::move(alloc), scratch_alloc);
    #endif
        }
    //}

    
    size_t rsz = u.size() + v.size();
    LimbT* r = std::allocator_traits<AllocatorT>::allocate(alloc, rsz);
    LimbT* re = umul_basecase(u.data(), u.size(), v.data(), v.size(), r);
    return { r, static_cast<size_t>(re - r), rsz };
    
}

// base case mul with explicit high limbs uh and vh
// precondition: |u| >= |v|, u.size() > 0
// returns iterator to one past the last written limb (r + u.size() + v.size() + 2)
template <std::unsigned_integral LimbT, typename AllocatorT>
requires(std::is_same_v<LimbT, typename std::allocator_traits<AllocatorT>::value_type>)
inline std::tuple<LimbT*, size_t, size_t> umul(LimbT uh, std::span<const LimbT> ul, LimbT vh, std::span<const LimbT> vl, AllocatorT alloc)
{
    assert(!ul.empty());

    const size_t n = ul.size();
    const size_t m = vl.size();

    if (!m) {
        size_t rsz = ul.size() + 2;
        LimbT* r = std::allocator_traits<AllocatorT>::allocate(alloc, rsz);
        LimbT* re = r;
        LimbT rh = umul1(uh, ul.data(), ul.data() + n, vh, re);
        if (rh) { *re = rh; ++re; }
        return { r, static_cast<size_t>(re - r), rsz };
    }

    size_t rsz = n + m;
    LimbT* r = std::allocator_traits<AllocatorT>::allocate(alloc, rsz);
    
    // Low product
    LimbT* re = umul_basecase(ul.data(), n, vl.data(), m, r);

    if (!uh && !vh) return { r, static_cast<size_t>(re - r), rsz };

    // Ensure top two limbs exist and start from zero
    LimbT* top = r; top += (n + m);
        
    // Clear intermediate limbs if any
    while (re != top) {
        *re++ = LimbT{0};
    }

    if (uh) {
        *re = LimbT{0};
        ++re;
        // Add uh * v at offset n
        auto rvhu = r + n;
        LimbT c = umul1_add(vl.data(), vl.data() + m, uh, rvhu);
        c = uadd1<LimbT>(rvhu, re - rvhu, c, rvhu);
        assert(c == 0);
    }
    if (vh) {
        *re = LimbT{0};
        ++re;
        // Add vh * u at offset m
        auto rvhv = r + m;
        LimbT c = umul1_add(ul.data(), ul.data() + n, vh, rvhv);
        c = uadd1<LimbT>(rvhv, re - rvhv, c, rvhv);
        assert(c == 0);
    
        // Add uh * vh at offset n + m
        if (uh) {
            auto [h, l] = numetron::arithmetic::umul1<LimbT>(uh, vh);
            r += n + m;
            auto [c, s] = numetron::arithmetic::uadd1<LimbT>(*r, l);
            *r = s; ++r;
            auto [c2, s2] = numetron::arithmetic::uadd1<LimbT>(*r, h, c);
            *r = s2;
            assert(c2 == 0);
        }
    }

    return { r, static_cast<size_t>(re - r), rsz };;
}

#if 0
template <std::unsigned_integral LimbT, typename ResultIteratorT>
inline ResultIteratorT umul(LimbT uh, std::span<const LimbT> ul, LimbT vh, std::span<const LimbT> vl, ResultIteratorT r) noexcept
{
    assert(!ul.empty());

    const size_t n = ul.size();
    const size_t m = vl.size();

    if (!m) {
        LimbT rh = umul1(uh, ul.data(), ul.data() + ul.size(), vh, r);
        if (rh) { *r = rh; ++r; }

        //if (ul.empty()) {
        //    auto [h, ph] = numetron::arithmetic::umul1(uh, vh);
        //    *r = ph; ++r;
        //    *r = h; ++r;
        //} else {
        //    LimbT rh = umul1(uh, ul.data(), ul.data() + ul.size(), vh, r);
        //    if (rh) { *r = rh; ++r; }
        //}
        return r;
    } 

    //else if (!ul.size()) {
    //    auto [pcl, ph] = umul1(vh, vl.data(), vl.data() + vl.size(), uh, r);

    //    if (ph) *r++ = ph;
    //    if (pcl) {
    //        assert(!pcl); // because cl = 0 in umul1 call
    //    }
    //    assert(!pcl); // because cl = 0 in umul1 call
    //    return r;
    //}

    // Low product
    ResultIteratorT re = umul<LimbT>(ul.data(), ul.data() + n, vl.data(), vl.data() + m, r); // may return r if one is empty

    if (!uh && !vh) return re;

    // Ensure top two limbs exist and start from zero
    ResultIteratorT top = r; top += (n + m);
        
    // Clear intermediate limbs if any
    while (re != top) {
        *re++ = LimbT{0};
    }

    if (uh) {
        *re = LimbT{0};
        ++re;
        // Add uh * v at offset n
        auto rvhu = r + n;
        LimbT c = umul1_add(vl.data(), vl.data() + m, uh, rvhu);
        c = uadd1<LimbT>(rvhu, re - rvhu, c, rvhu);
        assert(c == 0);
    }
    if (vh) {
        *re = LimbT{0};
        ++re;
        // Add vh * u at offset m
        auto rvhv = r + m;
        LimbT c = umul1_add(ul.data(), ul.data() + n, vh, rvhv);
        c = uadd1<LimbT>(rvhv, re - rvhv, c, rvhv);
        assert(c == 0);
    
        // Add uh * vh at offset n + m
        if (uh) {
            auto [h, l] = numetron::arithmetic::umul1<LimbT>(uh, vh);
            r += n + m;
            auto [c, s] = numetron::arithmetic::uadd1<LimbT>(*r, l);
            *r = s; ++r;
            auto [c2, s2] = numetron::arithmetic::uadd1<LimbT>(*r, h, c);
            *r = s2;
            assert(c2 == 0);
        }
    }

    return re;
}

// base case mul: {u} * {v} -> {rb, re}
// returns re
template <std::unsigned_integral LimbT, typename ResultIteratorT>
inline ResultIteratorT umul(LimbT const* ub, LimbT const* ue, LimbT const* vb, LimbT const* ve, ResultIteratorT r) noexcept
{
    assert(ub != ue && vb != ve);

    ResultIteratorT rb = r;

    // first line
    unsigned char cl = 0;
    // unrolled first iteration
    auto [previous, current] = numetron::arithmetic::umul1(*ub, *vb);
    *rb = current; ++rb;
    // unrolled second iteration
    auto ub_it = ub;
    if (++ub_it != ue) {
        auto [h, l] = numetron::arithmetic::umul1(*ub_it, *vb);
        auto [nc, current] = numetron::arithmetic::uadd1(l, previous);
        *rb = current; ++rb;
        cl = nc;
        previous = h;
        while (++ub_it != ue) {
            auto [h, l] = numetron::arithmetic::umul1(*ub_it, *vb);
            *rb = numetron::arithmetic::uadd1c(l, previous, cl);
            ++rb;
            previous = h;
        }
    }
    *rb = previous + cl;

    // next lines
    size_t dec_usz = ue - ub - 1;
    for (++vb; vb != ve; ++vb) {
        cl = 0;
        ub_it = ub;
        rb -= dec_usz;
        LimbT vb_val = *vb;
        // first iteration unrolled
        auto [previous, l] = numetron::arithmetic::umul1(*ub_it, vb_val);
        auto [cl2, current] = numetron::arithmetic::uadd1(l, *rb);
        *rb = current; ++rb;
        while (++ub_it != ue) {
            auto [h, l] = numetron::arithmetic::umul1(*ub_it, vb_val);
            *rb = numetron::arithmetic::uadd1c2(l, *rb, previous, cl, cl2);
            ++rb;
            previous = h;
        } 
        *rb = previous + cl + cl2;
    }
    return rb + 1;
}
#endif

}
