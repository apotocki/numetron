// Numetron — Compile-time and runtime arbitrary-precision arithmetic
// (c) 2025 Alexander Pototskiy
// Licensed under the MIT License. See LICENSE file for details.

#include "test_common.hpp"

#include "numetron/ct.hpp"

namespace numetron {

void ct_test()
{
    using namespace ct::literals;
    using ct::W;

    static_assert(getsize(normalize(W<1, 1, 0, 0, 0>)) == 2);

    //using r0_t = ct::add<ct::limbs<uint32_t, 0xffffFFFF>, ct::limbs<uint32_t, 3>>;
    using r0_t = decltype(ct::limbs<uint32_t, 0xffffFFFF>{} + ct::limbs<uint32_t, 3>{});
    static_assert(ct::at<r0_t, 0> == 2);
    static_assert(ct::at<r0_t, 1> == 1);
    static_assert(ct::back<r0_t> == 1);
    static_assert(ct::size<r0_t> == 2);

    //using r1_t = ct::mul1<ct::limbs<uint32_t, 0xffffFFFF, 0xffffFFFF>, uint32_t(16)>;
    using r1_t = decltype(ct::limbs<uint32_t, 0xffffFFFF, 0xffffFFFF>{} *ct::limbs<uint32_t, 16>{});
    static_assert(ct::at<r1_t, 0> == 0xffffFFF0);
    static_assert(ct::at<r1_t, 1> == 0xffffFFFF);
    static_assert(ct::at<r1_t, 2> == 0xf);
    static_assert(ct::back<r1_t> == 0xf);
    static_assert(ct::size<r1_t> == 3);

    using r1_1_t = decltype(100_W * 65536_W);
    static_assert(ct::size<r1_1_t> == 1);
    static_assert(ct::back<r1_1_t> == 6553600);

    //using r2_t = ct::mul1<ct::limbs<uint32_t, 0xffffFFFF, 0xffffFFFF>, uint32_t(0xffffFFFF)>;
    using r2_t = decltype(ct::W32<0xffffFFFF, 0xffffFFFF>* ct::W32<0xffffFFFF>);
    static_assert(ct::at<r2_t, 0> == 1);
    static_assert(ct::at<r2_t, 1> == 0xffffFFFF);
    static_assert(ct::at<r2_t, 2> == 0xffffFFFe);
    static_assert(ct::size<r2_t> == 3);

    //using r3_t = ct::mul1<ct::limbs<uint64_t, 0xffffFFFFffffFFFFULL, 0xffffFFFFffffFFFFULL>, uint64_t(0xffffFFFFffffFFFFULL)>;
    using r3_t = decltype(ct::limbs<uint64_t, 0xffffFFFFffffFFFFULL, 0xffffFFFFffffFFFFULL>{} *ct::limbs<uint64_t, 0xffffFFFFffffFFFFULL>{});
    static_assert(ct::at<r3_t, 0> == 1);
    static_assert(ct::at<r3_t, 1> == 0xffffFFFFffffFFFFULL);
    static_assert(ct::at<r3_t, 2> == 0xffffFFFFffffFFFE);
    static_assert(ct::size<r3_t> == 3);

    using r4_t = ct::mul<ct::limbs<uint64_t, 0xffffFFFFffffFFFFULL, 0xffffFFFFffffFFFFULL>, ct::limbs<uint64_t, 0xffffFFFFffffFFFFULL, 0xffffFFFFffffFFFFULL>>;
    static_assert(ct::at<r4_t, 0> == 1);
    static_assert(ct::at<r4_t, 1> == 0);
    static_assert(ct::at<r4_t, 2> == 0xffffFFFFffffFFFEULL);
    static_assert(ct::at<r4_t, 3> == 0xffffFFFFffffFFFFULL);
    static_assert(ct::size<r4_t> == 4);

    using r5_t = decltype(W<1> -W<3, 0>);
    static_assert(ct::size<r5_t> == 2);
    static_assert(ct::front<r5_t> == 0xffffFFFFffffFFFEULL);
    static_assert(ct::back<r5_t> == 0xffffFFFFffffFFFFULL);

    static_assert(ct::cmp<decltype(100_W), decltype(101_W)> == -1);
    static_assert(ct::cmp<ct::limbs<uint64_t, 1, 0, 0>, ct::limbs<uint64_t, 1>> == 0);
    static_assert(ct::cmp<ct::limbs<uint64_t, 1, 0, 1>, ct::limbs<uint64_t, 1>> == 1);
    static_assert(ct::cmp<ct::limbs<uint64_t, 1, 0, 1>, ct::limbs<uint64_t, 2, 0, 1>> == -1);

    using r6_t = decltype(upow(100_W, 12_W)); // = 54210 * 2 ^ 64 + 2003764205206896640 = 1000000000000000000000000
    static_assert(ct::size<r6_t> == 2);
    static_assert(ct::back<r6_t> == 54210);
    static_assert(ct::front<r6_t> == 2003764205206896640UL);

    using r7_t = ct::shift_left<decltype(W<0xffffFFFFffffFFFFULL, 0xffffFFFFffffFFFFULL>), 1>;
    static_assert(ct::size<r7_t> == 3);
    static_assert(ct::front<r7_t> == 0xffffFFFFffffFFFEULL);
    static_assert(ct::at<r7_t, 1> == 0xffffFFFFffffFFFFULL);
    static_assert(ct::back<r7_t> == 1);

    using r8_t = ct::shift_left<decltype(W<0xffffFFFFffffFFFFULL, 0x7fffFFFFffffFFFFULL>), 1>;
    static_assert(ct::size<r8_t> == 2);
    static_assert(ct::back<r8_t> == 0xffffFFFFffffFFFFULL);
    static_assert(ct::front<r8_t> == 0xffffFFFFffffFFFEULL);

    //using r9_t = ct::shift_left<decltype(W<0xffffFFFFffffFFFFULL, 0x7fffFFFFffffFFFFULL>), 65>;
    using r9_t = decltype(W<0xffffFFFFffffFFFFULL, 0x7fffFFFFffffFFFFULL> << W<65>);
    static_assert(ct::size<r9_t> == 3);
    static_assert(ct::back<r9_t> == 0xffffFFFFffffFFFFULL);
    static_assert(ct::at<r9_t, 1> == 0xffffFFFFffffFFFEULL);
    static_assert(ct::front<r9_t> == 0);

    {
        constexpr auto max64 = 0xffffFFFFffffFFFFULL;
        using r0_t = decltype(W<max64, max64-1, max64-2, max64-3, max64-4, max64-5>);
        static_assert(ct::back<ct::slice_limbs<r0_t, 1>> == (uint64_t)-1);
        static_assert(ct::back<ct::slice_limbs<r0_t, 2>> == (uint64_t)-2);
        static_assert(ct::back<ct::slice_limbs<r0_t, 3>> == (uint64_t)-3);
        static_assert(ct::back<ct::slice_limbs<r0_t, 4>> == (uint64_t)-4);
        static_assert(ct::back<ct::slice_limbs<r0_t, 5>> == (uint64_t)-5);
        static_assert(ct::back<ct::slice_limbs<r0_t, 6>> == (uint64_t)-6);
        static_assert(ct::back<ct::slice<r0_t, 128>> == (uint64_t)-2);
        static_assert(ct::back<ct::slice<r0_t, 124>> == 0x0fffFFFFffffFFFEULL);
    }
    {
        using r0_t = ct::shift_right<decltype(W<0xffffFFFFffffFFFFULL, 0x7fffFFFFffffFFFFULL>), 0>;
        static_assert(ct::size<r0_t> == 2);
        static_assert(ct::at<r0_t, 0> == 0xffffFFFFffffFFFFULL);
        static_assert(ct::at<r0_t, 1> == 0x7fffFFFFffffFFFFULL);

        using r1_t = ct::shift_right<decltype(W<0xffffFFFFffffFFFFULL, 0x7fffFFFFffffFFFFULL>), 1>;
        static_assert(ct::size<r1_t> == 2);
        static_assert(ct::at<r1_t, 0> == 0xffffFFFFffffFFFFULL);
        static_assert(ct::at<r1_t, 1> == 0x3fffFFFFffffFFFFULL);

        using r2_t = ct::shift_right<decltype(W<0xffffFFFFffffFFFFULL, 0x7fffFFFFffffFFFFULL>), 63>;
        static_assert(ct::size<r2_t> == 1);
        static_assert(ct::at<r1_t, 0> == 0xffffFFFFffffFFFFULL);
    }

    using r10_t = ct::div1<r3_t, 0xffffFFFFffffFFFFULL>;
    static_assert(ct::at<r10_t, 0> == 0xffffFFFFffffFFFFULL);
    static_assert(ct::at<r10_t, 1> == 0xffffFFFFffffFFFFULL);
    static_assert(ct::size<r10_t> == 2);

    //constexpr auto c2_64 = W<2> ^ W<64>;
    //constexpr auto mdb = W<0x8ac7230489e80000ULL>;
    //using r11_t = decltype(c2_64 * c2_64);
    //using r11_t = decltype(c2_64* (c2_64 - mdb));
    //static_assert(ct::front<r11_t> == 0xd83c94fb6d2ac34aULL);
    //std::cout << typeid(ct::mul<decltype(c2_64), decltype(c2_64)>).name() << "\n";
    //std::cout << typeid(r6_t).name() << "\n" << numetron::arithmetic::consteval_log2<uint32_t>(16);


    //using r6_t = decltype(100_W);
    //using ct10 = ct::limbs<uint32_t, 10>;
    //using r5_t = ct::pow<ct10, 12>;
    //std::cout << typeid(ct::norm<ct::limbs<uint64_t, 1, 0, 0, 0, 0, 1>>).name() << "\n";
    //std::cout << typeid(r6_t).name() << "\n";

    /*
    using ct10 = ct::limbs<uint32_t, 10>;
    using ct16 = ct::limbs<uint32_t, 16>;
    using ct1010 = ct::limbs<uint32_t, 1010>;

    using ctx100 = ct::mul1<ct16, uint32_t(16)>;
    static_assert(ct::at<ctx100, 0> == 256);
    using ctx10000 = ct::mul1<ctx100, uint32_t(256)>;
    static_assert(ct::at<ctx10000, 0> == 0x10000);
    using ctx100000000 = ct::mul1<ctx10000, uint32_t(0x10000)>;
    static_assert(ct::at<ctx100000000, 0> == 0);
    static_assert(ct::at<ctx100000000, 1> == 1);

    using ct16nff = ct::limbs<uint32_t, 0xffffFFFF, 0xffffFFFF>;
    using r_t = ct::mul1<ct16nff, uint32_t(0xffffFFFF)>;
    static_assert(ct::at<r_t, 0> == 0xffffFFFE);
    static_assert(ct::at<r_t, 1> == 0xffffFFFF);
    //using r_t = ct::mul<ct10, ct1010>;
    //using r_t = ct::mul1<ct16, uint32_t(16)>;
    std::cout << typeid(r_t).name() << "\n";
    */
    //0x1ffffffffFFFFfffe
}

}
