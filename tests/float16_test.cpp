// Numetron — Compile-time and runtime arbitrary-precision arithmetic
// (c) 2025 Alexander Pototskiy
// Licensed under the MIT License. See LICENSE file for details.

#include "test_common.hpp"

#include "numetron/float16.hpp"
#include <iostream>
#include <cmath>
#include <limits>

namespace numetron {

void test_float16_basic_operations() {
    // Test basic construction and conversion
    float16 f1(1.0f);
    float16 f2(2.0f);
    float16 f3(-1.0f);
    float16 f4(0.0f);
    
    // Test integral constructor
    float16 fi1(1);
    float16 fi2(2);
    float16 fi3(-1);
    float16 fi4(0);
    
    // Check that conversions work correctly
    CHECK_EQUAL((float)f1, 1.0f);
    CHECK_EQUAL((float)f2, 2.0f);
    CHECK_EQUAL((float)f3, -1.0f);
    CHECK_EQUAL((float)f4, 0.0f);
    
    // Check that integral constructor works correctly
    CHECK_EQUAL((float)fi1, 1.0f);
    CHECK_EQUAL((float)fi2, 2.0f);
    CHECK_EQUAL((float)fi3, -1.0f);
    CHECK_EQUAL((float)fi4, 0.0f);
    
    // Test comparisons
    CHECK(f1 < f2);
    CHECK(f1 > f3);
    CHECK(f1 == f1);
    CHECK(f4 == float16(0.0f));
    CHECK(fi1 == f1);
    CHECK(fi2 == f2);
    
    // Test inequality
    CHECK(f1 != f2);
    CHECK(f1 != f3);
    CHECK(f2 != f3);
    
    // Test greater/less than or equal
    CHECK(f1 <= f2);
    CHECK(f1 >= f3);
    CHECK(f1 <= f1);
    CHECK(f1 >= f1);
}

void test_float16_special_values() {
    // Test infinity
    float16 pos_inf(std::numeric_limits<float>::infinity());
    float16 neg_inf(-std::numeric_limits<float>::infinity());
    
    CHECK(pos_inf > float16(1000.0f));
    CHECK(neg_inf < float16(-1000.0f));
    CHECK(pos_inf > neg_inf);
    CHECK(pos_inf != neg_inf);
    
    // Test that infinity equals itself
    CHECK(pos_inf == pos_inf);
    CHECK(neg_inf == neg_inf);
    
    // Test NaN
    float16 nan_val(std::numeric_limits<float>::quiet_NaN());
    CHECK(!(nan_val == nan_val)); // NaN should not be equal to itself
    CHECK(!(nan_val < float16(1.0f))); // NaN comparisons should be false
    CHECK(!(nan_val > float16(1.0f))); // NaN comparisons should be false
    CHECK(!(nan_val == float16(1.0f))); // NaN should not equal any value
    CHECK(!(float16(1.0f) == nan_val)); // NaN should not equal any value
    
    // Test NaN inequality
    CHECK(nan_val != float16(1.0f));
    CHECK(float16(1.0f) != nan_val);
}

void test_float16_zeros() {
    float16 pos_zero(0.0f);
    float16 neg_zero(-0.0f);
    
    // Both +0 and -0 should be equal according to IEEE 754
    CHECK(pos_zero == neg_zero);
    CHECK(!(pos_zero < neg_zero));
    CHECK(!(pos_zero > neg_zero));
    CHECK(pos_zero <= neg_zero);
    CHECK(pos_zero >= neg_zero);
    
    // Test that zero equals itself
    CHECK(pos_zero == pos_zero);
    CHECK(neg_zero == neg_zero);
    
    // Test conversion back to float
    CHECK_EQUAL((float)pos_zero, 0.0f);
    CHECK_EQUAL((float)neg_zero, -0.0f);
}

void test_float16_small_values() {
    float16 small1(1e-3f);
    float16 small2(1e-4f);
    float16 small3(8e-5f);
    
    // Test that small values are ordered correctly
    CHECK(small1 > small2);
    CHECK(small2 > small3);
    CHECK(small1 > small3);
    
    // Test transitivity
    CHECK(small1 >= small2);
    CHECK(small2 >= small3);
    CHECK(small1 >= small3);
    
    // Test inequality
    CHECK(small1 != small2);
    CHECK(small2 != small3);
    CHECK(small1 != small3);
    
    // Test that they are all positive
    float16 zero(0.0f);
    CHECK(small1 > zero);
    CHECK(small2 > zero);
    CHECK(small3 > zero);

    float16 very_small1 = float16::min();
    float16 very_small2 = float16::denorm_min();

    CHECK(very_small1 > zero);
    CHECK(very_small2 > zero);
    CHECK(very_small1 > very_small2);
}

void test_float16_precision_limits() {
    // Test values near the precision limits of float16
    float16 one(1.0f);
    float16 one_plus_eps(1.0f + 1.0f/1024.0f); // Smallest representable increment for 1.0 in float16
    float16 one_plus_half_eps(1.0f + 1.0f/2048.0f); // Should round to 1.0
    
    CHECK(one_plus_eps > one);
    CHECK(one_plus_half_eps == one); // Should round down to 1.0
    
    // Test maximum finite value
    float16 max_val(65504.0f); // Maximum finite float16 value
    float16 large_val(65000.0f);
    CHECK(max_val > large_val);
    
    // Test minimum positive normal value
    float16 min_normal(6.103515625e-5f); // Minimum positive normal float16
    float16 smaller_normal(7e-5f);
    CHECK(smaller_normal > min_normal);
}

void test_float16_arithmetic_properties() {
    float16 a(3.0f);
    float16 b(4.0f);
    float16 c(5.0f);
    
    // Test transitivity
    CHECK(a < b);
    CHECK(b < c);
    CHECK(a < c);
    
    // Test reflexivity
    CHECK(a == a);
    CHECK(b == b);
    CHECK(c == c);
    
    // Test symmetry of equality
    float16 a_copy(3.0f);
    CHECK(a == a_copy);
    CHECK(a_copy == a);
    
    // Test antisymmetry of ordering
    CHECK(!(a < b && b < a));
    CHECK(!(b < c && c < b));
}

void test_float16_negative_values() {
    float16 neg_one(-1.0f);
    float16 neg_two(-2.0f);
    float16 neg_half(-0.5f);
    float16 zero(0.0f);
    float16 pos_one(1.0f);
    
    // Test negative ordering
    CHECK(neg_two < neg_one);
    CHECK(neg_one < neg_half);
    CHECK(neg_half < zero);
    CHECK(zero < pos_one);
    
    // Test that all negative values are less than zero
    CHECK(neg_one < zero);
    CHECK(neg_two < zero);
    CHECK(neg_half < zero);
    
    // Test that all negative values are less than positive values
    CHECK(neg_one < pos_one);
    CHECK(neg_two < pos_one);
    CHECK(neg_half < pos_one);
}

void test_float16_integral_constructor() {
    // Test normal integral values
    float16 f1(0);
    float16 f2(1);
    float16 f3(-1);
    float16 f4(42);
    float16 f5(-42);
    float16 f6(1000);
    float16 f7(-1000);
    
    CHECK_EQUAL((float)f1, 0.0f);
    CHECK_EQUAL((float)f2, 1.0f);
    CHECK_EQUAL((float)f3, -1.0f);
    CHECK_EQUAL((float)f4, 42.0f);
    CHECK_EQUAL((float)f5, -42.0f);
    CHECK_EQUAL((float)f6, 1000.0f);
    CHECK_EQUAL((float)f7, -1000.0f);
    
    // Test edge values within range
    float16 f_max(65504);   // max finite positive
    float16 f_min(-65504);  // max finite negative
    
    CHECK_EQUAL((float)f_max, 65504.0f);
    CHECK_EQUAL((float)f_min, -65504.0f);
    
    // Test overflow behavior - values become infinity
    float16 f_pos_inf(100000);   // too large -> +infinity
    float16 f_neg_inf(-100000);  // too small -> -infinity
    
    CHECK(std::isinf((float)f_pos_inf));
    CHECK((float)f_pos_inf > 0);
    CHECK(std::isinf((float)f_neg_inf));
    CHECK((float)f_neg_inf < 0);
    
    // Test different integer types
    float16 fi8_pos(int8_t(127));
    float16 fi8_neg(int8_t(-128));
    float16 fui8(uint8_t(255));
    float16 fi16_pos(int16_t(32752));
    float16 fi16_neg(int16_t(-32768));
    float16 fui16_overflow(uint16_t(65535));  // This will overflow to infinity
    
    CHECK_EQUAL((float)fi8_pos, 127.0f);
    CHECK_EQUAL((float)fi8_neg, -128.0f);
    CHECK_EQUAL((float)fui8, 255.0f);
    CHECK_EQUAL((float)fi16_pos, 32752.0f);
    CHECK_EQUAL((float)fi16_neg, -32768.0f);
    CHECK(std::isinf((float)fui16_overflow));  // 65535 > 65504, so infinity
}

void test_float16_static_values() {
    // Test special infinity values
    float16 pos_inf = float16::infinity();
    float16 neg_inf = float16::negative_infinity();
    
    CHECK(std::isinf((float)pos_inf));
    CHECK((float)pos_inf > 0);
    CHECK(std::isinf((float)neg_inf));
    CHECK((float)neg_inf < 0);
    CHECK(pos_inf > neg_inf);
    CHECK(pos_inf != neg_inf);
    
    // Test NaN values
    float16 qnan = float16::quiet_NaN();
    float16 snan = float16::signaling_NaN();
    
    CHECK(std::isnan((float)qnan));
    CHECK(std::isnan((float)snan));
    CHECK(!(qnan == qnan));  // NaN should not equal itself
    CHECK(!(snan == snan));  // NaN should not equal itself
    CHECK(qnan != snan);     // Different NaN values should not be equal
    
    // Test finite extreme values
    float16 max_val = float16::max();
    float16 lowest_val = float16::lowest();
    
    CHECK_EQUAL((float)max_val, 65504.0f);
    CHECK_EQUAL((float)lowest_val, -65504.0f);
    CHECK(max_val > lowest_val);
    CHECK(max_val > float16::zero());
    CHECK(lowest_val < float16::zero());
    
    // Test minimum positive normal value
    float16 min_normal = float16::min();
    CHECK((float)min_normal > 0.0f);
    CHECK(min_normal > float16::zero());
    CHECK(min_normal < float16::one());
    
    // Test epsilon (machine precision)
    float16 eps = float16::epsilon();
    float16 one = float16::one();
    float16 one_plus_eps = float16((float)one + (float)eps);
    
    CHECK(eps > float16::zero());
    CHECK(one_plus_eps > one);
    CHECK(eps < one);
    
    // Test denormalized minimum
    float16 denorm_min = float16::denorm_min();
    CHECK((float)denorm_min > 0.0f);
    CHECK(denorm_min > float16::zero());
    CHECK(denorm_min < min_normal);
    
    // Test zeros
    float16 pos_zero = float16::zero();
    float16 neg_zero = float16::negative_zero();
    
    CHECK_EQUAL((float)pos_zero, 0.0f);
    CHECK_EQUAL((float)neg_zero, -0.0f);
    CHECK(pos_zero == neg_zero);  // IEEE 754: +0 == -0
    
    // Test one
    CHECK_EQUAL((float)one, 1.0f);
    CHECK(one > pos_zero);
    CHECK(one > neg_zero);
}

void test_float16_static_values_ordering() {
    // Test ordering of special values
    float16 neg_inf = float16::negative_infinity();
    float16 lowest = float16::lowest();
    float16 neg_zero = float16::negative_zero();
    float16 pos_zero = float16::zero();
    float16 denorm_min = float16::denorm_min();
    float16 min_normal = float16::min();
    float16 one = float16::one();
    float16 max_val = float16::max();
    float16 pos_inf = float16::infinity();
    
    // Test complete ordering (excluding NaN)
    CHECK(neg_inf < lowest);
    CHECK(lowest < neg_zero);
    CHECK(neg_zero == pos_zero);  // IEEE 754 equality
    CHECK(pos_zero < denorm_min);
    CHECK(denorm_min < min_normal);
    CHECK(min_normal < one);
    CHECK(one < max_val);
    CHECK(max_val < pos_inf);
    
    // Test transitivity with a few key comparisons
    CHECK(neg_inf < one);
    CHECK(lowest < max_val);
    CHECK(denorm_min < pos_inf);
}

void test_float16_static_values_bit_patterns() {
    // Test that static methods return expected bit patterns
    CHECK_EQUAL(float16::infinity().to_bits(), 0x7C00);
    CHECK_EQUAL(float16::negative_infinity().to_bits(), 0xFC00);
    CHECK_EQUAL(float16::quiet_NaN().to_bits(), 0x7E00);
    CHECK_EQUAL(float16::signaling_NaN().to_bits(), 0x7D00);
    CHECK_EQUAL(float16::max().to_bits(), 0x7BFF);
    CHECK_EQUAL(float16::lowest().to_bits(), 0xFBFF);
    CHECK_EQUAL(float16::min().to_bits(), 0x0400);
    CHECK_EQUAL(float16::epsilon().to_bits(), 0x1400);
    CHECK_EQUAL(float16::denorm_min().to_bits(), 0x0001);
    CHECK_EQUAL(float16::zero().to_bits(), 0x0000);
    CHECK_EQUAL(float16::negative_zero().to_bits(), 0x8000);
    CHECK_EQUAL(float16::one().to_bits(), 0x3C00);
}

void test_float16_integral_equality() {
    // Test equality with exact integer values
    float16 f1(1.0f);
    float16 f2(42.0f);
    float16 f3(-5.0f);
    float16 f0(0.0f);
    
    // Test direct comparison with integers
    CHECK(f1 == 1);
    CHECK(1 == f1);
    CHECK(f2 == 42);
    CHECK(42 == f2);
    CHECK(f3 == -5);
    CHECK(-5 == f3);
    CHECK(f0 == 0);
    CHECK(0 == f0);
    
    // Test inequality with different values
    CHECK(!(f1 == 2));
    CHECK(!(2 == f1));
    CHECK(!(f2 == 43));
    CHECK(!(43 == f2));
    CHECK(!(f3 == -6));
    CHECK(!(-6 == f3));
    
    // Test with different integer types
    CHECK(f1 == int8_t(1));
    CHECK(f1 == uint8_t(1));
    CHECK(f1 == int16_t(1));
    CHECK(f1 == uint16_t(1));
    CHECK(f1 == int32_t(1));
    CHECK(f1 == uint32_t(1));
    CHECK(f1 == int64_t(1));
    CHECK(f1 == uint64_t(1));
    
    // Test with large values within float16 range
    float16 large_pos(32000.0f);
    float16 large_neg(-32000.0f);
    CHECK(large_pos == 32000);
    CHECK(large_neg == -32000);
    
    // Test with values outside float16 range
    float16 max_val = float16::max();  // 65504
    float16 min_val = float16::lowest(); // -65504
    
    CHECK(max_val == 65504);
    CHECK(min_val == -65504);
    CHECK(!(max_val == 65505));  // Outside range
    CHECK(!(min_val == -65505)); // Outside range
    CHECK(!(max_val == 100000)); // Way outside range
    CHECK(!(min_val == -100000)); // Way outside range
}

void test_float16_integral_equality_special_values() {
    // Test with special float16 values
    float16 pos_inf = float16::infinity();
    float16 neg_inf = float16::negative_infinity();
    float16 qnan = float16::quiet_NaN();
    
    // Infinity should never equal any integer
    CHECK(!(pos_inf == 1));
    CHECK(!(1 == pos_inf));
    CHECK(!(neg_inf == -1));
    CHECK(!((-1) == neg_inf));
    CHECK(!(pos_inf == 0));
    CHECK(!(0 == pos_inf));
    
    // NaN should never equal any integer
    CHECK(!(qnan == 1));
    CHECK(!(1 == qnan));
    CHECK(!(qnan == 0));
    CHECK(!(0 == qnan));
    CHECK(!(qnan == -1));
    CHECK(!((-1) == qnan));
    
    // Test zeros
    float16 pos_zero = float16::zero();
    float16 neg_zero = float16::negative_zero();
    
    CHECK(pos_zero == 0);
    CHECK(0 == pos_zero);
    CHECK(neg_zero == 0);  // -0.0 should equal 0
    CHECK(0 == neg_zero);
}

void test_float16_integral_equality_precision() {
    // Test values that should be equal despite precision differences
    float16 f1(1024.0f);   // Exactly representable
    float16 f2(1025.0f);   // Exactly representable
    float16 f3(1023.0f);   // Exactly representable
    
    CHECK(f1 == 1024);
    CHECK(f2 == 1025);
    CHECK(f3 == 1023);
    
    // Test that values that don't have exact float16 representation
    // behave correctly (they should round and then compare)
    float16 f_rounded(1.1f);  // Will be rounded in float16
    CHECK(!(f_rounded == 1)); // Should not equal 1
    CHECK(f_rounded != 1);    // Should be unequal to 1
    
    // Test fractional values constructed from integers
    float16 f_from_int(5);
    CHECK(f_from_int == 5);
    CHECK(5 == f_from_int);
    CHECK(static_cast<float>(f_from_int) == 5.0f);
}

void test_float16_floating_point_equality() {
    // Test equality with exact floating-point values
    float16 f1(1.0f);
    float16 f2(42.5f);
    float16 f3(-3.14f);
    float16 f0(0.0f);
    
    // Test direct comparison with float
    CHECK(f1 == 1.0f);
    CHECK(1.0f == f1);
    CHECK(f2 == 42.5f);
    CHECK(42.5f == f2);
    CHECK(f3 == -3.14f);  // Note: this might be approximate due to float16 precision
    CHECK(-3.14f == f3);
    CHECK(f0 == 0.0f);
    CHECK(0.0f == f0);
    
    // Test with double
    CHECK(f1 == 1.0);
    CHECK(1.0 == f1);
    CHECK(f0 == 0.0);
    CHECK(0.0 == f0);
    
    // Test with long double
    CHECK(f1 == 1.0L);
    CHECK(1.0L == f1);
    
    // Test inequality with different values
    CHECK(!(f1 == 2.0f));
    CHECK(!(2.0f == f1));
    CHECK(!(f2 == 43.0f));
    CHECK(!(43.0f == f2));
    
    // Test with small differences that should be unequal
    float16 f_small(1.001f);  // Will be rounded in float16
    CHECK(!(f_small == 1.0f)); // Should not be exactly equal to 1.0
    
    // Test with values that round to the same float16 value
    float val1 = 1.0f;
    float val2 = 1.0f + 1e-8f; // Very small difference
    float16 f16_val1(val1);
    float16 f16_val2(val2);
    
    // If they round to the same float16 value, they should be equal to either original float
    if (f16_val1.to_bits() == f16_val2.to_bits()) {
        // Both should compare equal to the float16 representation
        float f16_as_float = static_cast<float>(f16_val1);
        CHECK(f16_val1 == f16_as_float);
        CHECK(f16_val2 == f16_as_float);
    }
}

void test_float16_floating_point_equality_special_values() {
    // Test with special floating-point values
    float16 pos_inf = float16::infinity();
    float16 neg_inf = float16::negative_infinity();
    float16 qnan = float16::quiet_NaN();
    
    // Test infinity comparisons
    CHECK(pos_inf == std::numeric_limits<float>::infinity());
    CHECK(std::numeric_limits<float>::infinity() == pos_inf);
    CHECK(neg_inf == -std::numeric_limits<float>::infinity());
    CHECK(-std::numeric_limits<float>::infinity() == neg_inf);
    
    // Test that different infinities are not equal
    CHECK(!(pos_inf == -std::numeric_limits<float>::infinity()));
    CHECK(!(neg_inf == std::numeric_limits<float>::infinity()));
    
    // Test NaN comparisons - NaN should never equal anything, including itself
    float float_nan = std::numeric_limits<float>::quiet_NaN();
    CHECK(!(qnan == float_nan));
    CHECK(!(float_nan == qnan));
    CHECK(!(qnan == 1.0f));
    CHECK(!(1.0f == qnan));
    CHECK(!(qnan == 0.0f));
    CHECK(!(0.0f == qnan));
    
    // Test with signaling NaN
    float16 snan = float16::signaling_NaN();
    CHECK(!(snan == float_nan));
    CHECK(!(float_nan == snan));
    CHECK(!(snan == 1.0f));
    CHECK(!(1.0f == snan));
    
    // Test zeros
    float16 pos_zero = float16::zero();
    float16 neg_zero = float16::negative_zero();
    
    CHECK(pos_zero == 0.0f);
    CHECK(0.0f == pos_zero);
    CHECK(neg_zero == -0.0f);
    CHECK(-0.0f == neg_zero);
    
    // IEEE 754: +0.0 should equal -0.0
    CHECK(pos_zero == -0.0f);
    CHECK(-0.0f == pos_zero);
    CHECK(neg_zero == 0.0f);
    CHECK(0.0f == neg_zero);
}

void test_float16_floating_point_equality_precision_limits() {
    // Test values near float16 precision limits
    float16 epsilon = float16::epsilon();
    
    // Test that epsilon comparisons work correctly
    float eps_as_float = static_cast<float>(epsilon);
    CHECK(epsilon == eps_as_float);
    CHECK(eps_as_float == epsilon);
    
    // Test with double precision
    double eps_as_double = static_cast<double>(epsilon);
    CHECK(epsilon == eps_as_double);
    CHECK(eps_as_double == epsilon);
    
    // Test that very small differences in float are detected
    float16 small_val(1e-4f);
    float slightly_different = static_cast<float>(small_val) + 1e-10f; // Tiny difference
    
    // The float16 should equal its exact float representation
    CHECK(small_val == static_cast<float>(small_val));
    
    // But may not equal a slightly different float (depends on precision)
    // This test checks that we properly handle precision differences
    float16 from_different(slightly_different);
    if (small_val.to_bits() != from_different.to_bits()) {
        CHECK(!(small_val == slightly_different));
    }
    
    // Test maximum and minimum finite values
    float16 max_val = float16::max();
    float16 min_val = float16::lowest();
    
    CHECK(max_val == 65504.0f);
    CHECK(65504.0f == max_val);
    CHECK(min_val == -65504.0f);
    CHECK(-65504.0f == min_val);
    
    // Test with double
    CHECK(max_val == 65504.0);
    CHECK(65504.0 == max_val);
    CHECK(min_val == -65504.0);
    CHECK(-65504.0 == min_val);
}

void test_float16_floating_point_mixed_precision() {
    // Test comparisons between different precisions
    float16 f16_val(3.14159f);
    
    // Convert to different floating-point types
    float as_float = static_cast<float>(f16_val);
    double as_double = static_cast<double>(f16_val);
    long double as_long_double = static_cast<long double>(f16_val);
    
    // All should be equal to the float16
    CHECK(f16_val == as_float);
    CHECK(as_float == f16_val);
    CHECK(f16_val == as_double);
    CHECK(as_double == f16_val);
    CHECK(f16_val == as_long_double);
    CHECK(as_long_double == f16_val);
    
    // Test with values that have different representations in different precisions
    float16 precise_val(1.0f/3.0f); // 1/3 - will be rounded in float16
    float float_third = 1.0f/3.0f;
    double double_third = 1.0/3.0;
    
    // The float16 should equal its exact conversion to float
    CHECK(precise_val == static_cast<float>(precise_val));
    
    // But the original 1/3 calculations might be different due to precision
    // This depends on how the rounding works
    float16 from_float_third(float_third);
    float16 from_double_third(static_cast<float>(double_third));
    
    // These should be consistent
    CHECK(precise_val == static_cast<float>(precise_val));
    CHECK(from_float_third == float_third);
    CHECK(from_double_third == float_third);
}

void test_float16_double_constructor() {
    // Test basic double to float16 conversion
    float16 f1(1.0);    // double literal
    float16 f2(2.5);
    float16 f3(-3.14159);
    float16 f0(0.0);
    float16 fneg0(-0.0);
    
    // Check basic conversions
    CHECK_EQUAL(static_cast<float>(f1), 1.0f);
    CHECK_EQUAL(static_cast<float>(f2), 2.5f);
    CHECK_EQUAL(static_cast<float>(f0), 0.0f);
    CHECK_EQUAL(static_cast<float>(fneg0), -0.0f);
    
    // Test that pi gets rounded appropriately
    float pi_as_float16 = static_cast<float>(f3);
    CHECK(pi_as_float16 < -3.0f && pi_as_float16 > -3.5f); // Rough range check
    
    // Test special double values
    float16 pos_inf(std::numeric_limits<double>::infinity());
    float16 neg_inf(-std::numeric_limits<double>::infinity());
    float16 qnan(std::numeric_limits<double>::quiet_NaN());
    
    CHECK(std::isinf(static_cast<float>(pos_inf)));
    CHECK(static_cast<float>(pos_inf) > 0);
    CHECK(std::isinf(static_cast<float>(neg_inf)));
    CHECK(static_cast<float>(neg_inf) < 0);
    CHECK(std::isnan(static_cast<float>(qnan)));
    
    // Test values near float16 limits - use values that are exactly representable
    float16 large_pos(32000.0);   // This should be exactly representable
    float16 large_neg(-32000.0);
    float16 too_large(1e10);      // Should overflow to infinity
    float16 too_small(-1e10);     // Should overflow to negative infinity
    
    // Check that 32000 is representable (or close enough)
    float converted_pos = static_cast<float>(large_pos);
    float converted_neg = static_cast<float>(large_neg);
    CHECK(std::abs(converted_pos - 32000.0f) < 100.0f); // Allow some rounding error
    CHECK(std::abs(converted_neg - (-32000.0f)) < 100.0f); // Allow some rounding error
    
    CHECK(std::isinf(static_cast<float>(too_large)));
    CHECK(static_cast<float>(too_large) > 0);
    CHECK(std::isinf(static_cast<float>(too_small)));
    CHECK(static_cast<float>(too_small) < 0);
    
    // Test very small values (should underflow to zero)
    float16 tiny_pos(1e-50);      // Much smaller than float16 can represent
    float16 tiny_neg(-1e-50);
    
    CHECK_EQUAL(static_cast<float>(tiny_pos), 0.0f);
    CHECK_EQUAL(static_cast<float>(tiny_neg), -0.0f);
    
    // Test values that should be exactly representable in float16
    float16 exact1(1024.0);       // 2^10
    float16 exact2(512.0);        // 2^9  
    float16 exact3(256.0);        // 2^8
    
    CHECK_EQUAL(static_cast<float>(exact1), 1024.0f);
    CHECK_EQUAL(static_cast<float>(exact2), 512.0f);
    CHECK_EQUAL(static_cast<float>(exact3), 256.0f);
}

void test_float16_double_precision_conversion() {
    // Test precision handling during double->float16 conversion
    
    // Test values that should round correctly
    double precise_values[] = {
        1.0, 2.0, 3.0, 4.0, 5.0,
        0.5, 0.25, 0.125, 0.0625,
        1.5, 2.5, 3.5, 4.5,
        10.0, 100.0, 1000.0
    };
    
    for (double val : precise_values) {
        float16 f16_pos(val);
        float16 f16_neg(-val);
        
        float converted_pos = static_cast<float>(f16_pos);
        float converted_neg = static_cast<float>(f16_neg);
        
        // These should be exactly representable or very close
        CHECK(std::abs(converted_pos - static_cast<float>(val)) < 0.01f);
        CHECK(std::abs(converted_neg - static_cast<float>(-val)) < 0.01f);
    }
    
    // Test that double precision values get properly rounded
    double high_precision = 1.23456789012345; // More precision than float16 can hold
    float16 f16_rounded(high_precision);
    float converted = static_cast<float>(f16_rounded);
    
    // Should be close to original but not exact due to rounding
    CHECK(std::abs(converted - static_cast<float>(high_precision)) < 0.1f);
    
    // Test subnormal handling
    double very_small = 1e-6; // Might become subnormal in float16
    float16 f16_small(very_small);
    float converted_small = static_cast<float>(f16_small);
    
    // Should be representable as a small positive number
    CHECK(converted_small >= 0.0f);
    CHECK(converted_small <= static_cast<float>(very_small) * 2.0f); // Allow for rounding
}

void test_float16_double_comparison() {
    // Test that double constructor creates values that compare correctly
    // with the floating-point equality operator
    
    double test_values[] = {
        0.0, 1.0, -1.0, 2.5, -2.5,
        3.125, -3.125,  // Use values exactly representable in float16
        1000.0, -1000.0,
        32768.0, -32768.0  // Use smaller values that are more likely to be exactly representable
    };
    
    for (double val : test_values) {
        float16 f16_from_double(val);
        
        // Should be equal to the original double value (within float16 precision)
        CHECK(f16_from_double == val);
        CHECK(val == f16_from_double);
        
        // Should also be equal to the float version
        float float_val = static_cast<float>(val);
        CHECK(f16_from_double == float_val);
        CHECK(float_val == f16_from_double);
    }
    
    // Test special values
    float16 inf_from_double(std::numeric_limits<double>::infinity());
    float16 ninf_from_double(-std::numeric_limits<double>::infinity());
    
    CHECK(inf_from_double == std::numeric_limits<double>::infinity());
    CHECK(ninf_from_double == -std::numeric_limits<double>::infinity());
    CHECK(inf_from_double == std::numeric_limits<float>::infinity());
    CHECK(ninf_from_double == -std::numeric_limits<float>::infinity());
}

void test_float16_integral_three_way_comparison() {
    // Test basic three-way comparisons with integers
    float16 f1(5.0f);
    float16 f2(-3.0f);
    float16 f0(0.0f);
    
    // Test less than
    CHECK((f1 <=> 10) == std::partial_ordering::less);
    CHECK((f1 <=> 6) == std::partial_ordering::less);
    CHECK((f2 <=> 0) == std::partial_ordering::less);
    CHECK((f2 <=> -1) == std::partial_ordering::less);
    
    // Test greater than
    CHECK((f1 <=> 3) == std::partial_ordering::greater);
    CHECK((f1 <=> 0) == std::partial_ordering::greater);
    CHECK((f2 <=> -5) == std::partial_ordering::greater);
    CHECK((f0 <=> -1) == std::partial_ordering::greater);
    
    // Test equivalent
    CHECK((f1 <=> 5) == std::partial_ordering::equivalent);
    CHECK((f2 <=> -3) == std::partial_ordering::equivalent);
    CHECK((f0 <=> 0) == std::partial_ordering::equivalent);
    
    // Test with different integer types
    CHECK((f1 <=> int8_t(5)) == std::partial_ordering::equivalent);
    CHECK((f1 <=> uint8_t(5)) == std::partial_ordering::equivalent);
    CHECK((f1 <=> int16_t(5)) == std::partial_ordering::equivalent);
    CHECK((f1 <=> uint16_t(5)) == std::partial_ordering::equivalent);
    CHECK((f1 <=> int32_t(5)) == std::partial_ordering::equivalent);
    CHECK((f1 <=> uint32_t(5)) == std::partial_ordering::equivalent);
    CHECK((f1 <=> int64_t(5)) == std::partial_ordering::equivalent);
    CHECK((f1 <=> uint64_t(5)) == std::partial_ordering::equivalent);
}

void test_float16_integral_three_way_comparison_special_values() {
    // Test with special float16 values
    float16 pos_inf = float16::infinity();
    float16 neg_inf = float16::negative_infinity();
    float16 qnan = float16::quiet_NaN();
    float16 snan = float16::signaling_NaN();
    
    // Infinity comparisons
    CHECK((pos_inf <=> 1000000) == std::partial_ordering::greater);
    CHECK((pos_inf <=> -1000000) == std::partial_ordering::greater);
    CHECK((pos_inf <=> 0) == std::partial_ordering::greater);
    
    CHECK((neg_inf <=> 1000000) == std::partial_ordering::less);
    CHECK((neg_inf <=> -1000000) == std::partial_ordering::less);
    CHECK((neg_inf <=> 0) == std::partial_ordering::less);
    
    // Test with double infinity
    CHECK((pos_inf <=> std::numeric_limits<double>::infinity()) == std::partial_ordering::equivalent);
    CHECK((neg_inf <=> -std::numeric_limits<double>::infinity()) == std::partial_ordering::equivalent);
    
    // NaN comparisons should be unordered
    CHECK((qnan <=> 42) == std::partial_ordering::unordered);
    CHECK((qnan <=> 0) == std::partial_ordering::unordered);
    CHECK((qnan <=> -42) == std::partial_ordering::unordered);
    
    CHECK((snan <=> 42) == std::partial_ordering::unordered);
    CHECK((snan <=> 0) == std::partial_ordering::unordered);
    CHECK((snan <=> -42) == std::partial_ordering::unordered);
    
    // Test zeros
    float16 pos_zero = float16::zero();
    float16 neg_zero = float16::negative_zero();
    
    CHECK((pos_zero <=> 0) == std::partial_ordering::equivalent);
    CHECK((neg_zero <=> 0) == std::partial_ordering::equivalent);
    CHECK((pos_zero <=> 1) == std::partial_ordering::less);
    CHECK((pos_zero <=> -1) == std::partial_ordering::greater);
    CHECK((neg_zero <=> 1) == std::partial_ordering::less);
    CHECK((neg_zero <=> -1) == std::partial_ordering::greater);
}

void test_float16_integral_three_way_comparison_boundary_values() {
    // Test with values at the boundary of float16 range
    float16 max_val = float16::max();     // 65504
    float16 min_val = float16::lowest();  // -65504
    
    // Values within range
    CHECK((max_val <=> 65504) == std::partial_ordering::equivalent);
    CHECK((min_val <=> -65504) == std::partial_ordering::equivalent);
    CHECK((max_val <=> 65503) == std::partial_ordering::greater);
    CHECK((min_val <=> -65503) == std::partial_ordering::less);
    
    // Values outside float16 range
    CHECK((max_val <=> 100000) == std::partial_ordering::less);
    CHECK((min_val <=> -100000) == std::partial_ordering::greater);
    CHECK((max_val <=> 65505) == std::partial_ordering::less);
    CHECK((min_val <=> -65505) == std::partial_ordering::greater);
    
    // Test with unsigned integers that are too large
    float16 small_positive(1.0f);
    CHECK((small_positive <=> uint32_t(100000)) == std::partial_ordering::less);
    CHECK((small_positive <=> uint64_t(100000)) == std::partial_ordering::less);
    
    // Test negative float16 with unsigned integers
    float16 negative(-5.0f);
    CHECK((negative <=> uint32_t(0)) == std::partial_ordering::less);
    CHECK((negative <=> uint32_t(10)) == std::partial_ordering::less);
}

void test_float16_integral_three_way_comparison_precision() {
    // Test values that test float16 precision limits
    float16 f1(1024.0f);   // Exactly representable
    float16 f2(1025.0f);   // Exactly representable  
    float16 f3(1023.0f);   // Exactly representable
    
    CHECK((f1 <=> 1024) == std::partial_ordering::equivalent);
    CHECK((f2 <=> 1025) == std::partial_ordering::equivalent);
    CHECK((f3 <=> 1023) == std::partial_ordering::equivalent);
    
    CHECK((f1 <=> 1023) == std::partial_ordering::greater);
    CHECK((f1 <=> 1025) == std::partial_ordering::less);
    
    // Test with values that get rounded in float16
    float16 rounded(1.1f);  // Gets rounded to closest representable value
    // The exact comparison depends on how 1.1f rounds in float16
    auto result = rounded <=> 1;
    CHECK(result != std::partial_ordering::unordered); // Should not be unordered
    
    // Test small values
    float16 small(0.5f);
    CHECK((small <=> 0) == std::partial_ordering::greater);
    CHECK((small <=> 1) == std::partial_ordering::less);
    
    // Test with subnormal values
    float16 denorm_min = float16::denorm_min();
    CHECK((denorm_min <=> 0) == std::partial_ordering::greater);
    CHECK((denorm_min <=> 1) == std::partial_ordering::less);
}

void test_float16_integral_three_way_comparison_derived_operators() {
    // Test that the three-way comparison enables all other comparison operators
    float16 f1(5.0f);
    float16 f2(-3.0f);
    
    // Less than
    CHECK(f1 < 10);
    CHECK(f1 < 6);
    CHECK(f2 < 0);
    CHECK(f2 < -1);
    
    // Greater than
    CHECK(f1 > 3);
    CHECK(f1 > 0);
    CHECK(f2 > -5);
    CHECK(f2 > -10);
    
    // Less than or equal
    CHECK(f1 <= 10);
    CHECK(f1 <= 5);
    CHECK(f2 <= 0);
    CHECK(f2 <= -3);
    
    // Greater than or equal
    CHECK(f1 >= 3);
    CHECK(f1 >= 5);
    CHECK(f2 >= -5);
    CHECK(f2 >= -3);
    
    // Test that NaN comparisons return false for all derived operators
    float16 qnan = float16::quiet_NaN();
    CHECK(!(qnan < 42));
    CHECK(!(qnan > 42));
    CHECK(!(qnan <= 42));
    CHECK(!(qnan >= 42));
    CHECK(!(42 < qnan));
    CHECK(!(42 > qnan));
    CHECK(!(42 <= qnan));
    CHECK(!(42 >= qnan));
}

void test_float16_floating_point_three_way_comparison() {
    // Test basic three-way comparisons with floating-point types
    float16 f1(5.0f);
    float16 f2(-3.14f);
    float16 f0(0.0f);

    // Test with float
    CHECK((f1 <=> 10.0f) == std::partial_ordering::less);
    CHECK((f1 <=> 3.0f) == std::partial_ordering::greater);
    CHECK((f1 <=> 5.0f) == std::partial_ordering::equivalent);
    CHECK((f2 <=> 0.0f) == std::partial_ordering::less);
    CHECK((f2 <=> -5.0f) == std::partial_ordering::greater);
    CHECK((f0 <=> 0.0f) == std::partial_ordering::equivalent);

    // Test with double
    CHECK((f1 <=> 10.0) == std::partial_ordering::less);
    CHECK((f1 <=> 3.0) == std::partial_ordering::greater);
    CHECK((f1 <=> 5.0) == std::partial_ordering::equivalent);
    //CHECK((f2 <=> -3.14) == std::partial_ordering::equivalent); // Note: might be approximate due to precision

    // Test with long double
    CHECK((f1 <=> 5.0L) == std::partial_ordering::equivalent);
    CHECK((f0 <=> 0.0L) == std::partial_ordering::equivalent);

    // Test zeros (positive and negative should be equivalent)
    float16 pos_zero = float16::zero();
    float16 neg_zero = float16::negative_zero();
    CHECK((pos_zero <=> 0.0f) == std::partial_ordering::equivalent);
    CHECK((neg_zero <=> -0.0f) == std::partial_ordering::equivalent);
    CHECK((pos_zero <=> -0.0f) == std::partial_ordering::equivalent); // IEEE 754: +0 == -0
    CHECK((neg_zero <=> 0.0f) == std::partial_ordering::equivalent);
}

void test_float16_floating_point_three_way_comparison_special_values() {
    // Test with infinity
    float16 pos_inf = float16::infinity();
    float16 neg_inf = float16::negative_infinity();

    // Infinity vs infinity
    CHECK((pos_inf <=> std::numeric_limits<float>::infinity()) == std::partial_ordering::equivalent);
    CHECK((neg_inf <=> -std::numeric_limits<float>::infinity()) == std::partial_ordering::equivalent);
    CHECK((pos_inf <=> -std::numeric_limits<float>::infinity()) == std::partial_ordering::greater);
    CHECK((neg_inf <=> std::numeric_limits<float>::infinity()) == std::partial_ordering::less);

    // Infinity vs finite values
    CHECK((pos_inf <=> 1000000.0f) == std::partial_ordering::greater);
    CHECK((pos_inf <=> -1000000.0f) == std::partial_ordering::greater);
    CHECK((neg_inf <=> 1000000.0f) == std::partial_ordering::less);
    CHECK((neg_inf <=> -1000000.0f) == std::partial_ordering::less);

    // Test with double infinity
    CHECK((pos_inf <=> std::numeric_limits<double>::infinity()) == std::partial_ordering::equivalent);
    CHECK((neg_inf <=> -std::numeric_limits<double>::infinity()) == std::partial_ordering::equivalent);

    // NaN comparisons should be unordered
    float16 qnan = float16::quiet_NaN();
    float16 snan = float16::signaling_NaN();

    float float_nan = std::numeric_limits<float>::quiet_NaN();
    double double_nan = std::numeric_limits<double>::quiet_NaN();

    CHECK((qnan <=> 42.0f) == std::partial_ordering::unordered);
    CHECK((qnan <=> float_nan) == std::partial_ordering::unordered);
    CHECK((qnan <=> double_nan) == std::partial_ordering::unordered);
    CHECK((snan <=> 42.0f) == std::partial_ordering::unordered);
    CHECK((snan <=> float_nan) == std::partial_ordering::unordered);

    // Test finite float16 vs NaN
    float16 finite_val(3.14f);
    CHECK((finite_val <=> float_nan) == std::partial_ordering::unordered);
    CHECK((finite_val <=> double_nan) == std::partial_ordering::unordered);
}

void test_float16_floating_point_three_way_comparison_precision() {
    // Test precision-related comparisons
    float16 precise_val(1.0f);

    // Test with values that should be exactly equal
    CHECK((precise_val <=> 1.0f) == std::partial_ordering::equivalent);
    CHECK((precise_val <=> 1.0) == std::partial_ordering::equivalent);
    CHECK((precise_val <=> 1.0L) == std::partial_ordering::equivalent);

    // Test with small differences
    float16 small_val(0.5f);
    CHECK((small_val <=> 0.5f) == std::partial_ordering::equivalent);
    CHECK((small_val <=> 0.499f) == std::partial_ordering::greater);
    CHECK((small_val <=> 0.501f) == std::partial_ordering::less);

    // Test with larger values where float16 precision matters
    float16 large_val(1024.0f);  // Exactly representable
    CHECK((large_val <=> 1024.0f) == std::partial_ordering::equivalent);
    CHECK((large_val <=> 1024.0) == std::partial_ordering::equivalent);
    CHECK((large_val <=> 1023.0f) == std::partial_ordering::greater);
    CHECK((large_val <=> 1025.0f) == std::partial_ordering::less);

    // Test boundary values
    float16 max_val = float16::max();
    float16 min_val = float16::lowest();

    CHECK((max_val <=> 65504.0f) == std::partial_ordering::equivalent);
    CHECK((min_val <=> -65504.0f) == std::partial_ordering::equivalent);
    CHECK((max_val <=> 100000.0f) == std::partial_ordering::less);
    CHECK((min_val <=> -100000.0f) == std::partial_ordering::greater);
}

void test_float16_floating_point_three_way_comparison_derived_operators() {
    // Test that the three-way comparison enables all other comparison operators
    float16 f1(5.5f);
    float16 f2(-2.3f);

    // Less than
    CHECK(f1 < 10.0f);
    CHECK(f1 < 6.0f);
    CHECK(f2 < 0.0f);
    CHECK(f2 < -1.0f);

    // Greater than
    CHECK(f1 > 3.0f);
    CHECK(f1 > 5.0f);
    CHECK(f2 > -5.0f);
    CHECK(f2 > -3.0f);

    // Less than or equal
    CHECK(f1 <= 10.0f);
    CHECK(f1 <= 5.5f);
    CHECK(f2 <= 0.0f);
    CHECK(f2 <= -2.3f);

    // Greater than or equal
    CHECK(f1 >= 3.0f);
    CHECK(f1 >= 5.5f);
    CHECK(f2 >= -5.0f);
    CHECK(f2 >= -2.3f);

    // Test with double
    CHECK(f1 < 10.0);
    CHECK(f1 > 3.0);
    CHECK(f1 <= 5.5);
    CHECK(f1 >= 5.5);

    // Test that NaN comparisons return false for all derived operators
    float16 qnan = float16::quiet_NaN();
    float float_nan = std::numeric_limits<float>::quiet_NaN();

    CHECK(!(qnan < 42.0f));
    CHECK(!(qnan > 42.0f));
    CHECK(!(qnan <= 42.0f));
    CHECK(!(qnan >= 42.0f));
    CHECK(!(qnan < float_nan));
    CHECK(!(qnan > float_nan));
    CHECK(!(qnan <= float_nan));
    CHECK(!(qnan >= float_nan));

    // Test reverse comparisons (should work automatically)
    CHECK(10.0f > f1);
    CHECK(3.0f < f1);
    CHECK(5.5f >= f1);
    CHECK(5.5f <= f1);
}

} // namespace numetron
