//  Sonia.one framework (c) by Alexander A Pototskiy
//  Sonia.one is licensed under the terms of the Open Source GPL 3.0 license.
//  For a license to use the Sonia.one software under conditions other than those described here, please contact me at admin@sonia.one

#pragma once
#include <cstdint>
#include <cmath>
#include <ostream>
#include <concepts>
#include <bit>
#include <compare>
#include <utility>
#include <functional>
#include <type_traits>

namespace numetron {

class float16
{
    uint16_t data;

public:
    float16() = default;

    template <std::floating_point T>
    requires(sizeof(T) == 4)
    explicit float16(T) noexcept;

    template <std::floating_point T>
    requires(sizeof(T) == 8)
    explicit float16(T val) noexcept;

    template <std::floating_point T>
    requires(sizeof(T) != 4 && sizeof(T) != 8)
    inline explicit float16(T val) noexcept
        : float16(static_cast<double>(val))
    {}

    template <std::integral T>
    constexpr float16(T val) noexcept;

    float16(float16 const&) = default;

    float16& operator= (float16 const&) = default;

    template <std::integral T>
    inline constexpr float16& operator= (T val) noexcept
    {
        *this = float16(val);
        return *this;
    }

    template <std::floating_point T>
    inline constexpr float16& operator= (T val) noexcept
    {
        *this = float16(val);
        return *this;
    }

    // Static method to create float16 from raw bit representation
    inline static constexpr float16 from_bits(uint16_t bits) noexcept
    {
        float16 result;
        result.data = bits;
        return result;
    }

    // Method to get the raw bit representation
    inline uint16_t to_bits() const noexcept
    {
        return data;
    }

    // Static methods for special values
    static constexpr float16 infinity() noexcept
    {
        return from_bits(0x7C00);  // +infinity: sign=0, exp=0x1F, mantissa=0
    }
    
    static constexpr float16 negative_infinity() noexcept
    {
        return from_bits(0xFC00);  // -infinity: sign=1, exp=0x1F, mantissa=0
    }
    
    static constexpr float16 quiet_NaN() noexcept
    {
        return from_bits(0x7E00);  // quiet NaN: sign=0, exp=0x1F, mantissa=0x200
    }
    
    static constexpr float16 signaling_NaN() noexcept
    {
        return from_bits(0x7D00);  // signaling NaN: sign=0, exp=0x1F, mantissa=0x100
    }
    
#if defined(_MSC_VER)
    static constexpr float16 (max)() noexcept
    {
        return from_bits(0x7BFF);  // Maximum finite value: 65504.0
    }
    static constexpr float16 (min)() noexcept
    {
        return from_bits(0x0400);  // Minimum positive normal value: 6.103515625e-5
    }
#else
    static constexpr float16 max() noexcept
    {
        return from_bits(0x7BFF);  // Maximum finite value: 65504.0
    }
    static constexpr float16 min() noexcept
    {
        return from_bits(0x0400);  // Minimum positive normal value: 6.103515625e-5
    }
#endif

    static constexpr float16 lowest() noexcept
    {
        return from_bits(0xFBFF);  // Minimum finite value: -65504.0
    }
    
    static constexpr float16 epsilon() noexcept
    {
        return from_bits(0x1400);  // Machine epsilon: 0.0009765625 (1/1024)
    }
    
    static constexpr float16 denorm_min() noexcept
    {
        return from_bits(0x0001);  // Smallest positive subnormal value: 5.9604644775390625e-8
    }
    
    static constexpr float16 zero() noexcept
    {
        return from_bits(0x0000);  // Positive zero
    }
    
    static constexpr float16 negative_zero() noexcept
    {
        return from_bits(0x8000);  // Negative zero
    }
    
    static constexpr float16 one() noexcept
    {
        return from_bits(0x3C00);  // 1.0
    }

    template <std::floating_point T>
    explicit operator T() const noexcept;

    friend bool operator== (float16 const& l, float16 const& r) noexcept;
    friend std::partial_ordering operator <=> (float16 const& l, float16 const& r) noexcept;

    template <std::integral T>
    friend bool operator== (float16 const& l, T const& r) noexcept;

    template <std::integral T>
    friend std::partial_ordering operator <=> (float16 const& l, T const& r) noexcept;

    template <std::floating_point T>
    friend bool operator== (float16 const& l, T const& r) noexcept;

    template <std::floating_point T>
    friend std::partial_ordering operator <=> (float16 const& l, T const& r) noexcept;

    friend inline size_t hash_value(float16 const& v) noexcept
    {
        return std::hash<uint16_t>{}(v.data);
    }
};

inline bool operator== (float16 const& l, float16 const& r) noexcept
{ 
    // NaN should not be equal to anything, including itself
    uint16_t l_exp = (l.data >> 10) & 0x1F;
    uint16_t r_exp = (r.data >> 10) & 0x1F;
    uint16_t l_mant = l.data & 0x3FF;
    uint16_t r_mant = r.data & 0x3FF;
    
    // Check for NaN
    bool l_is_nan = (l_exp == 0x1F) && (l_mant != 0);
    bool r_is_nan = (r_exp == 0x1F) && (r_mant != 0);
    
    if (l_is_nan || r_is_nan) {
        return false;
    }
    
    // For zeros, both +0 and -0 should be equal
    bool l_is_zero = (l_exp == 0) && (l_mant == 0);
    bool r_is_zero = (r_exp == 0) && (r_mant == 0);
    
    if (l_is_zero && r_is_zero) {
        return true;
    }
    
    return l.data == r.data; 
}

inline std::partial_ordering operator <=> (float16 const& l, float16 const& r) noexcept
{
    // Extract components for direct comparison
    uint16_t l_data = l.data;
    uint16_t r_data = r.data;
    
    // Extract sign, exponent, and mantissa
    bool l_sign = (l_data & 0x8000) != 0;
    bool r_sign = (r_data & 0x8000) != 0;
    uint16_t l_exp = (l_data >> 10) & 0x1F;
    uint16_t r_exp = (r_data >> 10) & 0x1F;
    uint16_t l_mant = l_data & 0x3FF;
    uint16_t r_mant = r_data & 0x3FF;
    
    // Handle NaN cases first
    bool l_is_nan = (l_exp == 0x1F) && (l_mant != 0);
    bool r_is_nan = (r_exp == 0x1F) && (r_mant != 0);
    
    if (l_is_nan || r_is_nan) {
        return std::partial_ordering::unordered;
    }
    
    // Handle infinity cases
    bool l_is_inf = (l_exp == 0x1F) && (l_mant == 0);
    bool r_is_inf = (r_exp == 0x1F) && (r_mant == 0);
    
    if (l_is_inf && r_is_inf) {
        if (l_sign == r_sign) {
            return std::partial_ordering::equivalent;
        }
        return l_sign ? std::partial_ordering::less : std::partial_ordering::greater;
    }
    
    if (l_is_inf) {
        return l_sign ? std::partial_ordering::less : std::partial_ordering::greater;
    }
    
    if (r_is_inf) {
        return r_sign ? std::partial_ordering::greater : std::partial_ordering::less;
    }
    
    // Handle zero cases (both +0 and -0 are equal)
    bool l_is_zero = (l_exp == 0) && (l_mant == 0);
    bool r_is_zero = (r_exp == 0) && (r_mant == 0);
    
    if (l_is_zero && r_is_zero) {
        return std::partial_ordering::equivalent;
    }
    
    // Different signs comparison
    if (l_sign != r_sign) {
        if (l_is_zero) return r_sign ? std::partial_ordering::greater : std::partial_ordering::less;
        if (r_is_zero) return l_sign ? std::partial_ordering::less : std::partial_ordering::greater;
        return l_sign ? std::partial_ordering::less : std::partial_ordering::greater;
    }
    
    // Same sign comparison - need to handle subnormal numbers properly
    // For same-sign numbers, we can use a simpler approach:
    // Convert both to a standardized form for comparison
    
    auto get_magnitude_bits = [](uint16_t exp, uint16_t mant) -> uint32_t {
        if (exp == 0) {
            // Subnormal or zero
            if (mant == 0) return 0;
            // For subnormal numbers, find the leading bit position
            uint32_t shift = 0;
            uint32_t temp_mant = mant;
            while ((temp_mant & 0x200) == 0) {
                temp_mant <<= 1;
                shift++;
            }
            // Effective exponent is 1 - shift, but we need positive values for comparison
            // Use (1 - shift + 1000) to make it positive, and include the normalized mantissa
            uint32_t eff_exp = 1000 + 1 - shift; // This ensures positive values
            uint32_t norm_mant = temp_mant & 0x1FF; // Remove the leading 1
            return (eff_exp << 10) | norm_mant;
        } else {
            // Normal numbers: use (exp + 1000) to avoid issues
            return ((exp + 1000) << 10) | mant;
        }
    };
    
    uint32_t l_mag = get_magnitude_bits(l_exp, l_mant);
    uint32_t r_mag = get_magnitude_bits(r_exp, r_mant);
    
    std::partial_ordering mag_cmp = l_mag <=> r_mag;
    //if (l_mag < r_mag) {
    //    mag_cmp = std::partial_ordering::less;
    //} else if (l_mag > r_mag) {
    //    mag_cmp = std::partial_ordering::greater;
    //} else {
    //    mag_cmp = std::partial_ordering::equivalent;
    //}
    
    // If both are negative, reverse the comparison
    if (l_sign) {
        if (mag_cmp == std::partial_ordering::less) return std::partial_ordering::greater;
        if (mag_cmp == std::partial_ordering::greater) return std::partial_ordering::less;
    }
    
    return mag_cmp;
}

template <typename E, typename T>
inline std::basic_ostream<E, T>& operator<<(std::basic_ostream<E, T>& os, float16 const& v)
{
    return os << (float)v;
}


template <std::floating_point T>
requires(sizeof(T) == 4)
inline float16::float16(T f32val) noexcept
{
    uint32_t f32data = std::bit_cast<uint32_t>(f32val);
#ifndef __F16CINTRIN_H
    const uint32_t frac32 = f32data & 0x7fffff;
    const uint8_t exp32 = f32data >> 23;
    const int8_t exp32_diff = exp32 - 127;

    uint16_t exp16 = 0;
    uint16_t frac16 = frac32 >> 13;

    if (exp32 == 0xff || exp32_diff > 15) {
        exp16 = 0x1f;
    } else if (!exp32 || exp32_diff < -14) {
        exp16 = 0;
    } else {
        exp16 = exp32_diff + 15;
    }

    if (exp32 == 0xff && frac32 && !frac16) {
        frac16 = 0x200;
    } else if (!exp32 || (exp16 == 0x1f && exp32 != 0xff)) {
        frac16 = 0;
    } else if (!exp16 && exp32) {
        frac16 = (0x200 | (frac16 >> 1)) >> (17 - exp32_diff);
    }

    data = (f32data & 0x80000000) >> 16; // sign
    data |= exp16 << 10;
    data |= frac16;
#else
    data = _cvtss_sh(f32val, 0);
#endif
}

template <std::floating_point T>
requires(sizeof(T) == 8)
inline float16::float16(T f64val) noexcept
{
    uint64_t f64data = std::bit_cast<uint64_t>(f64val);
    
    // Extract components from double (IEEE 754 binary64)
    bool sign = (f64data & 0x8000000000000000ULL) != 0;
    uint32_t exp64 = static_cast<uint32_t>((f64data >> 52) & 0x7FF); // 11 bits
    uint64_t mant64 = f64data & 0xFFFFFFFFFFFFFULL; // 52 bits
    
    uint16_t result = 0;
    
    // Handle special cases first
    if (exp64 == 0x7FF) { // Infinity or NaN
        result = 0x7C00; // Set exponent to all 1s
        if (mant64 != 0) { // NaN
            // Preserve NaN-ness but fit into 10-bit mantissa
            // Take the top bits of the mantissa to preserve NaN payload if possible
            uint16_t nan_payload = static_cast<uint16_t>((mant64 >> 42) & 0x3FF);
            if (nan_payload == 0) {
                nan_payload = 0x200; // Ensure it's still NaN (quiet NaN)
            }
            result |= nan_payload;
        }
        // For infinity, mantissa stays 0
    }
    else if (exp64 == 0) { // Zero or subnormal
        if (mant64 == 0) {
            result = 0; // Zero
        } else {
            // Subnormal double -> likely underflows to zero in float16
            // Double subnormals are much smaller than float16 can represent
            result = 0; // Underflow to zero
        }
    }
    else { // Normal number
        // Convert exponent: double bias = 1023, float16 bias = 15
        int32_t exp16 = static_cast<int32_t>(exp64) - 1023 + 15;
        
        if (exp16 <= 0) {
            // Underflow - might become subnormal or zero
            if (exp16 < -10) {
                result = 0; // Too small, underflow to zero
            } else {
                // Might be representable as subnormal
                // Subnormal float16: implicit leading bit becomes explicit
                // and we shift the mantissa based on how far below normal range we are
                uint64_t full_mant = (1ULL << 52) | mant64; // Add implicit leading 1
                int shift = 1 - exp16; // How much to shift right
                
                if (shift < 52) {
                    uint64_t shifted_mant = full_mant >> (shift + 42); // 52 - 10 = 42
                    if (shifted_mant > 0 && shifted_mant <= 0x3FF) {
                        result = static_cast<uint16_t>(shifted_mant);
                    } else {
                        result = 0; // Underflow
                    }
                } else {
                    result = 0; // Underflow
                }
            }
        }
        else if (exp16 >= 0x1F) {
            // Overflow to infinity
            result = 0x7C00;
        }
        else {
            // Normal range - extract top 10 bits of mantissa
            uint16_t mant16 = static_cast<uint16_t>(mant64 >> 42); // 52 - 10 = 42
            
            // Rounding: check the bit just below our precision (bit 41)
            if ((mant64 >> 41) & 1) {
                // Round up
                mant16++;
                if (mant16 > 0x3FF) { // Mantissa overflow
                    mant16 = 0;
                    exp16++;
                    if (exp16 >= 0x1F) { // Exponent overflow
                        result = 0x7C00; // Infinity
                    } else {
                        result = static_cast<uint16_t>((exp16 << 10) | mant16);
                    }
                } else {
                    result = static_cast<uint16_t>((exp16 << 10) | mant16);
                }
            } else {
                result = static_cast<uint16_t>((exp16 << 10) | mant16);
            }
        }
    }
    
    // Apply sign bit
    if (sign) {
        result |= 0x8000;
    }
    
    data = result;
}

template <std::floating_point T>
inline float16::operator T() const noexcept
{
#ifndef __F16CINTRIN_H
    uint32_t exponent = (data >> 10) & 0x1F;
    uint32_t mantissa = (data & 0x3FF);
    
    uint32_t f32data;

    if (exponent == 0x1f) {
        f32data = 0xff;
    } else if (!exponent) {
        f32data = 0;
    } else {
        f32data = exponent + 112;
    }

    if (!exponent && mantissa) {
        uint8_t offset = 0;
        do {
            ++offset;
            mantissa <<= 1;
        } while (!(mantissa & 0x400));
        mantissa &= 0x3ff;
        f32data = 113 - offset;
    }

    f32data <<= 23;
    f32data |= static_cast<uint32_t>(data & 0x8000) << 16;
    f32data |= (mantissa << 13);

    return std::bit_cast<float>(f32data);
#else
    return _cvtsh_ss(data);
#endif

}

// Helper function for compile-time validation
template <std::integral T>
constexpr bool is_valid_for_float16(T val) noexcept {
    return val >= -65504 && val <= 65504;
}

template <std::integral T>
inline constexpr float16::float16(T val) noexcept
{
    // float16 finite range: approximately -65504 to +65504
    constexpr int64_t max_finite_value = 65504;
    constexpr int64_t min_finite_value = -65504;
    
    if (std::is_constant_evaluated()) {
        // Compile-time: strict validation
        if (val > max_finite_value) {
            // In C++23 constexpr context, this will cause compilation failure
            std::unreachable(); 
        }
        if (val < min_finite_value) {
            // In C++23 constexpr context, this will cause compilation failure
            std::unreachable();
        }

        // Compile-time float16 conversion for integral values
        if (val == 0) {
            data = 0;
        } else {
            bool negative = val < 0;
            uint64_t abs_val = negative ? static_cast<uint64_t>(-static_cast<int64_t>(val)) : static_cast<uint64_t>(val);
            
            // Count leading zeros to find MSB position
            int leading_zeros = std::countl_zero(abs_val);
            int msb_pos = 63 - leading_zeros;
            
            if (msb_pos <= 10) {
                // Value fits directly in mantissa
                data = static_cast<uint16_t>(abs_val);
            } else {
                // Need exponent
                int exponent = msb_pos - 10;
                if (exponent >= 31) {
                    // Overflow - but this should not happen due to range check above
                    data = 0x7C00; // infinity
                } else {
                    // Extract mantissa (top 10 bits after MSB)
                    uint16_t mantissa = static_cast<uint16_t>((abs_val >> (msb_pos - 10)) & 0x3FF);
                    uint16_t exp_field = static_cast<uint16_t>(exponent + 15);
                    data = (exp_field << 10) | mantissa;
                }
            }
            
            if (negative) {
                data |= 0x8000; // Set sign bit
            }
        }
    } else {
        // Runtime behavior with overflow handling
        if (val > max_finite_value) {
            data = 0x7C00; // +infinity
        } else if (val < min_finite_value) {
            data = 0xFC00; // -infinity
        } else {
            // Use existing float constructor for runtime
            *this = float16(static_cast<float>(val));
        }
    }
}

template <std::integral T>
inline bool operator== (numetron::float16 const& l, T const& r) noexcept
{
    if constexpr (std::is_signed_v<T>) {
        constexpr int64_t max_finite_value = 65504;
        constexpr int64_t min_finite_value = -65504;
        if (r > max_finite_value || r < min_finite_value) {
            return false; // Integer is outside float16 range
        }
    } else {
        constexpr uint64_t max_finite_value = 65504;
        if (r > max_finite_value) {
            return false; // Integer is outside float16 range
        } else if (l.data & 0x8000) {
            return false; // negative float16 cannot equal unsigned integer
        }
    }

    // Handle special float16 values
    uint16_t l_exp = (l.data >> 10) & 0x1F;
    if (l_exp == 0x1F) { // NaN or infinity => not equal to any integer
        return false;
    }

    // For finite values, convert to float for comparison
    // This handles both normal and subnormal float16 values correctly
    float l_as_float = static_cast<float>(l);
    
    // Compare as floats
    return l_as_float == static_cast<float>(r);
}

template <std::floating_point T>
inline bool operator== (numetron::float16 const& l, T const& r) noexcept
{
    // Handle NaN cases first - NaN should not equal anything, including itself
    if (std::isnan(r)) {
        return false; // NaN is never equal to anything
    }
    
    // Check if float16 is NaN or infinity
    uint16_t l_exp = (l.data >> 10) & 0x1F;
    if (l_exp == 0x1F) { // NaN or infinity
        uint16_t l_mant = l.data & 0x3FF;
        if (l_mant) return false; // NaN is never equal to anything
        // Infinity case
        if (std::isfinite(r)) return false;
    }

    return l == float16{ r };
}

template <std::integral T>
inline std::partial_ordering operator <=> (numetron::float16 const& l, T const& r) noexcept
{
    // Handle special float16 values first
    uint16_t l_exp = (l.data >> 10) & 0x1F;
    uint16_t l_mant = l.data & 0x3FF;
    
    // Check for NaN - NaN comparisons are always unordered
    bool l_is_nan = (l_exp == 0x1F) && (l_mant != 0);
    if (l_is_nan) {
        return std::partial_ordering::unordered;
    }
    
    // Check for infinity
    bool l_is_inf = (l_exp == 0x1F) && (l_mant == 0);
    if (l_is_inf) {
        bool l_is_negative = (l.data & 0x8000) != 0;
        // +infinity is greater than any finite integer
        // -infinity is less than any finite integer
        return l_is_negative ? std::partial_ordering::less : std::partial_ordering::greater;
    }
    
    // Check if integer is outside float16 representable range
    if constexpr (std::is_signed_v<T>) {
        constexpr int64_t max_finite_value = 65504;
        constexpr int64_t min_finite_value = -65504;
        
        if (r > max_finite_value) {
            // Integer is larger than any finite float16
            return std::partial_ordering::less;
        }
        if (r < min_finite_value) {
            // Integer is smaller than any finite float16
            return std::partial_ordering::greater;
        }
    } else {
        constexpr uint64_t max_finite_value = 65504;
        if (r > max_finite_value) {
            // Unsigned integer is larger than any finite float16
            return std::partial_ordering::less;
        }
        // For unsigned integers, we know r >= 0, so no need to check lower bound
        // against negative float16 values - we'll handle that in the general case
    }
    
    // For finite values, convert float16 to the comparison type and compare
    // This handles both normal and subnormal float16 values correctly
    if constexpr (sizeof(T) <= sizeof(long long)) {
        // For smaller integer types, convert to double for precise comparison
        double l_as_double = static_cast<double>(static_cast<float>(l));
        double r_as_double = static_cast<double>(r);
        
        if (l_as_double < r_as_double) {
            return std::partial_ordering::less;
        } else if (l_as_double > r_as_double) {
            return std::partial_ordering::greater;
        } else {
            return std::partial_ordering::equivalent;
        }
    } else {
        // For very large integer types, be more careful about precision
        // Convert both to long double for maximum precision
        long double l_as_ld = static_cast<long double>(static_cast<float>(l));
        long double r_as_ld = static_cast<long double>(r);
        
        if (l_as_ld < r_as_ld) {
            return std::partial_ordering::less;
        } else if (l_as_ld > r_as_ld) {
            return std::partial_ordering::greater;
        } else {
            return std::partial_ordering::equivalent;
        }
    }
}

template <std::floating_point T>
inline std::partial_ordering operator <=> (numetron::float16 const& l, T const& r) noexcept
{
    // Handle NaN cases first - any comparison with NaN is unordered
    if (std::isnan(r)) {
        return std::partial_ordering::unordered;
    }
    
    // Check if float16 is NaN
    uint16_t l_exp = (l.data >> 10) & 0x1F;
    uint16_t l_mant = l.data & 0x3FF;
    bool l_is_nan = (l_exp == 0x1F) && (l_mant != 0);
    if (l_is_nan) {
        return std::partial_ordering::unordered;
    }
    
    // Handle infinity cases
    bool l_is_inf = (l_exp == 0x1F) && (l_mant == 0);
    if (l_is_inf) {
        bool l_is_negative = (l.data & 0x8000) != 0;
        
        if (std::isinf(r)) {
            // Both are infinity - compare signs
            bool r_is_negative = r < 0;
            if (l_is_negative == r_is_negative) {
                return std::partial_ordering::equivalent;
            }
            return l_is_negative ? std::partial_ordering::less : std::partial_ordering::greater;
        } else {
            // float16 is infinity, r is finite
            return l_is_negative ? std::partial_ordering::less : std::partial_ordering::greater;
        }
    } else if (std::isinf(r)) {
        // r is infinity, float16 is finite
        return r > 0 ? std::partial_ordering::less : std::partial_ordering::greater;
    }

    return l <=> float16{ r };
    //// Both are finite - convert float16 to the same type as r for comparison
    //T l_as_T = static_cast<T>(l);
    //
    //// Use standard floating-point comparison
    //if (l_as_T < r) {
    //    return std::partial_ordering::less;
    //} else if (l_as_T > r) {
    //    return std::partial_ordering::greater;
    //} else {
    //    return std::partial_ordering::equivalent;
    //}
}

} // namespace numetron

namespace std {

template <>
struct hash<numetron::float16>
{
    inline std::size_t operator()(numetron::float16 const& obj) const noexcept
    {
        return hash_value(obj);
    }
};

}

template <typename T>
inline numetron::float16 float16_cast(T f) noexcept
{
    return numetron::float16{ static_cast<float>(f) };
}
