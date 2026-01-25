// Numetron — Compile-time and runtime arbitrary-precision arithmetic
// (c) 2025 Alexander Pototskiy
// Licensed under the MIT License. See LICENSE file for details.

#pragma once

#include <memory>
#include <concepts>
#include <type_traits>
#include <new>

namespace std {

template<class T>
requires (std::is_trivially_copyable_v<T> /*&& std::is_implicit_lifetime_v<T> */)
inline T* start_lifetime_as(void* p) noexcept
{
    return std::launder(static_cast<T*>(std::memmove(p, p, sizeof(T))));
}

template<class T>
requires (std::is_trivially_copyable_v<T> /*&& std::is_implicit_lifetime_v<T> */)
inline T const* start_lifetime_as(void const* p) noexcept
{
    return std::launder(static_cast<T const*>(std::memmove(const_cast<void*>(p), p, sizeof(T))));
}

template<class T>
requires (std::is_trivially_copyable_v<T> /*&& std::is_implicit_lifetime_v<T> */)
inline T* start_lifetime_as_array(void* p, size_t n) noexcept
{
    return std::launder(static_cast<T*>(std::memmove(p, p, n * sizeof(T))));
}

template<class T>
requires (std::is_trivially_copyable_v<T> /*&& std::is_implicit_lifetime_v<T> */)
inline T const* start_lifetime_as_array(void const* p, size_t n) noexcept
{
    return std::launder(static_cast<T const*>(std::memmove(const_cast<void*>(p), p, n * sizeof(T))));
}

}
