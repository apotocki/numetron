// Numetron — Compile-time and runtime arbitrary-precision arithmetic
// (c) 2025 Alexander Pototskiy
// Licensed under the MIT License. See LICENSE file for details.

#pragma once

#include <functional>
#include <typeindex>

namespace numetron::detail {

inline constexpr size_t hash_init_value()
{
    if constexpr (sizeof(size_t) <= 4) {
        return 2166136261U;
    } else {
        return 14695981039346656037ULL;
    }
}

template <class T>
inline void hash_combine(std::size_t& seed, const T& v) noexcept
{
    std::hash<T> hasher;
    seed ^= hasher(v) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
}

template <size_t N, typename ArgT0, typename ... ArgsT> struct forward_at_impl : forward_at_impl<N - 1, ArgsT ...>
{
    using base_type = forward_at_impl<N - 1, ArgsT ...>;
    inline auto&& operator()(ArgT0&&, ArgsT&& ... args) const { return base_type::operator()(std::forward<ArgsT>(args) ...); }
};

template <typename ArgT0, typename ... ArgsT> struct forward_at_impl<0, ArgT0, ArgsT ...>
{
    using result_type = ArgT0;
    inline auto&& operator()(ArgT0&& arg0, ArgsT&& ...) const { return std::forward<ArgT0>(arg0); }
};

template <size_t N, typename ... ArgsT>
inline auto&& forward_at(ArgsT&& ... args)
{
    return forward_at_impl<N, ArgsT...>()(std::forward<ArgsT>(args) ...);
}


struct hasher
{
    template <typename T>
    inline size_t operator()(T const& arg) const noexcept
    {
        return std::hash<T>{}(arg);
    }

    template <typename ... Ts>
    requires(sizeof...(Ts) > 1)
    inline size_t operator()(Ts const& ... vs) const noexcept
    {
        constexpr size_t seed = hash_init_value();
        return do_work(std::make_index_sequence<sizeof ...(Ts)>(), seed, vs ...);
    }

    template <typename ... Ts, size_t ... Idxs>
    inline size_t do_work(std::index_sequence<Idxs...>, size_t seed, Ts const& ... vs) const noexcept
    {
        (hash_combine(seed, (forward_at<Idxs>(vs ...))), ...);
        return seed;
    }
};

#if 0
template <typename T> struct hash
{
    inline size_t operator()(T const& v) const noexcept
    {
        return hash_value(v);
    }
};

template <std::integral T> struct hash<T> : std::hash<T> {};
template <std::floating_point T> struct hash<T> : std::hash<T> {};
template <> struct hash<std::type_index> : std::hash<std::type_index> {};
template <> struct hash<nullptr_t> { inline size_t operator()(nullptr_t) const noexcept { return 0; } };





inline constexpr size_t hash_prime_value()
{
    if constexpr (sizeof(size_t) <= 4)
    {
        return 16777619U;
    } else {
        return 1099511628211ULL;
    }
}


#endif

}
