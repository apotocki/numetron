// Numetron — Compile-time and runtime arbitrary-precision arithmetic
// (c) 2025 Alexander Pototskiy
// Licensed under the MIT License. See LICENSE file for details.

// impl based on https://stackoverflow.com/questions/3669833/c11-scope-exit-guard-a-good-idea

#pragma once

#include <utility>

#ifndef NUMETRON_ERROR_REPORTING
#include <iostream>
#define NUMETRON_ERROR_REPORTING(msg) \
    std::cerr << "Error: " << msg << std::endl;
#endif

namespace numetron::detail {

enum class scope_exit_type
{
    always_e, exceptional_e, noexceptional_e
};

template <typename T, scope_exit_type SV = scope_exit_type::always_e>
class scope_exit
{
public:
    scope_exit(scope_exit const&) = delete;
    scope_exit(T&& exitScope) : exitScope_(std::forward<T>(exitScope)){}
    ~scope_exit() noexcept
    {
        try {
            if constexpr (SV == scope_exit_type::always_e) {
                exitScope_(); 
            } else if constexpr (SV == scope_exit_type::exceptional_e) {
                if (std::uncaught_exceptions()) {
                    exitScope_();
                }
            } else if constexpr (SV == scope_exit_type::noexceptional_e) {
                if (!std::uncaught_exceptions()) {
                    exitScope_();
                }
            }
        } catch (std::exception const& e) {
            NUMETRON_ERROR_REPORTING("error during the scope exit: " << e.what());
        } catch (...) {
            NUMETRON_ERROR_REPORTING("error during the scope exit: unknown exception");
        }
    }

private:
    T exitScope_;
};          

template <typename T>
scope_exit<T> scope_exit_create(T&& exitScope)
{
    return scope_exit<T>(std::forward<T>(exitScope));
}

template <typename T>
scope_exit<T, scope_exit_type::exceptional_e> scope_exception_exit_create(T&& exitScope)
{
    return scope_exit<T, scope_exit_type::exceptional_e>(std::forward<T>(exitScope));
}

template <typename T>
scope_exit<T, scope_exit_type::noexceptional_e> scope_noexcept_exit_create(T&& exitScope)
{
    return scope_exit<T, scope_exit_type::noexceptional_e>(std::forward<T>(exitScope));
}

}

#define NUMETRON_EXIT_SCOPE_LINENAME_CAT(name, line) name##line
#define NUMETRON_EXIT_SCOPE_LINENAME(name, line) NUMETRON_EXIT_SCOPE_LINENAME_CAT(name, line)
#define NUMETRON_SCOPE_EXIT(...) const auto& NUMETRON_EXIT_SCOPE_LINENAME(EXIT, __LINE__) = ::numetron::detail::scope_exit_create(__VA_ARGS__)
#define NUMETRON_SCOPE_EXCEPTIONAL_EXIT(...) const auto& NUMETRON_EXIT_SCOPE_LINENAME(EXIT, __LINE__) = ::numetron::detail::scope_exception_exit_create(__VA_ARGS__)
#define NUMETRON_NOEXCEPT_EXIT(...) const auto& NUMETRON_EXIT_SCOPE_LINENAME(EXIT, __LINE__) = ::numetron::detail::scope_noexcept_exit_create(__VA_ARGS__)
#define numetron_defer const ::numetron::detail::scope_exit NUMETRON__EXIT_SCOPE_LINENAME(EXIT, __LINE__) = [&]
