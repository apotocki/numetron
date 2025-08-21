// Numetron — Compile-time and runtime arbitrary-precision arithmetic
// (c) 2025 Alexander Pototskiy
// Licensed under the MIT License. See LICENSE file for details.

#pragma once

#include "basic_decimal.hpp"

namespace numetron {

//void numeric_round(decimal &);
//void ceil(decimal &);
//void floor(decimal &);
    /*
    void numeric_round(decimal& val)
    {
        if (auto exp = val.raw_exp(); exp < 0) {
            double tmpval = std::round(val.get<double>());
            if (exp == -1 && val < 0 && (-val.raw_value()) % 10 <= 5) {
                tmpval += 1;
            }
            val = tmpval;
        }
    }

    void ceil(decimal& val)
    {
        if (auto exp = val.raw_exp(); exp < 0) {
            val = std::ceil(val.get<double>());
        }
    }

    void floor(decimal& val)
    {
        if (auto exp = val.raw_exp(); exp < 0) {
            val = std::floor(val.get<double>());
        }
    }
    */
}
