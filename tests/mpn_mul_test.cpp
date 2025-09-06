// Numetron — Compile - time and runtime arbitrary - precision arithmetic
// (c) 2025 Alexander Pototskiy
// Licensed under the MIT License. See LICENSE file for details.

#include "test_common.hpp"

#include <iostream>
#include <filesystem>
#include <fstream>
#include <string>
#include <tuple>
#include <vector>

#ifdef _WIN32
#   pragma warning(disable : 4244 4146)
#endif
#include "gmp.h"

#include "numetron/basic_integer.hpp"

namespace numetron {

namespace fs = std::filesystem;

void mpn_mul_test()
{
    char const* path = std::getenv("TESTS_HOME");
    fs::path suitedir{ (path ? fs::path(path) / "testdata" : fs::path("testdata")) };
    fs::path filepath = suitedir / "test_mul_data.txt";

    if (!fs::exists(filepath)) {
        std::cout << "file not found: " << filepath.string() << "\n";
    }
    CHECK(fs::exists(filepath));

    std::ifstream file;
    file.exceptions(std::ifstream::badbit);
    file.open(filepath.string().c_str());
    using data_tpl_t = std::tuple<mpz_t, mpz_t, mpz_t>;
    std::vector<data_tpl_t> data_set;
    for (;;)
    {
        std::string u_str, v_str, r_str, emt_str;
        if (!std::getline(file, u_str) || u_str.empty()) break;
        std::getline(file, v_str);
        std::getline(file, r_str);
        std::getline(file, emt_str);
        CHECK(emt_str.empty());


        numetron::integer u{ u_str };
        numetron::integer v{ v_str };

        bool cond = (u.size() >= v.size() && u.size() < 30 && NUMETRON_USZCOND(u.size())) ||
            (v.size() > u.size() && v.size() < 30 && NUMETRON_USZCOND(v.size()));

        if (NUMETRON_NO30 || cond) {
            data_set.emplace_back();
            data_tpl_t& tpl = data_set.back();
            mpz_init_set_str(get<0>(tpl), u_str.c_str(), 10);
            mpz_init_set_str(get<1>(tpl), v_str.c_str(), 10);
            mpz_init_set_str(get<2>(tpl), r_str.c_str(), 10);
        }
    }
    std::cout << "loaded #" << data_set.size() << "\n";

    // test multiplication
        
    for (int attempt = 0; attempt < 5; ++attempt) {
        auto start = std::chrono::steady_clock::now();
        for (int test_cnt = 0; test_cnt < NUMETRON_TEST_COUNT; ++test_cnt)
        {
            for (auto & tpl : data_set)
            {
                mpz_t& u = get<0>(tpl);
                mpz_t& v = get<1>(tpl);
                mpz_t& r = get<2>(tpl);
        
                mpz_t r_calc;
                mpz_init(r_calc);
                //mpz_add(r_calc, u, v);
                mpz_mul(r_calc, u, v);
                int res = mpz_cmp(r_calc, r);
                CHECK(!res);

                mpz_clear(r_calc);
            }
        }
        auto finish = std::chrono::steady_clock::now();
        std::cout << "done in: " << std::chrono::duration_cast<std::chrono::microseconds>(finish - start).count() << " microseconds \n";
    }
    for (auto& tpl : data_set)
    {
        mpz_t& u = get<0>(tpl);
        mpz_t& v = get<1>(tpl);
        mpz_t& r = get<2>(tpl);

        mpz_clear(u);
        mpz_clear(v);
        mpz_clear(r);
    }
}

}
