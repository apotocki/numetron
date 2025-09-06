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

#include "numetron/basic_integer.hpp"

namespace numetron {

namespace fs = std::filesystem;

void mul_subtest()
{
    using integer_t = basic_integer<uint64_t, 1>;
    //integer u00{ "62771"sv };
    //integer v00{ "62771"sv };
    //integer r00 = u00 * v00;
    //CHECK_EQUAL(r00, 3940198441);

    integer_t u0{ "39402006196394479212279040092186442022523152995979330697599163027629505090517363849060896925176132585842098598510592"sv };
    integer_t v0{ "340282366920938463463086377055616499712"sv };
    integer_t r0 = u0 * v0;
    CHECK_EQUAL(r0, integer_t{ "13407807929942597099562668140431042021649674374789391703788046732797909808112131073751320842035306269075028071288656077186249683270904537386723943596949504"sv });
}

void mul_test()
{
    //mul_subtest();

    using integer_t = basic_integer<uint64_t, 1>;

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
    using data_tpl_t = std::tuple<integer_t, integer_t, integer_t>;
    std::vector<data_tpl_t> data_set;
    for (;;)
    {
        std::string u_str, v_str, r_str, emt_str;
        if (!std::getline(file, u_str) || u_str.empty()) break;
        std::getline(file, v_str);
        std::getline(file, r_str);
        std::getline(file, emt_str);
        CHECK(emt_str.empty());

        integer_t u{ u_str, 10 };
        integer_t v{ v_str, 10 };
        integer_t r{ r_str, 10 };

        bool cond = (u.size() >= v.size() && u.size() < 30 && NUMETRON_USZCOND(u.size())) ||
            (v.size() > u.size() && v.size() < 30 && NUMETRON_USZCOND(v.size()));
            
        if (NUMETRON_NO30 || cond) {
            data_set.emplace_back(std::move(u), std::move(v), std::move(r));
        }
    }
    std::cout << "loaded #" << data_set.size() << "\n";

    // test multiplication
    for (int attempt = 0; attempt < 5; ++attempt) {
        auto start = std::chrono::steady_clock::now();
        for (int test_cnt = 0; test_cnt < NUMETRON_TEST_COUNT; ++test_cnt)
        {
            for (auto const& tpl : data_set)
            {
                integer_t const& u = get<0>(tpl);
                integer_t const& v = get<1>(tpl);
                integer_t const& r = get<2>(tpl);
        
                auto r_calc = u * v;
                //CHECK(r_calc.size() >= 0);
                //CHECK_EQUAL(r_calc.size(), r.size());
                CHECK_EQUAL(r_calc, r);
            }
        }
        auto finish = std::chrono::steady_clock::now();
        std::cout << "done in: " << std::chrono::duration_cast<std::chrono::microseconds>(finish - start).count() << " microseconds \n";
    }
}

}