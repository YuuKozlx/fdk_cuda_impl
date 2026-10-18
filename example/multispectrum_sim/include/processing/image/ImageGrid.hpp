#pragma once
#include <array>
namespace yk::spectral::image {
struct Grid {
    std::array<int, 3> dimensions { 1, 1, 1 };
    std::array<double, 3> spacing_mm { 1, 1, 1 }, origin_mm {};
    std::array<double, 9> direction { 1, 0, 0, 0, 1, 0, 0, 0, 1 };
};
}
