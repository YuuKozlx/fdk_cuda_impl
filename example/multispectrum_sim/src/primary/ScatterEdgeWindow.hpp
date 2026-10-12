#pragma once
#include "FasksRuntimeModel.hpp"
#include <algorithm>
#include <cmath>
#include <limits>
#include <vector>
namespace yk::spectral::scatter_detail {
inline std::vector<double> edt1d(const std::vector<double>& input, double spacing) {
    const int n = static_cast<int>(input.size());
    std::vector<int> sites;
    for (int i = 0; i < n; ++i)
        if (std::isfinite(input[i])) sites.push_back(i);
    std::vector<double> output(n, std::numeric_limits<double>::infinity());
    if (sites.empty()) return output;
    std::vector<int> v(sites.size());
    std::vector<double> z(sites.size() + 1);
    int k = 0;
    v[0] = sites[0];
    z[0] = -std::numeric_limits<double>::infinity();
    z[1] = std::numeric_limits<double>::infinity();
    const double a = spacing * spacing;
    for (size_t site_index = 1; site_index < sites.size(); ++site_index) {
        const int q = sites[site_index];
        double intersection = 0.0;
        for (;;) {
            const int p = v[k];
            intersection = ((input[q] + a * q * q) -
                            (input[p] + a * p * p)) /
                           (2.0 * a * (q - p));
            if (intersection > z[k]) break;
            --k; // Recompute the intersection against the new envelope site.
        }
        ++k;
        v[k] = q;
        z[k] = intersection;
        z[k + 1] = std::numeric_limits<double>::infinity();
    }
    k = 0;
    for (int q = 0; q < n; ++q) {
        while (z[k + 1] < q) ++k;
        const double delta = q - v[k];
        output[q] = input[v[k]] + a * delta * delta;
    }
    return output;
}

inline std::vector<float> edgeWindow(const std::vector<unsigned char>& mask, int width,
                              int height, double pixel_u, double pixel_v,
                              const FasksRuntimeModel& model) {
    std::vector<double> distance(static_cast<size_t>(width) * height,
                                 std::numeric_limits<double>::infinity());
    for (size_t i = 0; i < distance.size(); ++i)
        if (!mask[i]) distance[i] = 0.0;
    for (int y = 0; y < height; ++y) {
        std::vector<double> row(width);
        for (int x = 0; x < width; ++x) row[x] = distance[static_cast<size_t>(y) * width + x];
        row = edt1d(row, pixel_u);
        for (int x = 0; x < width; ++x) distance[static_cast<size_t>(y) * width + x] = row[x];
    }
    for (int x = 0; x < width; ++x) {
        std::vector<double> column(height);
        for (int y = 0; y < height; ++y) column[y] = distance[static_cast<size_t>(y) * width + x];
        column = edt1d(column, pixel_v);
        for (int y = 0; y < height; ++y) distance[static_cast<size_t>(y) * width + x] = column[y];
    }
    std::vector<float> result(distance.size(), 0.0f);
    for (size_t i = 0; i < result.size(); ++i) {
        if (!mask[i]) continue;
        const double normalized = std::min(
            distance[i] / model.edgeRampMm(), 1.0);
        result[i] = static_cast<float>(model.edgeBoundaryScale()
                                       + (1.0 - model.edgeBoundaryScale()) * normalized);
    }
    return result;
}

}
