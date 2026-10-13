#pragma once
#include <cmath>
#include <cstddef>
#ifdef __CUDACC__
#define IMG_HD __host__ __device__
#else
#define IMG_HD
#endif
namespace yk::spectral::image::detail {
struct Shape {
    int x, y, z;
    IMG_HD size_t count() const { return size_t(x) * y * z; }
    IMG_HD int dim(int a) const { return a == 0 ? x : a == 1 ? y
                                                             : z; }
    IMG_HD size_t stride(int a) const { return a == 0 ? 1 : a == 1 ? size_t(x)
                                                                   : size_t(x) * y; }
};
IMG_HD inline int clamp(int v, int n) { return v < 0 ? 0 : v >= n ? n - 1
                                                                  : v; }
IMG_HD inline size_t index(Shape s, int x, int y, int z) { return (size_t(clamp(z, s.z)) * s.y + clamp(y, s.y)) * s.x + clamp(x, s.x); }
// Finite, replicated boundaries. Double accumulation avoids cancellation in moments.
struct AxisFilter {
    const float* in;
    Shape s;
    int axis, radius;
    double sigma;
    bool gaussian;
    IMG_HD float operator()(size_t i) const
    {
        const size_t stride = s.stride(axis);
        const int c = int(i / stride % s.dim(axis));
        double sum = 0, weight = 0;
        for (int d = -radius; d <= radius; ++d) {
            const double q = d / sigma;
            double w = gaussian ? exp(-0.5 * q * q) : 1.;
            sum += w * in[i - size_t(c) * stride + size_t(clamp(c + d, s.dim(axis))) * stride];
            weight += w;
        }
        return float(sum / weight);
    }
};
struct ResizeAxis {
    const float* in;
    Shape src, dst;
    int axis;
    bool nearest, area;
    IMG_HD float operator()(size_t i) const
    {
        int p[3] = { int(i % dst.x), int(i / dst.x % dst.y), int(i / (size_t(dst.x) * dst.y)) };
        const int c = p[axis];
        const double scale = double(src.dim(axis)) / dst.dim(axis);
        double sum = 0;
        if (nearest) {
            p[axis] = clamp(int(floor((c + 0.5) * scale)), src.dim(axis));
            return in[index(src, p[0], p[1], p[2])];
        }
        if (area && scale > 1.) {
            const double lo = c * scale, hi = (c + 1) * scale;
            for (int j = int(floor(lo)); j < int(ceil(hi)); ++j) {
                double a = lo > j ? lo : j, b = hi < j + 1 ? hi : j + 1;
                p[axis] = j;
                sum += (b - a) * in[index(src, p[0], p[1], p[2])];
            }
            return float(sum / scale);
        }
        const double u = (c + 0.5) * scale - 0.5;
        const int j = int(floor(u));
        const double t = u - j;
        p[axis] = j;
        sum = (1 - t) * in[index(src, p[0], p[1], p[2])];
        p[axis] = j + 1;
        return float(sum + t * in[index(src, p[0], p[1], p[2])]);
    }
};
struct Bilateral {
    const float* in;
    Shape s;
    int rx, ry, rz;
    double sx, sy, sz, range;
    IMG_HD float operator()(size_t i) const
    {
        int x = int(i % s.x), y = int(i / s.x % s.y), z = int(i / (size_t(s.x) * s.y));
        double sum = 0, weights = 0;
        for (int dz = -rz; dz <= rz; ++dz)
            for (int dy = -ry; dy <= ry; ++dy)
                for (int dx = -rx; dx <= rx; ++dx) {
                    double v = in[index(s, x + dx, y + dy, z + dz)], d = (v - in[i]) / range;
                    const double qx = dx / sx, qy = dy / sy, qz = dz / sz;
                    double w = exp(-0.5 * (qx * qx + qy * qy + qz * qz + d * d));
                    sum += w * v;
                    weights += w;
                }
        return float(sum / weights);
    }
};
struct Pointwise {
    const float *x, *y, *z;
    int mode;
    double epsilon;
    IMG_HD float operator()(size_t i) const
    {
        if (mode == 4)
            return x[i] - y[0];
        if (mode == 5)
            return x[i] + y[0];
        if (mode == 0)
            return x[i] * x[i];
        if (mode == 1) {
            double v = double(x[i]) - double(y[i]) * y[i];
            v = v > 0 ? v : 0;
            return float(v / (v + epsilon));
        }
        if (mode == 2)
            return (1 - x[i]) * y[i];
        return x[i] * y[i] + z[i];
    }
};
} // namespace
#undef IMG_HD
