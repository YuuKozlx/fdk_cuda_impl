#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

namespace YK::TestImage {

struct GrayPanel {
    const std::vector<float>* volume = nullptr;
    int nx = 0;
    int ny = 0;
    int nz = 0;
    int z = 0;
    float scale = 1.f;
    float window_min = 0.f;
    float window_max = 1.f;
    bool absolute_value = false;
};

inline void writeLe16(std::ofstream& output, std::uint16_t value)
{
    const char bytes[2] = {
        static_cast<char>(value & 0xffu),
        static_cast<char>((value >> 8) & 0xffu)
    };
    output.write(bytes, sizeof(bytes));
}

inline void writeLe32(std::ofstream& output, std::uint32_t value)
{
    const char bytes[4] = {
        static_cast<char>(value & 0xffu),
        static_cast<char>((value >> 8) & 0xffu),
        static_cast<char>((value >> 16) & 0xffu),
        static_cast<char>((value >> 24) & 0xffu)
    };
    output.write(bytes, sizeof(bytes));
}

// 写出无第三方依赖的 24-bit BMP。每行是一组测试，每列是一种结果；
// panel 之间的亮分隔线用于防止相邻图像在统一窗宽下视觉粘连。
inline bool writeGrayMontageBmp(const std::filesystem::path& path,
    const std::vector<GrayPanel>& panels, int panel_columns, int gap = 2,
    int pixel_scale = 1)
{
    if (panels.empty() || panel_columns <= 0 ||
        panels.size() % static_cast<size_t>(panel_columns) != 0 || pixel_scale <= 0)
        return false;
    const int panel_width = panels.front().nx;
    const int panel_height = panels.front().ny;
    if (panel_width <= 0 || panel_height <= 0) return false;
    for (const auto& panel : panels) {
        if (!panel.volume || panel.nx != panel_width || panel.ny != panel_height ||
            panel.nz <= 0 || panel.z < 0 || panel.z >= panel.nz ||
            panel.volume->size() != static_cast<size_t>(panel.nx) * panel.ny * panel.nz ||
            !(panel.window_max > panel.window_min)) return false;
    }

    const int panel_rows = static_cast<int>(panels.size()) / panel_columns;
    const int scaled_width = panel_width * pixel_scale;
    const int scaled_height = panel_height * pixel_scale;
    const int width = panel_columns * scaled_width + (panel_columns - 1) * gap;
    const int height = panel_rows * scaled_height + (panel_rows - 1) * gap;
    const int row_bytes = ((width * 3 + 3) / 4) * 4;
    std::vector<unsigned char> pixels(static_cast<size_t>(row_bytes) * height, 235u);

    for (int panel_index = 0; panel_index < static_cast<int>(panels.size());
         ++panel_index) {
        const auto& panel = panels[panel_index];
        const int panel_x = (panel_index % panel_columns) * (scaled_width + gap);
        const int panel_y = (panel_index / panel_columns) * (scaled_height + gap);
        const float inverse_window = 1.f / (panel.window_max - panel.window_min);
        for (int y = 0; y < panel_height; ++y) {
            for (int x = 0; x < panel_width; ++x) {
                const size_t source = (static_cast<size_t>(panel.z) * panel.ny + y) *
                    panel.nx + x;
                float value = (*panel.volume)[source] * panel.scale;
                if (panel.absolute_value) value = std::fabs(value);
                value = std::clamp((value - panel.window_min) * inverse_window, 0.f, 1.f);
                const unsigned char gray = static_cast<unsigned char>(value * 255.f + 0.5f);
                // BMP 按底行优先存储；三个颜色通道相等即为灰度图。
                for (int sy = 0; sy < pixel_scale; ++sy) {
                    const int destination_y = height - 1 -
                        (panel_y + y * pixel_scale + sy);
                    for (int sx = 0; sx < pixel_scale; ++sx) {
                        const size_t destination = static_cast<size_t>(destination_y) *
                            row_bytes + static_cast<size_t>(panel_x + x * pixel_scale + sx) * 3;
                        pixels[destination] = gray;
                        pixels[destination + 1] = gray;
                        pixels[destination + 2] = gray;
                    }
                }
            }
        }
    }

    std::filesystem::create_directories(path.parent_path());
    std::ofstream output(path, std::ios::binary);
    if (!output) return false;
    const std::uint32_t pixel_bytes = static_cast<std::uint32_t>(pixels.size());
    output.put('B'); output.put('M');
    writeLe32(output, 54u + pixel_bytes);
    writeLe16(output, 0); writeLe16(output, 0); writeLe32(output, 54);
    writeLe32(output, 40); writeLe32(output, static_cast<std::uint32_t>(width));
    writeLe32(output, static_cast<std::uint32_t>(height));
    writeLe16(output, 1); writeLe16(output, 24); writeLe32(output, 0);
    writeLe32(output, pixel_bytes); writeLe32(output, 2835); writeLe32(output, 2835);
    writeLe32(output, 0); writeLe32(output, 0);
    output.write(reinterpret_cast<const char*>(pixels.data()), pixels.size());
    return output.good();
}

} // namespace YK::TestImage
