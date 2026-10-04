#pragma once

#include "cbct_calibration/types.hpp"

#include <fstream>

namespace cbct::calibration {

class RawStackReader {
public:
    RawStackReader(const std::filesystem::path& path, int views, ImageShape shape);
    void readFrame(int view, std::vector<float>& destination);
    int views() const noexcept { return views_; }
    ImageShape shape() const noexcept { return shape_; }

private:
    std::ifstream stream_;
    int views_;
    ImageShape shape_;
};

void writeCalibrationJson(const std::filesystem::path& path,
                          const CalibrationResult& result);

}  // namespace cbct::calibration
