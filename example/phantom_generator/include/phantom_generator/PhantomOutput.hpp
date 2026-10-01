#pragma once

#include "phantom_generator/PhantomModel.hpp"

#include <filesystem>

namespace phantom_generator {

class PhantomOutputWriter {
public:
    static void validate(const Phantom& phantom);
    static void writeRaw(const Phantom& phantom, const std::filesystem::path& output);
    static void writeDescription(const Phantom& phantom,
        const std::filesystem::path& output);
    static void write(const Phantom& phantom, const std::filesystem::path& output);
};

} // namespace phantom_generator
