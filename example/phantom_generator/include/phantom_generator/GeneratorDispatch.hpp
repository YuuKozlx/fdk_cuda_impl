#pragma once

#include "phantom_generator/PhantomModel.hpp"

#include <functional>
#include <string>
#include <unordered_map>
#include <vector>

namespace phantom_generator {

using Generator = std::function<void(Phantom&, const std::vector<std::string>&)>;

class GeneratorRegistry {
public:
    void add(std::string type, Generator generator);
    bool contains(const std::string& type) const;
    void generate(const std::string& type, Phantom& phantom,
        const std::vector<std::string>& arguments) const;
    std::vector<std::string> types() const;

private:
    std::unordered_map<std::string, Generator> generators_;
};

} // namespace phantom_generator
