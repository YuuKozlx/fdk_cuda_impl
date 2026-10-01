#include "phantom_generator/GeneratorDispatch.hpp"

#include <algorithm>
#include <stdexcept>

namespace phantom_generator {

void GeneratorRegistry::add(std::string type, Generator generator)
{
    if (type.empty() || !generator)
        throw std::invalid_argument("generator type and callback are required");
    if (!generators_.emplace(std::move(type), std::move(generator)).second)
        throw std::invalid_argument("duplicate phantom generator type");
}

bool GeneratorRegistry::contains(const std::string& type) const
{
    return generators_.find(type) != generators_.end();
}

void GeneratorRegistry::generate(const std::string& type, Phantom& phantom,
    const std::vector<std::string>& arguments) const
{
    const auto it = generators_.find(type);
    if (it == generators_.end())
        throw std::invalid_argument("unknown phantom type: " + type);
    it->second(phantom, arguments);
}

std::vector<std::string> GeneratorRegistry::types() const
{
    std::vector<std::string> result;
    result.reserve(generators_.size());
    for (const auto& item : generators_)
        result.push_back(item.first);
    std::sort(result.begin(), result.end());
    return result;
}

} // namespace phantom_generator
