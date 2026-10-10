#pragma once
#include <cmath>
#include <stdexcept>
#include <string>
namespace primary_test {
inline void require(bool condition, const std::string& name) {
    if (!condition) throw std::runtime_error(name);
}
inline void near(double actual, double expected, double tolerance, const std::string& name) {
    if (!std::isfinite(actual) || std::abs(actual - expected) > tolerance)
        throw std::runtime_error(name + ": actual=" + std::to_string(actual) +
            ", expected=" + std::to_string(expected) + ", tolerance=" + std::to_string(tolerance));
}
template<class Function> void rejects(Function function, const std::string& name) {
    bool rejected = false;
    try { function(); } catch (const std::exception&) { rejected = true; }
    require(rejected, name + " did not reject invalid input");
}
}
