#pragma once
#include "PluginInfo.hpp"
#include "TaskProgress.hpp"
#include <chrono>
#include <functional>
#include <map>
#include <memory>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
namespace yk::spectral::pipeline {
enum class Location { Host,
    Device };
template <class Domain>
struct Contract {
    Domain input_domain, output_domain;
    Location input_location = Location::Host, output_location = Location::Host;
};
// All processing stages share this contract; the payload and context stay typed.
// Pipeline owns domain/location transitions. Stages modify values, not state flags.
template <class Data, class Context, class Domain>
class IStage : public IPlugin {
public:
    virtual ~IStage() = default;
    virtual Contract<Domain> contract() const = 0;
    virtual void validateContext(const Context&) const { }
    virtual void prepare(const Context&) { }
    virtual void process(Data&, const Context&) = 0;
    virtual void reset() noexcept { }
};
template <class Domain>
struct SourceContract {
    Domain output_domain;
    Location output_location = Location::Host;
};
// A source generates data rather than modifying an existing payload.
// Preparation owns its input snapshot/resources; generate preserves batch ordering.
template <class Data, class Context, class Domain>
class ISource : public IPlugin {
public:
    virtual ~ISource() = default;
    virtual SourceContract<Domain> contract() const = 0;
    virtual void prepare(const Context&) = 0;
    virtual int batchCapacity() const = 0;
    virtual void generate(int first, int count, std::vector<Data>&) = 0;
    virtual void reset() noexcept = 0;
};
template <class Data, class Context, class Domain>
class IOutput : public IPlugin {
public:
    virtual ~IOutput() = default;
    virtual bool acceptsDomain(Domain) const = 0;
    virtual Location inputLocation() const { return Location::Host; }
    virtual void prepare(const Context&) { }
    virtual void write(const Data&, const Context&) = 0;
    virtual void finish() { }
    virtual void reset() noexcept { }
};
class RunScope {
    bool& running_;

public:
    explicit RunScope(bool& running)
        : running_(running)
    {
        if (running_)
            throw std::runtime_error("pipeline is already running");
        running_ = true;
    }
    ~RunScope() { running_ = false; }
    RunScope(const RunScope&) = delete;
    RunScope& operator=(const RunScope&) = delete;
};
template <class F>
class Cleanup {
    F action_;

public:
    explicit Cleanup(F f)
        : action_(std::move(f))
    {
    }
    ~Cleanup() { action_(); }
    Cleanup(const Cleanup&) = delete;
    Cleanup& operator=(const Cleanup&) = delete;
};
template <class F>
Cleanup<F> onExit(F f) { return Cleanup<F>(std::move(f)); }
template <class Stages>
void resetReverse(Stages& stages) noexcept
{
    for (auto it = stages.rbegin(); it != stages.rend(); ++it)
        (*it)->reset();
}
template <class Domain>
Domain advanceDomain(Domain current, const Contract<Domain>& c, const char* name)
{
    if (current != c.input_domain)
        throw std::runtime_error(std::string(name) + ": input domain mismatch");
    return c.output_domain;
}
using Clock = std::chrono::steady_clock;
inline double elapsedMilliseconds(Clock::time_point begin) { return std::chrono::duration<double, std::milli>(Clock::now() - begin).count(); }
struct ExecutionTiming {
    double transfer_ms = 0, process_ms = 0;
    size_t transfers = 0;
};
// Transfer is invoked once per batch only when any payload needs a location change.
// synchronize is supplied by the business pipeline; runtime never introduces CUDA waits.
// Process receives payloads in order, preserving temporal state across batches.
template <class Data, class Domain, class Transfer, class Process, class Synchronize>
ExecutionTiming execute(const char* name, const Contract<Domain>& c, Data* data, size_t count,
    Transfer transfer, Process process, Synchronize synchronize)
{
    if (!data || !count)
        throw std::invalid_argument("empty stage batch");
    for (size_t i = 0; i < count; ++i)
        advanceDomain(data[i].domain, c, name);
    ExecutionTiming timing;
    bool needs_transfer = false;
    for (size_t i = 0; i < count; ++i)
        needs_transfer = needs_transfer || data[i].location != c.input_location;
    if (needs_transfer) {
        const auto tick = Clock::now();
        transfer(c.input_location);
        timing.transfer_ms = elapsedMilliseconds(tick);
        timing.transfers = 1;
    }
    for (size_t i = 0; i < count; ++i)
        if (data[i].location != c.input_location || data[i].domain != c.input_domain)
            throw std::runtime_error(std::string(name) + ": invalid transferred state");
    const auto tick = Clock::now();
    for (size_t i = 0; i < count; ++i) {
        process(data[i]);
        if (data[i].domain != c.input_domain || data[i].location != c.input_location)
            throw std::runtime_error(std::string(name) + ": stage changed pipeline-owned state");
        data[i].domain = c.output_domain;
        data[i].location = c.output_location;
    }
    synchronize();
    timing.process_ms = elapsedMilliseconds(tick);
    return timing;
}
class Progress {
    const TaskProgressCallback& callback_;
    const char* phase_;
    int total_, completed_ = 0;

public:
    Progress(const TaskProgressCallback& cb, const char* phase, int total)
        : callback_(cb)
        , phase_(phase)
        , total_(total)
    {
        if (total < 0 || !phase)
            throw std::invalid_argument("invalid pipeline progress");
    }
    void start() const
    {
        if (callback_)
            callback_({ phase_, 0, total_ });
    }
    void advance()
    {
        if (completed_ >= total_)
            throw std::runtime_error("pipeline progress exceeds total");
        if (callback_)
            callback_({ phase_, ++completed_, total_ });
        else
            ++completed_;
    }
};
template <class Product>
class FactoryRegistry {
    static_assert(std::is_base_of<IPlugin, Product>::value, "registry products must implement IPlugin");

public:
    using Factory = std::function<std::unique_ptr<Product>()>;

private:
    struct Entry {
        PluginInfo info;
        Factory factory;
    };
    std::map<std::string, Entry> entries_;

public:
    void add(PluginInfo info, Factory factory)
    {
        validatePluginInfo(info);
        if (!factory)
            throw std::invalid_argument("empty plugin factory");
        const auto id = info.id;
        if (!entries_.emplace(id, Entry { std::move(info), std::move(factory) }).second)
            throw std::invalid_argument("duplicate plugin ID: " + id);
    }
    const PluginInfo* find(const std::string& id) const
    {
        const auto it = entries_.find(id);
        return it == entries_.end() ? nullptr : &it->second.info;
    }
    std::vector<PluginInfo> list() const
    {
        std::vector<PluginInfo> result;
        result.reserve(entries_.size());
        for (const auto& entry : entries_)
            result.push_back(entry.second.info);
        return result;
    }
    std::unique_ptr<Product> create(const std::string& id) const
    {
        const auto it = entries_.find(id);
        if (it == entries_.end())
            throw std::invalid_argument("unknown plugin: " + id);
        auto plugin = it->second.factory();
        if (!plugin)
            throw std::runtime_error("plugin factory returned null");
        if (!(plugin->info() == it->second.info))
            throw std::runtime_error("plugin factory metadata mismatch: " + id);
        return plugin;
    }
};
} // namespace yk::spectral::pipeline
