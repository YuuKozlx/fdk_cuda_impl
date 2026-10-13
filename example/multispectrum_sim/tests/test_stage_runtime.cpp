#include "pipeline/Stage.hpp"
#include <iostream>
#include <vector>
using namespace yk::spectral::pipeline;
enum class Domain { Signal,
    Log };
struct Data {
    Domain domain = Domain::Signal;
    Location location = Location::Host;
    int value = 0;
};
void require(bool b, const char* m)
{
    if (!b)
        throw std::runtime_error(m);
}
template <class F>
void reject(F f)
{
    bool bad = false;
    try {
        f();
    } catch (const std::exception&) {
        bad = true;
    }
    require(bad, "invalid runtime state accepted");
}
struct ResetProbe {
    std::vector<int>& trace;
    int id;
    void reset() noexcept { trace.push_back(id); }
};
int main()
{
    try {
        Data data[2];
        int transfers = 0, synchronizes = 0, history = 0;
        auto transfer = [&](Location l) {++transfers;for(auto& d:data)d.location=l; };
        const Contract<Domain> device { Domain::Signal, Domain::Signal, Location::Device, Location::Device };
        auto process = [&](Data& d) { d.value = ++history; };
        auto sync = [&] { ++synchronizes; };
        const auto first = execute("temporal", device, data, 2, transfer, process, sync);
        const auto second = execute("temporal", device, data, 2, transfer, process, sync);
        require(transfers == 1 && first.transfers == 1 && second.transfers == 0, "resident batch retransferred");
        require(data[0].value == 3 && data[1].value == 4 && synchronizes == 2, "temporal order or synchronization changed");
        const Contract<Domain> log { Domain::Signal, Domain::Log, Location::Device, Location::Device };
        execute("convert", log, data, 2, transfer, [](Data&) { }, [] { });
        require(data[0].domain == Domain::Log && data[1].domain == Domain::Log, "declared domain transition missing");
        reject([&] { execute("wrong domain", device, data, 2, transfer, process, sync); });
        require(history == 4, "domain failure executed process");
        reject([&] { execute("empty", device, data, 0, transfer, process, sync); });
        Data bad;
        reject([&] { execute("failed transfer", device, &bad, 1, [](Location) { }, process, sync); });
        require(history == 4, "failed transfer executed process");
        reject([&] { execute("metadata mutation", Contract<Domain> { Domain::Signal, Domain::Signal }, &bad, 1, [](Location) {}, [](Data& d) { d.domain = Domain::Log; }, [] {}); });
        bool running = false;
        std::vector<int> trace;
        ResetProbe a { trace, 1 }, b { trace, 2 };
        std::vector<ResetProbe*> stages { &a, &b };
        reject([&] {RunScope scope(running);auto cleanup=onExit([&]{resetReverse(stages);});require(running,"run flag not set");RunScope nested(running); });
        require(!running && trace == std::vector<int>({ 2, 1 }), "exception cleanup order/flag broken");
        yk::spectral::TaskProgressCallback callback = [&](yk::spectral::TaskProgress p) { trace.push_back(p.completed); };
        Progress progress(callback, "test", 2);
        progress.start();
        progress.advance();
        progress.advance();
        reject([&] { progress.advance(); });
        require(trace == std::vector<int>({ 2, 1, 0, 1, 2 }), "progress changed");
        std::cout << "Shared contracts, ordered batching, residency, metadata guards, lifecycle, progress and factories passed\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}
