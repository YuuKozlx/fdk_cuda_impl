#pragma once

#include "YKCBCT/global/YkExport.hpp"
#include "YKCBCT/interface/YkTaskTypes.hpp"

namespace YK {

// A session owns one CUDA stream and all algorithm workspace.  It is not
// thread-safe: use one session per concurrent reconstruction/projection.
class YK_API ISession {
public:
    virtual ~ISession() = default;

    // Geometry and algorithm selection are immutable after initialization.
    virtual bool initialize(const SessionDesc& desc) = 0;
    virtual bool execute(const ExecuteRequest& request) = 0;
    virtual void reset() = 0;
    virtual void release() = 0;
    virtual bool isInitialized() const = 0;
};

// The library retains ownership of the returned object.  Destroy it only by
// this function so allocation and deletion always happen inside the DLL.
struct YK_API SessionFactory {
    static ISession* create();
    static void destroy(ISession* session);
};

} // namespace YK
