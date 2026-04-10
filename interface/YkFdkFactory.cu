#include <cstdio>
#include "../FDK/YkFdkReconstructorHandle.hpp"
#include "../interface/YkFdkFactory.hpp"
#include "IReconstructor.hpp"

namespace YK {

    IReconstructor* FdkFactory::create()
    {
        std::printf("[FdkFactory] create reconstructor\n");
        return new FdkReconstructorHandle();
    }

    void FdkFactory::destroy(IReconstructor* p)
    {
        std::printf("[FdkFactory] destroy reconstructor %p\n", (void*)p);
        delete p;
    }

} // namespace YK