#include <iostream>
#include <vector>

#include "global/YkMem3d.hpp"
#include "global/YkDataObj.hpp"

using namespace YK;
using namespace YK::Mem;
using namespace YK::DataObj;

////////////////////////////////////////////////////////////
/// GPU kernel
////////////////////////////////////////////////////////////
namespace YKTest {



    ////////////////////////////////////////////////////////////
    /// Integration test with wrap
    ////////////////////////////////////////////////////////////

    void test_mem_data_integration_wrap()
    {
        std::cout << "===== Integration Test with CPU Wrap =====\n";

        MemoryController mem;
        DataController data(mem);

        const int NX = 8;
        const int NY = 8;
        const int NZ = 8;

        ////////////////////////////////////////////////////////////
        // create CPU volume
        ////////////////////////////////////////////////////////////

        auto cpuVol = data.createCpuVolume<float>(NX, NY, NZ);
        auto cpuView = cpuVol.view();

        // 初始化数据
        for (int z = 0; z < NZ; z++)
            for (int y = 0; y < NY; y++)
                for (int x = 0; x < NX; x++)
                    cpuView(x, y, z) = float(x + y + z);

        std::cout << "CPU init sample: " << cpuView(1, 1, 1) << "\n";

        ////////////////////////////////////////////////////////////
        // wrap external CPU array
        ////////////////////////////////////////////////////////////

        std::vector<float> externalData(size_t(NX) * NY * NZ);
        for (int i = 0; i < externalData.size(); ++i)
            externalData[i] = float(i); // 任意初始化

        auto cpuVolWrap = data.wrapCpuVolume<float>(externalData.data(), NX, NY, NZ);
        auto wrapView = cpuVolWrap.view();

        std::cout << "Wrap sample before copy: " << wrapView(1, 1, 1) << "\n";

        ////////////////////////////////////////////////////////////
        // copy wrapped CPU → GPU
        ////////////////////////////////////////////////////////////
        int device = 0;
        cudaStream_t stream;

        auto gpuVol = data.createGpuVolume<float>(NX, NY, NZ, device);

        mem.upload3D(
            gpuVol.buffer(),
            cpuVolWrap.buffer()  // 使用 wrap 数据上传
        );

        ////////////////////////////////////////////////////////////
        // GPU → wrapped CPU copy
        ////////////////////////////////////////////////////////////

        mem.download3D(
            cpuVolWrap.buffer(),
            gpuVol.buffer()
        );

        ////////////////////////////////////////////////////////////
        // verify wrap CPU
        ////////////////////////////////////////////////////////////

        bool ok = true;
        for (int z = 0; z < NZ; z++)
            for (int y = 0; y < NY; y++)
                for (int x = 0; x < NX; x++)
                {

                    if (wrapView(x, y, z) != 0.f)
                    {
                        ok = false;
                        break;
                    }
                }

        if (ok)
            std::cout << "Wrap TEST PASS\n";
        else
            std::cout << "Wrap TEST FAIL\n";

        std::cout << "Wrap CPU result sample: " << wrapView(1, 1, 1) << "\n";

        std::cout << "===== Done =====\n";
    }

}; // namespace YKTest




namespace YKTest {

    void test_cpu_wrap_copy()
    {
        std::cout << "===== CPU Wrap + Copy Test =====\n";

        MemoryController mem;
        DataController data(mem);

        const int NX = 4;
        const int NY = 4;
        const int NZ = 4;

        // 外部数据
        std::vector<float> externalData(size_t(NX) * NY * NZ);
        for (int i = 0; i < externalData.size(); ++i)
            externalData[i] = float(i);

        // wrap 外部数据
        auto cpuVolWrap = data.wrapCpuVolume<float>(externalData.data(), NX, NY, NZ);
        auto wrapView = cpuVolWrap.view();

        std::cout << "Initial wrap sample: " << wrapView(1, 1, 1) << "\n";

        // 创建常规 CPU volume
        auto cpuVol = data.createCpuVolume<float>(NX, NY, NZ);
        auto cpuView = cpuVol.view();

        // copy wrap -> cpuVol
        for (int z = 0; z < NZ; ++z)
            for (int y = 0; y < NY; ++y)
                for (int x = 0; x < NX; ++x)
                    cpuView(x, y, z) = wrapView(x, y, z);

        // 验证复制
        bool ok = true;
        for (int z = 0; z < NZ; ++z)
            for (int y = 0; y < NY; ++y)
                for (int x = 0; x < NX; ++x)
                    if (cpuView(x, y, z) != wrapView(x, y, z))
                        ok = false;

        std::cout << "Copy wrap->cpuVol test: " << (ok ? "PASS" : "FAIL") << "\n";

        // 修改 cpuVol
        cpuView(1, 1, 1) = 999.0f;

        // wrap 数据不受影响
        std::cout << "After modifying cpuVol, wrap sample: " << wrapView(1, 1, 1) << "\n";
        std::cout << "cpuVol sample: " << cpuView(1, 1, 1) << "\n";

        if (wrapView(1, 1, 1) != 999.0f)
            std::cout << "Wrap memory correctly unaffected by cpuVol modifications.\n";
        else
            std::cout << "ERROR: Wrap memory was modified!\n";

        std::cout << "===== Done =====\n";
    }

} // namespace YKTest