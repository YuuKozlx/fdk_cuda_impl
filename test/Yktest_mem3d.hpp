#pragma once
#include <iostream>
#include <vector>
#include <cassert>
#include "global/YkMem3d.hpp"  // 新版头文件

namespace YKTest {

    void testGpuBuffer3D() {
        using namespace YK::Mem;

        MemoryController ctrl;

        // 获取 GPU 数量
        int deviceCount = 0;
        cudaGetDeviceCount(&deviceCount);
        if (deviceCount == 0) {
            std::cerr << "No CUDA devices found!" << std::endl;
            return;
        }

        cudaStream_t s = nullptr; // 这里不使用 stream，保持同步调用
        YK_CUDA_CHECK(cudaStreamCreate(&s)); // 创建 stream，虽然不使用，但确保 API 调用正确


        int deviceId = 0; // 测试第一个 GPU

        // 1. 分配 3D DeviceBuffer
        auto buf = ctrl.allocateDevice3D<float>(16, 8, 4, deviceId, true);
        assert(buf);
        std::cout << "Allocated DeviceBuffer3D: "
            << buf.shape().nx << "x" << buf.shape().ny << "x" << buf.shape().nz
            << " (pitchBytes=" << buf.pitch() << ")" << std::endl;

        // 2. 准备 CPU 数据
        CpuBuffer3D<float> hostBuf(16, 8, 4);
        auto hostView = hostBuf.view(); // 非 const view 用于写入

        for (int z = 0; z < hostBuf.shape().nz; ++z)
            for (int y = 0; y < hostBuf.shape().ny; ++y)
                for (int x = 0; x < hostBuf.shape().nx; ++x)
                    hostView(x, y, z) = 42.0f;

        // 3. 上传到 GPU
        ctrl.upload3D(buf, hostBuf);

        // 4. 下载回 CPU
        CpuBuffer3D<float> hostOut(16, 8, 4);
        auto outView = hostOut.view();
        ctrl.download3D(hostOut, buf);

        // 验证
        auto outCView = hostOut.cview(); // const view 用于只读
        for (int z = 0; z < hostOut.shape().nz; ++z)
            for (int y = 0; y < hostOut.shape().ny; ++y)
                for (int x = 0; x < hostOut.shape().nx; ++x)
                    assert(outCView(x, y, z) == 42.0f);

        std::cout << "Upload/download test passed." << std::endl;

        // 5. 移动构造
        DeviceBuffer3D<float> buf2(std::move(buf));
        assert(buf2);
        assert(!buf);
        std::cout << "Move constructor test passed." << std::endl;

        // 6. 移动赋值
        auto buf3 = ctrl.allocateDevice3D<float>(16, 8, 4, deviceId, s);
        buf3 = std::move(buf2);
        assert(buf3);
        assert(!buf2);
        std::cout << "Move assignment test passed." << std::endl;

        // 7. Zero buffer
        std::cout << "buf.ptr = " << buf3.data()
            << ", nx=" << buf3.shape().nx
            << ", ny=" << buf3.shape().ny
            << ", nz=" << buf3.shape().nz
            << std::endl;
        ctrl.download3D(hostOut, buf3);
        for (int z = 0; z < hostOut.shape().nz; ++z)
            for (int y = 0; y < hostOut.shape().ny; ++y)
                for (int x = 0; x < hostOut.shape().nx; ++x) {
                    assert(outCView(x, y, z) == 42.0f);
                }



        std::cout << "Zero buffer test passed." << std::endl;
        std::cout << "All tests passed!" << std::endl;
    }

    int test_gpumem3d() {
        try {
            testGpuBuffer3D();
        }
        catch (const std::exception& e) {
            std::cerr << "Exception: " << e.what() << std::endl;
            return 1;
        }
        return 0;
    }

} // namespace YKTest