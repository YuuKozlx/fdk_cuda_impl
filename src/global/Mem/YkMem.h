#pragma once

// ============================================================
// YkMem.h — 统一入口
// ============================================================
//
// 包含顺序（依赖关系）：
//   YkMemTypes.h      → Shape3D, Region3D, OwnerTag, CpuView3D, DeviceView3D
//   YkBuffer.h        → 所有 Buffer 类
//   YkMemController.h → MemoryController, PodDataController
//
// 一般只需 #include "YkMem.h"
// 若只用 Buffer 类型（如在 kernel launcher 头文件中），可单独 #include "YkBuffer.h"
// ============================================================

#include "YkMemTypes.h"
#include "YkBuffer.h"
#include "YkMemController.h"
