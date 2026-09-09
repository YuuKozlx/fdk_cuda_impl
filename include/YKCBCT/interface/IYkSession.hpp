#pragma once

#include "YKCBCT/interface/YkReconstructionApi.hpp"

namespace YK {
// 公共 Session 只保留一套名称；旧头文件继续转发，避免出现第二套参数模型。
using ISession = IReconstructionSession;
using SessionFactory = ReconstructionSessionFactory;
} // namespace YK
