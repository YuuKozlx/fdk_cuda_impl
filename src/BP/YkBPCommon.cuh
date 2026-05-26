#pragma once

namespace YK {
    namespace Bp {
        enum class BpStepSuperSample {
            x1 = 1,   // 步长 1.0，原始
            x2 = 2,   // 步长 0.5
            x4 = 4,   // 步长 0.25
        };

        //enum class FpDetSuperSample {
        //    x1 = 1,   // 无探测器超采
        //    x2 = 2,   // 2x2 子射线
        //    x4 = 4,   // 4x4 子射线
        //};
    }

}

