#pragma once
#include <cstdint>
#include <limits>

namespace YK {

enum class EPwlsRegularizerSpec : int32_t {
    None,
    Quadratic,
    Huber,
};

struct SPwlsAlgoParams {
    EPwlsRegularizerSpec regularizer = EPwlsRegularizerSpec::Quadratic;
    float regularization = 1e-3f;
    float huber_delta = 3e-3f;
    float epsilon = 1e-6f;
    float lower_bound = 0.f;
    float upper_bound = std::numeric_limits<float>::max();
    // 投影统计权重 W 默认全 1；实际数组由执行请求提供，配置只声明
    // 是否消费该数组，避免在算法描述中保存有生命周期的设备指针。
    bool use_projection_weights = false;
};

} // namespace YK
