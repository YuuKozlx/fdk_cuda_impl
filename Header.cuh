#include <matx.h>
using namespace matx;

class MatXCommonConv {
public:
    /**
     * 通用 1D 频域卷积 (针对批量数据优化)
     * @param d_input      输入信号 [Batch, N]
     * @param d_kernel     卷积核 (空域) [K]
     * @param d_output     输出信号 [Batch, N]
     */
    static void convolve1D(float* d_input, int N, int Batch,
        float* d_kernel, int K,
        float* d_output) {

        // 1. 计算线性卷积所需的最小长度
        int minL = N + K - 1;
        int paddedL = 1;
        while (paddedL < minL) paddedL <<= 1;

        // 2. 创建视图
        auto input_view = make_tensor<float>(d_input, { Batch, N });
        auto output_view = make_tensor<float>(d_output, { Batch, N });
        auto kernel_view = make_tensor<float>(d_kernel, { K });

        // 3. 准备临时 Padding 空间 (MatX 会在作用域结束后释放或使用缓存)
        auto padded_input = make_tensor<float>({ Batch, paddedL });
        auto padded_kernel = make_tensor<float>({ paddedL });

        // 4. 执行 Padding
        (padded_input = 0.0f).run();
        (padded_kernel = 0.0f).run();

        // 将数据靠左放置 (Zero-padding)
        (padded_input.Slice({ 0, 0 }, { Batch, N }) = input_view).run();
        (padded_kernel.Slice({ 0 }, { K }) = kernel_view).run();

        // 5. 频域点乘 (卷积定理: FFT(x * h) = FFT(x) . FFT(h))
        // MatX 自动处理 R2C 和 1D 广播
        auto fft_res = fft(padded_input) * fft(padded_kernel);

        // 6. 逆变换并截取回原始长度 'same'
        auto full_conv = ifft(fft_res);

        // 注意：FFT 卷积后需要除以 paddedL 进行归一化
        // 截取前 N 个点返回
        (output_view = full_conv.Slice({ 0, 0 }, { Batch, N }) / static_cast<float>(paddedL)).run();
    }
};