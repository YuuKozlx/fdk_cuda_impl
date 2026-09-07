#pragma once
#include <cuda_runtime.h>
#include <cufft.h>
#include <device_launch_parameters.h>
#include "YkFFT.hpp"  // 上面的 CudaFFTPlan
#include "../global/YkMacro.hpp"
#include "../global/YkMem3d.hpp"

namespace YK {

    namespace Filter {

        void launch_pointwise_mul(cufftComplex* data, const float* weights, int n_complex, int batch, cudaStream_t stream);



        class FrequencyFilter {
        public:
            bool init(int paddedN, int batch, cudaStream_t stream = 0) {
                release();
                paddedN_ = paddedN;
                batch_ = batch;
                stream_ = stream;

                n_complex_ = paddedN_ / 2 + 1;

                // 复用 FFT plan
                fft_.init(paddedN_, batch_, stream_);

                // 复用频域缓冲
                d_complex_.alloc(batch_ * n_complex_, 0);
                return true;
            }

            // d_padded_data: [batch * paddedN]
            void apply(float* d_padded_data, const float* d_filter_fft /*[n_complex]*/) {
                fft_.fft(d_padded_data, d_complex_.data());

                launch_pointwise_mul(d_complex_.data(), d_filter_fft, n_complex_, batch_, stream_);

                fft_.ifft(d_complex_.data(), d_padded_data);
            }

            void setStream(cudaStream_t s) {
                stream_ = s;
                fft_.setStream(s);
            }

            void release() {
                fft_.release();
                d_complex_.reset();
                paddedN_ = 0; n_complex_ = 0; batch_ = 0; stream_ = 0;
            }

            ~FrequencyFilter() { release(); }

        private:
            int paddedN_ = 0;
            int n_complex_ = 0;
            int batch_ = 0;
            cudaStream_t stream_ = 0;

            CudaFFT fft_;
            Mem::DeviceLinearBuffer<cufftComplex> d_complex_;
        };


    }


} // namespace YK
