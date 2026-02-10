#pragma once
#include <cstddef>
#include <cuda_runtime.h>
#include <driver_types.h>
#include <string>

namespace YK {
    namespace IO {
        class DumpManager {
        public:
            DumpManager() = default;
            ~DumpManager();

            DumpManager(const DumpManager&) = delete;
            DumpManager& operator=(const DumpManager&) = delete;

            DumpManager(DumpManager&& o) noexcept { move_from(o); }
            DumpManager& operator=(DumpManager&& o) noexcept {
                if (this != &o) { release(); move_from(o); }
                return *this;
            }

            // output_dir: e.g. "./dbg/"
            // enabled: false -> all dump calls become no-op
            bool init(std::string output_dir, bool enabled = true);

            void setEnabled(bool on) { enabled_ = on; }
            bool enabled() const { return enabled_; }

            // Ensure internal pinned host buffer size >= bytes
            bool reserveBytes(size_t bytes);

            // Dump a contiguous float32 device buffer to: <output_dir>/<prefix>_aXXXX.raw
            // count_f32: number of float elements
            // stream: the same stream that produced d_src (will sync before writing)
            bool dumpDeviceF32_A(
                const char* prefix,
                int a,
                const float* d_src,
                size_t count_f32,
                cudaStream_t stream);

            // Optional: dump a small text file: <output_dir>/<prefix>_aXXXX.txt
            bool dumpText_A(
                const char* prefix,
                int a,
                const std::string& text);

            // Utility: join path safely (very simple)
            static std::string joinPath(const std::string& dir, const std::string& name);

        private:
            void release();
            void move_from(DumpManager& o) noexcept;

            std::string out_dir_;
            bool enabled_ = false;

            void* h_pinned_ = nullptr;
            size_t pinned_bytes_ = 0;
        };
    };

  

} // namespace YK::IO
