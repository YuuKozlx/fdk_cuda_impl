
#include <fstream>
#include <sstream>
#include <iomanip>
#include <vector>

#include "../global/YkGlobals.h" // YK_CUDA_CHECK
#include "../global/YkMacro.hpp"
#include "YkIoDump.hpp"

namespace YK {
    namespace IO {
        static inline std::string ensureTrailingSlash(std::string s) {
            if (!s.empty()) {
                char c = s.back();
                if (c != '/' && c != '\\') s.push_back('/');
            }
            return s;
        }

        std::string DumpManager::joinPath(const std::string& dir, const std::string& name) {
            if (dir.empty()) return name;
            std::string d = dir;
            d = ensureTrailingSlash(d);
            return d + name;
        }

        DumpManager::~DumpManager() { release(); }

        bool DumpManager::init(std::string output_dir, bool enabled) {
            out_dir_ = std::move(output_dir);
            out_dir_ = ensureTrailingSlash(out_dir_);
            enabled_ = enabled;
            return true;
        }

        void DumpManager::release() {
            if (h_pinned_) {
                cudaFreeHost(h_pinned_);
                h_pinned_ = nullptr;
            }
            pinned_bytes_ = 0;
        }

        void DumpManager::move_from(DumpManager& o) noexcept {
            out_dir_ = std::move(o.out_dir_);
            enabled_ = o.enabled_;
            h_pinned_ = o.h_pinned_;
            pinned_bytes_ = o.pinned_bytes_;
            o.h_pinned_ = nullptr;
            o.pinned_bytes_ = 0;
        }

        bool DumpManager::reserveBytes(size_t bytes) {
            if (!enabled_) return true;
            if (bytes <= pinned_bytes_) return true;

            // realloc pinned
            if (h_pinned_) {
                cudaFreeHost(h_pinned_);
                h_pinned_ = nullptr;
                pinned_bytes_ = 0;
            }
            void* p = nullptr;
            cudaError_t err = cudaMallocHost(&p, bytes);
            if (err != cudaSuccess) {
                return false;
            }
            h_pinned_ = p;
            pinned_bytes_ = bytes;
            return true;
        }

        static inline std::string makeNameA(const char* prefix, int a, const char* ext) {
            std::ostringstream ss;
            ss << prefix << "_a" << std::setw(4) << std::setfill('0') << a << ext;
            return ss.str();
        }

        bool DumpManager::dumpDeviceF32_A(
            const char* prefix,
            int a,
            const float* d_src,
            size_t count_f32,
            cudaStream_t stream)
        {
            if (!enabled_) return true;
            if (!prefix || !d_src || count_f32 == 0) return false;

            const size_t bytes = count_f32 * sizeof(float);
            if (!reserveBytes(bytes)) return false;

            // D2H
            YK_CUDA_CHECK(cudaMemcpyAsync(h_pinned_, d_src, bytes, cudaMemcpyDeviceToHost, stream));
            YK_CUDA_CHECK(cudaStreamSynchronize(stream));

            // Write raw
            const std::string fname = joinPath(out_dir_, makeNameA(prefix, a, ".raw"));
            std::ofstream os(fname, std::ios::binary);
            if (!os) return false;
            os.write(reinterpret_cast<const char*>(h_pinned_), (std::streamsize)bytes);
            return (bool)os;
        }

        bool DumpManager::dumpText_A(const char* prefix, int a, const std::string& text) {
            if (!enabled_) return true;
            if (!prefix) return false;

            const std::string fname = joinPath(out_dir_, makeNameA(prefix, a, ".txt"));
            std::ofstream os(fname);
            if (!os) return false;
            os << text;
            return (bool)os;
        }
    };

} // namespace YK::IO
