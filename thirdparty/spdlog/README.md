# spdlog

Repository-pinned spdlog 1.17.0 headers, relocated unchanged from `src/util/spdlog`.
Includes spdlog's bundled fmt; the separate `src/util/fmt` dependency is unrelated.
Original copyright notices are retained in every upstream header.

Consumers link the CMake interface target `yk::spdlog` privately. This supplies only
include paths; spdlog selects its normal header-only implementation. No download,
binary library, runtime DLL, or global include directory is required.

Both the parent build and the standalone multispectrum application register this
target. The installed YKCBCT public API does not expose spdlog types.
