# Repository-pinned spdlog headers, including their bundled fmt implementation.
# Consumers opt in explicitly; no compiled library or runtime DLL is introduced.
add_library(yk_spdlog_headers INTERFACE)
add_library(yk::spdlog ALIAS yk_spdlog_headers)
target_include_directories(yk_spdlog_headers SYSTEM INTERFACE "${CMAKE_CURRENT_SOURCE_DIR}/include")
