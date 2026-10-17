cmake_minimum_required(VERSION 3.24)

project(CbctCalibrationCpp VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

option(CBCT_CALIBRATION_BUILD_TESTS "Build calibration regression tests" ON)
option(CBCT_CALIBRATION_BUILD_REAL_DATA_TEST "Build the repository RAW regression" OFF)

if(NOT TARGET Eigen3::Eigen)
    find_package(Eigen3 CONFIG QUIET)
endif()
if(NOT TARGET Eigen3::Eigen)
    include(FetchContent)
    FetchContent_Declare(calibration_eigen
        URL https://gitlab.com/libeigen/eigen/-/archive/3.4.0/eigen-3.4.0.tar.gz
        URL_HASH SHA256=8586084f71f9bde545ee7fa6d00288b264a2b7ac3607b974e54d13e7162c1c72
        DOWNLOAD_EXTRACT_TIMESTAMP TRUE)
    FetchContent_GetProperties(calibration_eigen)
    if(NOT calibration_eigen_POPULATED)
        FetchContent_Populate(calibration_eigen)
    endif()
    add_library(calibration_eigen_headers INTERFACE)
    target_include_directories(calibration_eigen_headers SYSTEM INTERFACE "${calibration_eigen_SOURCE_DIR}")
    add_library(Eigen3::Eigen ALIAS calibration_eigen_headers)
endif()

add_library(cbct_calibration STATIC
    src/dlt.cpp
    src/io.cpp
    src/optimizer.cpp
    src/phantom.cpp
    src/pipeline.cpp
    src/source_circle.cpp
    src/spiral.cpp
    src/single_row_fit.cpp
    src/single_row.cpp
    src/tracker.cpp
)
add_library(cbct_calibration::cbct_calibration ALIAS cbct_calibration)

target_include_directories(cbct_calibration PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>)
target_link_libraries(cbct_calibration PUBLIC Eigen3::Eigen)

if(MSVC)
    target_compile_options(cbct_calibration PRIVATE /W4 /permissive- /utf-8)
else()
    target_compile_options(cbct_calibration PRIVATE -Wall -Wextra -Wpedantic)
endif()

if(CBCT_CALIBRATION_BUILD_REAL_DATA_TEST)
    add_executable(cbct_calibration_real_data_test tests/test_real_data.cpp)
    target_link_libraries(cbct_calibration_real_data_test PRIVATE cbct_calibration)
endif()

if(CBCT_CALIBRATION_BUILD_TESTS)
    enable_testing()
    add_executable(cbct_calibration_tests tests/test_pipeline.cpp)
    target_link_libraries(cbct_calibration_tests PRIVATE cbct_calibration)
    add_test(NAME cbct_calibration_tests COMMAND cbct_calibration_tests)
    add_executable(cbct_calibration_phantom_tests tests/test_phantom_families.cpp)
    target_link_libraries(cbct_calibration_phantom_tests PRIVATE cbct_calibration)
    add_test(NAME cbct_calibration_phantom_tests COMMAND cbct_calibration_phantom_tests)
endif()
