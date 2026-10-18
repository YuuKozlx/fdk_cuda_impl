cmake_minimum_required(VERSION 3.20)
project(spek_spectrum VERSION 0.1.0 LANGUAGES CXX)

add_library(spek_spectrum STATIC
  src/spectrum.cpp
  src/spek_tables_embedded.cpp)
add_library(spek_spectrum::core ALIAS spek_spectrum)
target_compile_features(spek_spectrum PUBLIC cxx_std_17)
target_include_directories(spek_spectrum PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>)
set_target_properties(spek_spectrum PROPERTIES
  POSITION_INDEPENDENT_CODE ON
  EXPORT_NAME core)

if(MSVC)
  target_compile_options(spek_spectrum PRIVATE /W4 /utf-8)
else()
  target_compile_options(spek_spectrum PRIVATE -Wall -Wextra -Wpedantic)
endif()

option(SPEK_SPECTRUM_BUILD_TESTS "Build spek_spectrum tests" ${PROJECT_IS_TOP_LEVEL})
if(SPEK_SPECTRUM_BUILD_TESTS)
  enable_testing()
  add_executable(test_spek_spectrum tests/test_spectrum.cpp)
  target_link_libraries(test_spek_spectrum PRIVATE spek_spectrum::core)
  add_test(NAME spek_spectrum COMMAND test_spek_spectrum)
  add_executable(spek_spectrum_cli tools/spek_spectrum_cli.cpp)
  target_link_libraries(spek_spectrum_cli PRIVATE spek_spectrum::core)
endif()

include(GNUInstallDirs)
install(TARGETS spek_spectrum EXPORT SpekSpectrumTargets)
install(DIRECTORY include/ DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
install(EXPORT SpekSpectrumTargets NAMESPACE spek_spectrum::
  DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/spek_spectrum)
