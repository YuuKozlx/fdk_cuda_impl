cmake_minimum_required(VERSION 3.24)
project(multispectrum_plot_tools LANGUAGES CXX)

find_package(Matplot++ CONFIG REQUIRED)
find_program(GNUPLOT_EXECUTABLE gnuplot REQUIRED)

add_executable(multispectrum_plot_reconstruction
    plot_reconstruction.cpp)
target_compile_features(multispectrum_plot_reconstruction PRIVATE cxx_std_17)
target_link_libraries(multispectrum_plot_reconstruction PRIVATE Matplot++::matplot)
if(MSVC)
    target_compile_options(multispectrum_plot_reconstruction PRIVATE /utf-8)
endif()
