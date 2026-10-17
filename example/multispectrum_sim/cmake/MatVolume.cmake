# MAT import belongs to the application; daily builds consume the installed SDK.
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_LIST_DIR}")
find_package(matio 1.5.28 REQUIRED)
add_library(multispectrum_mat_volume STATIC src/io/MatLabelVolume.cpp)
target_include_directories(multispectrum_mat_volume PUBLIC include)
target_link_libraries(multispectrum_mat_volume PRIVATE matio::matio)
if(MSVC)
    target_compile_options(multispectrum_mat_volume PRIVATE /utf-8)
endif()
