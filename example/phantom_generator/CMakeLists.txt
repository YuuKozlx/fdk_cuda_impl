cmake_minimum_required(VERSION 3.24)
project(phantom_generator LANGUAGES CXX)

add_executable(phantom_generator
    src/main.cpp
    src/GeneratorDispatch.cpp
    src/PhantomOutput.cpp)
target_include_directories(phantom_generator PRIVATE include)
target_compile_features(phantom_generator PRIVATE cxx_std_17)
if(MSVC)
    target_compile_options(phantom_generator PRIVATE /utf-8)
endif()

include(CTest)
if(BUILD_TESTING)
    foreach(type IN ITEMS shepp_logan tungsten_wire tungsten_wire_slanted gold_foil water_cylinder
            water_ellipse water_cylinder_200mm catphan low_contrast
            tungsten_bead_line)
        add_test(NAME phantom_generator_${type}
            COMMAND ${CMAKE_COMMAND}
                -DGENERATOR=$<TARGET_FILE:phantom_generator>
                -DTYPE=${type}
                -DOUTPUT_DIR=${CMAKE_CURRENT_BINARY_DIR}/test-output
                -P ${CMAKE_CURRENT_SOURCE_DIR}/tests/verify_output.cmake)
    endforeach()
endif()
