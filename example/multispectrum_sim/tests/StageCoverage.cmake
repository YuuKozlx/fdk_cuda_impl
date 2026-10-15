# Shared by the full application build and the standalone stage-test project.
set(STAGE_TEST_ROOT "${CMAKE_CURRENT_LIST_DIR}/..")
add_executable(multispectrum_calibration_fits_test
    "${STAGE_TEST_ROOT}/tests/test_calibration_fits.cpp"
    "${STAGE_TEST_ROOT}/src/recon/preprocess/BeamHardeningFit.cpp"
    "${STAGE_TEST_ROOT}/src/WaterReference.cpp"
    "${STAGE_TEST_ROOT}/src/XcomAttenuationProvider.cpp"
    "${STAGE_TEST_ROOT}/src/config/SimConfig.cpp"
    "${STAGE_TEST_ROOT}/src/config/SpectrumSource.cpp")
target_include_directories(multispectrum_calibration_fits_test PRIVATE
    "${STAGE_TEST_ROOT}/include" "${STAGE_TEST_ROOT}/src"
    "${STAGE_TEST_ROOT}/../../thirdparty/tomlplusplus/include"
    "${STAGE_TEST_ROOT}/../../thirdparty/nlohmann/include")
target_link_libraries(multispectrum_calibration_fits_test PRIVATE
    multispectrum_two_material_calibration multispectrum_ebhc_fit nist_xcom_core spek_spectrum::core)
if(MSVC)
    target_compile_options(multispectrum_calibration_fits_test PRIVATE /utf-8)
endif()
foreach(case water_mono water_eid water_invalid empirical_recovery empirical_invalid material_holdout material_invalid water_matrix empirical_matrix material_matrix)
    add_test(NAME calibration_fit_${case} COMMAND multispectrum_calibration_fits_test ${case} "${CMAKE_CURRENT_BINARY_DIR}/fit-tests")
    set_tests_properties(calibration_fit_${case} PROPERTIES LABELS "cpu;unit;calibration")
endforeach()
if(TARGET multispectrum_image_processing)
    add_executable(multispectrum_image_oracles_test "${STAGE_TEST_ROOT}/tests/test_image_oracles.cpp")
    target_link_libraries(multispectrum_image_oracles_test PRIVATE multispectrum_image_processing)
    if(MSVC)
        target_compile_options(multispectrum_image_oracles_test PRIVATE /utf-8)
    endif()
    foreach(case mean gaussian bilateral guided resample invalid)
        add_test(NAME image_oracle_${case} COMMAND multispectrum_image_oracles_test ${case})
        set_tests_properties(image_oracle_${case} PROPERTIES LABELS "gpu;unit;image")
    endforeach()
endif()
if(TARGET multispectrum_device_pool)
    add_executable(multispectrum_pixel_helpers_test
        "${STAGE_TEST_ROOT}/tests/test_pixel_helpers.cpp"
        "${STAGE_TEST_ROOT}/src/recon/preprocess/CudaPixelOps.cu")
    target_include_directories(multispectrum_pixel_helpers_test PRIVATE "${STAGE_TEST_ROOT}/src")
    target_link_libraries(multispectrum_pixel_helpers_test PRIVATE multispectrum_device_pool CUDA::cudart)
    set_property(TARGET multispectrum_pixel_helpers_test PROPERTY CUDA_RUNTIME_LIBRARY Shared)
    if(MSVC)
        target_compile_options(multispectrum_pixel_helpers_test PRIVATE $<$<COMPILE_LANGUAGE:CXX>:/utf-8> $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/utf-8>)
    endif()
    foreach(case polynomial roundtrip residual mask invalid)
        add_test(NAME pixel_helper_${case} COMMAND multispectrum_pixel_helpers_test ${case})
        set_tests_properties(pixel_helper_${case} PROPERTIES LABELS "gpu;unit;preprocess")
    endforeach()
endif()
