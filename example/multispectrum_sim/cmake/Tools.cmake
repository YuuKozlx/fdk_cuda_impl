# Image and intensity processing
if(TARGET YKCBCT)
    add_executable(multispectrum_bone_bhc_singlepass EXCLUDE_FROM_ALL
        app/experiments/IdealBoneCorrection.cu src/input/config/SimConfig.cpp src/input/config/ConfigSerialization.cpp src/input/config/ConfigValidation.cpp src/input/config/SpectrumSource.cpp
        src/input/materials/WaterReference.cpp src/input/calibration/WaterBhcTable.cpp src/input/materials/XcomAttenuationProvider.cpp src/processing/support/SimLogger.cpp
        src/input/calibration/WaterBhcCalibration.cpp)
    target_link_libraries(multispectrum_bone_bhc_singlepass PRIVATE Eigen3::Eigen)
target_include_directories(multispectrum_bone_bhc_singlepass PRIVATE include src
        "${CMAKE_CURRENT_SOURCE_DIR}/../../thirdparty/tomlplusplus/include"
        "${CMAKE_CURRENT_SOURCE_DIR}/../../thirdparty/nlohmann/include")
    target_link_libraries(multispectrum_bone_bhc_singlepass PRIVATE yk::spdlog nist_xcom_core spek_spectrum::core CUDA::cudart)
    multispectrum_use_device_memory(multispectrum_bone_bhc_singlepass)
    set_property(TARGET multispectrum_bone_bhc_singlepass PROPERTY CUDA_RUNTIME_LIBRARY Shared)
    if(MSVC)
        target_compile_options(multispectrum_bone_bhc_singlepass PRIVATE
            $<$<COMPILE_LANGUAGE:CXX>:/utf-8> $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/utf-8>)
    endif()
    add_executable(multispectrum_bone_bhc_prepare EXCLUDE_FROM_ALL
        app/experiments/PrepareBoneBhc.cpp src/input/config/SimConfig.cpp src/input/config/ConfigSerialization.cpp src/input/config/ConfigValidation.cpp src/input/config/SpectrumSource.cpp
        src/input/materials/WaterReference.cpp src/input/calibration/WaterBhcTable.cpp src/input/materials/XcomAttenuationProvider.cpp
        src/recon/postprocess/PostprocessPipeline.cpp src/recon/postprocess/OutputStages.cpp src/processing/support/SimLogger.cpp)
    target_include_directories(multispectrum_bone_bhc_prepare PRIVATE include
        "${CMAKE_CURRENT_SOURCE_DIR}/../../thirdparty/tomlplusplus/include"
        "${CMAKE_CURRENT_SOURCE_DIR}/../../thirdparty/nlohmann/include")
    target_link_libraries(multispectrum_bone_bhc_prepare PRIVATE yk::spdlog multispectrum_intensity_stages nist_xcom_core spek_spectrum::core)
    multispectrum_use_image_processing(multispectrum_bone_bhc_prepare)
    if(MSVC)
        target_compile_options(multispectrum_bone_bhc_prepare PRIVATE $<$<COMPILE_LANGUAGE:CXX>:/utf-8>)
    endif()
endif()

# Bone-hardening calibration and correction
add_executable(multispectrum_fit_material_phantom EXCLUDE_FROM_ALL app/experiments/FitMaterialPhantom.cpp)
target_link_libraries(multispectrum_fit_material_phantom PRIVATE multispectrum_two_material_calibration)
target_include_directories(multispectrum_fit_material_phantom PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/../../thirdparty/nlohmann/include")
if(TARGET YKCBCT)
    add_executable(multispectrum_bone_bhc_2dcal EXCLUDE_FROM_ALL
        app/experiments/TwoMaterialBoneCorrection.cu src/input/config/SimConfig.cpp src/input/config/ConfigSerialization.cpp src/input/config/ConfigValidation.cpp src/input/config/SpectrumSource.cpp
        src/input/materials/WaterReference.cpp src/input/calibration/WaterBhcTable.cpp src/input/materials/XcomAttenuationProvider.cpp src/processing/support/SimLogger.cpp src/recon/preprocess/CudaPixelOps.cu)
    target_include_directories(multispectrum_bone_bhc_2dcal PRIVATE include src "${CMAKE_CURRENT_SOURCE_DIR}/../../thirdparty/tomlplusplus/include" "${CMAKE_CURRENT_SOURCE_DIR}/../../thirdparty/nlohmann/include")
    target_link_libraries(multispectrum_bone_bhc_2dcal PRIVATE yk::spdlog multispectrum_two_material_bhc nist_xcom_core spek_spectrum::core)
    set_property(TARGET multispectrum_bone_bhc_2dcal PROPERTY CUDA_RUNTIME_LIBRARY Shared)
endif()
if(MSVC AND TARGET multispectrum_bone_bhc_2dcal)
    target_compile_options(multispectrum_bone_bhc_2dcal PRIVATE $<$<COMPILE_LANGUAGE:CXX>:/utf-8> $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/utf-8>)
endif()
if(TARGET YKCBCT)
    add_executable(multispectrum_ebhc EXCLUDE_FROM_ALL app/experiments/EmpiricalBoneCorrection.cu
        src/input/config/SimConfig.cpp src/input/config/ConfigSerialization.cpp src/input/config/ConfigValidation.cpp src/input/config/SpectrumSource.cpp src/recon/preprocess/CudaPixelOps.cu)
    target_include_directories(multispectrum_ebhc PRIVATE include src "${CMAKE_CURRENT_SOURCE_DIR}/../../thirdparty/tomlplusplus/include" "${CMAKE_CURRENT_SOURCE_DIR}/../../thirdparty/nlohmann/include")
    target_sources(multispectrum_ebhc PRIVATE src/input/materials/XcomAttenuationProvider.cpp)
    target_link_libraries(multispectrum_ebhc PRIVATE multispectrum_empirical_bhc nist_xcom_core spek_spectrum::core)
    set_property(TARGET multispectrum_ebhc PROPERTY CUDA_RUNTIME_LIBRARY Shared)
    if(MSVC)
        target_compile_options(multispectrum_ebhc PRIVATE $<$<COMPILE_LANGUAGE:CXX>:/utf-8> $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/utf-8>)
    endif()
endif()
add_executable(multispectrum_fit_ebhc EXCLUDE_FROM_ALL app/experiments/FitEmpiricalBhc.cpp)
target_link_libraries(multispectrum_fit_ebhc PRIVATE multispectrum_ebhc_fit)
target_include_directories(multispectrum_fit_ebhc PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/../../thirdparty/nlohmann/include")
if(TARGET YKCBCT)
    add_executable(multispectrum_cone_two_pass EXCLUDE_FROM_ALL app/experiments/ConeArtifactTwoPass.cu src/input/config/SimConfig.cpp src/input/config/ConfigSerialization.cpp src/input/config/ConfigValidation.cpp src/input/config/SpectrumSource.cpp src/input/materials/XcomAttenuationProvider.cpp)
    target_include_directories(multispectrum_cone_two_pass PRIVATE include src "${CMAKE_CURRENT_SOURCE_DIR}/../../thirdparty/tomlplusplus/include" "${CMAKE_CURRENT_SOURCE_DIR}/../../thirdparty/nlohmann/include")
    target_link_libraries(multispectrum_cone_two_pass PRIVATE nist_xcom_core spek_spectrum::core)
    multispectrum_use_image_processing(multispectrum_cone_two_pass)
    set_property(TARGET multispectrum_cone_two_pass PROPERTY CUDA_RUNTIME_LIBRARY Shared)
    if(MSVC)
        target_compile_options(multispectrum_cone_two_pass PRIVATE $<$<COMPILE_LANGUAGE:CXX>:/utf-8> $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/utf-8>)
    endif()
    add_executable(multispectrum_ebhc_paper EXCLUDE_FROM_ALL app/experiments/EbhcPaper.cu src/input/config/SimConfig.cpp src/input/config/ConfigSerialization.cpp src/input/config/ConfigValidation.cpp src/input/config/SpectrumSource.cpp src/input/materials/XcomAttenuationProvider.cpp)
    target_include_directories(multispectrum_ebhc_paper PRIVATE include src "${CMAKE_CURRENT_SOURCE_DIR}/../../thirdparty/tomlplusplus/include" "${CMAKE_CURRENT_SOURCE_DIR}/../../thirdparty/nlohmann/include")
    target_link_libraries(multispectrum_ebhc_paper PRIVATE multispectrum_ebhc_paper_core nist_xcom_core spek_spectrum::core)
    set_property(TARGET multispectrum_ebhc_paper PROPERTY CUDA_RUNTIME_LIBRARY Shared)
    if(MSVC)
        target_compile_options(multispectrum_ebhc_paper PRIVATE $<$<COMPILE_LANGUAGE:CXX>:/utf-8> $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/utf-8>)
    endif()
endif()

# Built-in examples and MAT input
add_executable(multispectrum_pack_mat app/PackMatVolume.cpp src/input/materials/XcomAttenuationProvider.cpp)
target_link_libraries(multispectrum_pack_mat PRIVATE multispectrum_data_examples nist_xcom_core)
if(MSVC)
    target_compile_options(multispectrum_pack_mat PRIVATE /utf-8)
endif()
add_executable(multispectrum_icrp_examples app/PrepareIcrpExamples.cpp src/client/ProjectionUiConfig.cpp src/input/config/SimConfig.cpp src/input/config/ConfigSerialization.cpp src/input/config/ConfigValidation.cpp src/input/config/SpectrumSource.cpp src/input/materials/XcomAttenuationProvider.cpp)
target_compile_definitions(multispectrum_icrp_examples PRIVATE MULTISPECTRUM_ICRP_MATERIAL_DIRECTORY="${CMAKE_CURRENT_SOURCE_DIR}/../../data/materials/icrp145")
target_include_directories(multispectrum_icrp_examples PRIVATE include "${CMAKE_CURRENT_SOURCE_DIR}/../../thirdparty/tomlplusplus/include")
target_link_libraries(multispectrum_icrp_examples PRIVATE multispectrum_icrp nist_xcom_core spek_spectrum::core)
if(MSVC)
    target_compile_options(multispectrum_icrp_examples PRIVATE /utf-8)
endif()

# Projection benchmarks
if(CMAKE_CUDA_COMPILER)
    add_executable(multispectrum_benchmark_warp EXCLUDE_FROM_ALL app/BenchmarkWarpPrimary.cu
        src/input/config/SimConfig.cpp src/input/config/ConfigSerialization.cpp src/input/config/ConfigValidation.cpp src/input/config/SpectrumSource.cpp src/input/materials/XcomAttenuationProvider.cpp)
    target_include_directories(multispectrum_benchmark_warp PRIVATE include src "${CMAKE_CURRENT_SOURCE_DIR}/../../thirdparty/tomlplusplus/include")
    target_link_libraries(multispectrum_benchmark_warp PRIVATE CUDA::cudart nist_xcom_core spek_spectrum::core)
    if(MSVC)
        target_compile_options(multispectrum_benchmark_warp PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/utf-8> $<$<COMPILE_LANGUAGE:CXX>:/utf-8>)
    endif()
endif()
if(TARGET multispectrum_scalar_fp)
    add_executable(multispectrum_benchmark_scalar_fp EXCLUDE_FROM_ALL app/BenchmarkScalarFp.cpp
        src/input/config/SimConfig.cpp src/input/config/ConfigSerialization.cpp src/input/config/ConfigValidation.cpp src/input/config/SpectrumSource.cpp src/input/materials/XcomAttenuationProvider.cpp)
    target_include_directories(multispectrum_benchmark_scalar_fp PRIVATE include src "${CMAKE_CURRENT_SOURCE_DIR}/../../thirdparty/tomlplusplus/include")
    target_link_libraries(multispectrum_benchmark_scalar_fp PRIVATE multispectrum_scalar_fp nist_xcom_core spek_spectrum::core)
    if(MSVC)
        target_compile_options(multispectrum_benchmark_scalar_fp PRIVATE /utf-8)
    endif()
endif()
