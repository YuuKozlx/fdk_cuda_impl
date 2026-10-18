# Bone-hardening stage contract
add_library(multispectrum_bone_bhc STATIC src/recon/bhc/BoneBhcStage.cpp)
target_include_directories(multispectrum_bone_bhc PUBLIC include)
if(MSVC)
    target_compile_options(multispectrum_bone_bhc PRIVATE $<$<COMPILE_LANGUAGE:CXX>:/utf-8>)
endif()

# Main application and UI
add_executable(multispectrum_sim
    app/main.cpp
    src/processing/support/SimLogger.cpp
    src/input/config/SimConfig.cpp src/input/config/ConfigSerialization.cpp src/input/config/ConfigValidation.cpp
    src/input/config/SpectrumSource.cpp
    src/input/materials/WaterReference.cpp src/input/calibration/WaterBhcTable.cpp src/input/calibration/WaterBhcCalibration.cpp
    src/input/materials/SpectralModel.cpp
    src/processing/scatter/FasksRuntimeModel.cpp
    src/input/materials/XcomAttenuationProvider.cpp
    src/projection/spectral/ProjectionSimulator.cpp)
target_link_libraries(multispectrum_sim PRIVATE Eigen3::Eigen)
target_include_directories(multispectrum_sim PRIVATE
    include src "${SPECTRAL_XCOM_ROOT}" "${CMAKE_CURRENT_SOURCE_DIR}/../../thirdparty/tomlplusplus/include"
    "${CMAKE_CURRENT_SOURCE_DIR}/../../thirdparty/nlohmann/include")
target_link_libraries(multispectrum_sim PRIVATE yk::spdlog nist_xcom_core spek_spectrum::core)
if(MULTISPECTRUM_ENABLE_EXPERIMENTAL_ADAPTIVE_SCATTER)
    multispectrum_use_adaptive_scatter(multispectrum_sim)
endif()
if(TARGET YKCBCT)
    target_sources(multispectrum_sim PRIVATE
        src/recon/bhc/BoneBhcJob.cpp
        src/recon/backend/LibraryGeometry.cpp src/recon/backend/LibraryReconstructor.cpp src/recon/preprocess/ReconstructionPreprocessor.cpp
        src/recon/postprocess/PostprocessPipeline.cpp src/recon/postprocess/OutputStages.cpp
        src/recon/preprocess/ProjectionCache.cpp
        src/projection/spectral/CudaPrimaryProjector.cu src/projection/spectral/SpectralProjectionPipeline.cpp src/projection/spectral/MultiSpecFpStage.cpp
        src/projection/spectral/CudaProjectionBackend.cu src/projection/spectral/DetectorSignal.cpp
        src/projection/spectral/ProjectionStages.cpp src/projection/spectral/ProjectionOutput.cpp
        src/processing/scatter/CudaFasksScatter.cu
        src/recon/preprocess/PreprocessPipeline.cpp src/recon/preprocess/Stages.cpp
        src/recon/preprocess/CudaPixelOps.cu
        src/projection/spectral/CudaSignalStages.cu)
    multispectrum_use_image_processing(multispectrum_sim)
    target_include_directories(multispectrum_sim PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}/../../include")
    target_link_libraries(multispectrum_sim PRIVATE YKCBCT)
    target_link_libraries(multispectrum_sim PRIVATE multispectrum_bone_bhc)
    target_link_libraries(multispectrum_sim PRIVATE CUDA::cufft)
    target_compile_definitions(multispectrum_sim PRIVATE
        YK_MULTISPECTRUM_CUDA_PRIMARY=1)
    if(TARGET ykcbct_visualization)
        target_sources(multispectrum_sim PRIVATE src/client/InteractiveProjection.cpp)
        target_link_libraries(multispectrum_sim PRIVATE ykcbct_visualization)
        target_compile_definitions(multispectrum_sim PRIVATE YK_SPECTRAL_VTK=1)
        if(WIN32)
            add_custom_command(TARGET multispectrum_sim POST_BUILD
                COMMAND ${CMAKE_COMMAND} -E copy_if_different
                    $<TARGET_RUNTIME_DLLS:multispectrum_sim>
                    $<TARGET_FILE_DIR:multispectrum_sim>
                COMMAND_EXPAND_LISTS)
        endif()
        if(TARGET yk_imgui)
            target_sources(multispectrum_sim PRIVATE src/client/ImGuiProjection.cpp src/client/ProjectionUiConfig.cpp src/client/DisplayGeometry.cpp src/client/ReconstructionDisplay.cpp)
            target_link_libraries(multispectrum_sim PRIVATE yk_imgui comdlg32)
            target_compile_definitions(multispectrum_sim PRIVATE YK_SPECTRAL_IMGUI=1 NOMINMAX)
        endif()
    endif()
    add_custom_target(multispectrum_runtime_sync
        COMMAND ${CMAKE_COMMAND} -E make_directory
            $<TARGET_FILE_DIR:multispectrum_sim>
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            $<TARGET_FILE:YKCBCT> $<TARGET_FILE_DIR:multispectrum_sim>
        DEPENDS YKCBCT)
    add_dependencies(multispectrum_sim multispectrum_runtime_sync)
    add_custom_command(TARGET multispectrum_sim POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            $<TARGET_FILE:YKCBCT> $<TARGET_FILE_DIR:multispectrum_sim>)
else()
    target_sources(multispectrum_sim PRIVATE src/recon/backend/LibraryReconstructorStub.cpp src/projection/spectral/CudaPrimaryProjectorStub.cpp)
endif()
if(MSVC)
    target_compile_options(nist_xcom_core PRIVATE /utf-8)
    target_compile_options(multispectrum_sim PRIVATE
        $<$<COMPILE_LANGUAGE:CXX>:/utf-8>
        $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/utf-8>)
endif()

# The same arithmetic policy applies to application and test consumers.
set_source_files_properties(src/projection/spectral/CudaSignalStages.cu PROPERTIES COMPILE_OPTIONS "--fmad=false")

# Image and intensity processing
add_library(multispectrum_intensity_stages STATIC
    src/recon/postprocess/stages/IntensityOps.cpp
    src/recon/postprocess/stages/correction/HuTransforms.cpp
    src/recon/postprocess/stages/calibration/HuCalibration.cpp
    src/input/calibration/HuCalibrationFile.cpp
    src/recon/postprocess/stages/segmentation/HuThresholds.cpp)
target_include_directories(multispectrum_intensity_stages PUBLIC include PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/../../thirdparty/tomlplusplus/include")
if(MSVC)
    target_compile_options(multispectrum_intensity_stages PRIVATE
        $<$<COMPILE_LANGUAGE:CXX>:/utf-8>
        $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/utf-8>)
endif()
if(TARGET YKCBCT)
    target_sources(multispectrum_intensity_stages PRIVATE src/recon/postprocess/stages/IntensityKernels.cu)
    target_compile_definitions(multispectrum_intensity_stages PRIVATE YK_INTENSITY_CUDA=1)
    target_link_libraries(multispectrum_intensity_stages PRIVATE CUDA::cudart)
    multispectrum_use_device_memory(multispectrum_intensity_stages)
    set_property(TARGET multispectrum_intensity_stages PROPERTY CUDA_RUNTIME_LIBRARY Shared)
endif()
target_link_libraries(multispectrum_sim PRIVATE multispectrum_intensity_stages)

# Bone-hardening calibration and correction
add_library(multispectrum_two_material_calibration STATIC src/recon/preprocess/calibration/TwoMaterialCalibration.cpp src/recon/preprocess/calibration/EmpiricalTwoMaterialFit.cpp)
target_link_libraries(multispectrum_two_material_calibration PRIVATE Eigen3::Eigen)
target_include_directories(multispectrum_two_material_calibration PUBLIC include PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/../../thirdparty/nlohmann/include")
if(TARGET YKCBCT)
    add_library(multispectrum_two_material_bhc STATIC src/recon/preprocess/correction/TwoMaterialBhcStage.cu)
    target_include_directories(multispectrum_two_material_bhc PUBLIC include)
    target_link_libraries(multispectrum_two_material_bhc PUBLIC multispectrum_two_material_calibration CUDA::cudart)
    multispectrum_use_device_memory(multispectrum_two_material_bhc)
    target_sources(multispectrum_bone_bhc PRIVATE src/recon/bhc/BoneBhcAlgorithms.cu)
    target_include_directories(multispectrum_bone_bhc PRIVATE src "${CMAKE_CURRENT_SOURCE_DIR}/../../thirdparty/nlohmann/include")
    target_link_libraries(multispectrum_bone_bhc PUBLIC multispectrum_two_material_bhc multispectrum_scalar_fp CUDA::cudart)
    multispectrum_use_device_memory(multispectrum_bone_bhc)
    set_property(TARGET multispectrum_bone_bhc PROPERTY CUDA_RUNTIME_LIBRARY Shared)
    if(MSVC)
        target_compile_options(multispectrum_bone_bhc PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/utf-8>)
    endif()
    set_property(TARGET multispectrum_two_material_bhc PROPERTY CUDA_RUNTIME_LIBRARY Shared)
endif()
if(MSVC AND TARGET multispectrum_two_material_calibration)
    target_compile_options(multispectrum_two_material_calibration PRIVATE $<$<COMPILE_LANGUAGE:CXX>:/utf-8> $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/utf-8>)
endif()
if(MSVC AND TARGET multispectrum_two_material_bhc)
    target_compile_options(multispectrum_two_material_bhc PRIVATE $<$<COMPILE_LANGUAGE:CXX>:/utf-8> $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/utf-8>)
endif()
if(TARGET YKCBCT)
    add_library(multispectrum_empirical_bhc STATIC src/recon/preprocess/correction/EmpiricalBhcStage.cu)
    target_include_directories(multispectrum_empirical_bhc PUBLIC include)
    target_link_libraries(multispectrum_empirical_bhc PUBLIC CUDA::cudart)
    multispectrum_use_device_memory(multispectrum_empirical_bhc)
    set_property(TARGET multispectrum_empirical_bhc PROPERTY CUDA_RUNTIME_LIBRARY Shared)
    if(MSVC)
        target_compile_options(multispectrum_empirical_bhc PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/utf-8>)
    endif()
endif()
add_library(multispectrum_ebhc_fit STATIC src/recon/preprocess/calibration/EmpiricalBhcFit.cpp)
target_link_libraries(multispectrum_ebhc_fit PRIVATE Eigen3::Eigen)
target_include_directories(multispectrum_ebhc_fit PUBLIC include)
if(TARGET YKCBCT)
    add_library(multispectrum_ebhc_paper_core STATIC src/recon/postprocess/stages/correction/EmpiricalBoneCorrection.cu)
    target_include_directories(multispectrum_ebhc_paper_core PUBLIC include)
    target_link_libraries(multispectrum_ebhc_paper_core PUBLIC CUDA::cudart)
    multispectrum_use_device_memory(multispectrum_ebhc_paper_core)
    set_property(TARGET multispectrum_ebhc_paper_core PROPERTY CUDA_RUNTIME_LIBRARY Shared)
    if(MSVC)
        target_compile_options(multispectrum_ebhc_paper_core PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/utf-8>)
    endif()
endif()
if(TARGET YKCBCT)
    target_link_libraries(multispectrum_sim PRIVATE multispectrum_scalar_fp)
endif()

# Built-in examples and MAT input
add_library(multispectrum_examples STATIC src/input/examples/ExampleStage.cpp)
target_include_directories(multispectrum_examples PUBLIC include)
target_link_libraries(multispectrum_sim PRIVATE multispectrum_examples)
if(MSVC)
    target_compile_options(multispectrum_examples PRIVATE /utf-8)
endif()
add_library(multispectrum_mat_volume STATIC src/input/io/MatLabelVolume.cpp)
target_include_directories(multispectrum_mat_volume PUBLIC include)
target_link_libraries(multispectrum_mat_volume PRIVATE matio::matio)
if(MSVC)
    target_compile_options(multispectrum_mat_volume PRIVATE /utf-8)
endif()
add_library(multispectrum_data_examples STATIC src/input/examples/DataExamples.cpp)
target_include_directories(multispectrum_data_examples PUBLIC include)
target_compile_definitions(multispectrum_data_examples PRIVATE MULTISPECTRUM_EXAMPLE_DATA_ROOT="${CMAKE_CURRENT_SOURCE_DIR}/../../data")
target_link_libraries(multispectrum_data_examples PUBLIC multispectrum_examples multispectrum_icrp multispectrum_mat_volume)
target_link_libraries(multispectrum_sim PRIVATE multispectrum_data_examples)
if(MSVC)
    target_compile_options(multispectrum_data_examples PRIVATE /utf-8)
endif()
add_library(multispectrum_icrp STATIC src/input/examples/IcrpExamples.cpp)
target_include_directories(multispectrum_icrp PUBLIC include PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/../../thirdparty/tomlplusplus/include")
if(MSVC)
    target_compile_options(multispectrum_icrp PRIVATE /utf-8)
endif()

# Geometry calibration
add_library(multispectrum_geometry_calibration STATIC src/recon/geometry/GeometryCalibrationStage.cpp src/recon/geometry/GeometryCorrectionStage.cpp src/input/calibration/GeometryCalibrationFile.cpp)
target_include_directories(multispectrum_geometry_calibration PUBLIC include PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/../../thirdparty/tomlplusplus/include")
target_link_libraries(multispectrum_geometry_calibration PUBLIC cbct_calibration::cbct_calibration)
target_link_libraries(multispectrum_sim PRIVATE multispectrum_geometry_calibration)
if(MSVC)
    target_compile_options(multispectrum_geometry_calibration PRIVATE /utf-8)
endif()
