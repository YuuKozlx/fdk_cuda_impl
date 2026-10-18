# Internal implementation assembly. These helpers create no library targets.
# Keep absolute paths so the same source lists serve standalone numerical tests.
include_guard(GLOBAL)

function(multispectrum_use_device_memory target)
    get_target_property(already_added ${target} MULTISPECTRUM_DEVICE_SOURCES)
    if(already_added)
        return()
    endif()
    set_property(TARGET ${target} PROPERTY MULTISPECTRUM_DEVICE_SOURCES TRUE)
    set(sim "${CMAKE_CURRENT_FUNCTION_LIST_DIR}/..")
    target_sources(${target} PRIVATE "${sim}/src/processing/device/DeviceMemoryPool.cpp")
    target_include_directories(${target} PRIVATE "${sim}/include")
    target_link_libraries(${target} PRIVATE CUDA::cudart)
    set_property(TARGET ${target} PROPERTY CUDA_RUNTIME_LIBRARY Shared)
    if(MSVC)
        target_compile_options(${target} PRIVATE
            $<$<COMPILE_LANGUAGE:CXX>:/utf-8> $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/utf-8>)
    endif()
endfunction()

function(multispectrum_use_image_processing target)
    get_target_property(already_added ${target} MULTISPECTRUM_IMAGE_SOURCES)
    if(already_added)
        return()
    endif()
    set_property(TARGET ${target} PROPERTY MULTISPECTRUM_IMAGE_SOURCES TRUE)
    set(sim "${CMAKE_CURRENT_FUNCTION_LIST_DIR}/..")
    target_sources(${target} PRIVATE
        "${sim}/src/processing/image/ImageProcessing.cpp"
        "${sim}/src/processing/image/ImageKernels.cu"
        "${sim}/src/recon/ImageStages.cpp")
    multispectrum_use_device_memory(${target})
endfunction()

function(multispectrum_use_adaptive_scatter target)
    get_target_property(already_added ${target} MULTISPECTRUM_ADAPTIVE_SOURCES)
    if(already_added)
        return()
    endif()
    set_property(TARGET ${target} PROPERTY MULTISPECTRUM_ADAPTIVE_SOURCES TRUE)
    set(sim "${CMAKE_CURRENT_FUNCTION_LIST_DIR}/..")
    target_sources(${target} PRIVATE "${sim}/src/processing/scatter/AdaptiveScatterSolver.cu")
    target_compile_definitions(${target} PRIVATE YK_SPECTRAL_ADAPTIVE_SCATTER=1)
    multispectrum_use_device_memory(${target})
endfunction()
