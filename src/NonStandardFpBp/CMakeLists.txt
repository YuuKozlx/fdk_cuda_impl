# Non-standard projection/backprojection families are kept separate from the
# circular analytic FDK implementations.  The first family is source-orbit
# tomography: the detector and object stay fixed while only the source moves.
target_sources(YKCBCT PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/SourceOrbitTomography/YkSourceOrbitTomography.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/SourceOrbitTomography/YkSourceOrbitDerivedGeometry.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/SourceOrbitTomography/YkSourceOrbitTomographyPipeline.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/SourceOrbitTomography/kernels/YkSourceOrbitResampleKernels.cu
)
