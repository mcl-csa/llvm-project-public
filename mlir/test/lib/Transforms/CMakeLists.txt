# Exclude tests from libMLIR.so
add_mlir_library(MLIRTestTransforms
  TestConstantFold.cpp
  TestConvertMatmulParallelLoopsToGPU.cpp
  TestGpuMatmulParallelLoopMapping.cpp
  TestInlining.cpp
  TestLoopFusion.cpp
  TestLoopMapping.cpp
  TestLoopParametricTiling.cpp
  TestLoopUnrolling.cpp
  TestSpecializeAffineForWMMA.cpp

  EXCLUDE_FROM_LIBMLIR

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Transforms
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Conversion/SCFToGPU

  LINK_LIBS PUBLIC
  MLIRTestDialect
  MLIRTransforms
  MLIRGPU
  )

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../Dialect/Test)
include_directories(${CMAKE_CURRENT_BINARY_DIR}/../Dialect/Test)
