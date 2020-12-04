# Exclude tests from libMLIR.so
add_mlir_library(MLIRTestTransforms
  TestConstantFold.cpp
  TestConvertMatmulParallelLoopsToGPU.cpp
  TestGpuMarkGlobalAsWorkgroupMemory.cpp
  TestGpuMatmulFastBufferPlacement.cpp
  TestGPUMatmulBarrierInsertion.cpp
  TestGpuMatmulParallelLoopMapping.cpp
  TestCollapseAffineParallel.cpp
  TestInlining.cpp
  TestLoopFusion.cpp
  TestLoopMapping.cpp
  TestLoopParametricTiling.cpp
  TestLoopUnrolling.cpp
  TestUnrollAndDelayCopies.cpp
  TestMarkParallelLoops.cpp
  TestSplitLoop.cpp
  TestVectorizeGPUMatmulCopyLoops.cpp
  TestCopyLoopNormalization.cpp
  TestSpecializeAffineForWMMA.cpp

  EXCLUDE_FROM_LIBMLIR

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Transforms
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Conversion/SCFToGPU

  LINK_LIBS PUBLIC
  MLIRAffineTransforms
  MLIRTestDialect
  MLIRTransforms
  MLIRGPU
  )

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../Dialect/Test)
include_directories(${CMAKE_CURRENT_BINARY_DIR}/../Dialect/Test)
