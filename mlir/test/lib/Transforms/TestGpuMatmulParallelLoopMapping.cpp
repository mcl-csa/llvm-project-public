//===- ParallelLoopMapper.cpp - Utilities for mapping parallel loops to GPU =//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements pass to map parallel loops to GPU devices.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/GPU/GPUDialect.h"
#include "mlir/Dialect/GPU/ParallelLoopMapper.h"
#include "mlir/Dialect/GPU/ParallelLoopMapperAttr.cpp.inc"
#include "mlir/Dialect/GPU/ParallelLoopMapperEnums.cpp.inc"
#include "mlir/Dialect/GPU/Passes.h"
#include "mlir/Dialect/SCF/SCF.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/LoopUtils.h"

using namespace mlir;
using namespace mlir::gpu;
using namespace mlir::scf;

#define DEBUG_TYPE "test-gpu-matmul-parallel-loop-mapping"

namespace {
/// Simple pass for testing the mapping of parallel loops to hardware ids using
/// a mapping strategy specialized for matmul op.
struct TestGpuMatmulParallelLoopMapping
    : public PassWrapper<TestGpuMatmulParallelLoopMapping, FunctionPass> {
  TestGpuMatmulParallelLoopMapping() = default;
  TestGpuMatmulParallelLoopMapping(
      const TestGpuMatmulParallelLoopMapping &pass) {}
  void runOnFunction() override;
};

enum MappingLevel { MapGrid = 0, MapWarp = 1, MapBlock = 2, Sequential = 3 };

static constexpr int kNumHardwareIds = 3;

} // namespace

static StringRef getMappingAttributeName() { return "mapping"; }

static ParallelLoopDimMapping
getParallelLoopDimMappingAttribute(Processor processor, AffineMap map,
                                   AffineMap bound) {
  MLIRContext *context = map.getContext();
  OpBuilder builder(context);
  return ParallelLoopDimMapping::get(
      ProcessorAttr::get(builder.getContext(), processor),
      AffineMapAttr::get(map), AffineMapAttr::get(bound), context);
}

/// Sets mapping attribute `mapping` as an attribute of `ploopOp`.
static LogicalResult
setMappingAttribute(scf::ParallelOp ploopOp,
                    ArrayRef<ParallelLoopDimMapping> mapping) {
  // Verify that each processor is mapped to only once.
  llvm::DenseSet<gpu::Processor> specifiedMappings;
  for (auto dimAttr : mapping) {
    gpu::Processor processor = getProcessor(dimAttr);
    if (processor != gpu::Processor::Sequential &&
        specifiedMappings.count(processor))
      return ploopOp.emitError(
          "invalid mapping multiple loops to same processor");
  }
  ArrayRef<Attribute> mappingAsAttrs(mapping.data(), mapping.size());
  ploopOp->setAttr(gpu::getMappingAttrName(),
                   ArrayAttr::get(ploopOp.getContext(), mappingAsAttrs));
  return success();
}

/// Bounded increment on MappingLevel. Increments to the next
/// level unless Sequential was already reached.
static MappingLevel &operator++(MappingLevel &mappingLevel) {
  if (mappingLevel < Sequential) {
    mappingLevel = static_cast<MappingLevel>(mappingLevel + 1);
  }
  return mappingLevel;
}

/// Computed the hardware id to use for a given mapping level. Will
/// assign x,y and z hardware ids for the first 3 dimensions and use
/// sequential after.
static Processor getHardwareIdForMapping(MappingLevel level, int dimension) {

  if (dimension >= kNumHardwareIds || level == Sequential)
    return Processor::Sequential;
  switch (level) {
  case MapGrid:
    switch (dimension) {
    case 0:
      return Processor::BlockX;
    case 1:
      return Processor::BlockY;
    case 2:
      return Processor::BlockZ;
    default:
      return Processor::Sequential;
    }
    break;
  case MapWarp:
    switch (dimension) {
    case 0:
      return Processor::WarpX;
    case 1:
      return Processor::WarpY;
    case 2:
      return Processor::WarpZ;
    default:
      return Processor::Sequential;
    }
    break;
  case MapBlock:
    switch (dimension) {
    case 0:
      return Processor::ThreadX;
    case 1:
      return Processor::ThreadY;
    case 2:
      return Processor::ThreadZ;
    default:
      return Processor::Sequential;
    }
  default:;
  }
  return Processor::Sequential;
}

/// Checks if `loop` is a copy loop/loop nest or not. Here we have taken a
/// conservative approach for identifying copy loop. We define a loop as a
/// copy loop if it consists of exactly one load op and one store op.
static bool isCopyLoop(ParallelOp loop) {
  unsigned numLoad = 0, numStore = 0;
  loop.walk([&](memref::LoadOp load) {
    if (load)
      numLoad++;
  });
  loop.walk([&](memref::StoreOp store) {
    if (store)
      numStore++;
  });
  if (numLoad == 1 && numStore == 1)
    return true;
  return false;
}

/// Collapses a n-dimensional parallel loop into a 1-d parallel loop.
static ParallelOp collapseParallelLoop(ParallelOp parallelLoop) {
  // `combinedDimensions` stores all the dimensions of parallel loop which has
  // to be collapsed.
  std::vector<unsigned> combinedDimensions;
  for (unsigned i = 0, e = parallelLoop.getNumLoops(); i < e; ++i)
    combinedDimensions.push_back(i);
  return collapseParallelLoops(parallelLoop, combinedDimensions);
}

/// Add mapping information to the given parallel loop. Do not add
/// mapping information if the loop already has it. Also, don't
/// start a mapping at a nested loop.
static void mapParallelOp(ParallelOp parallelOp,
                          MappingLevel mappingLevel = MapGrid) {

  // Do not try to add a mapping to already mapped loops or nested loops (if
  // it is not a copy loop).
  if (parallelOp->getAttr(getMappingAttributeName()) ||
      ((mappingLevel == MapGrid) && parallelOp->getParentOfType<ParallelOp>() &&
       !isCopyLoop(parallelOp)))
    return;

  // The copy loop nests are mapped to the thread blocks.
  if (isCopyLoop(parallelOp)) {
    mappingLevel = MapGrid;
    // Collapsing an n-dimensional parallel loop into 1-dimensional parallel
    // loop.
    parallelOp = collapseParallelLoop(parallelOp);
  }

  MLIRContext *ctx = parallelOp.getContext();
  Builder b(ctx);
  SmallVector<ParallelLoopDimMapping, 4> attrs;
  attrs.reserve(parallelOp.getNumLoops());
  for (int i = 0, e = parallelOp.getNumLoops(); i < e; ++i) {
    attrs.push_back(getParallelLoopDimMappingAttribute(
        getHardwareIdForMapping(mappingLevel, i), b.getDimIdentityMap(),
        b.getDimIdentityMap()));
  }
  // If the `parallelOp` is 2 or more dimensional then interchanging the first
  // two mapping attributes in order to map the i loop to y-dimension and
  // j loop to the x-dimension because it is more intuitive.
  if (parallelOp.getNumLoops() > 1) {
    ParallelLoopDimMapping attr = attrs[0];
    attrs[0] = attrs[1];
    attrs[1] = attr;
  }
  (void)setMappingAttribute(parallelOp, attrs);
  ++mappingLevel;
  // Parallel loop operations are immediately nested, so do not use
  // walk but just iterate over the operations.
  for (Operation &op : *parallelOp.getBody()) {
    if (ParallelOp nested = dyn_cast<ParallelOp>(op))
      mapParallelOp(nested, mappingLevel);
  }
}

void TestGpuMatmulParallelLoopMapping::runOnFunction() {
  FuncOp funcOp = getFunction();
  funcOp.getRegion().walk(
      [](ParallelOp parallelOp) { mapParallelOp(parallelOp); });
}

namespace mlir {
void registerTestGpuMatmulParallelLoopMappingPass() {
  PassRegistration<TestGpuMatmulParallelLoopMapping>(
      "test-gpu-matmul-parallel-loop-mapping",
      "Maps all parallel loops to gpu hardware ids.");
}
} // namespace mlir
