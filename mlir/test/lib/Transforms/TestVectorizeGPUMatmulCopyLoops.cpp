//== -TestVectorizeGPUMatmulCopyLoops.cpp - Test copy loops vectorization - ==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file performs vectorization of copy loops in matmul from slow(global) to
// fast(shared) memory.
//
//===----------------------------------------------------------------------===//

#include "mlir/Analysis/AffineAnalysis.h"
#include "mlir/Analysis/LoopAnalysis.h"
#include "mlir/Analysis/Utils.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Affine/Passes.h"
#include "mlir/Dialect/Affine/Utils.h"
#include "mlir/Dialect/GPU/GPUDialect.h"
#include "mlir/Dialect/Vector/VectorOps.h"
#include "mlir/IR/BlockAndValueMapping.h"
#include "mlir/Transforms/LoopUtils.h"

using namespace mlir;

#define DEBUG_TYPE "test-vectorize-gpu-matmul-copy-loops"

namespace {
struct TestVectorizeGPUMatmulCopyLoops
    : public PassWrapper<TestVectorizeGPUMatmulCopyLoops, FunctionPass> {
  void runOnFunction() override;

  TestVectorizeGPUMatmulCopyLoops(){};
  TestVectorizeGPUMatmulCopyLoops(const TestVectorizeGPUMatmulCopyLoops &) {}
  explicit TestVectorizeGPUMatmulCopyLoops(unsigned loadStoreWidth) {
    clLoadStoreWidth = loadStoreWidth;
  }

  /// CL option to specify vector width to use for global memory loads.
  Option<unsigned> clLoadStoreWidth{
      *this, "load-store-width",
      llvm::cl::desc(
          "Vector width in bits to use for load/store operations. "
          "Valid widths are 32, 64 and 128. No vectorization if option"
          "is unspecified."),
      llvm::cl::init(0)};

  /// Copy loop-nest string identifier.
  static const std::string isCopyLoopNest;
};
} // namespace

void TestVectorizeGPUMatmulCopyLoops::runOnFunction() {
  FuncOp funcOp = getFunction();

  // Find and vectorize copy loops.
  if (clLoadStoreWidth != 0) {
    DenseSet<Operation *> toVectorize;
    funcOp->walk([&](AffineForOp forOp) {
      if (BoolAttr attr = forOp->getAttrOfType<BoolAttr>(
              TestVectorizeGPUMatmulCopyLoops::isCopyLoopNest)) {
        // Walk and get all the load ops.
        SmallVector<AffineLoadOp> loadOps;
        forOp.walk([&](AffineLoadOp loadOp) { loadOps.push_back(loadOp); });
        // Loop corresponding to the fastest varying dimension has to be
        // vectorized. Check all the forOps that are vectorizable and find the
        // one which corresponds to the fastest varying dimension of the loadOp.
        // Vectorizing this loop would enable vectorized accesses and also
        // ensure coalescing of requests.
        int maxDimInx = -1, memRefDim, invertedDimInx;
        AffineForOp fastestVaryingLoop;
        forOp.walk([&](AffineForOp nestedFor) {
          for (auto memOp : loadOps) {
            if (isContiguousAccess(nestedFor.getInductionVar(), memOp,
                                   &memRefDim)) {
              invertedDimInx = memOp.getMemRefType().getRank() - memRefDim - 1;
              if (invertedDimInx > maxDimInx) {
                fastestVaryingLoop = nestedFor;
                maxDimInx = invertedDimInx;
              }
            }
          }
        });
        toVectorize.insert(fastestVaryingLoop);
      }
    });

    // Vectorize the collected loops.
    for (auto loop : toVectorize) {
      AffineForOp forOp = dyn_cast<AffineForOp>(*loop);
      if (forOp)
        (void)loopVectorize(forOp, clLoadStoreWidth);
    }
  }

  // Convert the marked forOps to parallel. Currently, It is assumed that all
  // the transformations done above do not change the nature of loops, i.e., a
  // sequential loop remains sequential and a parallel loop remains parallel.
  // TODO: Currently this works because the parallel loops really doesn't
  // yield anything and affineParallelize also does not handle the case when
  // the ops yield somehing. Whenever this breaks we need to handle this.
  funcOp.walk([&](AffineForOp forOp) {
    if (auto isParallel = forOp->getAttrOfType<BoolAttr>("isParallel")) {
      if (isParallel.getValue() == true)
        affineParallelize(forOp);
    }
  });
}

namespace mlir {
void registerTestVectorizeGPUMatmulCopyLoops() {
  PassRegistration<TestVectorizeGPUMatmulCopyLoops>(
      "test-vectorize-gpu-matmul-copy-loops",
      "vectorize GPU matmul copy loops");
}
} // namespace mlir

const std::string TestVectorizeGPUMatmulCopyLoops::isCopyLoopNest =
    "isCopyLoopNest";
