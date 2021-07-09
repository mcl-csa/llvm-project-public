//== -----TestMarkParallelLoops.cpp - Test marking of parallel loops ------ ==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements a test pass to find and mark parallel loops in a given
// IR. A boolean attribute named isParallel is attatched with the loop that is
// parallel. Marking parallel loops in advance helps in converting affine.for to
// affine.parallel later assuming that any transformation applied on that code
// didn't change the nature of the loops. I.e., sequential loop stays sequential
// and parallel loop stays parallel.
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
struct TestMarkParallelLoops
    : public PassWrapper<TestMarkParallelLoops, FunctionPass> {
  void runOnFunction() override;

  /// Copy loop-nest string identifier.
  static const std::string isParallel;
};
} // namespace

void TestMarkParallelLoops::runOnFunction() {
  FuncOp funcOp = getFunction();

  // Check and mark the parallel loops in the IR.
  funcOp.walk([&](AffineForOp loop) {
    if (isLoopParallel(loop)) {
      loop->setAttr(TestMarkParallelLoops::isParallel,
                    BoolAttr::get(loop.getContext(), true));
    }
  });
}

namespace mlir {
void registerTestMarkParallelLoops() {
  PassRegistration<TestMarkParallelLoops>(
      "test-mark-parallel-loops", "mark parallel loops in the given IR");
}
} // namespace mlir

const std::string TestMarkParallelLoops::isParallel = "isParallel";
