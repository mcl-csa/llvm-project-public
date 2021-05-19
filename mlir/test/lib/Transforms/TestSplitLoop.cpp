//==-TestSplitLoop.cpp - Test loop splitting to overlap copy with compute -==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements a test pass to split a given loop which has a copy and
// compute part to overlap the copy and compute.
//
//===----------------------------------------------------------------------===//
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/GPU/GPUDialect.h"
#include "mlir/Transforms/LoopUtils.h"
#include "mlir/Transforms/Passes.h"
#include "llvm/Support/CommandLine.h"

using namespace mlir;

namespace {

/// Splits a loop which consists of a copy-in part, which moves data to a faster
/// memory and a compute part which uses the data copied in. Both parts are
/// expected to be inside the loop being split. `split` essentially means moving
/// out one iteration of the copy-in part. Now inside the copy the copy-in
/// instructions for instruction `i + 1` will be issued while the compute will
/// be done for iteration `i`. This now means that the upper bound of the loop
/// has to be adjusted to prevent illegal memory acceses in the copy-in part.
/// Doing this will consequently lead to a peeled out iteration of the compute
/// part just after the loop being split.

/// The pass uses `clIdentifyCopyLoopNestAttribute` to identify that the given
/// loop nest is a copy loop nest which copies data from a slower to a fast
/// memory.
///
/// `clIdentifyComputeLoopNestAttribute` is used to identify that the given loop
/// nest is a computation loop nest which performs computation and pipelining is
struct TestSplitLoop : public PassWrapper<TestSplitLoop, FunctionPass> {
  TestSplitLoop() = default;
  TestSplitLoop(const TestSplitLoop &pass) {}

  void runOnFunction() override;
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<gpu::GPUDialect>();
  }

private:
  Option<std::string> clIdentifyCopyLoopNestAttribute{
      *this, "copy-loop-nest-attr",
      llvm::cl::desc("Used to identify loop nest(s) which copy data"),
      llvm::cl::init("isCopyLoopNest")};
  Option<std::string> clIdentifyComputeLoopNestAttribute{
      *this, "compute-loop-nest-attr",
      llvm::cl::desc(
          "Used to identify loop nest(s) which performs computation"),
      llvm::cl::init("isComputeLoopNest")};
};

} // end of namespace

void TestSplitLoop::runOnFunction() {
  FuncOp funcOp = getFunction();
  OpBuilder b(funcOp.getContext());
  Operation *toSplit = nullptr;
  funcOp.walk([&](AffineForOp forOp) {
    BoolAttr attr =
        forOp->getAttrOfType<BoolAttr>(clIdentifyComputeLoopNestAttribute);
    if (attr && attr.getValue() == true) {
      splitLoop(forOp, clIdentifyCopyLoopNestAttribute,
                clIdentifyComputeLoopNestAttribute);
      toSplit = forOp.getOperation();
    }
  });
}

namespace mlir {
void registerTestSplitLoopPass() {
  PassRegistration<TestSplitLoop>(
      "test-split-compute-loop",
      "Tests splitting of a particular loop to overlap copy with compute");
}
} // namespace mlir
