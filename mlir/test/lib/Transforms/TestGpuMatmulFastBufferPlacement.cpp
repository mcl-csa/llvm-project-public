//= TestGpuMatmulFastBufferPlacement.cpp - Places matrices into fast buffer ==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements pass that places matrices into fast buffer.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Affine/Passes.h"
#include "mlir/Transforms/GpuUtils.h"
#include "mlir/Transforms/LoopUtils.h"
#include "llvm/Support/CommandLine.h"

using namespace mlir;

#define DEBUG_TYPE "test-gpu-matmul-fast-buffer-placement"

namespace {

/// This pass places the input and output matrices of matrix multiplication
/// of the form C = A*B + C into the gpu's fast buffer (shared memory).
/// Matrices to be placed can be specified using the pass option `matrices`.
/// If nothing is specified then it places A and B matrix into the fast buffer.
/// The pass works only in case of perfectly nested loop nest. The pass can be
/// extended easily for other forms of matrix multiplication.
struct TestGpuMatmulFastBufferPlacement
    : public PassWrapper<TestGpuMatmulFastBufferPlacement, FunctionPass> {
  TestGpuMatmulFastBufferPlacement() = default;
  TestGpuMatmulFastBufferPlacement(
      const TestGpuMatmulFastBufferPlacement &pass) {}
  void runOnFunction() override;
  ListOption<std::string> matrices{
      *this, "matrices", llvm::cl::MiscFlags::CommaSeparated,
      llvm::cl::desc("Specifies which matrices to place in the GPU Buffer.")};
  Option<bool> stackAllocation{
      *this, "stack-allocation",
      llvm::cl::desc(
          "Specifies whether to allocate buffers in the stack or not."),
      llvm::cl::init(false)};
  Option<bool> globalAllocation{
      *this, "global-allocation",
      llvm::cl::desc(
          "Specifies whether to allocate buffers as the global memref or not."),
      llvm::cl::init(false)};
};
// Names of matrices to place in fast buffer.
SmallVector<std::string, 3> matricesToPlace;
bool useStackAllocation;
bool useGlobalAllocation;
} // end anonymous namespace

static void runOnBlock(Block &block, ArrayRef<std::string> matricesToPlace,
                       bool useStackAllocation, bool useGlobalAllocation) {
  for (Operation &op : block) {
    // Finding the topmost for loop.
    if (AffineForOp forOp = dyn_cast<AffineForOp>(op)) {
      if (!forOp->getParentOfType<AffineForOp>()) {
        OpBuilder opBuilder(forOp);
        createAndPlaceFastBuffersForGpuMatmul(
            forOp, matricesToPlace, useStackAllocation, useGlobalAllocation);
      }
    }
  }
}

void TestGpuMatmulFastBufferPlacement::runOnFunction() {
  FuncOp funcOp = getFunction();

  useStackAllocation = stackAllocation;
  useGlobalAllocation = globalAllocation;

  for (auto mat : matrices)
    // This condition ensures that only those matrices are placed in fast
    // memory which are specified correctly in the option `matrices`.
    if (mat == "A" || mat == "B" || mat == "C")
      matricesToPlace.push_back(mat);

  // If no matrix is specified (correctly) then by default A and B matrix are
  // placed in fast memory.
  if (matricesToPlace.empty())
    matricesToPlace.insert(matricesToPlace.begin(), {"A", "B"});

  for (Block &block : funcOp) {
    runOnBlock(block, matricesToPlace, useStackAllocation, useGlobalAllocation);
  }
}

namespace mlir {
void registerTestGpuMatmulFastBufferPlacementPass() {
  PassRegistration<TestGpuMatmulFastBufferPlacement>(
      "test-gpu-matmul-fast-buffer-placement",
      "Place fast memory(SMEM) buffers right inside kernel");
}
} // namespace mlir
