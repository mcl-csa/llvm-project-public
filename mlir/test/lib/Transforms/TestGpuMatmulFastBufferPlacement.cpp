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
// It contains matrices name that has to be placed in fast buffer.
SmallVector<std::string, 3> matricesToPlace;
bool useStackAllocation;
bool useGlobalAllocation;
} // end anonymous namespace

/// Creates fast buffers (in memory space == 3) and places the specified
/// matrices into them.
static void createAndPlaceFastBuffers(AffineForOp rootForOp,
                                      OpBuilder opBuilder) {
  SmallVector<AffineForOp, 6> loopNest;
  getPerfectlyNestedLoops(loopNest, rootForOp);

  // Checks if the loop nest is perfectly nested or not. The pass doesn't work
  // in case of imperfect loop nest.
  assert(loopNest.size() > 2 && "Expected perfectly nested loop nest.");

  SmallVector<Value, 4> inputMemrefs;
  Value outputMemRef, lhsMemRef, rhsMemRef;
  // Identify the input and output matrices (memrefs).
  rootForOp.walk(
      [&](AffineStoreOp storeOp) { outputMemRef = storeOp.getMemRef(); });
  rootForOp.walk([&](AffineLoadOp loadOp) {
    // Checks if the loadOp's memref is equal to output memref, if yes then
    // it's the output matrix's memref and skip it.
    if (outputMemRef == loadOp.getMemRef())
      return;
    inputMemrefs.push_back(loadOp.getMemRef());
  });

  // Intialize the copy options for placing matrices into fast buffers.
  AffineCopyOptions copyOptions = {
      /*generateDma=*/false,
      /*slowMemorySpace=*/0,
      /*fastMemorySpace=*/3,
      /*tagMemorySpace=*/0,
      /*fastMemCapacityBytes=*/UINT_MAX,
      /*fastBufferLayout*/ AffineMap(),
      /*fastBufferPlacementBlock*/ loopNest[1].getBody(),
      /*useStackAllocation*/ useStackAllocation,
      /*useGlobalAllocation*/ useGlobalAllocation,
      /*globalMemrefName*/ "global_mem"};

  // It contains loop nests which copies data from gpu's slow memory into
  // fast buffers.
  DenseSet<Operation *> copyNests;

  // Checks whether the matrix has to be placed or not, if yes then place it
  // in the fast memory.
  if (llvm::is_contained(matricesToPlace, "A")) {
    copyOptions.globalMemrefName = "frag_A";
    affineDataCopyGenerate(loopNest[2].getBody()->begin(),
                           std::prev(loopNest[2].getBody()->end()), copyOptions,
                           inputMemrefs[0], copyNests);
  }

  if (llvm::is_contained(matricesToPlace, "B")) {
    copyOptions.globalMemrefName = "frag_B";
    affineDataCopyGenerate(loopNest[2].getBody()->begin(),
                           std::prev(loopNest[2].getBody()->end()), copyOptions,
                           inputMemrefs[1], copyNests);
  }

  if (llvm::is_contained(matricesToPlace, "C")) {
    copyOptions.globalMemrefName = "frag_C";
    affineDataCopyGenerate(loopNest[2].getBody()->begin(),
                           std::prev(loopNest[2].getBody()->end()), copyOptions,
                           outputMemRef, copyNests);
  }

  // Attaches attributes with the loop nests copying input matrices A and B
  // (if present), and the loop nest which performs computation. These
  // attribtes are used by the pipelining pass.
  for (Operation *copyNest : copyNests)
    copyNest->setAttr("isCopyLoopNest", opBuilder.getBoolAttr(true));

  // Checks if the loop nest to be marked is present or not.
  if (loopNest[3])
    loopNest[3]->setAttr("isComputeLoopNest", opBuilder.getBoolAttr(true));
}

static void runOnBlock(Block &block) {
  for (Operation &op : block) {
    // Finding the topmost for loop.
    if (AffineForOp forOp = dyn_cast<AffineForOp>(op)) {
      if (!forOp->getParentOfType<AffineForOp>()) {
        OpBuilder opBuilder(forOp);
        createAndPlaceFastBuffers(forOp, opBuilder);
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
    runOnBlock(block);
  }
}

namespace mlir {
void registerTestGpuMatmulFastBufferPlacementPass() {
  PassRegistration<TestGpuMatmulFastBufferPlacement>(
      "test-gpu-matmul-fast-buffer-placement",
      "Place fast memory(SMEM) buffers right inside kernel");
}
} // namespace mlir
