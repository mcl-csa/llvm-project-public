//===- GpuUtils.cpp ---- Utilities for lowering to GPUs -------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains mid-level utilities for lowering to GPUs.
//
//===----------------------------------------------------------------------===//

#include "mlir/Transforms/GpuUtils.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Transforms/LoopUtils.h"

using namespace mlir;

/// Creates fast buffers (in memory space == 3) and places the specified
/// matrices into them.
void mlir::createAndPlaceFastBuffersForGpuMatmul(
    AffineForOp rootForOp, ArrayRef<std::string> matricesToPlace,
    bool useStackAllocation, bool useGlobalAllocation) {
  SmallVector<AffineForOp, 6> loopNest;
  getPerfectlyNestedLoops(loopNest, rootForOp);

  // Checks if the loop nest is perfectly nested or not. The pass doesn't work
  // in case of imperfect loop nest.
  assert(loopNest.size() > 5 && "Expected perfectly nested loop nest.");

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
  MLIRContext *context = rootForOp.getContext();
  for (Operation *copyNest : copyNests)
    copyNest->setAttr("isCopyLoopNest", BoolAttr::get(context, true));

  // Mark the compute loop nest.
  loopNest[2]->setAttr("isComputeLoopNest", BoolAttr::get(context, true));
}
