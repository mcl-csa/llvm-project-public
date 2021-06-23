//===- GpuUtils.h - transformation utilities for GPUs -----------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This header file defines prototypes for various transformation utilities for
// lowering IR at the mid-level (memrefs and loops) to GPUs.
//
//===----------------------------------------------------------------------===//

#include "mlir/Support/LLVM.h"
#include "llvm/ADT/StringRef.h"
#include <string>

namespace mlir {

class AffineForOp;
struct LogicalResult;

/// Creates fast buffers (in memory space == 3) and places the specified
/// matrices into them.
void createAndPlaceFastBuffersForGpuMatmul(
    AffineForOp forOp, ArrayRef<std::string> matricesToPlace,
    bool useStackAllocation, bool useGlobalAllocation);

/// Map an affine.for nest to WMMA ops.
LogicalResult mapAffineNestToWmma(AffineForOp rootForOp,
                                  StringRef accumulateType,
                                  bool enableUnroll = true);

} // namespace mlir
