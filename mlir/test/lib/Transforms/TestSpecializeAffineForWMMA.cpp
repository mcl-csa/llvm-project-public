//== ---TestSpecializeAffineForWMMA.cpp - Test affine to GPU WMMA matmul -- ==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains specilaization patterns for matmul targetting tensor cores
// on Nvidia GPUs. It inserts WMMA ops and also moves loop-invariant load/stores
// outside the loops. It also inserts synchronization barriers wherever
// necessary.
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
#include "mlir/Transforms/GpuUtils.h"
#include "mlir/Transforms/LoopUtils.h"
#include "llvm/Support/Debug.h"

using namespace mlir;

#define DEBUG_TYPE "test-specialize-affine-matmul-for-wmma"

namespace {
struct TestSpecializeAffineForWMMA
    : public PassWrapper<TestSpecializeAffineForWMMA, FunctionPass> {
  void runOnFunction() override;

  TestSpecializeAffineForWMMA(){};
  TestSpecializeAffineForWMMA(const TestSpecializeAffineForWMMA &) {}
  explicit TestSpecializeAffineForWMMA(StringRef accumulateType) {
    clAccumulateType = accumulateType.str();
  }

  explicit TestSpecializeAffineForWMMA(unsigned loadStoreWidth) {
    clLoadStoreWidth = loadStoreWidth;
  }

  explicit TestSpecializeAffineForWMMA(StringRef accumulateType,
                                       unsigned loadStoreWidth) {
    clAccumulateType = accumulateType.str();
    clLoadStoreWidth = loadStoreWidth;
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<gpu::GPUDialect, mlir::vector::VectorDialect>();
  }

  /// Order of loops required in the input IR in their relative order. The
  /// increasing order of values represents increasing depth in the nest, i.e,
  /// TbI is the outermost loop while ThreadK is the innermost loop.
  enum LoopStructure {
    TbI,
    TbJ,
    TbK,
    WarpI,
    WarpJ,
    WarpK,
    ThreadI,
    ThreadJ,
    ThreadK
  };

  /// Number of total Matmul operands.
  constexpr static unsigned kNumMatmulOperands = 4;

  /// Enum containing GEMM operands, usefull in accessing certain containers.
  enum WmmaOps { AOp, BOp, COp, DOp };

  /// CL option to specify the accumulate type to use in matmul.
  Option<std::string> clAccumulateType{
      *this, "accum",
      llvm::cl::desc("Accumulate type(f16/f32) to use for matmul."),
      llvm::cl::init("f32")};

  /// CL option to specify vector width to use for global memory loads.
  Option<unsigned> clLoadStoreWidth{
      *this, "load-store-width",
      llvm::cl::desc(
          "Vector width in bits to use for load/store operations. "
          "Valid widths are 32, 64 and 128. No vectorization if option"
          "is unspecified."),
      llvm::cl::init(0)};

  /// CL option to specify padding factor for A matrix in shared memory.
  Option<unsigned> clPaddingA{
      *this, "padding-a",
      llvm::cl::desc(
          "Padding in number of elements for A matrix. Minimum padding factor "
          "is 8 f16 elements, and must be a multiple of 8."),
      llvm::cl::init(0)};

  /// CL option to specify padding factor for B matrix in shared memory.
  Option<unsigned> clPaddingB{
      *this, "padding-b",
      llvm::cl::desc(
          "Padding in number of elements for B matrix. Minimum padding factor "
          "is 8 f16 elements, and must be a multiple of 8."),
      llvm::cl::init(0)};

  /// Debug option to disable unrolling and unroll-and-jam.
  Option<unsigned> clDisableUnroll{
      *this, "disable-unroll",
      llvm::cl::desc("Disable unroll and unroll-and-jam"),
      llvm::cl::init(false)};
};
} // end anonymous namespace

void TestSpecializeAffineForWMMA::runOnFunction() {
  FuncOp funcOp = getFunction();
  MLIRContext *context = funcOp.getContext();

  Type cdOpType, abOpType;
  abOpType = FloatType::getF16(context);

  Type operandElemTypes[kNumMatmulOperands];

  // Initialize the numElems array to hold the correct number of elements by
  // inspecting the accumulate version.
  if (clAccumulateType.getValue().compare("f16") == 0) {
    cdOpType = abOpType;
  } else if (clAccumulateType.getValue().compare("f32") == 0) {
    cdOpType = FloatType::getF32(context);
  } else {
    assert(false && "unknown accumulate type");
  }

  // Set element type for GPU WmmaOps.
  operandElemTypes[AOp] = abOpType;
  operandElemTypes[BOp] = abOpType;
  operandElemTypes[COp] = cdOpType;
  operandElemTypes[DOp] = cdOpType;

  // Walk and set padding for elements in A and B matrix. Shared memory buffers
  // are modeled as global memrefs, walk and find any getGlobalMemref ops which
  // maybe present.
  if (clPaddingA.getValue() != 0 || clPaddingB.getValue() != 0) {
    funcOp.walk([&](memref::GetGlobalOp getGlobalMemrefOp) {
      if (getGlobalMemrefOp.name().equals("frag_A") ||
          getGlobalMemrefOp.name().equals("frag_B")) {
        std::string newName;
        unsigned paddingFactor;

        if (getGlobalMemrefOp.name().equals("frag_A")) {
          newName = "frag_A_padded";
          paddingFactor = clPaddingA.getValue();
        } else {
          newName = "frag_B_padded";
          paddingFactor = clPaddingB.getValue();
        }

        MemRefType fragType =
            getGlobalMemrefOp.getResult().getType().cast<MemRefType>();
        ArrayRef<int64_t> fragShape = fragType.getShape();

        if (fragType.getAffineMaps().empty() ||
            fragType.getAffineMaps().front().isIdentity()) {
          // Save and Drop the fastest varying dimesnion.
          int64_t leadDimension = fragShape.back();
          fragShape = fragShape.drop_back();
          SmallVector<int64_t> newfragShape(fragShape.begin(), fragShape.end());

          // Insert new dimension which is oldLeadDimension + Padding.
          newfragShape.push_back(leadDimension + paddingFactor);

          // Create new fragType.
          MemRefType paddedAFragType = MemRefType::get(
              newfragShape, fragType.getElementType(), fragType.getAffineMaps(),
              fragType.getMemorySpaceAsInt());

          OpBuilder b(funcOp.getContext());
          b.setInsertionPoint(funcOp);
          b.create<memref::GlobalOp>(funcOp.getLoc(), b.getStringAttr(newName),
                                     b.getStringAttr("public"),
                                     TypeAttr::get(paddedAFragType),
                                     mlir::Attribute(), mlir::UnitAttr());

          b.setInsertionPointAfter(getGlobalMemrefOp);
          Value paddedAFrag = b.create<memref::GetGlobalOp>(
              getGlobalMemrefOp.getLoc(), paddedAFragType, newName);

          getGlobalMemrefOp.replaceAllUsesWith(paddedAFrag);
          getGlobalMemrefOp.erase();
        }
      }
    });
  }

  // Get the root for op first.
  AffineForOp rootForOp;
  funcOp.walk<WalkOrder::PreOrder>([&](AffineForOp forOp) {
    rootForOp = forOp;
    return WalkResult::interrupt();
  });
  if (!rootForOp) {
    LLVM_DEBUG(llvm::dbgs() << "No root for op to work with\n");
    return;
  }

  if (failed(mapAffineNestToWmma(rootForOp, clAccumulateType.getValue(),
                                 !clDisableUnroll))) {
    LLVM_DEBUG(llvm::dbgs() << "Specialization for WMMA failed");
  }
}

namespace mlir {
namespace test {
void registerTestSpecializeAffineForWMMAPass() {
  PassRegistration<TestSpecializeAffineForWMMA>(
      "test-specialize-affine-matmul-for-wmma",
      "specialize affine matmul loops to use GPU WMMA ops");
}
} // namespace test
} // namespace mlir
