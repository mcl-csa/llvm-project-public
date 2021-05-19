//== --TestGPUMatmulBarrierInsertion.cpp - Test sync barrier insertion ---- ==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file inserts synchronization barriers in mamtul pattern that is
// targetted to run on GPUs.
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

#define DEBUG_TYPE "test-gpu-matmul-barrier-insertion"

namespace {
struct TestGPUMatmulBarrierPlacement
    : public PassWrapper<TestGPUMatmulBarrierPlacement, FunctionPass> {
  void runOnFunction() override;

  TestGPUMatmulBarrierPlacement(){};
  TestGPUMatmulBarrierPlacement(const TestGPUMatmulBarrierPlacement &) {}
  explicit TestGPUMatmulBarrierPlacement(StringRef accumulateType) {
    clAccumulateType = accumulateType.str();
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<gpu::GPUDialect, mlir::vector::VectorDialect>();
  }

  /// CL option to specify the accumulate type to use in matmul.
  Option<std::string> clAccumulateType{
      *this, "accum",
      llvm::cl::desc("Accumulate type(f16/f32) to use for matmul."),
      llvm::cl::init("f32")};

  /// Copy loop-nest string identifier.
  static const std::string isParallel;
};
} // namespace

void placeBarriersAutomatically(FuncOp &funcOp, StringRef accumType) {
  Location loc = funcOp.getLoc();
  OpBuilder b(funcOp.getContext());

  // Find the maximum depth in the new IR. This nesting depth calculated
  // here will be later used in dependence analysis.
  unsigned maxNestingDepth = 0;
  funcOp.walk([&](AffineForOp forOp) {
    unsigned curDepth = getNestingDepth(forOp);
    if (curDepth > maxNestingDepth)
      maxNestingDepth = curDepth;
  });

  // Walk and create affine.load/store ops corresponding to each
  // gpuWmmaLoad/Store op. This will help in dependence analysis and barrier
  // placement.
  SmallVector<Operation *> dummyOps;
  b.setInsertionPointToStart(&funcOp.body().front());
  Value dummyToStore;
  if (accumType.compare("f16") == 0)
    dummyToStore = b.create<ConstantOp>(loc, b.getF16FloatAttr(1.0f));
  else
    dummyToStore = b.create<ConstantOp>(loc, b.getF32FloatAttr(1.0f));

  funcOp.walk([&](gpu::SubgroupMmaLoadMatrixOp loadOp) {
    b.setInsertionPointAfter(loadOp);
    dummyOps.push_back(
        b.create<AffineLoadOp>(loc, loadOp.srcMemref(), loadOp.indices())
            .getOperation());
  });

  funcOp.walk([&](gpu::SubgroupMmaStoreMatrixOp storeOp) {
    b.setInsertionPointAfter(storeOp);
    dummyOps.push_back(b.create<AffineStoreOp>(loc, dummyToStore,
                                               storeOp.dstMemref(),
                                               storeOp.indices())
                           .getOperation());
  });

  // Synchronization barriers need to be inserted in two scenarios:-
  // 1.) We have loops following each other at the same nesting depth,
  //      parallelFor {...}   |         parallelFor {...}
  //        ...               |           ...
  //      parallelFor {...}   |         sequentialFor {...}
  // 2.) We have nested loops like,
  //      sequentialFor {
  //        ...
  //        parallelFor {
  //          ...
  //        }
  //      }
  // In the first case we need to only insert synchronization barrier between
  // the loops only when there is a dependence between the accesses in the
  // first and second loop. While in the second case we need to insert barrier
  // after the sequential loop.

  funcOp.dump();

  // Handle the first case of synchronization insertion.
  funcOp.walk([&](AffineForOp forOp) {
    // if (isLoopParallel(forOp)) {
    // Check if a sequential/parallel loop exists at the same nesting depth,
    // after this for. Traverse the enclosing block and check if there is a
    // sequential loop just after this for loop.
    Block *enclosingBlock = forOp->getBlock();
    auto start = enclosingBlock->begin();

    // Advance the iterator till the forOp.
    while (&*start != forOp.getOperation()) {
      ++start;
    }

    // Increment again to get past the forOp.
    ++start;

    bool isBarrierInserted = false;
    while (start != enclosingBlock->end() && !isBarrierInserted) {
      if (auto nextForOp = dyn_cast<AffineForOp>(start)) {
        // Check if dependences exist between this forOp and nextForOp.
        // If yes then we need to place synchronization between forOp and
        // nextForOp.
        // Prepare a list of loadStore ops in both of these forOps.
        SmallVector<Operation *> forOpLoadStores;
        SmallVector<Operation *> nextForOpLoadStores;
        forOp.walk([&](Operation *op) {
          if (isa<AffineLoadOp, AffineStoreOp>(op)) {
            forOpLoadStores.push_back(op);
          }
        });
        nextForOp.walk([&](Operation *op) {
          if (isa<AffineLoadOp, AffineStoreOp>(op)) {
            nextForOpLoadStores.push_back(op);
          }
        });
        // Check if there is a dependence from any op of forOp to any
        // op of nextForOp. If yes then place a barrier and exit.
        for (auto srcOp : forOpLoadStores) {
          MemRefAccess srcAccess(srcOp);
          for (auto dstOp : nextForOpLoadStores) {
            MemRefAccess dstAccess(dstOp);
            unsigned numCommonLoops =
                getNumCommonSurroundingLoops(*srcOp, *dstOp);
            for (unsigned d = 1; d <= numCommonLoops + 1; ++d) {
              FlatAffineConstraints depConstraints;
              SmallVector<DependenceComponent, 2> depComps;
              DependenceResult res = checkMemrefAccessDependence(
                  srcAccess, dstAccess, d, &depConstraints, &depComps);
              if (hasDependence(res)) {
                forOp.getLoc().dump();
                nextForOp.getLoc().dump();
                srcOp->dump();
                dstOp->dump();
                // Place a sync just before the nextForOp. This will ensure
                // synchronization for before entering nextForOp or any
                // other op that follows nextForOp.
                b.setInsertionPoint(nextForOp);
                // b.setInsertionPointAfter(forOp);
                b.create<gpu::BarrierOp>(loc);
                isBarrierInserted = true;
              }
            }
          }
        }
      }
      ++start;
    }
    //}
  });

  // Erase the dummy ops those were inserted to make the affine analysis work.
  for (auto *dummyOp : dummyOps)
    dummyOp->erase();

  // Convert the marked forOps to parallel. Currently, It is assumed that all
  // the transformations done above do not change the nature of loops, i.e., a
  // sequential loop remains sequential and a parallel loop remains parallel.
  // TODO: Currently this works because the parallel loops really doesn't
  // yield anything and affineParallelize also does not handle the case when
  // the ops yield somehing. Whenever this breaks we need to handle this.
  // funcOp.walk([&](AffineForOp forOp) {
  //  if (auto isParallel = forOp->getAttrOfType<BoolAttr>(
  //          TestGPUMatmulBarrierPlacement::isParallel)) {
  //    if (isParallel.getValue() == true)
  //      affineParallelize(forOp);
  //  }
  //});

  // Handle the second case of synchronization insertion.
  // funcOp.walk([&](AffineParallelOp parOp) {
  //  if (auto forOp = dyn_cast<AffineForOp>(parOp->getParentOp())) {
  //    b.setInsertionPointToStart(&forOp.getLoopBody().front());
  //    b.create<gpu::BarrierOp>(loc);
  //  }
  //});

  // funcOp.walk([&](AffineForOp parOp) {
  //  if (BoolAttr isPar = parOp->getAttrOfType<BoolAttr>(isParallel)) {
  //    if (isPar.getValue() == true) {
  //      if (auto forOp = dyn_cast<AffineForOp>(parOp->getParentOp())) {
  //        if (!forOp->getAttrOfType<BoolAttr>(isParallel)) {
  //          b.setInsertionPointToStart(&forOp.getLoopBody().front());
  //          b.create<gpu::BarrierOp>(loc);
  //        }
  //      }
  //    }
  //  }
  //});

  // Remove redundant synchronizations that may have been inserted.
  DenseSet<Operation *> barriersToErase;
  funcOp.walk([&](gpu::BarrierOp barrier) {
    Block *enclosingBlock = barrier->getBlock();
    auto start = enclosingBlock->begin();

    // Advance the iterator till the barrier.
    while (&*start != barrier.getOperation()) {
      ++start;
    }

    // Increment again to get past the barrier.
    ++start;

    // Continue till a barrier is found.
    while (start != enclosingBlock->end() && isa<gpu::BarrierOp>(*start)) {
      barriersToErase.insert(&*start);
      ++start;
    }
  });

  // Erase the collected duplicates.
  for (auto *barrier : barriersToErase)
    barrier->erase();
}

void placeBarriersByHeuristics(FuncOp funcOp) {
  OpBuilder b(funcOp.getContext());

  // Place synchronization after the compute loops.
  funcOp.walk([&](AffineForOp forOp) {
    auto boolAttr = forOp->getAttrOfType<BoolAttr>("isComputeLoopNest");
    if (boolAttr && boolAttr.getValue() == true) {
      b.setInsertionPoint(forOp);
      b.create<gpu::BarrierOp>(forOp.getLoc());
    }
  });
}

void TestGPUMatmulBarrierPlacement::runOnFunction() {
  FuncOp funcOp = getFunction();
  // placeBarriersAutomatically(funcOp, clAccumulateType.getValue());
  placeBarriersByHeuristics(funcOp);
}

namespace mlir {
void registerTestGPUMatmulBarrierPlacement() {
  PassRegistration<TestGPUMatmulBarrierPlacement>(
      "test-gpu-matmul-barrier-insertion",
      "Insert synchronization barriers in GPU matmul");
}
} // namespace mlir

const std::string TestGPUMatmulBarrierPlacement::isParallel = "isParallel";
