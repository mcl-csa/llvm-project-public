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
#include "mlir/Analysis/LoopAnalysis.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/GPU/GPUDialect.h"
#include "mlir/IR/BlockAndValueMapping.h"
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

  assert(inputMemrefs.size() >= 2 &&
         "Expected at least two memrefs corresponding to input matrices.");
  assert(outputMemRef && "Expected one memref corresponding to output matrix.");

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

/// Find tile space loops. The three outermost loops are the tile space
/// loops. Three loops are the minimum number of loops that are to be present
/// in matrix multiplication. Since copy loops may also be present in the code,
/// The input may not be perfectly nested. Assuming that the copy loops are
/// annotated and we find the loops without such an attribute from outermost to
/// innermost in `computeLoops`.
//  TODO/FIXME: make this more systematic.
static void findComputeLoops(AffineForOp rootForOp,
                             SmallVector<AffineForOp> &computeLoops) {
  bool nestedForExists = true;
  while (nestedForExists) {
    nestedForExists = false;
    computeLoops.push_back(rootForOp);
    // Scan for other for loops in the body which are not copy loops.
    for (auto forOp : rootForOp.getOps<AffineForOp>()) {
      // FIXME: get rid of hardcoded attributes
      // isCopyLoopNest/isComputeLoopNest.
      auto attr = forOp->getAttrOfType<BoolAttr>("isCopyLoopNest");
      if (!attr || !attr.getValue()) {
        // Make this forOp the next root.
        // TODO: Insert assertion for multiple non-copy loop children of this
        // for op.
        rootForOp = forOp;
        nestedForExists = true;
      }
    }
  }
}

/// Number of total Matmul operands.
constexpr static unsigned kNumMatmulOperands = 4;

/// Enum containing GEMM operands, usefull in accessing certain containers.
enum WmmaOps { AOp, BOp, COp, DOp };

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

/// Constant representing the number of loops in untiled matmul.
constexpr static unsigned kNumIntialLoops = 3;

/// Constant representing the shape of WMMA op in M dimension.
constexpr static unsigned kWMMAM = 16;

/// Constant representing the shape of WMMA op in N dimension.
constexpr static unsigned kWMMAN = 16;

/// Constant representing the shape of WMMA op in K dimension.
constexpr static unsigned kWMMAK = 16;

/// Check that the loops are in the desired ordered, i.e.,
///		    Inter Thread-Block loops(i,j,k)
///		      Inter Warp loops(ii, jj, kk)
///			Intra Warp loops(iii, jjj, kkk)
static LogicalResult inspectTileStructure(ArrayRef<AffineForOp> computeLoops,
                                          ArrayRef<Value> loopsIVs) {
  unsigned curMapStage = 0;
  for (AffineForOp loop : computeLoops.drop_front(kNumIntialLoops)) {
    if (loop.hasConstantBounds())
      continue;
    // Insert lower/upper bound operands.
    SmallVector<Value> ivOperands(loop.getLowerBoundOperands());
    ivOperands.append(loop.getUpperBoundOperands().begin(),
                      loop.getUpperBoundOperands().end());

    // The loops must be dependent from the outermost to the innermost loops.
    bool foundDependentLoopIV = false;
    for (Value operand : ivOperands) {
      if (operand == loopsIVs[curMapStage] ||
          operand == loopsIVs[curMapStage + kNumIntialLoops])
        foundDependentLoopIV = true;
    }

    if (foundDependentLoopIV) {
      computeLoops.front()->emitError(
          "Recipe for tensor core matmul failed, improperly tiled loop nest");
      return failure();
    }
    ++curMapStage;
    curMapStage %= kNumIntialLoops;
  }
  return success();
}

/// Checks whether a given op is hoistable with respect to a forOp.
static bool canBeHoisted(Operation *op, AffineForOp forOp,
                         SmallVector<AffineMap> &affineMaps,
                         SmallVector<SmallVector<Value>> &mapOprs) {
  auto isIndependentOfLoopIV = [&](ValueRange operands) {
    for (Value operand : operands) {
      // TODO: Handle cases where the operands to the op may not be results of
      // AffineApplyOp.
      if (auto defOp = operand.getDefiningOp<AffineApplyOp>()) {
        AffineMap inxMap = defOp.getAffineMap();
        SmallVector<Value> mapOpr(defOp.getMapOperands());
        fullyComposeAffineMapAndOperands(&inxMap, &mapOpr);
        canonicalizeMapAndOperands(&inxMap, &mapOpr);
        // After compostion check whether all the operands are independant of
        // the surrounding AffineForOp.
        if (!llvm::is_contained(mapOpr, forOp.getInductionVar())) {
          affineMaps.push_back(inxMap);
          mapOprs.push_back(mapOpr);
        } else
          return false;
      }
    }
    return true;
  };

  if (auto mmaOp = dyn_cast<gpu::SubgroupMmaLoadMatrixOp>(op)) {
    // Check if the indices of the mmaLoadOp have any dependency to an affine
    // apply op.
    return isIndependentOfLoopIV(mmaOp->getOperands());
  }

  return false;
}

/// Fetches all the uses of an op until the use converges into an op
/// with no results.
static void getRecursiveUses(
    Operation *source, Operation *op, Operation *target,
    SmallVector<std::pair<Operation *, Operation *>> &loadStoreOps) {
  auto allUses = op->getUses();
  if (allUses.empty())
    return;
  for (OpOperand &use : allUses) {
    // Inspect ops wihtout any regions, i.e., avoid forops, ifops etc.
    if (use.getOwner()->getNumRegions() == 0) {
      if (use.getOwner() == target) {
        loadStoreOps.push_back(std::make_pair(source, target));
      } else {
        getRecursiveUses(source, use.getOwner(), target, loadStoreOps);
      }
    }
  }
}

/// Find all pairs of load/store ops that can be moved and move them just
/// before/after the forOp.
static void findAndMoveLoadStorePairs(
    AffineForOp forOp, OpBuilder &b,
    SmallVector<std::pair<Operation *, Operation *>> &loadStoreOps) {
  auto &loopBody = forOp.getLoopBody();

  SmallVector<gpu::SubgroupMmaLoadMatrixOp> loadOps;
  SmallVector<gpu::SubgroupMmaStoreMatrixOp> storeOps;

  // Collect all the WMMALoad/StoreOps in the body of the loop.
  for (auto &op : loopBody.getOps()) {
    if (auto mmaOp = dyn_cast<gpu::SubgroupMmaLoadMatrixOp>(op))
      loadOps.push_back(mmaOp);
    else if (auto mmaOp = dyn_cast<gpu::SubgroupMmaStoreMatrixOp>(op))
      storeOps.push_back(mmaOp);
  }

  // Find pairs of load/stores such that the value being stored is somehow
  // dependent on the load.
  for (auto loadOp : loadOps) {
    for (auto storeOp : storeOps) {
      getRecursiveUses(loadOp.getOperation(), loadOp.getOperation(),
                       storeOp.getOperation(), loadStoreOps);
    }
  }

  // If no load/store pair found, then return.
  if (loadStoreOps.empty())
    return;

  SmallVector<Value> newLoadOps;
  SmallVector<Operation *> newStoreOps;
  SmallVector<Operation *> movableOps;
  SmallVector<SmallVector<Value>> newIndices;

  // Check if the load/store op pairs are hoistable.
  for (auto &memOp : loadStoreOps) {
    SmallVector<AffineMap> indexMaps;
    SmallVector<SmallVector<Value>> mapOprs;
    // TODO: Insert check for storeOp also.
    if (canBeHoisted(memOp.first, forOp, indexMaps, mapOprs)) {
      movableOps.push_back(memOp.first);

      b.setInsertionPoint(forOp);
      SmallVector<Value> indices;
      for (auto inx : llvm::zip(indexMaps, mapOprs)) {
        AffineMap affMap;
        SmallVector<Value> oprs;
        std::tie(affMap, oprs) = inx;
        indices.push_back(
            b.create<AffineApplyOp>(forOp.getLoc(), affMap, oprs));
      }

      // Store these new indices for use later, while moving the store ops.
      newIndices.push_back(indices);

      // To move this pair of ops we need to to move the operands too. We have
      // already fetched the operands using the affine map composition and we
      // can safely create the same ops outside this for loop. Create new load
      // ops. These ops will be used as iter_args for the forOp.
      auto origLoadop = cast<gpu::SubgroupMmaLoadMatrixOp>(memOp.first);
      newLoadOps.push_back(b.create<gpu::SubgroupMmaLoadMatrixOp>(
          forOp.getLoc(), origLoadop->getResultTypes()[0],
          origLoadop.srcMemref(), indices, origLoadop.leadDimension()));
    }
  }

  if (movableOps.empty())
    return;

  // Insert newly created ops as operands for the for op.
  SmallVector<Value> newOperands(forOp.getLowerBoundOperands());
  newOperands.append(forOp.getUpperBoundOperands().begin(),
                     forOp.getUpperBoundOperands().end());
  newOperands.append(newLoadOps);
  forOp->setOperands(newOperands);

  // Add newly created ops as arguments to the basic block containing the loop
  // body.
  SmallVector<BlockArgument> newArgs;
  for (auto newOp : newLoadOps) {
    newArgs.push_back(loopBody.front().addArgument(newOp.getType()));
  }

  // Set the newly created ops as iter_args for the forOp.
  for (unsigned i = 0, e = movableOps.size(); i < e; ++i) {
    movableOps[i]->getResult(0).replaceAllUsesWith(newArgs[i]);
  }

  // Create a new affine forOp with body and clone the ops from the original
  // nest to this loop and then erase the original nest.
  b.setInsertionPointAfter(forOp);
  AffineForOp newForop = b.create<AffineForOp>(
      forOp.getLoc(), forOp.getLowerBoundOperands(), forOp.getLowerBoundMap(),
      forOp.getUpperBoundOperands(), forOp.getUpperBoundMap(), forOp.getStep(),
      forOp.getIterOperands(),
      [&](OpBuilder &builder, Location loc, Value iv, ValueRange iterArgs) {
        builder.create<AffineYieldOp>(loc, iterArgs);
      });

  // Preserve computeLoopNest Attribute if present.
  newForop->setAttrs(forOp->getAttrs());

  // Clone the body of the original forop into the newly create for op. First
  // add the iterArgs and loopIV into the clonigMap.
  BlockAndValueMapping mapping;
  mapping.map(forOp.getInductionVar(), newForop.getInductionVar());
  mapping.map(forOp.getRegionIterArgs(), newForop.getRegionIterArgs());

  b.setInsertionPointToStart(&newForop.getLoopBody().front());

  // Create storeOps just after the forOp. The operands to these store ops
  // will be the results yielded by the forOp.
  SmallVector<gpu::SubgroupMmaStoreMatrixOp> clonedStoreOps;
  for (auto &op : forOp.getLoopBody().front().without_terminator()) {
    Operation *clonedOp = b.clone(op, mapping);
    if (auto storeOp = dyn_cast<gpu::SubgroupMmaStoreMatrixOp>(clonedOp))
      clonedStoreOps.push_back(storeOp);
  }

  // Erase the original for op.
  forOp.erase();

  // Set the correct operands for the yield op.
  SmallVector<Value> toYield;
  AffineYieldOp yieldOp =
      dyn_cast<AffineYieldOp>(newForop.getLoopBody().front().back());

  for (auto op : clonedStoreOps) {
    toYield.push_back(op.src());
  }

  yieldOp->setOperands(toYield);

  // Place newly created storeOps just outside the for ops body and set their
  // operands to be the ops yeileded by the newly created AffineForOp.
  SmallVector<Value> newForOpRes(newForop.getResults());
  b.setInsertionPointAfter(newForop);
  for (auto resSrc : llvm::zip(newForOpRes, clonedStoreOps, newIndices)) {
    Value newRes;
    gpu::SubgroupMmaStoreMatrixOp clonedStoreOp;
    SmallVector<Value> indices;
    std::tie(newRes, clonedStoreOp, indices) = resSrc;
    newStoreOps.push_back(b.create<gpu::SubgroupMmaStoreMatrixOp>(
        newForop.getLoc(), newRes, clonedStoreOp.dstMemref(), indices,
        clonedStoreOp.leadDimension()));
    clonedStoreOp.erase();
  }

  // Update loadStoreOps to contain newly created load/store ops. Newly created
  // load/store ops are always candidates for further movement.
  loadStoreOps.clear();
  for (auto lSPair : llvm::zip(newLoadOps, newStoreOps)) {
    Value load;
    Operation *store;
    std::tie(load, store) = lSPair;
    loadStoreOps.push_back(std::make_pair(load.getDefiningOp(), store));
  }
}

/// Utility to move invariant load/store pairs with respect to a forOp
/// before/after respectively. The utility finds loads such that the result of
/// the store is somehow dependent on the what was loaded. The loads are moved
/// just before the forOp and the load results are set as iter_args for the
/// forOp. The result of the computation is then yielded by the forOp and result
/// of the forOp is then store by the storeOps which are moved just after the
/// forOp. In this way redundant load/stores can be moved out of a given forOp.
static void moveInvariantLoadStorePairs(FuncOp funcOp, OpBuilder b) {
  SmallVector<std::pair<Operation *, Operation *>> loadStoreOps;
  funcOp->walk([&](AffineForOp forOp) {
    findAndMoveLoadStorePairs(forOp, b, loadStoreOps);
  });
}

LogicalResult mlir::mapAffineNestToWmma(AffineForOp rootForOp,
                                        StringRef accumulateType,
                                        bool enableUnroll) {
  auto funcOp = rootForOp->getParentOfType<FuncOp>();

  /// Constant representing the maximum number of tiled loops that can be
  /// present in the input IR.
  constexpr static unsigned kMaxTiledLoops = 9;

  MLIRContext *context = funcOp.getContext();

  Type cdOpType, abOpType;
  abOpType = FloatType::getF16(context);

  Type operandElemTypes[kNumMatmulOperands];

  // Initialize the numElems array to hold the correct number of elements by
  // inspecting the accumulate version.
  if (accumulateType.compare("f16") == 0) {
    cdOpType = abOpType;
  } else if (accumulateType.compare("f32") == 0) {
    cdOpType = FloatType::getF32(context);
  } else {
    assert(false && "unknown accumulate type");
  }

  // Set element type for GPU WmmaOps.
  operandElemTypes[AOp] = abOpType;
  operandElemTypes[BOp] = abOpType;
  operandElemTypes[COp] = cdOpType;
  operandElemTypes[DOp] = cdOpType;

  // Find All the compute loops in the rootForOp.
  SmallVector<AffineForOp> computeLoops;
  findComputeLoops(rootForOp, computeLoops);

  // The expected number of loops is 9 i.e., all matmul loops are tiled two
  // times.
  // TODO: Add cases when all the loops are not tiled.
  assert(computeLoops.size() == kMaxTiledLoops &&
         "Recipe for tensor core matmul failed, improperly tiled loop nest");

  // Find the different type of loops. When mapped to GPU there may be three
  // different types of loops present. 1.) Inter ThreadBlock-tile loops, 2.)
  // Inter Warp-tile loops 3.) Intra Warp-tile loops.
  SmallVector<Value> loopsIVs;
  for (auto loop : computeLoops) {
    loopsIVs.push_back(loop.getInductionVar());
  }

  // Check the tiling order of loops.
  if (failed(inspectTileStructure(computeLoops, loopsIVs)))
    return failure();

  // Insert GPU MMA ops in the innermost loop nest. This involves changing the
  // loop steps of the surrounding loops. To the size of WMMA operation and
  // then caluclating the right indices for load/store operations and also
  // identifying the leading dimensions of the source/destination memrefs.
  // First change the loop steps to MMA size.
  // TODO: Add CL option to get WMMA size once more versions of WMMA ops are
  // introcuced.
  computeLoops[ThreadI].setStep(kWMMAM);
  computeLoops[ThreadJ].setStep(kWMMAN);
  computeLoops[ThreadK].setStep(kWMMAK);

  // Create a new loop which will be the innermostLoop. This loop will have
  // gpuSubgroup WmmaOps instead of scalar loads/stores and compute ops.
  AffineForOp innermostLoop = computeLoops[ThreadK];
  Block &body = innermostLoop.getLoopBody().front();
  OpBuilder b(rootForOp.getContext());

  b.setInsertionPointAfter(innermostLoop);
  AffineForOp newInnermostLoop =
      b.create<AffineForOp>(rootForOp.getLoc(), 0, 0, innermostLoop.getStep());

  // Copy bounds from the original innermostLoop.
  newInnermostLoop.setLowerBound(innermostLoop.getLowerBoundOperands(),
                                 innermostLoop.getLowerBoundMap());
  newInnermostLoop.setUpperBound(innermostLoop.getUpperBoundOperands(),
                                 innermostLoop.getUpperBoundMap());

  // Insert wmmaOps in the newly created loop.
  b.setInsertionPointToStart(&newInnermostLoop.getLoopBody().front());
  Location loc = rootForOp.getLoc();
  SmallVector<Value> wmmaOps;
  unsigned numOpsProcessed = 0;

  // Helper to emit indices as affine.apply's for a load/store op.
  auto emitIndices = [&](SmallVector<Value> &index,
                         SmallVector<Value> &valueOperands, AffineMap opMap,
                         ValueRange operands) {
    for (auto operand : operands) {
      if (operand == innermostLoop.getInductionVar()) {
        valueOperands.push_back(newInnermostLoop.getInductionVar());
      } else {
        valueOperands.push_back(operand);
      }
    }

    // Emit affine.apply's for each result expr in the map.
    for (unsigned i = 0, e = opMap.getNumResults(); i < e; ++i) {
      index.push_back(
          b.create<AffineApplyOp>(loc, opMap.getSubMap(i), valueOperands));
    }
  };

  /// String array representing the standard operands of matmul.
  const std::string kOpsName[4] = {"AOp", "BOp", "COp", "DOp"};

  // Now try to get the source/destination matrices for matmul in the original
  // innermostLoop and create corresponding ops in the new innermostLoop. We
  // do so by inspecting the innermost loop body. We'll assume the first load
  // to be the `A` operand, second to be the `B` operand, Third to be the `C`
  // operand. We also emit affine.apply's for each index of the load/store op
  // in order to use them as indices for gpuWmmaOps.
  for (auto op = body.begin(), e = body.end(); op != e; ++op) {
    SmallVector<Value> index;
    SmallVector<Value> valueOperands;
    AffineMap opMap;
    ValueRange operands;
    if (auto loadOp = dyn_cast<AffineLoadOp>(op)) {
      assert(numOpsProcessed <= 2 &&
             "Recipe for tensor core matmul failed, "
             "innermost body probably doesn't represent a matmul");
      opMap = loadOp.getAffineMap();
      operands = loadOp.indices();

      // Emit affine.apply's for each index.
      emitIndices(index, valueOperands, opMap, operands);
      // Cases when higher dimensional memrefs are present are unimplemented.
      if (index.size() != 2)
        return failure();

      MemRefType opType = loadOp.memref().getType().cast<MemRefType>();
      if (!opType.getAffineMaps().empty() &&
          !opType.getAffineMaps().front().isIdentity()) {
        // TODO: Handle such cases.
      } else {
        // Create GPU WMMA loadOp.
        wmmaOps.push_back(b.create<gpu::SubgroupMmaLoadMatrixOp>(
            loc,
            gpu::MMAMatrixType::get({16, 16}, operandElemTypes[numOpsProcessed],
                                    kOpsName[numOpsProcessed]),
            loadOp.memref(), index, b.getIndexAttr(opType.getDimSize(1))));
        ++numOpsProcessed;
      }
    } else if (auto storeOp = dyn_cast<AffineStoreOp>(op)) {
      if (numOpsProcessed == 3) {
        // If this is the last loadOp that is being processed, Then we can
        // emit the compute and store op also.
        wmmaOps.push_back(b.create<gpu::SubgroupMmaComputeOp>(
            loc,
            gpu::MMAMatrixType::get({16, 16}, operandElemTypes[COp],
                                    kOpsName[COp]),
            wmmaOps[numOpsProcessed - 3], wmmaOps[numOpsProcessed - 2],
            wmmaOps[numOpsProcessed - 1]));

        opMap = storeOp.getAffineMap();
        operands = storeOp.indices();

        // Emit affine.apply's for each index.
        emitIndices(index, valueOperands, opMap, operands);

        MemRefType opType = storeOp.memref().getType().cast<MemRefType>();
        b.create<gpu::SubgroupMmaStoreMatrixOp>(
            loc, wmmaOps[numOpsProcessed], storeOp.memref(), index,
            b.getIndexAttr(opType.getDimSize(1)));
      }
    }
  }

  // Erase this for op and the newInnermostLoop at the correct position.
  innermostLoop.erase();
  computeLoops[ThreadK] = newInnermostLoop;

  // Sink down the K-loop to an inner level, just inside the warp space `i`
  // and `j` loops. The operations in `k` loop, before the warp space `i`,
  // must be moved with it. This, makes those operations get executed more
  // number of times than before, but, when these loops are mapped to a warp
  // in GPU, It is ideally expected that they have a single iteration. So They
  // will get executed only once.
  b.setInsertionPointToStart(&computeLoops[WarpJ].getLoopBody().front());

  SmallVector<Operation *> toErase;
  Block &kBody = computeLoops[TbK].getLoopBody().front();

  // Gather all operations between the global `k` loop and the warp-space
  // loops. Clone them just inside the sunken `k` loop.
  for (auto op = kBody.begin(), e = kBody.end();
       op != e && &*op != computeLoops[WarpI].getOperation(); ++op) {
    // TODO: Inspect the loop structure and guarantee that only copy loops
    // exist here.
    b.clone(*op);
    toErase.push_back(&*op);
  }

  // Erase all the gathered ops.
  for (auto *op : toErase)
    op->erase();

  // Interchange the warp space loops with `k` loop.
  interchangeLoops(computeLoops[TbK], computeLoops[WarpI]);
  interchangeLoops(computeLoops[TbK], computeLoops[WarpJ]);

  // Update positions in the original list.
  std::swap(computeLoops[WarpI], computeLoops[TbK]);
  std::swap(computeLoops[WarpK], computeLoops[WarpI]);

  // Permute the innermost loop nest to bring `k` at the outermost position.
  MutableArrayRef<AffineForOp> toPermute(computeLoops.begin(),
                                         computeLoops.end());

  permuteLoops(toPermute.drop_front(6), {1, 2, 0});

  // Update positions in the original list.
  std::swap(computeLoops[ThreadK], computeLoops[ThreadJ]);
  std::swap(computeLoops[ThreadI], computeLoops[ThreadJ]);

  // Unroll-Jam the innermost `i` loop by factor equal to trip count.
  if (enableUnroll && getConstantTripCount(computeLoops[ThreadJ]).hasValue()) {
    auto status = loopUnrollJamByFactor(
        computeLoops[ThreadJ],
        getConstantTripCount(computeLoops[ThreadJ]).getValue());
    assert(succeeded(status) && "Unable to unroll loop, please inspect the IR");
  }

  // Unroll the innermostLoop completely.
  if (enableUnroll) {
    auto status = loopUnrollFull(computeLoops[ThreadK]);
    assert(succeeded(status) && "Unable to unroll loop, please inspect the IR");

    // Promote the now innermostLoop, which is the `k` loop.
    (void)promoteIfSingleIteration(computeLoops[ThreadI]);

    // We need to move ops from inside to the outside level which are invariant
    // on the surrounding loop ivs. We handle side effecting operations in a
    // special way, if the side effecting operations are loop invariant than
    // they can be moved out. If the side effecting operations read and write to
    // the same location then they can still be moved out of the loops using
    // appropriate yield ops and also supplying loaded values back into the
    // invariant loop as iter_args. This would also require substituing the
    // usign values with the iter_arg.
    moveInvariantLoadStorePairs(funcOp, b);
  }
  return success();
}
