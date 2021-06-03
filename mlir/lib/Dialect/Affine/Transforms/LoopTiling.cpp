//===- LoopTiling.cpp --- Loop tiling pass ------------------------------*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements a pass to tile loop nests.
//
//===----------------------------------------------------------------------===//

#include "PassDetail.h"
#include "mlir/Analysis/AffineAnalysis.h"
#include "mlir/Analysis/AffineStructures.h"
#include "mlir/Analysis/LoopAnalysis.h"
#include "mlir/Analysis/Utils.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Affine/IR/AffineValueMap.h"
#include "mlir/Dialect/Affine/Passes.h"
#include "mlir/IR/BlockAndValueMapping.h"
#include "mlir/IR/Builders.h"
#include "mlir/Transforms/LoopUtils.h"
#include "mlir/Transforms/Utils.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
using namespace mlir;

#define DEBUG_TYPE "affine-loop-tile"

namespace {

/// A pass to perform loop tiling on all suitable loop nests of a Function.
struct LoopTiling : public AffineLoopTilingBase<LoopTiling> {
  LoopTiling() = default;
  explicit LoopTiling(uint64_t cacheSizeBytes, bool avoidMaxMinBounds = true)
      : avoidMaxMinBounds(avoidMaxMinBounds) {
    this->cacheSizeInKiB = cacheSizeBytes / 1024;
  }

  void runOnFunction() override;
  void getTileSizes(ArrayRef<AffineForOp> band, unsigned tilingLevelIndex,
                    SmallVectorImpl<unsigned> *tileSizes);
  mlir::LogicalResult checkTileSizesValidity(FuncOp func);

  /// Default tile size is chosen to be 16 because the wmma intrinsic requires
  /// it be 16 or more.
  constexpr static unsigned kDefaultTileSize = 16;

  /// Default tiling levels if nothing is provided.
  constexpr static unsigned kDefaultNumTilingLevels = 1;

  // If true, tile sizes are set to avoid max/min in bounds if possible.
  bool avoidMaxMinBounds = true;
};

} // end anonymous namespace

/// Creates a pass to perform loop tiling on all suitable loop nests of a
/// Function.
std::unique_ptr<OperationPass<FuncOp>>
mlir::createLoopTilingPass(uint64_t cacheSizeBytes) {
  return std::make_unique<LoopTiling>(cacheSizeBytes);
}
std::unique_ptr<OperationPass<FuncOp>> mlir::createLoopTilingPass() {
  return std::make_unique<LoopTiling>();
}

/// Reduces each tile size to the largest divisor of the corresponding trip
/// count (if the trip count is known).
static void adjustToDivisorsOfTripCounts(ArrayRef<AffineForOp> band,
                                         SmallVectorImpl<unsigned> *tileSizes) {
  assert(band.size() == tileSizes->size() && "invalid tile size count");
  for (unsigned i = 0, e = band.size(); i < e; i++) {
    unsigned &tSizeAdjusted = (*tileSizes)[i];
    Optional<uint64_t> mayConst = getConstantTripCount(band[i]);
    if (!mayConst)
      continue;
    // Adjust the tile size to largest factor of the trip count less than
    // tSize.
    uint64_t constTripCount = mayConst.getValue();
    if (constTripCount > 1 && tSizeAdjusted > constTripCount / 2)
      tSizeAdjusted = constTripCount / 2;
    while (constTripCount % tSizeAdjusted != 0)
      tSizeAdjusted--;
  }
}

// Returns tile sizes to use. Checks CL options; if none are specified, sets it
// based on a simple model that looks at the memory footprint and determines
// tile sizes assuming identity accesses / 1:1 tile size proportional footprint
// along each of the dimensions being tiled. Second argument `curTilingLevel`[0,
// numTilingLevels - 1] represents the tiling level for which this method is
// called.
// TODO: evolve this model. Tile size determination is a large area
// to play with in general.
void LoopTiling::getTileSizes(ArrayRef<AffineForOp> band,
                              unsigned tilingLevelIndex,
                              SmallVectorImpl<unsigned> *tileSizes) {
  if (band.empty())
    return;

  // Use command-line tileSize if specified. Scale Down the tileSizes for each
  // level by a factor equal to the tilingLevel.
  if (tileSize) {
    tileSizes->assign(
        band.size(),
        std::max(1U, static_cast<unsigned>(tileSize / (tilingLevelIndex + 1))));
    return;
  }

  // Use tileSizes and fill them with default tile size if it's short.
  if (!this->tileSizes.empty()) {
    // Fill in with the tile sizes supplied.
    if (this->tileSizes.size() > (tilingLevelIndex * band.size())) {
      for (unsigned i = tilingLevelIndex * band.size(),
                    e = this->tileSizes.size();
           i < e; ++i) {
        tileSizes->push_back(this->tileSizes[i]);
      }
    }
    // If sufficient tile sizes were provided then return.
    if (tileSizes->size() == band.size())
      return;

    // Fill the remaining places with default tile size.
    tileSizes->resize(band.size(), kDefaultTileSize);
    return;
  }
  tileSizes->resize(band.size());

  // The first loop in the band.
  AffineForOp rootForOp = band[0];
  (void)rootForOp;

  // Obtain memory footprint and set tile sizes so that a tile fits in
  // the cache size. This is an approximation with the assumption that the
  // footprint increases with the tile size linearly in that dimension (i.e.,
  // assumes one-to-one access function).
  Optional<int64_t> fp = getMemoryFootprintBytes(band[0], 0);
  if (!fp) {
    // Fill with default tile sizes if footprint is unknown.
    std::fill(tileSizes->begin(), tileSizes->end(),
              LoopTiling::kDefaultTileSize);
    adjustToDivisorsOfTripCounts(band, tileSizes);
    LLVM_DEBUG(
        rootForOp.emitWarning("memory footprint unknown: using default tile "
                              "sizes adjusted to trip count divisors"));
    return;
  }

  // Check how many times larger the cache size is when compared to footprint.
  uint64_t cacheSizeBytes = cacheSizeInKiB * 1024;
  uint64_t excessFactor = llvm::divideCeil(fp.getValue(), cacheSizeBytes);
  if (excessFactor <= 1) {
    // No need of any tiling - set tile size to 1.
    std::fill(tileSizes->begin(), tileSizes->end(), 1);
    return;
  }

  // Divide all loops equally in an attempt to reduce footprint.
  // TODO: this is approximate. Ideally, obtain reuse factor /
  // profitability along each dimension and weight tile sizes based on that as
  // one possible approach. Or compute a polynomial in tile sizes and solve for
  // it.

  // For an n-d tileable band, compute the n^th root of the excess.
  unsigned tSize =
      static_cast<unsigned>(floorl(std::pow(excessFactor, 1.0 / band.size())));
  // We'll keep a running product to determine the last tile size better.
  unsigned cumulProductOfTileSizes = 1;
  for (unsigned i = 0, e = band.size(); i < e; i++) {
    if (i < e - 1)
      (*tileSizes)[i] = tSize;
    else
      // Set last tile size to cover the balance.
      (*tileSizes)[i] = std::max(
          1U, static_cast<unsigned>(excessFactor / cumulProductOfTileSizes));
    cumulProductOfTileSizes *= (*tileSizes)[i];
  }
  if (avoidMaxMinBounds)
    adjustToDivisorsOfTripCounts(band, tileSizes);
}

/// Checks the validity of tile sizes. There are two conditions for tile sizes
/// to be valid: 1.) The thread block level tile size must be a multiple of warp
/// level tile size and strictly greater than warp level tile size. 2.) The warp
/// tile size must be at least 16 for each of the dimensions.
LogicalResult LoopTiling::checkTileSizesValidity(FuncOp func) {
  auto checkCondition = [](unsigned tbDim, unsigned wDim, FuncOp func,
                           Location loc,
                           const ListOption<unsigned> &tileSizes) {
    if (tileSizes[wDim] < 16)
      func.emitError("warp level tile size must be at least 16");
    if (tileSizes[tbDim] <= tileSizes[wDim] ||
        (tileSizes[tbDim] % tileSizes[wDim] != 0))
      func.emitError(
          "thread block level tile size must be greater than the warp level "
          "tile size and it must be a multiple of warp level tile size");
  };
  if (tileSizes.size() > 3)
    checkCondition(0, 3, func, func.getLoc(), tileSizes);
  if (tileSizes.size() > 4)
    checkCondition(1, 4, func, func.getLoc(), tileSizes);
  if (tileSizes.size() > 5)
    checkCondition(2, 5, func, func.getLoc(), tileSizes);
  return LogicalResult::success();
}

void LoopTiling::runOnFunction() {
  // Bands of loops to tile.
  std::vector<SmallVector<AffineForOp, 6>> bands;
  FuncOp func = getFunction();
  getTileableBands(func, &bands);

  // Number of times to tile a band.
  unsigned numTilingLevels;
  numTilingLevels =
      tilingLevels ? tilingLevels : LoopTiling::kDefaultNumTilingLevels;
  // Check for tile sizes validity only if the tiling levels is greater than
  // one and perform tiling only if the tile sizes are valid.
  if (numTilingLevels > 1 && failed(checkTileSizesValidity(func)))
    return;
  // Tile each band.
  for (auto &band : bands) {
    unsigned bandSize = band.size();
    AffineForOp rootForOp = band[0];
    SmallVector<AffineForOp, 6> tiledNest;
    // Tile the band `numTilingLevels` times.
    for (unsigned curTilingLevel = 0; curTilingLevel < numTilingLevels;
         ++curTilingLevel) {
      // Set up tile sizes; fill missing tile sizes at the end with default
      // tile size or tileSize if one was provided.
      SmallVector<unsigned, 6> tileSizes;
      getTileSizes(band, curTilingLevel, &tileSizes);
      if (llvm::DebugFlag) {
        auto diag = rootForOp.emitRemark("using tile sizes [");
        for (unsigned tSize : tileSizes)
          diag << tSize << ' ';
        diag << "] for tiling level " << curTilingLevel << "\n";
      }
      // If not the first level of tiling then take out the loops to tile
      // next form the already tiled loop nest.
      if (curTilingLevel != 0) {
        band.clear();
        band.insert(band.begin(), tiledNest.begin() + bandSize,
                    tiledNest.end());
      }
      if (failed(tilePerfectlyNested(band, tileSizes, &tiledNest,
                                     this->hasToDoRelativeIndexing)))
        return signalPassFailure();
      // Separate full and partial tiles. Separation only supported at the
      // last level of tiling.
      if (separate && (curTilingLevel == numTilingLevels - 1)) {
        auto intraTileLoops =
            MutableArrayRef<AffineForOp>(tiledNest).drop_front(band.size());
        (void)separateFullTiles(intraTileLoops);
      }
    }
  }
}

constexpr unsigned LoopTiling::kDefaultTileSize;
constexpr unsigned LoopTiling::kDefaultNumTilingLevels;
