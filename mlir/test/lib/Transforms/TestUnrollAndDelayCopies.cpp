#include "mlir/Dialect/GPU/GPUDialect.h"
#include "mlir/Dialect/SCF/SCF.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/Transforms/LoopUtils.h"
#include "mlir/Transforms/Passes.h"
#include "llvm/Support/CommandLine.h"

using namespace mlir;

namespace {
struct TestUnrollAndDelayCopies
    : public PassWrapper<TestUnrollAndDelayCopies, FunctionPass> {
  TestUnrollAndDelayCopies() = default;
  TestUnrollAndDelayCopies(const TestUnrollAndDelayCopies &pass) {}
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<gpu::GPUDialect>();
  }

  void runOnFunction() override;

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

void TestUnrollAndDelayCopies::runOnFunction() {
  FuncOp funcOp = getFunction();
  SmallPtrSet<Operation *, 2> copyLoops;

  // Find and unroll copy loops.
  funcOp.walk([&](scf::ForOp copyLoop) {
    BoolAttr isCopyLoop = copyLoop->getAttrOfType<BoolAttr>(
        TestUnrollAndDelayCopies::clIdentifyCopyLoopNestAttribute);
    if (isCopyLoop && isCopyLoop.getValue() == true) {
      // This is a specialization for our matmul pattern. We only want to
      // delay the copy loops in the reduction-dimension that is the k-loop.
      // This down in the codegen pipeline it is not possible to determine
      // that sequential loop. We can however heuristically check for the
      // loop the we want to process. The copy loops which have as parent a
      // loop which yields values is our target.
      auto parentOp = dyn_cast<scf::ForOp>(copyLoop->getParentOp());
      if (parentOp && parentOp.getNumResults() != 0) {
        (void)loopUnrollFull(copyLoop, false /*promoteSingleIteration*/);
        copyLoops.insert(copyLoop);
      }
    }
  });

  // After unrolling the copy loops we need to delay the store in the copy
  // loops to the end of the parent forOp.
  SmallPtrSet<Operation *, 2> isBarrierInserted;
  OpBuilder b(funcOp.getContext());
  for (auto op : copyLoops) {
    scf::ForOp copyLoop = static_cast<scf::ForOp>(op);
    scf::ForOp parentOp = static_cast<scf::ForOp>(copyLoop->getParentOp());
    b.setInsertionPoint(&parentOp.getBody()->back());
    if (isBarrierInserted.find(parentOp.getOperation()) ==
        isBarrierInserted.end()) {
      b.create<gpu::BarrierOp>(parentOp.getLoc());
      // isBarrierInserted.insert(parentOp.getOperation());
    }
    SmallVector<memref::StoreOp> toDelay;
    for (auto &nestedOp : *copyLoop.getBody()) {
      if (auto storeOp = dyn_cast<memref::StoreOp>(nestedOp)) {
        b.clone(*storeOp.getOperation());
        toDelay.push_back(storeOp);
      }
    }
    b.setInsertionPointToStart(parentOp.getBody());
    if (isBarrierInserted.find(parentOp.getOperation()) ==
        isBarrierInserted.end()) {
      b.create<gpu::BarrierOp>(parentOp.getLoc());
      isBarrierInserted.insert(parentOp.getOperation());
    }
    for (auto storeOp : toDelay)
      storeOp.erase();
    (void)promoteIfSingleIteration(copyLoop);
  }
}

namespace mlir {
void registerTestUnrollAndDelayCopiesPass() {
  PassRegistration<TestUnrollAndDelayCopies>(
      "test-unroll-and-delay-copies",
      "Unroll copy loops completely and also delay stores in copy loops to "
      "happen after compute.");
}
} // namespace mlir
