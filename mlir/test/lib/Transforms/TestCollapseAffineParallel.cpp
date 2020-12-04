#include "mlir/Transforms/LoopUtils.h"
#include "mlir/Transforms/Passes.h"

#define PASS_NAME "test-collapse-affine-parallel"

using namespace mlir;

namespace {

/// The 'affine.parallel' op represents a hyper-rectangluar 'affine.parallel'
/// region, by allowing to have multi dimensional lb, ub and corresponding
/// steps. Exploiting this functionality the perfectly nested parallel ops can
/// be collapsed together and converted into a single multi dimensional
/// 'affine.parallel' op. This pass finds such perfectly nested
/// multi-dimensional parallel ops and converts them into a single n-dimensional
/// parallel op.
struct TestCollapseAffineParallel
    : public PassWrapper<TestCollapseAffineParallel, FunctionPass> {
  TestCollapseAffineParallel() = default;
  TestCollapseAffineParallel(const TestCollapseAffineParallel &pass) {}

  void runOnFunction() override;
};

} // end of namespace

// Runs on a function 'func' and finds all the perfectly nested
// multi-dimensional parallel opsinside func and converts them into a sinle
// n-d parallel op.
void TestCollapseAffineParallel::runOnFunction() {
  FuncOp func = getFunction();

  collapseAffineParallelOps(func);
}

namespace mlir {
void registerTestCollapseAffineParallelPass() {
  PassRegistration<TestCollapseAffineParallel>(
      PASS_NAME, "tests collapsing of affine parallel ops");
}
} // namespace mlir
