#include "mlir/Dialect/GPU/GPUDialect.h"
#include "mlir/Dialect/GPU/Passes.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "llvm/Support/CommandLine.h"

using namespace mlir;

#define DEBUG_TYPE "test-gpu-mark-global-as-workgroup-memory"

namespace {

/// This pass adds global memrefs used inside the gpu.func as its workgroup
/// memory.
struct TestGpuMarkGlobalAsWorkgroupMemory
    : public PassWrapper<TestGpuMarkGlobalAsWorkgroupMemory,
                         OperationPass<ModuleOp>> {
  TestGpuMarkGlobalAsWorkgroupMemory() = default;
  TestGpuMarkGlobalAsWorkgroupMemory(
      const TestGpuMarkGlobalAsWorkgroupMemory &pass) {}
  void runOnOperation() override;
  Option<unsigned> memorySpace{
      *this, "memory-space",
      llvm::cl::desc(
          "Specifies which memory space global memrefs to be marked as the "
          "workgroup memory"),
      llvm::cl::init(3)};
};
// Contains names pf global memrefs which after being marked as workgroup
// memory needs to be erased.
SmallVector<llvm::StringRef, 2> nameOfMemrefsToErase;
// Specifies which memory space global memrefs to be marked as the workgroup
// memory.
unsigned memSpace;
} // end anonymous namespace

/// Marks global memref extracted by `memrefOp` as the workgroup memory
/// of `funcOp`.
static void
markGlobalMemrefAsWorkgroupMemoryOfFunc(gpu::GPUFuncOp funcOp,
                                        memref::GetGlobalOp memrefOp) {
  // Add memref type as the workgroup memory attribute.
  funcOp.addWorkgroupAttribution(memrefOp.getType());
  BlockArgument workgroupMemory = funcOp.getWorkgroupAttributions().back();
  Value globalMemref = memrefOp.getResult();
  // Replacing all the uses of `globalMemref` with the `workgroupMemory`.
  globalMemref.replaceAllUsesWith(workgroupMemory);
  // Adding the global memref which is being marked as the workgroup memory
  // to the list of memrefs to be erased.
  nameOfMemrefsToErase.push_back(memrefOp.name());
  // Erasing the get_global_memref op which was using the global memref
  // which is now being marked as the workgroup memory.
  memrefOp.erase();
}

/// Finds all global memrefs inside `funcOp` and marks them as the workgroup
/// memory.
static void findAndMarkAllGlobalMemrefsInsideFunc(gpu::GPUFuncOp funcOp) {
  // Walks over all the get_global_memref ops inside the `funcOp`.
  funcOp.walk([&](memref::GetGlobalOp memrefOp) {
    // If the global memref extracted by the `memrefOp` belongs to the
    // memory space == `memSpace` mark it as the workgroup memory.
    if (memrefOp.getType().getMemorySpaceAsInt() == memSpace)
      markGlobalMemrefAsWorkgroupMemoryOfFunc(funcOp, memrefOp);
  });
}

void TestGpuMarkGlobalAsWorkgroupMemory::runOnOperation() {
  auto moduleOp = getOperation();
  // `memSpace` specifies which memory space memrefs to be marked as the
  // workgroup memory.
  memSpace = memorySpace;
  // Walk over all the gpu.func ops.
  moduleOp.walk([&](gpu::GPUFuncOp funcOp) {
    // Finds and mark global memrefs as workgroup memory.
    findAndMarkAllGlobalMemrefsInsideFunc(funcOp);
  });
  // Traverse all global memrefs.
  moduleOp.walk([&](memref::GlobalOp globalMemref) {
    // Erase `globalMemref` if it is in the list of memrefs to be erased.
    if (llvm::find(nameOfMemrefsToErase, globalMemref.getName()))
      globalMemref.erase();
  });
}

namespace mlir {
void registerTestGpuMarkGlobalAsWorkgroupMemoryPass() {
  PassRegistration<TestGpuMarkGlobalAsWorkgroupMemory>(
      "test-gpu-mark-global-as-workgroup-memory",
      "Marks global memrefs belonging to a particular memory space as the "
      "workgroup memory");
}
} // namespace mlir
