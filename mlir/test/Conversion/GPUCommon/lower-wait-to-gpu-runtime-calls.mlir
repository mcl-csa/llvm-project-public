// RUN: mlir-opt %s --gpu-to-llvm | FileCheck %s

module attributes {gpu.container_module} {

  func @foo() {
    // CHECK: %[[t0:.*]] = llvm.call @mgpuStreamCreate
    // CHECK: %[[e0:.*]] = llvm.call @mgpuEventCreate
    // CHECK: llvm.call @mgpuEventRecord(%[[e0]], %[[t0]])
    %t0 = gpu.wait async
    // CHECK: %[[t1:.*]] = llvm.call @mgpuStreamCreate
    // CHECK: llvm.call @mgpuStreamWaitEvent(%[[t1]], %[[e0]])
    // CHECK: llvm.call @mgpuEventDestroy(%[[e0]])
    %t1 = gpu.wait async [%t0]
    // CHECK: llvm.call @mgpuStreamSynchronize(%[[t0]])
    // CHECK: llvm.call @mgpuStreamDestroy(%[[t0]])
    // CHECK: llvm.call @mgpuStreamSynchronize(%[[t1]])
    // CHECK: llvm.call @mgpuStreamDestroy(%[[t1]])
    gpu.wait [%t0, %t1]

    // CHECK: %[[s0:.*]] = llvm.call @mgpuStreamCreate()
    %t2 = gpu.wait async
    // CHECK: %[[e1:.*]] = llvm.call @mgpuEventCreate()
    %e1 = gpu.event_create : !gpu.event
    // CHECK: %[[e2:.*]] = llvm.call @mgpuEventCreate()
    %e2 = gpu.event_create : !gpu.event
    // CHECK: llvm.call @mgpuEventRecord(%[[e1]], %[[s0]])
    gpu.event_record [%t2], %e1
    // CHECK: llvm.call @mgpuEventRecord(%[[e2]], %[[s0]])
    gpu.event_record [%t2], %e2
    // CHECK: llvm.call @mgpuEventSynchronize(%[[e2]])
    gpu.event_synchronize %e2
    // CHECK: %[[time:.*]] = llvm.call @mgpuEventElapsedTime(%[[e1]], %[[e2]])
    %time = gpu.event_elapsed_time %e1, %e2 : !gpu.event, !gpu.event -> f32
    // CHECK: llvm.call @mgpuEventDestroy(%[[e1]])
    gpu.event_destroy %e1
    // CHECK: llvm.call @mgpuEventDestroy(%[[e2]])
    gpu.event_destroy %e2

    return
  }
}
