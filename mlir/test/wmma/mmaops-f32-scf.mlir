// RUN: mlir-opt %s --test-gpu-matmul-parallel-loop-mapping --canonicalize

module  {
  global_memref @asmem : memref<64x64xf16, 3>
  global_memref @bsmem : memref<64x64xf16, 3>
  func @main() {
    %c32 = constant 32 : index
    %c1024 = constant 1024 : index
    %c1 = constant 1 : index
    %c-1 = constant -1 : index
    %c0 = constant 0 : index
    %c64 = constant 64 : index
    %c16 = constant 16 : index
    %0 = alloc() : memref<1024x1024xf16>
    %1 = alloc() : memref<1024x1024xf16>
    %2 = alloc() : memref<1024x1024xf32>
    %3 = memref_cast %0 : memref<1024x1024xf16> to memref<*xf16>
    gpu.host_register %3 : memref<*xf16>
    %4 = memref_cast %1 : memref<1024x1024xf16> to memref<*xf16>
    gpu.host_register %4 : memref<*xf16>
    %5 = memref_cast %2 : memref<1024x1024xf32> to memref<*xf32>
    gpu.host_register %5 : memref<*xf32>
    scf.parallel (%arg0, %arg1) = (%c0, %c0) to (%c1024, %c1024) step (%c64, %c64) {
      %6 = get_global_memref @asmem : memref<64x64xf16, 3>
      %7 = get_global_memref @bsmem : memref<64x64xf16, 3>
      scf.parallel (%arg2, %arg3) = (%c0, %c0) to (%c64, %c64) step (%c32, %c32) {
        %8 = addi %arg0, %arg2 : index
        %9 = addi %arg1, %arg3 : index
        %10 = gpu.subgroup_mma_load_matrix %2[%8, %9] {leadDimension = 1024 : index, operand = "COp"} : memref<1024x1024xf32> -> !gpu.mmafragment<8, f32>
        %11 = addi %arg0, %arg2 : index
        %12 = addi %11, %c16 : index
        %13 = addi %arg1, %arg3 : index
        %14 = gpu.subgroup_mma_load_matrix %2[%12, %13] {leadDimension = 1024 : index, operand = "COp"} : memref<1024x1024xf32> -> !gpu.mmafragment<8, f32>
        %15 = addi %arg0, %arg2 : index
        %16 = addi %arg1, %arg3 : index
        %17 = addi %16, %c16 : index
        %18 = gpu.subgroup_mma_load_matrix %2[%15, %17] {leadDimension = 1024 : index, operand = "COp"} : memref<1024x1024xf32> -> !gpu.mmafragment<8, f32>
        %19 = addi %arg0, %arg2 : index
        %20 = addi %19, %c16 : index
        %21 = addi %arg1, %arg3 : index
        %22 = addi %21, %c16 : index
        %23 = gpu.subgroup_mma_load_matrix %2[%20, %22] {leadDimension = 1024 : index, operand = "COp"} : memref<1024x1024xf32> -> !gpu.mmafragment<8, f32>
        %24:4 = scf.for %arg4 = %c0 to %c1024 step %c64 iter_args(%arg5 = %10, %arg6 = %14, %arg7 = %18, %arg8 = %23) -> (!gpu.mmafragment<8, f32>, !gpu.mmafragment<8, f32>, !gpu.mmafragment<8, f32>, !gpu.mmafragment<8, f32>) {
          %25 = addi %arg4, %c64 : index
          %26 = addi %arg1, %c64 : index
          scf.parallel (%arg9, %arg10) = (%arg4, %arg1) to (%25, %26) step (%c1, %c1) {
            %30 = load %1[%arg9, %arg10] : memref<1024x1024xf16>
            %31 = muli %arg4, %c-1 : index
            %32 = addi %31, %arg9 : index
            %33 = muli %arg1, %c-1 : index
            %34 = addi %33, %arg10 : index
            store %30, %6[%32, %34] : memref<64x64xf16, 3>
            scf.yield
          }{isCopyLoopNest = true}
          %27 = addi %arg0, %c64 : index
          %28 = addi %arg4, %c64 : index
          scf.parallel (%arg9, %arg10) = (%arg0, %arg4) to (%27, %28) step (%c1, %c1) {
            %30 = load %0[%arg9, %arg10] : memref<1024x1024xf16>
            %31 = muli %arg0, %c-1 : index
            %32 = addi %31, %arg9 : index
            %33 = muli %arg4, %c-1 : index
            %34 = addi %33, %arg10 : index
            store %30, %7[%32, %34] : memref<64x64xf16, 3>
            scf.yield
          }{isCopyLoopNest = true}
          gpu.barrier
          %29:4 = scf.for %arg9 = %c0 to %c64 step %c16 iter_args(%arg10 = %arg5, %arg11 = %arg6, %arg12 = %arg7, %arg13 = %arg8) -> (!gpu.mmafragment<8, f32>, !gpu.mmafragment<8, f32>, !gpu.mmafragment<8, f32>, !gpu.mmafragment<8, f32>) {
            %30 = gpu.subgroup_mma_load_matrix %7[%arg2, %arg9] {leadDimension = 64 : index, operand = "AOp"} : memref<64x64xf16, 3> -> !gpu.mmafragment<8, vector<2xf16>>
            %31 = gpu.subgroup_mma_load_matrix %6[%arg9, %arg3] {leadDimension = 64 : index, operand = "BOp"} : memref<64x64xf16, 3> -> !gpu.mmafragment<8, vector<2xf16>>
            %32 = gpu.subgroup_mma_compute %30, %31, %arg10 : !gpu.mmafragment<8, vector<2xf16>>, !gpu.mmafragment<8, vector<2xf16>>, !gpu.mmafragment<8, f32> -> !gpu.mmafragment<8, f32>
            %33 = addi %arg2, %c16 : index
            %34 = gpu.subgroup_mma_load_matrix %7[%33, %arg9] {leadDimension = 64 : index, operand = "AOp"} : memref<64x64xf16, 3> -> !gpu.mmafragment<8, vector<2xf16>>
            %35 = gpu.subgroup_mma_load_matrix %6[%arg9, %arg3] {leadDimension = 64 : index, operand = "BOp"} : memref<64x64xf16, 3> -> !gpu.mmafragment<8, vector<2xf16>>
            %36 = gpu.subgroup_mma_compute %34, %35, %arg11 : !gpu.mmafragment<8, vector<2xf16>>, !gpu.mmafragment<8, vector<2xf16>>, !gpu.mmafragment<8, f32> -> !gpu.mmafragment<8, f32>
            %37 = gpu.subgroup_mma_load_matrix %7[%arg2, %arg9] {leadDimension = 64 : index, operand = "AOp"} : memref<64x64xf16, 3> -> !gpu.mmafragment<8, vector<2xf16>>
            %38 = addi %arg3, %c16 : index
            %39 = gpu.subgroup_mma_load_matrix %6[%arg9, %38] {leadDimension = 64 : index, operand = "BOp"} : memref<64x64xf16, 3> -> !gpu.mmafragment<8, vector<2xf16>>
            %40 = gpu.subgroup_mma_compute %37, %39, %arg12 : !gpu.mmafragment<8, vector<2xf16>>, !gpu.mmafragment<8, vector<2xf16>>, !gpu.mmafragment<8, f32> -> !gpu.mmafragment<8, f32>
            %41 = addi %arg2, %c16 : index
            %42 = gpu.subgroup_mma_load_matrix %7[%41, %arg9] {leadDimension = 64 : index, operand = "AOp"} : memref<64x64xf16, 3> -> !gpu.mmafragment<8, vector<2xf16>>
            %43 = addi %arg3, %c16 : index
            %44 = gpu.subgroup_mma_load_matrix %6[%arg9, %43] {leadDimension = 64 : index, operand = "BOp"} : memref<64x64xf16, 3> -> !gpu.mmafragment<8, vector<2xf16>>
            %45 = gpu.subgroup_mma_compute %42, %44, %arg13 : !gpu.mmafragment<8, vector<2xf16>>, !gpu.mmafragment<8, vector<2xf16>>, !gpu.mmafragment<8, f32> -> !gpu.mmafragment<8, f32>
            scf.yield %32, %36, %40, %45 : !gpu.mmafragment<8, f32>, !gpu.mmafragment<8, f32>, !gpu.mmafragment<8, f32>, !gpu.mmafragment<8, f32>
          }
          gpu.barrier
          scf.yield %29#0, %29#1, %29#2, %29#3 : !gpu.mmafragment<8, f32>, !gpu.mmafragment<8, f32>, !gpu.mmafragment<8, f32>, !gpu.mmafragment<8, f32>
        }
        gpu.subgroup_mma_store_matrix %24#0, %2[%8, %9] {leadDimension = 1024 : index} : !gpu.mmafragment<8, f32>, memref<1024x1024xf32>
        gpu.subgroup_mma_store_matrix %24#1, %2[%12, %13] {leadDimension = 1024 : index} : !gpu.mmafragment<8, f32>, memref<1024x1024xf32>
        gpu.subgroup_mma_store_matrix %24#2, %2[%15, %17] {leadDimension = 1024 : index} : !gpu.mmafragment<8, f32>, memref<1024x1024xf32>
        gpu.subgroup_mma_store_matrix %24#3, %2[%20, %22] {leadDimension = 1024 : index} : !gpu.mmafragment<8, f32>, memref<1024x1024xf32>
        scf.yield
      }
      scf.yield
    }
    return
  }
}

