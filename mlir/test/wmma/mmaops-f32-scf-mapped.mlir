// RUN: mlir-opt %s --test-convert-matmul-parallel-loops-to-gpu  --canonicalize | FileCheck %s

#map = affine_map<(d0) -> (d0)>
module  {
  global_memref @asmem : memref<64x64xf16, 3>
  global_memref @bsmem : memref<64x64xf16, 3>
  func @main() {
    %c32 = constant 32 : index
    %c1024 = constant 1024 : index
    %c-1 = constant -1 : index
    %c64 = constant 64 : index
    %c16 = constant 16 : index
    %c0 = constant 0 : index
    %c1 = constant 1 : index
    %0 = alloc() : memref<1024x1024xf16>
    %1 = alloc() : memref<1024x1024xf16>
    %2 = alloc() : memref<1024x1024xf16>
    %3 = memref_cast %0 : memref<1024x1024xf16> to memref<*xf16>
    gpu.host_register %3 : memref<*xf16>
    %4 = memref_cast %1 : memref<1024x1024xf16> to memref<*xf16>
    gpu.host_register %4 : memref<*xf16>
    %5 = memref_cast %2 : memref<1024x1024xf16> to memref<*xf16>
    gpu.host_register %5 : memref<*xf16>
    scf.parallel (%arg0, %arg1) = (%c0, %c0) to (%c1024, %c1024) step (%c64, %c64) {
      %6 = get_global_memref @asmem : memref<64x64xf16, 3>
      %7 = get_global_memref @bsmem : memref<64x64xf16, 3>
      scf.parallel (%arg2, %arg3) = (%c0, %c0) to (%c64, %c64) step (%c32, %c32) {
        %8 = addi %arg0, %arg2 : index
        %9 = addi %arg1, %arg3 : index
        %10 = gpu.subgroup_mma_load_matrix %2[%8, %9] {leadDimension = 1024 : index, operand = "COp"} : memref<1024x1024xf16> -> !gpu.mmafragment<8, f32>
        %11 = addi %arg0, %arg2 : index
        %12 = addi %11, %c16 : index
        %13 = addi %arg1, %arg3 : index
        %14 = gpu.subgroup_mma_load_matrix %2[%12, %13] {leadDimension = 1024 : index, operand = "COp"} : memref<1024x1024xf16> -> !gpu.mmafragment<8, f32>
        %15 = addi %arg0, %arg2 : index
        %16 = addi %arg1, %arg3 : index
        %17 = addi %16, %c16 : index
        %18 = gpu.subgroup_mma_load_matrix %2[%15, %17] {leadDimension = 1024 : index, operand = "COp"} : memref<1024x1024xf16> -> !gpu.mmafragment<8, f32>
        %19 = addi %arg0, %arg2 : index
        %20 = addi %19, %c16 : index
        %21 = addi %arg1, %arg3 : index
        %22 = addi %21, %c16 : index
        %23 = gpu.subgroup_mma_load_matrix %2[%20, %22] {leadDimension = 1024 : index, operand = "COp"} : memref<1024x1024xf16> -> !gpu.mmafragment<8, f32>
        %24:4 = scf.for %arg4 = %c0 to %c1024 step %c64 iter_args(%arg5 = %10, %arg6 = %14, %arg7 = %18, %arg8 = %23) -> (!gpu.mmafragment<8, f32>, !gpu.mmafragment<8, f32>, !gpu.mmafragment<8, f32>, !gpu.mmafragment<8, f32>) {
          %25 = addi %arg4, %c64 : index
          %26 = addi %arg1, %c64 : index
          %27 = subi %25, %arg4 : index
          %28 = subi %26, %arg1 : index
          %29 = muli %27, %28 : index
          scf.parallel (%arg9) = (%c0) to (%29) step (%c1) {
            %36 = remi_signed %arg9, %28 : index
            %37 = divi_signed %arg9, %28 : index
            %38 = addi %36, %arg1 : index
            %39 = addi %37, %arg4 : index
            %40 = load %1[%39, %38] : memref<1024x1024xf16>
            %41 = muli %arg4, %c-1 : index
            %42 = addi %41, %39 : index
            %43 = muli %arg1, %c-1 : index
            %44 = addi %43, %38 : index
            store %40, %6[%42, %44] : memref<64x64xf16, 3>
            scf.yield
          } {isCopyLoopNest = true, mapping = [{bound = #map, map = #map, processor = 0 : i64}]}
          %30 = addi %arg0, %c64 : index
          %31 = addi %arg4, %c64 : index
          %32 = subi %30, %arg0 : index
          %33 = subi %31, %arg4 : index
          %34 = muli %32, %33 : index
          scf.parallel (%arg9) = (%c0) to (%34) step (%c1) {
            %36 = remi_signed %arg9, %33 : index
            %37 = divi_signed %arg9, %33 : index
            %38 = addi %36, %arg4 : index
            %39 = addi %37, %arg0 : index
            %40 = load %0[%39, %38] : memref<1024x1024xf16>
            %41 = muli %arg0, %c-1 : index
            %42 = addi %41, %39 : index
            %43 = muli %arg4, %c-1 : index
            %44 = addi %43, %38 : index
            store %40, %7[%42, %44] : memref<64x64xf16, 3>
            scf.yield
          } {isCopyLoopNest = true, mapping = [{bound = #map, map = #map, processor = 0 : i64}]}
          gpu.barrier
          %35:4 = scf.for %arg9 = %c0 to %c64 step %c16 iter_args(%arg10 = %arg5, %arg11 = %arg6, %arg12 = %arg7, %arg13 = %arg8) -> (!gpu.mmafragment<8, f32>, !gpu.mmafragment<8, f32>, !gpu.mmafragment<8, f32>, !gpu.mmafragment<8, f32>) {
            %36 = gpu.subgroup_mma_load_matrix %7[%arg2, %arg9] {leadDimension = 64 : index, operand = "AOp"} : memref<64x64xf16, 3> -> !gpu.mmafragment<8, vector<2xf16>>
            %37 = gpu.subgroup_mma_load_matrix %6[%arg9, %arg3] {leadDimension = 64 : index, operand = "BOp"} : memref<64x64xf16, 3> -> !gpu.mmafragment<8, vector<2xf16>>
            %38 = gpu.subgroup_mma_compute %36, %37, %arg10 : !gpu.mmafragment<8, vector<2xf16>>, !gpu.mmafragment<8, vector<2xf16>>, !gpu.mmafragment<8, f32> -> !gpu.mmafragment<8, f32>
            %39 = addi %arg2, %c16 : index
            %40 = gpu.subgroup_mma_load_matrix %7[%39, %arg9] {leadDimension = 64 : index, operand = "AOp"} : memref<64x64xf16, 3> -> !gpu.mmafragment<8, vector<2xf16>>
            %41 = gpu.subgroup_mma_load_matrix %6[%arg9, %arg3] {leadDimension = 64 : index, operand = "BOp"} : memref<64x64xf16, 3> -> !gpu.mmafragment<8, vector<2xf16>>
            %42 = gpu.subgroup_mma_compute %40, %41, %arg11 : !gpu.mmafragment<8, vector<2xf16>>, !gpu.mmafragment<8, vector<2xf16>>, !gpu.mmafragment<8, f32> -> !gpu.mmafragment<8, f32>
            %43 = gpu.subgroup_mma_load_matrix %7[%arg2, %arg9] {leadDimension = 64 : index, operand = "AOp"} : memref<64x64xf16, 3> -> !gpu.mmafragment<8, vector<2xf16>>
            %44 = addi %arg3, %c16 : index
            %45 = gpu.subgroup_mma_load_matrix %6[%arg9, %44] {leadDimension = 64 : index, operand = "BOp"} : memref<64x64xf16, 3> -> !gpu.mmafragment<8, vector<2xf16>>
            %46 = gpu.subgroup_mma_compute %43, %45, %arg12 : !gpu.mmafragment<8, vector<2xf16>>, !gpu.mmafragment<8, vector<2xf16>>, !gpu.mmafragment<8, f32> -> !gpu.mmafragment<8, f32>
            %47 = addi %arg2, %c16 : index
            %48 = gpu.subgroup_mma_load_matrix %7[%47, %arg9] {leadDimension = 64 : index, operand = "AOp"} : memref<64x64xf16, 3> -> !gpu.mmafragment<8, vector<2xf16>>
            %49 = addi %arg3, %c16 : index
            %50 = gpu.subgroup_mma_load_matrix %6[%arg9, %49] {leadDimension = 64 : index, operand = "BOp"} : memref<64x64xf16, 3> -> !gpu.mmafragment<8, vector<2xf16>>
            %51 = gpu.subgroup_mma_compute %48, %50, %arg13 : !gpu.mmafragment<8, vector<2xf16>>, !gpu.mmafragment<8, vector<2xf16>>, !gpu.mmafragment<8, f32> -> !gpu.mmafragment<8, f32>
            scf.yield %38, %42, %46, %51 : !gpu.mmafragment<8, f32>, !gpu.mmafragment<8, f32>, !gpu.mmafragment<8, f32>, !gpu.mmafragment<8, f32>
          }
          gpu.barrier
          scf.yield %35#0, %35#1, %35#2, %35#3 : !gpu.mmafragment<8, f32>, !gpu.mmafragment<8, f32>, !gpu.mmafragment<8, f32>, !gpu.mmafragment<8, f32>
        }
        gpu.subgroup_mma_store_matrix %24#0, %2[%8, %9] {leadDimension = 1024 : index} : !gpu.mmafragment<8, f32>, memref<1024x1024xf16>
        gpu.subgroup_mma_store_matrix %24#1, %2[%12, %13] {leadDimension = 1024 : index} : !gpu.mmafragment<8, f32>, memref<1024x1024xf16>
        gpu.subgroup_mma_store_matrix %24#2, %2[%15, %17] {leadDimension = 1024 : index} : !gpu.mmafragment<8, f32>, memref<1024x1024xf16>
        gpu.subgroup_mma_store_matrix %24#3, %2[%20, %22] {leadDimension = 1024 : index} : !gpu.mmafragment<8, f32>, memref<1024x1024xf16>
        scf.yield
      } {mapping = [{bound = #map, map = #map, processor = 4 : i64}, {bound = #map, map = #map, processor = 3 : i64}]}
      scf.yield
    } {mapping = [{bound = #map, map = #map, processor = 1 : i64}, {bound = #map, map = #map, processor = 0 : i64}]}
    return
  }
}

