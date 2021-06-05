// RUN: mlir-opt %s --lower-affine --canonicalize

#map0 = affine_map<(d0, d1) -> (d0 + d1)>
#map1 = affine_map<(d0, d1) -> (d0 + d1 + 16)>
#map2 = affine_map<(d0) -> (d0 + 16)>
module  {
  memref.global @asmem : memref<64x64xf16, 3>
  memref.global @bsmem : memref<64x64xf16, 3>
  func @main() {
    %0 = memref.alloc() : memref<1024x1024xf16>
    %1 = memref.alloc() : memref<1024x1024xf16>
    %2 = memref.alloc() : memref<1024x1024xf32>
    %3 = memref.cast %0 : memref<1024x1024xf16> to memref<*xf16>
    gpu.host_register %3 : memref<*xf16>
    %4 = memref.cast %1 : memref<1024x1024xf16> to memref<*xf16>
    gpu.host_register %4 : memref<*xf16>
    %5 = memref.cast %2 : memref<1024x1024xf32> to memref<*xf32>
    gpu.host_register %5 : memref<*xf32>
    affine.parallel (%arg0, %arg1) = (0, 0) to (1024, 1024) step (64, 64) {
      %6 = memref.get_global @asmem : memref<64x64xf16, 3>
      %7 = memref.get_global @bsmem : memref<64x64xf16, 3>
      affine.parallel (%arg2, %arg3) = (0, 0) to (64, 64) step (32, 32) {
        %8 = affine.apply #map0(%arg0, %arg2)
        %9 = affine.apply #map0(%arg1, %arg3)
        %10 = gpu.subgroup_mma_load_matrix %2[%8, %9] {leadDimension = 1024 : index, operand = "COp"} : memref<1024x1024xf32> -> !gpu.mmafragment<8, f32>
        %11 = affine.apply #map1(%arg0, %arg2)
        %12 = affine.apply #map0(%arg1, %arg3)
        %13 = gpu.subgroup_mma_load_matrix %2[%11, %12] {leadDimension = 1024 : index, operand = "COp"} : memref<1024x1024xf32> -> !gpu.mmafragment<8, f32>
        %14 = affine.apply #map0(%arg0, %arg2)
        %15 = affine.apply #map1(%arg1, %arg3)
        %16 = gpu.subgroup_mma_load_matrix %2[%14, %15] {leadDimension = 1024 : index, operand = "COp"} : memref<1024x1024xf32> -> !gpu.mmafragment<8, f32>
        %17 = affine.apply #map1(%arg0, %arg2)
        %18 = affine.apply #map1(%arg1, %arg3)
        %19 = gpu.subgroup_mma_load_matrix %2[%17, %18] {leadDimension = 1024 : index, operand = "COp"} : memref<1024x1024xf32> -> !gpu.mmafragment<8, f32>
        %20:4 = affine.for %arg4 = 0 to 1024 step 64 iter_args(%arg5 = %10, %arg6 = %13, %arg7 = %16, %arg8 = %19) -> (!gpu.mmafragment<8, f32>, !gpu.mmafragment<8, f32>, !gpu.mmafragment<8, f32>, !gpu.mmafragment<8, f32>) {
          affine.parallel (%arg9, %arg10) = (%arg4, %arg1) to (%arg4 + 64, %arg1 + 64) {
            %22 = affine.load %1[%arg9, %arg10] : memref<1024x1024xf16>
            affine.store %22, %6[-%arg4 + %arg9, -%arg1 + %arg10] : memref<64x64xf16, 3>
          }
          affine.parallel (%arg9, %arg10) = (%arg0, %arg4) to (%arg0 + 64, %arg4 + 64) {
            %22 = affine.load %0[%arg9, %arg10] : memref<1024x1024xf16>
            affine.store %22, %7[-%arg0 + %arg9, -%arg4 + %arg10] : memref<64x64xf16, 3>
          }
          gpu.barrier
          %21:4 = affine.for %arg9 = 0 to 64 step 16 iter_args(%arg10 = %arg5, %arg11 = %arg6, %arg12 = %arg7, %arg13 = %arg8) -> (!gpu.mmafragment<8, f32>, !gpu.mmafragment<8, f32>, !gpu.mmafragment<8, f32>, !gpu.mmafragment<8, f32>) {
            %22 = gpu.subgroup_mma_load_matrix %7[%arg2, %arg9] {leadDimension = 64 : index, operand = "AOp"} : memref<64x64xf16, 3> -> !gpu.mmafragment<8, vector<2xf16>>
            %23 = gpu.subgroup_mma_load_matrix %6[%arg9, %arg3] {leadDimension = 64 : index, operand = "BOp"} : memref<64x64xf16, 3> -> !gpu.mmafragment<8, vector<2xf16>>
            %24 = gpu.subgroup_mma_compute %22, %23, %arg10 : !gpu.mmafragment<8, vector<2xf16>>, !gpu.mmafragment<8, vector<2xf16>>, !gpu.mmafragment<8, f32> -> !gpu.mmafragment<8, f32>
            %25 = affine.apply #map2(%arg2)
            %26 = gpu.subgroup_mma_load_matrix %7[%25, %arg9] {leadDimension = 64 : index, operand = "AOp"} : memref<64x64xf16, 3> -> !gpu.mmafragment<8, vector<2xf16>>
            %27 = gpu.subgroup_mma_load_matrix %6[%arg9, %arg3] {leadDimension = 64 : index, operand = "BOp"} : memref<64x64xf16, 3> -> !gpu.mmafragment<8, vector<2xf16>>
            %28 = gpu.subgroup_mma_compute %26, %27, %arg11 : !gpu.mmafragment<8, vector<2xf16>>, !gpu.mmafragment<8, vector<2xf16>>, !gpu.mmafragment<8, f32> -> !gpu.mmafragment<8, f32>
            %29 = gpu.subgroup_mma_load_matrix %7[%arg2, %arg9] {leadDimension = 64 : index, operand = "AOp"} : memref<64x64xf16, 3> -> !gpu.mmafragment<8, vector<2xf16>>
            %30 = affine.apply #map2(%arg3)
            %31 = gpu.subgroup_mma_load_matrix %6[%arg9, %30] {leadDimension = 64 : index, operand = "BOp"} : memref<64x64xf16, 3> -> !gpu.mmafragment<8, vector<2xf16>>
            %32 = gpu.subgroup_mma_compute %29, %31, %arg12 : !gpu.mmafragment<8, vector<2xf16>>, !gpu.mmafragment<8, vector<2xf16>>, !gpu.mmafragment<8, f32> -> !gpu.mmafragment<8, f32>
            %33 = affine.apply #map2(%arg2)
            %34 = gpu.subgroup_mma_load_matrix %7[%33, %arg9] {leadDimension = 64 : index, operand = "AOp"} : memref<64x64xf16, 3> -> !gpu.mmafragment<8, vector<2xf16>>
            %35 = affine.apply #map2(%arg3)
            %36 = gpu.subgroup_mma_load_matrix %6[%arg9, %35] {leadDimension = 64 : index, operand = "BOp"} : memref<64x64xf16, 3> -> !gpu.mmafragment<8, vector<2xf16>>
            %37 = gpu.subgroup_mma_compute %34, %36, %arg13 : !gpu.mmafragment<8, vector<2xf16>>, !gpu.mmafragment<8, vector<2xf16>>, !gpu.mmafragment<8, f32> -> !gpu.mmafragment<8, f32>
            affine.yield %24, %28, %32, %37 : !gpu.mmafragment<8, f32>, !gpu.mmafragment<8, f32>, !gpu.mmafragment<8, f32>, !gpu.mmafragment<8, f32>
          }
          gpu.barrier
          affine.yield %21#0, %21#1, %21#2, %21#3 : !gpu.mmafragment<8, f32>, !gpu.mmafragment<8, f32>, !gpu.mmafragment<8, f32>, !gpu.mmafragment<8, f32>
        }
        gpu.subgroup_mma_store_matrix %20#0, %2[%8, %9] {leadDimension = 1024 : index} : !gpu.mmafragment<8, f32>, memref<1024x1024xf32>
        gpu.subgroup_mma_store_matrix %20#1, %2[%11, %12] {leadDimension = 1024 : index} : !gpu.mmafragment<8, f32>, memref<1024x1024xf32>
        gpu.subgroup_mma_store_matrix %20#2, %2[%14, %15] {leadDimension = 1024 : index} : !gpu.mmafragment<8, f32>, memref<1024x1024xf32>
        gpu.subgroup_mma_store_matrix %20#3, %2[%17, %18] {leadDimension = 1024 : index} : !gpu.mmafragment<8, f32>, memref<1024x1024xf32>
      }
    }
    return
  }
}

