// RUN: mlir-opt --test-specialize-affine-matmul-for-wmma --canonicalize %s | FileCheck %s

#map0 = affine_map<(d0) -> (d0)>
#map1 = affine_map<(d0) -> (d0 + 64)>
module  {
  memref.global @asmem : memref<64x64xf16, 3>
  memref.global @bsmem : memref<64x64xf16, 3>
  func @matmul() {
    %0 = memref.alloc() : memref<1024x1024xf16>
    %1 = memref.alloc() : memref<1024x1024xf16>
    %2 = memref.alloc() : memref<1024x1024xf16>
    affine.for %arg0 = 0 to 1024 step 64 {
      affine.for %arg1 = 0 to 1024 step 64 {
        %3 = memref.alloca() : memref<64x64xf16, 3>
        %4 = memref.alloca() : memref<64x64xf16, 3>
        affine.for %arg2 = 0 to 1024 step 64 {
          affine.for %arg3 = #map0(%arg2) to #map1(%arg2) {
            affine.for %arg4 = #map0(%arg1) to #map1(%arg1) {
              %5 = affine.load %1[%arg3, %arg4] : memref<1024x1024xf16>
              affine.store %5, %3[-%arg2 + %arg3, -%arg1 + %arg4] : memref<64x64xf16, 3>
            }
          } {isCopyLoopNest = true}
          affine.for %arg3 = #map0(%arg0) to #map1(%arg0) {
            affine.for %arg4 = #map0(%arg2) to #map1(%arg2) {
              %5 = affine.load %0[%arg3, %arg4] : memref<1024x1024xf16>
              affine.store %5, %4[-%arg0 + %arg3, -%arg2 + %arg4] : memref<64x64xf16, 3>
            }
          } {isCopyLoopNest = true}
          affine.for %arg3 = 0 to 64 step 32 {
            affine.for %arg4 = 0 to 64 step 32 {
              affine.for %arg5 = 0 to 64 step 16 {
                affine.for %arg6 = 0 to 32 {
                  affine.for %arg7 = 0 to 32 {
                    affine.for %arg8 = 0 to 16 {
                      %5 = affine.load %4[%arg3 + %arg6, %arg5 + %arg8] : memref<64x64xf16, 3>
                      %6 = affine.load %3[%arg5 + %arg8, %arg4 + %arg7] : memref<64x64xf16, 3>
                      %7 = affine.load %2[%arg0 + %arg3 + %arg6, %arg1 + %arg4 + %arg7] : memref<1024x1024xf16>
                      %8 = mulf %5, %6 : f16
                      %9 = addf %7, %8 : f16
                      affine.store %9, %2[%arg0 + %arg3 + %arg6, %arg1 + %arg4 + %arg7] : memref<1024x1024xf16>
                    }
                  }
                }
              }
            }
          }
        }
      }
    }
    return
  }
}

// CHECK-DAG: #map0 = affine_map<(d0, d1) -> (d0 + d1)>
// CHECK-DAG: #map1 = affine_map<(d0, d1) -> (d0 + d1 + 16)>
// CHECK-DAG: #map2 = affine_map<(d0) -> (d0 + 16)>
// CHECK: module  {
// CHECK-NEXT:   memref.global @asmem : memref<64x64xf16, 3>
// CHECK-NEXT:   memref.global @bsmem : memref<64x64xf16, 3>
// CHECK-LABEL:   func @matmul() {
// CHECK-NEXT:     %0 = memref.alloc() : memref<1024x1024xf16>
// CHECK-NEXT:     %1 = memref.alloc() : memref<1024x1024xf16>
// CHECK-NEXT:     %2 = memref.alloc() : memref<1024x1024xf16>
// CHECK-NEXT:     affine.for %arg0 = 0 to 1024 step 64 {
// CHECK-NEXT:       affine.for %arg1 = 0 to 1024 step 64 {
// CHECK-NEXT:         %3 = memref.alloca() : memref<64x64xf16, 3>
// CHECK-NEXT:         %4 = memref.alloca() : memref<64x64xf16, 3>
// CHECK-NEXT:         affine.parallel (%arg2) = (0) to (64) step (32) {
// CHECK-NEXT:           affine.parallel (%arg3) = (0) to (64) step (32) {
// CHECK-NEXT:             %5 = affine.apply #map0(%arg0, %arg2)
// CHECK-NEXT:             %6 = affine.apply #map0(%arg1, %arg3)
// CHECK-NEXT:             %7 = gpu.subgroup_mma_load_matrix %2[%5, %6] {leadDimension = 1024 : index} : memref<1024x1024xf16> -> !gpu.mma_matrix<16x16xf16, "COp">
// CHECK-NEXT:             %8 = affine.apply #map1(%arg0, %arg2)
// CHECK-NEXT:             %9 = affine.apply #map0(%arg1, %arg3)
// CHECK-NEXT:             %10 = gpu.subgroup_mma_load_matrix %2[%8, %9] {leadDimension = 1024 : index} : memref<1024x1024xf16> -> !gpu.mma_matrix<16x16xf16, "COp">
// CHECK-NEXT:             %11 = affine.apply #map0(%arg0, %arg2)
// CHECK-NEXT:             %12 = affine.apply #map1(%arg1, %arg3)
// CHECK-NEXT:             %13 = gpu.subgroup_mma_load_matrix %2[%11, %12] {leadDimension = 1024 : index} : memref<1024x1024xf16> -> !gpu.mma_matrix<16x16xf16, "COp">
// CHECK-NEXT:             %14 = affine.apply #map1(%arg0, %arg2)
// CHECK-NEXT:             %15 = affine.apply #map1(%arg1, %arg3)
// CHECK-NEXT:             %16 = gpu.subgroup_mma_load_matrix %2[%14, %15] {leadDimension = 1024 : index} : memref<1024x1024xf16> -> !gpu.mma_matrix<16x16xf16, "COp">
// CHECK-NEXT:             %17:4 = affine.for %arg4 = 0 to 1024 step 64 iter_args(%arg5 = %7, %arg6 = %10, %arg7 = %13, %arg8 = %16) -> (!gpu.mma_matrix<16x16xf16, "COp">, !gpu.mma_matrix<16x16xf16, "COp">, !gpu.mma_matrix<16x16xf16, "COp">, !gpu.mma_matrix<16x16xf16, "COp">) {
// CHECK-NEXT:               affine.parallel (%arg9) = (%arg4) to (%arg4 + 64) {
// CHECK-NEXT:                 affine.parallel (%arg10) = (%arg1) to (%arg1 + 64) {
// CHECK-NEXT:                   %19 = affine.load %1[%arg9, %arg10] : memref<1024x1024xf16>
// CHECK-NEXT:                   affine.store %19, %3[-%arg4 + %arg9, -%arg1 + %arg10] : memref<64x64xf16, 3>
// CHECK-NEXT:                 }
// CHECK-NEXT:               }
// CHECK-NEXT:               affine.parallel (%arg9) = (%arg0) to (%arg0 + 64) {
// CHECK-NEXT:                 affine.parallel (%arg10) = (%arg4) to (%arg4 + 64) {
// CHECK-NEXT:                   %19 = affine.load %0[%arg9, %arg10] : memref<1024x1024xf16>
// CHECK-NEXT:                   affine.store %19, %4[-%arg0 + %arg9, -%arg4 + %arg10] : memref<64x64xf16, 3>
// CHECK-NEXT:                 }
// CHECK-NEXT:               }
// CHECK-NEXT:               %18:4 = affine.for %arg9 = 0 to 64 step 16 iter_args(%arg10 = %arg5, %arg11 = %arg6, %arg12 = %arg7, %arg13 = %arg8) -> (!gpu.mma_matrix<16x16xf16, "COp">, !gpu.mma_matrix<16x16xf16, "COp">, !gpu.mma_matrix<16x16xf16, "COp">, !gpu.mma_matrix<16x16xf16, "COp">) {
// CHECK-NEXT:                 %19 = gpu.subgroup_mma_load_matrix %4[%arg2, %arg9] {leadDimension = 64 : index} : memref<64x64xf16, 3> -> !gpu.mma_matrix<16x16xf16, "AOp">
// CHECK-NEXT:                 %20 = gpu.subgroup_mma_load_matrix %3[%arg9, %arg3] {leadDimension = 64 : index} : memref<64x64xf16, 3> -> !gpu.mma_matrix<16x16xf16, "BOp">
// CHECK-NEXT:                 %21 = gpu.subgroup_mma_compute %19, %20, %arg10 : !gpu.mma_matrix<16x16xf16, "AOp">, !gpu.mma_matrix<16x16xf16, "BOp">, !gpu.mma_matrix<16x16xf16, "COp"> -> !gpu.mma_matrix<16x16xf16, "COp">
// CHECK-NEXT:                 %22 = affine.apply #map2(%arg2)
// CHECK-NEXT:                 %23 = gpu.subgroup_mma_load_matrix %4[%22, %arg9] {leadDimension = 64 : index} : memref<64x64xf16, 3> -> !gpu.mma_matrix<16x16xf16, "AOp">
// CHECK-NEXT:                 %24 = gpu.subgroup_mma_load_matrix %3[%arg9, %arg3] {leadDimension = 64 : index} : memref<64x64xf16, 3> -> !gpu.mma_matrix<16x16xf16, "BOp">
// CHECK-NEXT:                 %25 = gpu.subgroup_mma_compute %23, %24, %arg11 : !gpu.mma_matrix<16x16xf16, "AOp">, !gpu.mma_matrix<16x16xf16, "BOp">, !gpu.mma_matrix<16x16xf16, "COp"> -> !gpu.mma_matrix<16x16xf16, "COp">
// CHECK-NEXT:                 %26 = gpu.subgroup_mma_load_matrix %4[%arg2, %arg9] {leadDimension = 64 : index} : memref<64x64xf16, 3> -> !gpu.mma_matrix<16x16xf16, "AOp">
// CHECK-NEXT:                 %27 = affine.apply #map2(%arg3)
// CHECK-NEXT:                 %28 = gpu.subgroup_mma_load_matrix %3[%arg9, %27] {leadDimension = 64 : index} : memref<64x64xf16, 3> -> !gpu.mma_matrix<16x16xf16, "BOp">
// CHECK-NEXT:                 %29 = gpu.subgroup_mma_compute %26, %28, %arg12 : !gpu.mma_matrix<16x16xf16, "AOp">, !gpu.mma_matrix<16x16xf16, "BOp">, !gpu.mma_matrix<16x16xf16, "COp"> -> !gpu.mma_matrix<16x16xf16, "COp">
// CHECK-NEXT:                 %30 = affine.apply #map2(%arg2)
// CHECK-NEXT:                 %31 = gpu.subgroup_mma_load_matrix %4[%30, %arg9] {leadDimension = 64 : index} : memref<64x64xf16, 3> -> !gpu.mma_matrix<16x16xf16, "AOp">
// CHECK-NEXT:                 %32 = affine.apply #map2(%arg3)
// CHECK-NEXT:                 %33 = gpu.subgroup_mma_load_matrix %3[%arg9, %32] {leadDimension = 64 : index} : memref<64x64xf16, 3> -> !gpu.mma_matrix<16x16xf16, "BOp">
// CHECK-NEXT:                 %34 = gpu.subgroup_mma_compute %31, %33, %arg13 : !gpu.mma_matrix<16x16xf16, "AOp">, !gpu.mma_matrix<16x16xf16, "BOp">, !gpu.mma_matrix<16x16xf16, "COp"> -> !gpu.mma_matrix<16x16xf16, "COp">
// CHECK-NEXT:                 affine.yield %21, %25, %29, %34 : !gpu.mma_matrix<16x16xf16, "COp">, !gpu.mma_matrix<16x16xf16, "COp">, !gpu.mma_matrix<16x16xf16, "COp">, !gpu.mma_matrix<16x16xf16, "COp">
// CHECK-NEXT:               }
// CHECK-NEXT:               affine.yield %18#0, %18#1, %18#2, %18#3 : !gpu.mma_matrix<16x16xf16, "COp">, !gpu.mma_matrix<16x16xf16, "COp">, !gpu.mma_matrix<16x16xf16, "COp">, !gpu.mma_matrix<16x16xf16, "COp">
// CHECK-NEXT:             }
// CHECK-NEXT:             gpu.subgroup_mma_store_matrix %17#0, %2[%5, %6] {leadDimension = 1024 : index} : !gpu.mma_matrix<16x16xf16, "COp">, memref<1024x1024xf16>
// CHECK-NEXT:             gpu.subgroup_mma_store_matrix %17#1, %2[%8, %9] {leadDimension = 1024 : index} : !gpu.mma_matrix<16x16xf16, "COp">, memref<1024x1024xf16>
// CHECK-NEXT:             gpu.subgroup_mma_store_matrix %17#2, %2[%11, %12] {leadDimension = 1024 : index} : !gpu.mma_matrix<16x16xf16, "COp">, memref<1024x1024xf16>
// CHECK-NEXT:             gpu.subgroup_mma_store_matrix %17#3, %2[%14, %15] {leadDimension = 1024 : index} : !gpu.mma_matrix<16x16xf16, "COp">, memref<1024x1024xf16>
// CHECK-NEXT:           }
// CHECK-NEXT:         }
// CHECK-NEXT:       }
// CHECK-NEXT:     }
// CHECK-NEXT:     return
// CHECK-NEXT:   }
// CHECK-NEXT: }
