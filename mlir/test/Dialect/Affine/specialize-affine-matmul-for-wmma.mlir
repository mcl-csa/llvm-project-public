// RUN: mlir-opt --test-specialize-affine-matmul-for-wmma="accum=f32" --canonicalize --cse %s | FileCheck %s

#map0 = affine_map<(d0) -> (d0)>
#map1 = affine_map<(d0) -> (d0 + 16)>
#map2 = affine_map<(d0) -> (d0 + 128)>
module  {
  memref.global "public" @frag_A : memref<128x16xf16, 3>
  memref.global "public" @frag_B : memref<16x128xf16, 3>
  func @main() {
    %cst = constant 1.600000e+01 : f16
    %cst_0 = constant 0.000000e+00 : f32
    %c0 = constant 0 : index
    %c1 = constant 1 : index
    %c2147483648_i64 = constant 2147483648 : i64
    %c1024 = constant 1024 : index
    %0 = memref.alloc() : memref<1024x1024xf16>
    %1 = memref.alloc() : memref<1024x1024xf16>
    %2 = memref.alloc() : memref<1024x1024xf32>
    affine.for %arg0 = 0 to 1024 step 128 {
      affine.for %arg1 = 0 to 1024 step 128 {
        %14 = memref.get_global @frag_B : memref<16x128xf16, 3>
        %15 = memref.get_global @frag_A : memref<128x16xf16, 3>
        affine.for %arg2 = 0 to 1024 step 16 {
          affine.for %arg3 = #map0(%arg2) to #map1(%arg2) {
            affine.for %arg4 = #map0(%arg1) to #map2(%arg1) {
              %16 = affine.load %1[%arg3, %arg4] : memref<1024x1024xf16>
              affine.store %16, %14[%arg3 - %arg2, %arg4 - %arg1] : memref<16x128xf16, 3>
            }
          } {isCopyLoopNest = true}
          affine.for %arg3 = #map0(%arg0) to #map2(%arg0) {
            affine.for %arg4 = #map0(%arg2) to #map1(%arg2) {
              %16 = affine.load %0[%arg3, %arg4] : memref<1024x1024xf16>
              affine.store %16, %15[%arg3 - %arg0, %arg4 - %arg2] : memref<128x16xf16, 3>
            }
          } {isCopyLoopNest = true}
          affine.for %arg3 = 0 to 128 step 32 {
            affine.for %arg4 = 0 to 128 step 32 {
              affine.for %arg5 = 0 to 16 step 16 {
                affine.for %arg6 = 0 to 32 {
                  affine.for %arg7 = 0 to 32 {
                    affine.for %arg8 = 0 to 16 {
                      %16 = affine.load %15[%arg3 + %arg6, %arg5 + %arg8] : memref<128x16xf16, 3>
                      %17 = affine.load %14[%arg5 + %arg8, %arg4 + %arg7] : memref<16x128xf16, 3>
                      %18 = affine.load %2[%arg0 + %arg3 + %arg6, %arg1 + %arg4 + %arg7] : memref<1024x1024xf32>
                      %19 = mulf %16, %17 : f16
                      %20 = fpext %19 : f16 to f32
                      %21 = addf %18, %20 : f32
                      affine.store %21, %2[%arg0 + %arg3 + %arg6, %arg1 + %arg4 + %arg7] : memref<1024x1024xf32>
                    }
                  }
                }
              }
            }
          } {isComputeLoopNest = true}
        }
      }
    }
    return
  }
}

// CHECK-DAG: #map0 = affine_map<(d0, d1) -> (d0 + d1)>
// CHECK-DAG: #map1 = affine_map<(d0, d1) -> (d0 + d1 + 16)>
// CHECK-DAG: #map2 = affine_map<(d0) -> (d0)>
// CHECK-DAG: #map3 = affine_map<(d0) -> (d0 + 16)>
// CHECK-DAG: #map4 = affine_map<(d0) -> (d0 + 128)>
// CHECK-DAG: memref.global "public" @frag_A : memref<128x16xf16, 3>
// CHECK-DAG: memref.global "public" @frag_B : memref<16x128xf16, 3>
// CHECK-LABEL:   func @main()
// CHECK-NEXT:     %[[A:.*]] = memref.alloc() : memref<1024x1024xf16>
// CHECK-NEXT:     %[[B:.*]] = memref.alloc() : memref<1024x1024xf16>
// CHECK-NEXT:     %[[C:.*]] = memref.alloc() : memref<1024x1024xf32>
// CHECK-NEXT:     affine.for %[[I:.*]] = 0 to 1024 step 128 {
// CHECK-NEXT:       affine.for %[[J:.*]] = 0 to 1024 step 128 {
// CHECK-NEXT:         %[[BS:.*]] = memref.get_global @frag_B : memref<16x128xf16, 3>
// CHECK-NEXT:         %[[AS:.*]] = memref.get_global @frag_A : memref<128x16xf16, 3>
// CHECK-NEXT:         affine.for %[[II:.*]] = 0 to 128 step 32 {
// CHECK-NEXT:           affine.for %[[JJ:.*]] = 0 to 128 step 32 {
// CHECK-NEXT:             %{{.*}} = affine.apply #map0(%[[I]], %[[II]])
// CHECK-NEXT:             %{{.*}} = affine.apply #map0(%[[J]], %[[JJ]])
// CHECK-NEXT:             %{{.*}} = gpu.subgroup_mma_load_matrix %[[C]][%{{.*}}, %{{.*}}] {leadDimension = 1024 : index} : memref<1024x1024xf32> -> !gpu.mma_matrix<16x16xf32, "COp">
// CHECK-NEXT:             %{{.*}} = affine.apply #map1(%[[I]], %[[II]])
// CHECK-NEXT:             %{{.*}} = gpu.subgroup_mma_load_matrix %[[C]][%{{.*}}, %{{.*}}] {leadDimension = 1024 : index} : memref<1024x1024xf32> -> !gpu.mma_matrix<16x16xf32, "COp">
// CHECK-NEXT:             %{{.*}} = affine.apply #map1(%[[J]], %[[JJ]])
// CHECK-NEXT:             %{{.*}} = gpu.subgroup_mma_load_matrix %[[C]][%{{.*}}, %{{.*}}] {leadDimension = 1024 : index} : memref<1024x1024xf32> -> !gpu.mma_matrix<16x16xf32, "COp">
// CHECK-NEXT:             %{{.*}} = gpu.subgroup_mma_load_matrix %[[C]][%{{.*}}, %{{.*}}] {leadDimension = 1024 : index} : memref<1024x1024xf32> -> !gpu.mma_matrix<16x16xf32, "COp">
// CHECK-NEXT:             %{{.*}}:4 = affine.for %arg4 = 0 to 1024 step 16 iter_args(%arg5 = %{{.*}}, %arg6 = %{{.*}}, %arg7 = %{{.*}}, %arg8 = %{{.*}}) -> (!gpu.mma_matrix<16x16xf32, "COp">, !gpu.mma_matrix<16x16xf32, "COp">, !gpu.mma_matrix<16x16xf32, "COp">, !gpu.mma_matrix<16x16xf32, "COp">) {
// CHECK-NEXT:               affine.for %arg9 = #map2(%arg4) to #map3(%arg4) {
// CHECK-NEXT:                 affine.for %arg10 = #map2(%[[J]]) to #map4(%[[J]]) {
// CHECK-NEXT:                   %{{.*}} = affine.load %[[B]][%arg9, %arg10] : memref<1024x1024xf16>
// CHECK-NEXT:                   affine.store %{{.*}}, %[[BS]][%arg9 - %arg4, %arg10 - %[[J]]] : memref<16x128xf16, 3>
// CHECK-NEXT:                 }
// CHECK-NEXT:               } {isCopyLoopNest = true}
// CHECK-NEXT:               affine.for %arg9 = #map2(%[[I]]) to #map4(%[[I]]) {
// CHECK-NEXT:                 affine.for %arg10 = #map2(%arg4) to #map3(%arg4) {
// CHECK-NEXT:                   %{{.*}} = affine.load %[[A]][%arg9, %arg10] : memref<1024x1024xf16>
// CHECK-NEXT:                   affine.store %{{.*}}, %[[AS]][%arg9 - %[[I]], %arg10 - %arg4] : memref<128x16xf16, 3>
// CHECK-NEXT:                 }
// CHECK-NEXT:               } {isCopyLoopNest = true}
// CHECK-NEXT:               %{{.*}}:4 = affine.for %arg9 = 0 to 16 step 16 iter_args(%arg10 = %arg5, %arg11 = %arg6, %arg12 = %arg7, %arg13 = %arg8) ->  (!gpu.mma_matrix<16x16xf32, "COp">, !gpu.mma_matrix<16x16xf32, "COp">, !gpu.mma_matrix<16x16xf32, "COp">, !gpu.mma_matrix<16x16xf32, "COp">) {
// CHECK-NEXT:                 %{{.*}} = gpu.subgroup_mma_load_matrix %[[AS]][%[[II]], %arg9] {leadDimension = 16 : index} : memref<128x16xf16, 3> -> !gpu.mma_matrix<16x16xf16, "AOp">
// CHECK-NEXT:                 %{{.*}} = gpu.subgroup_mma_load_matrix %[[BS]][%arg9, %[[JJ]]] {leadDimension = 128 : index} : memref<16x128xf16, 3> -> !gpu.mma_matrix<16x16xf16, "BOp">
// CHECK-NEXT:                 %{{.*}} = gpu.subgroup_mma_compute %{{.*}}, %{{.*}}, %arg10 : !gpu.mma_matrix<16x16xf16, "AOp">, !gpu.mma_matrix<16x16xf16, "BOp"> -> !gpu.mma_matrix<16x16xf32, "COp">
// CHECK-NEXT:                 %{{.*}} = affine.apply #map3(%[[II]])
// CHECK-NEXT:                 %{{.*}} = gpu.subgroup_mma_load_matrix %[[AS]][%{{.*}}, %arg9] {leadDimension = 16 : index} : memref<128x16xf16, 3> -> !gpu.mma_matrix<16x16xf16, "AOp">
// CHECK-NEXT:                 %{{.*}} = gpu.subgroup_mma_load_matrix %[[BS]][%arg9, %[[JJ]]] {leadDimension = 128 : index} : memref<16x128xf16, 3> -> !gpu.mma_matrix<16x16xf16, "BOp">
// CHECK-NEXT:                 %{{.*}} = gpu.subgroup_mma_compute %{{.*}}, %{{.*}}, %arg11 : !gpu.mma_matrix<16x16xf16, "AOp">, !gpu.mma_matrix<16x16xf16, "BOp"> -> !gpu.mma_matrix<16x16xf32, "COp">
// CHECK-NEXT:                 %{{.*}} = gpu.subgroup_mma_load_matrix %[[AS]][%[[II]], %arg9] {leadDimension = 16 : index} : memref<128x16xf16, 3> -> !gpu.mma_matrix<16x16xf16, "AOp">
// CHECK-NEXT:                 %{{.*}} = affine.apply #map3(%[[JJ]])
// CHECK-NEXT:                 %{{.*}} = gpu.subgroup_mma_load_matrix %[[BS]][%arg9, %{{.*}}] {leadDimension = 128 : index} : memref<16x128xf16, 3> -> !gpu.mma_matrix<16x16xf16, "BOp">
// CHECK-NEXT:                 %{{.*}} = gpu.subgroup_mma_compute %{{.*}}, %{{.*}}, %arg12 : !gpu.mma_matrix<16x16xf16, "AOp">, !gpu.mma_matrix<16x16xf16, "BOp"> -> !gpu.mma_matrix<16x16xf32, "COp">
// CHECK-NEXT:                 %{{.*}} = gpu.subgroup_mma_load_matrix %[[AS]][%{{.*}}, %arg9] {leadDimension = 16 : index} : memref<128x16xf16, 3> -> !gpu.mma_matrix<16x16xf16, "AOp">
// CHECK-NEXT:                 %{{.*}} = gpu.subgroup_mma_load_matrix %[[BS]][%arg9, %{{.*}}] {leadDimension = 128 : index} : memref<16x128xf16, 3> -> !gpu.mma_matrix<16x16xf16, "BOp">
// CHECK-NEXT:                 %{{.*}} = gpu.subgroup_mma_compute %{{.*}}, %{{.*}}, %arg13 : !gpu.mma_matrix<16x16xf16, "AOp">, !gpu.mma_matrix<16x16xf16, "BOp"> -> !gpu.mma_matrix<16x16xf32, "COp">
// CHECK-NEXT:                 affine.yield %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}} : !gpu.mma_matrix<16x16xf32, "COp">, !gpu.mma_matrix<16x16xf32, "COp">, !gpu.mma_matrix<16x16xf32, "COp">, !gpu.mma_matrix<16x16xf32, "COp">
// CHECK-NEXT:               }
// CHECK-NEXT:               affine.yield %{{.*}}#0, %{{.*}}#1, %{{.*}}#2, %{{.*}}#3 : !gpu.mma_matrix<16x16xf32, "COp">, !gpu.mma_matrix<16x16xf32, "COp">, !gpu.mma_matrix<16x16xf32, "COp">, !gpu.mma_matrix<16x16xf32, "COp">
// CHECK-NEXT:             }
// CHECK-NEXT:             gpu.subgroup_mma_store_matrix %{{.*}}#0, %[[C]][%{{.*}}, %{{.*}}] {leadDimension = 1024 : index} : !gpu.mma_matrix<16x16xf32, "COp">, memref<1024x1024xf32>
// CHECK-NEXT:             gpu.subgroup_mma_store_matrix %{{.*}}#1, %[[C]][%{{.*}}, %{{.*}}] {leadDimension = 1024 : index} : !gpu.mma_matrix<16x16xf32, "COp">, memref<1024x1024xf32>
// CHECK-NEXT:             gpu.subgroup_mma_store_matrix %{{.*}}#2, %[[C]][%{{.*}}, %{{.*}}] {leadDimension = 1024 : index} : !gpu.mma_matrix<16x16xf32, "COp">, memref<1024x1024xf32>
// CHECK-NEXT:             gpu.subgroup_mma_store_matrix %{{.*}}#3, %[[C]][%{{.*}}, %{{.*}}] {leadDimension = 1024 : index} : !gpu.mma_matrix<16x16xf32, "COp">, memref<1024x1024xf32>
// CHECK-NEXT:           }
// CHECK-NEXT:         } {isComputeLoopNest = true}
// CHECK-NEXT:       }
// CHECK-NEXT:     }
// CHECK-NEXT:     return
// CHECK-NEXT:   }
// CHECK-NEXT: }
