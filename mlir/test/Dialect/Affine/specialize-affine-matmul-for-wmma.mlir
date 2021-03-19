// RUN: mlir-opt --test-specialize-affine-matmul-for-wmma="accum=f32 load-store-width=128" --canonicalize --cse %s | FileCheck %s

#map0 = affine_map<(d0) -> (d0)>
#map1 = affine_map<(d0) -> (d0 + 16)>
#map2 = affine_map<(d0) -> (d0 + 128)>
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
