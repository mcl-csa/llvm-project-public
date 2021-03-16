// mlir-opt affine-multi-tiled.mlir --test-specialize-affine-matmul-for-wmma=accum=f32 --canonicalize

#map0 = affine_map<(d0) -> (d0)>
#map1 = affine_map<(d0) -> (d0 + 64)>
module  {
  global_memref @asmem : memref<64x64xf16, 3>
  global_memref @bsmem : memref<64x64xf16, 3>
  func @main() {
    %0 = alloc() : memref<1024x1024xf16>
    %1 = alloc() : memref<1024x1024xf16>
    %2 = alloc() : memref<1024x1024xf16>
    %00 = memref_cast %0 : memref<1024x1024xf16> to memref<*xf16>
    gpu.host_register %00 : memref<*xf16>
    %11 = memref_cast %1 : memref<1024x1024xf16> to memref<*xf16>
    gpu.host_register %11 : memref<*xf16>
    %22 = memref_cast %2 : memref<1024x1024xf16> to memref<*xf16>
    gpu.host_register %22 : memref<*xf16>

    affine.for %arg0 = 0 to 1024 step 64 {
      affine.for %arg1 = 0 to 1024 step 64 {
        %3 = get_global_memref @asmem : memref<64x64xf16, 3>
        %4 = get_global_memref @bsmem : memref<64x64xf16, 3>
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
