// RUN: mlir-opt %s -test-gpu-matmul-fast-buffer-placement -canonicalize | FileCheck %s

// Which matrices to place can be specified by -test-gpu-fast-buffer-placement='matrices=str'
// where str = A/B/C/A,B/A,C/B,C/A,B,C. Any of the following options can be used.

// This pass places the matrices on matmul operation in gpu's fast buffer which
// are specified using `matrices` option.

#map0 = affine_map<(d0) -> (d0)>
#map1 = affine_map<(d0)[s0] -> (d0 + 64, s0)>
#map2 = affine_map<(d0)[s0] -> (d0 + 32, s0)>
#map3 = affine_map<(d0, d1)[s0] -> (d1 + 32, d0 + 64, s0)>
#map4 = affine_map<(d0, d1)[s0] -> (d1 + 16, d0 + 32, s0)>

func @matmul() {
  %0 = memref.alloc() : memref<1024x1024xf16>
  %1 = memref.alloc() : memref<1024x1024xf16>
  %2 = memref.alloc() : memref<1024x1024xf16>
  %c0 = constant 0 : index
  %c1 = constant 1 : index
  %3 = memref.dim %0, %c0 : memref<1024x1024xf16>
  %4 = memref.dim %1, %c1 : memref<1024x1024xf16>
  %5 = memref.dim %0, %c1 : memref<1024x1024xf16>
  affine.for %arg0 = 0 to %3 step 64 {
    affine.for %arg1 = 0 to %4 step 64 {
      affine.for %arg2 = 0 to %5 step 32 {
        affine.for %arg3 = #map0(%arg0) to min #map1(%arg0)[%3] step 32 {
          affine.for %arg4 = #map0(%arg1) to min #map1(%arg1)[%4] step 32 {
            affine.for %arg5 = #map0(%arg2) to min #map2(%arg2)[%5] step 16 {
              affine.for %arg6 = #map0(%arg3) to min #map3(%arg0, %arg3)[%3] {
                affine.for %arg7 = #map0(%arg4) to min #map3(%arg1, %arg4)[%4] {
                  affine.for %arg8 = #map0(%arg5) to min #map4(%arg2, %arg5)[%5] {
                    %6 = affine.load %0[%arg6, %arg8] : memref<1024x1024xf16>
                    %7 = affine.load %1[%arg8, %arg7] : memref<1024x1024xf16>
                    %8 = affine.load %2[%arg6, %arg7] : memref<1024x1024xf16>
                    %9 = mulf %6, %7 : f16
                    %10 = addf %8, %9 : f16
                    affine.store %10, %2[%arg6, %arg7] : memref<1024x1024xf16>
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

// CHECK-LABEL:func @matmul() {
// CHECK:  memref.alloc() : memref<1024x1024xf16>
// CHECK-NEXT:  memref.alloc() : memref<1024x1024xf16>
// CHECK-NEXT:  memref.alloc() : memref<1024x1024xf16>
// CHECK-NEXT:  affine.for %arg{{.*}} = 0 to 1024 step 64 {
// CHECK-NEXT:    affine.for %arg{{.*}} = 0 to 1024 step 64 {
// CHECK-NEXT:      memref.alloca() : memref<32x64xf16, 3>
// CHECK-NEXT:      memref.alloca() : memref<64x32xf16, 3>
// CHECK-NEXT:      affine.for %arg{{.*}} = 0 to 1024 step 32 {
// CHECK-NEXT:        affine.for %arg{{.*}} = #map{{.*}}(%arg{{.*}}) to min #map{{.*}}(%arg{{.*}}) {
// CHECK-NEXT:          affine.for %arg{{.*}} = #map{{.*}}(%arg{{.*}}) to min #map{{.*}}(%arg{{.*}}) {
// CHECK-NEXT:            affine.load %{{.*}}[%arg{{.*}}, %arg{{.*}}] : memref<1024x1024xf16>
// CHECK-NEXT:            affine.store %{{.*}}, %{{.*}}[%arg{{.*}} - %arg{{.*}},
// %arg{{.*}} - %arg{{.*}}] : memref<32x64xf16, 3>
// CHECK-NEXT:          }
// CHECK-NEXT:        } {isCopyLoopNest = true}
// CHECK-NEXT:        affine.for %arg{{.*}} = #map{{.*}}(%arg{{.*}}) to min #map{{.*}}(%arg{{.*}}) {
// CHECK-NEXT:          affine.for %arg{{.*}} = #map{{.*}}(%arg{{.*}}) to min #map{{.*}}(%arg{{.*}}) {
// CHECK-NEXT:            affine.load %{{.*}}[%arg{{.*}}, %arg{{.*}}] : memref<1024x1024xf16>
// CHECK-NEXT:            affine.store %{{.*}}, %{{.*}}[%arg{{.*}} - %arg{{.*}},
// %arg{{.*}} - %arg{{.*}}] : memref<64x32xf16, 3>
// CHECK-NEXT:          }
// CHECK-NEXT:        } {isCopyLoopNest = true}
// CHECK-NEXT:        affine.for %arg{{.*}} = #map{{.*}}(%arg{{.*}}) to min #map{{.*}}(%arg{{.*}}) step 32 {
// CHECK-NEXT:          affine.for %arg{{.*}} = #map{{.*}}(%arg{{.*}}) to min #map{{.*}}(%arg{{.*}}) step 32 {
// CHECK-NEXT:            affine.for %arg{{.*}} = #map{{.*}}(%arg{{.*}}) to min #map{{.*}}(%arg{{.*}}) step 16 {
// CHECK-NEXT:              affine.for %arg{{.*}} = #map{{.*}}(%arg{{.*}}) to min #map{{.*}}(%arg{{.*}}, %arg{{.*}}) {
// CHECK-NEXT:                affine.for %arg{{.*}} = #map{{.*}}(%arg{{.*}}) to min #map{{.*}}(%arg{{.*}}, %arg{{.*}}) {
// CHECK-NEXT:                  affine.for %arg{{.*}} = #map{{.*}}(%arg{{.*}}) to min #map{{.*}}(%arg{{.*}}, %arg{{.*}}) {
// CHECK-NEXT:                    affine.load %{{.*}}[-%arg{{.*}} + %arg{{.*}}, -%arg{{.*}} + %arg{{.*}}] : memref<64x32xf16, 3>
// CHECK-NEXT:                    affine.load %{{.*}}[-%arg{{.*}} + %arg{{.*}}, -%arg{{.*}} + %arg{{.*}}] : memref<32x64xf16, 3>
// CHECK-NEXT:                    affine.load %{{.*}}[%arg{{.*}}, %arg{{.*}}] : memref<1024x1024xf16>
// CHECK-NEXT:                    mulf %{{.*}}, %{{.*}} : f16
// CHECK-NEXT:                    addf %{{.*}}, %{{.*}} : f16
// CHECK-NEXT:                    affine.store %{{.*}}, %{{.*}}[%arg{{.*}}, %arg{{.*}}] : memref<1024x1024xf16>
// CHECK-NEXT:                  }
// CHECK-NEXT:                }
// CHECK-NEXT:              }
// CHECK-NEXT:            }
// CHECK-NEXT:          }
// CHECK-NEXT:        } {isComputeLoopNest = true}
// CHECK-NEXT:      }
// CHECK-NEXT:    }
// CHECK-NEXT:  }
// CHECK-NEXT:  return
// CHECK-NEXT:}
