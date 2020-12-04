// RUN: mlir-opt -test-collapse-affine-parallel %s -split-input-file | FileCheck %s

#map0 = affine_map<(d0)[s0] -> (d0 + s0)>
#map1 = affine_map<(d0) -> (d0)>
#map2 = affine_map<(d0)[s0] -> (d0 + 6, s0)>

func @collapse_affine_parallel1() {
  %0 = memref.alloc() : memref<1024x1024xf32>
  %c0 = constant 0 : index
  %c1 = constant 1 : index
  %3 = memref.dim %0, %c0 : memref<1024x1024xf32>
  %4 = memref.dim %0, %c1 : memref<1024x1024xf32>
  affine.for %arg0 = 0 to %4 step 6 {
    %6 = affine.apply #map0(%arg0)[%3]
    %7 = affine.apply #map0(%arg0)[%4]
    affine.parallel (%arg1) = (%arg0) to (%6) {
      affine.parallel (%arg2) = (0) to (%7) {
        affine.for %arg3 = #map1(%arg0) to min #map2(%arg0)[%4] {
          %8 = affine.load %0[%arg1, %arg3] : memref<1024x1024xf32>
          %9 = affine.load %0[%arg3, %arg2] : memref<1024x1024xf32>
          %11 = mulf %8, %9 : f32
          affine.store %11, %0[%arg1, %arg2] : memref<1024x1024xf32>
        }
      }
    }
  }
  return
}

// CHECK-LABEL:func @collapse_affine_parallel1() {
// CHECK:    affine.parallel (%arg{{.*}}, %arg{{.*}}) = (%arg{{.*}}, 0) to (%{{.*}}, %{{.*}}) {
// CHECK-NEXT:      affine.for %arg{{.*}} = #map{{.*}}(%arg{{.*}}) to min #map{{.*}}(%arg{{.*}})[%{{.*}}] {

// -----

#map0 = affine_map<(d0)[s0] -> (d0 + s0)>
#map1 = affine_map<(d0) -> (d0)>
#map2 = affine_map<(d0)[s0] -> (d0 + 6, s0)>

func @collapse_affine_parallel2(%arg0: index, %arg1: index, %arg2: index, %arg3: index) {
  %0 = memref.alloc() : memref<1024x1024xf32>
  %1 = memref.alloc() : memref<1024x1024xf32>
  %2 = memref.alloc() : memref<1024x1024xf32>
  %c0 = constant 0 : index
  %c1 = constant 1 : index
  %3 = memref.dim %0, %c0 : memref<1024x1024xf32>
  %4 = memref.dim %1, %c1 : memref<1024x1024xf32>
  %5 = memref.dim %0, %c1 : memref<1024x1024xf32>
  affine.parallel (%arg4) = (%arg3 + 32) to (%arg0) {
    affine.parallel (%arg5) = (%arg2) to (%arg1) {
      affine.for %arg6 = 0 to %5 step 6 {
        %6 = affine.apply #map0(%arg4)[%3]
        %7 = affine.apply #map0(%arg5)[%4]
        affine.parallel (%arg7) = (%arg4) to (%6) {
          affine.parallel (%arg8) = (%arg5) to (%7) {
            affine.for %arg9 = #map1(%arg6) to min #map2(%arg6)[%5] {
              %8 = affine.load %0[%arg7, %arg9] : memref<1024x1024xf32>
              %9 = affine.load %1[%arg9, %arg8] : memref<1024x1024xf32>
              %10 = affine.load %2[%arg7, %arg8] : memref<1024x1024xf32>
              %11 = mulf %8, %9 : f32
              %12 = addf %10, %11 : f32
              affine.store %12, %2[%arg7, %arg8] : memref<1024x1024xf32>
            }
          }
        }
      }
    }
  }
  return
}

// CHECK-LABEL:func @collapse_affine_parallel2(%arg{{.*}}: index, %arg{{.*}}: index, %arg{{.*}}: index, %arg{{.*}}: index) {
// CHECK:  affine.parallel (%arg{{.*}}, %arg{{.*}}) = (%arg{{.*}} + 32, %arg{{.*}}) to (%arg{{.*}}, %arg{{.*}}) {
// CHECK-NEXT:    affine.for %arg{{.*}} = 0 to %{{.*}} step 6 {
// CHECK-NEXT:      affine.apply #map{{.*}}(%arg{{.*}})[%{{.*}}]
// CHECK-NEXT:      affine.apply #map{{.*}}(%arg{{.*}})[%{{.*}}]
// CHECK-NEXT:      affine.parallel (%arg{{.*}}, %arg{{.*}}) = (%arg{{.*}}, %arg{{.*}}) to (%{{.*}}, %{{.*}}) {
// CHECK-NEXT:        affine.for %arg{{.*}} = #map{{.*}}(%arg{{.*}}) to min #map{{.*}}(%arg{{.*}})[%{{.*}}] {

// -----

#map0 = affine_map<(d0)[s0] -> (d0 + s0)>
#map1 = affine_map<(d0) -> (d0)>
#map2 = affine_map<(d0)[s0] -> (d0 + 6, s0)>

func @collapse_imperfect_affine_parallel(%arg0: index, %arg1: index, %arg2: index, %arg3: index) {
  %0 = memref.alloc() : memref<1024x1024xf32>
  %1 = memref.alloc() : memref<1024x1024xf32>
  %2 = memref.alloc() : memref<1024x1024xf32>
  %c0 = constant 0 : index
  %c1 = constant 1 : index
  %3 = memref.dim %0, %c0 : memref<1024x1024xf32>
  %4 = memref.dim %1, %c1 : memref<1024x1024xf32>
  %5 = memref.dim %0, %c1 : memref<1024x1024xf32>
  affine.parallel (%arg4) = (%arg3) to (%arg0) {
    affine.parallel (%arg5) = (%arg2) to (%arg1) {
      affine.parallel (%arg6) = (0) to (%5) step (6) {
        %6 = affine.apply #map0(%arg4)[%3]
        %7 = affine.apply #map0(%arg5)[%4]
        affine.parallel (%arg7) = (%arg4) to (%6) {
          affine.parallel (%arg8) = (%arg5) to (%7) {
            affine.for %arg9 = #map1(%arg6) to min #map2(%arg6)[%5] {
              %8 = affine.load %0[%arg7, %arg9] : memref<1024x1024xf32>
              %9 = affine.load %1[%arg9, %arg8] : memref<1024x1024xf32>
              %10 = affine.load %2[%arg7, %arg8] : memref<1024x1024xf32>
              %11 = mulf %8, %9 : f32
              %12 = addf %10, %11 : f32
              affine.store %12, %2[%arg7, %arg8] : memref<1024x1024xf32>
            }
          }
        }
      }
      affine.parallel (%arg6) = (0) to (%5) step (6) {
        %6 = affine.apply #map0(%arg4)[%3]
        %7 = affine.apply #map0(%arg5)[%4]
        affine.parallel (%arg7) = (%arg4) to (%6) {
          affine.parallel (%arg8) = (%arg5) to (%7) {
            affine.for %arg9 = #map1(%arg6) to min #map2(%arg6)[%5] {
              %8 = affine.load %0[%arg7, %arg9] : memref<1024x1024xf32>
              %9 = affine.load %1[%arg9, %arg8] : memref<1024x1024xf32>
              %10 = affine.load %2[%arg7, %arg8] : memref<1024x1024xf32>
              %11 = mulf %8, %9 : f32
              %12 = addf %10, %11 : f32
              affine.store %12, %2[%arg7, %arg8] : memref<1024x1024xf32>
            }
          }
        }
      }
    }
  }
  return
}

// CHECK-LABEL:func @collapse_imperfect_affine_parallel(%arg{{.*}}: index, %arg{{.*}}: index, %arg{{.*}}: index, %arg{{.*}}: index) {
// CHECK:  affine.parallel (%arg{{.*}}, %arg{{.*}}) = (%arg{{.*}}, %arg{{.*}}) to (%arg{{.*}}, %arg{{.*}}) {
// CHECK-NEXT:    affine.parallel (%arg{{.*}}) = (0) to (%{{.*}}) step (6) {
// CHECK-NEXT:      affine.apply #map{{.*}}(%arg{{.*}})[%{{.*}}]
// CHECK-NEXT:      affine.apply #map{{.*}}(%arg{{.*}})[%{{.*}}]
// CHECK-NEXT:      affine.parallel (%arg{{.*}}, %arg{{.*}}) = (%arg{{.*}}, %arg{{.*}}) to (%{{.*}}, %{{.*}}) {
// CHECK-NEXT:        affine.for %arg{{.*}} = #map{{.*}}(%arg{{.*}}) to min #map{{.*}}(%arg{{.*}})[%{{.*}}] {
// CHECK:    affine.parallel (%arg{{.*}}) = (0) to (%{{.*}}) step (6) {
// CHECK-NEXT:      affine.apply #map{{.*}}(%arg{{.*}})[%{{.*}}]
// CHECK-NEXT:      affine.apply #map{{.*}}(%arg{{.*}})[%{{.*}}]
// CHECK-NEXT:      affine.parallel (%arg{{.*}}, %arg{{.*}}) = (%arg{{.*}}, %arg{{.*}}) to (%{{.*}}, %{{.*}}) {
// CHECK-NEXT:        affine.for %arg{{.*}} = #map{{.*}}(%arg{{.*}}) to min #map{{.*}}(%arg{{.*}})[%{{.*}}] {

// -----

#map0 = affine_map<(d0)[s0] -> (d0 + s0)>
#map1 = affine_map<(d0) -> (d0)>
#map2 = affine_map<(d0)[s0] -> (d0 + 6, s0)>

func @collapse_affine_parallel_3d() {
  %0 = memref.alloc() : memref<1024x1024xf32>
  %1 = memref.alloc() : memref<1024x1024xf32>
  %2 = memref.alloc() : memref<1024x1024xf32>
  %c0 = constant 0 : index
  %c1 = constant 1 : index
  %3 = memref.dim %0, %c0 : memref<1024x1024xf32>
  %4 = memref.dim %1, %c1 : memref<1024x1024xf32>
  %5 = memref.dim %0, %c1 : memref<1024x1024xf32>
  affine.parallel (%arg0) = (0) to (symbol(%3)) {
    affine.parallel (%arg1) = (0) to (symbol(%4)) {
      affine.parallel (%arg2) = (0) to (%5) step (6) {
        %6 = affine.apply #map0(%arg0)[%3]
        %7 = affine.apply #map0(%arg1)[%4]
        affine.parallel (%arg3) = (%arg0) to (%6) {
          affine.parallel (%arg4) = (%arg1) to (%7) {
            affine.for %arg5 = #map1(%arg2) to min #map2(%arg2)[%5] {
              %8 = affine.load %0[%arg3, %arg5] : memref<1024x1024xf32>
              %9 = affine.load %1[%arg5, %arg4] : memref<1024x1024xf32>
              %10 = affine.load %2[%arg3, %arg4] : memref<1024x1024xf32>
              %11 = mulf %8, %9 : f32
              %12 = addf %10, %11 : f32
              affine.store %12, %2[%arg3, %arg4] : memref<1024x1024xf32>
            }
          }
        }
      } {isBoundary = true}
    }
  }
  return
}

// CHECK-LABEL:func @collapse_affine_parallel_3d() {
// CHECK:  affine.parallel (%arg{{.*}}, %arg{{.*}}, %arg{{.*}}) = (0, 0, 0) to (symbol(%{{.*}}), symbol(%{{.*}}), %{{.*}}) step (1, 1, 6) {
// CHECK-NEXT:    affine.apply #map{{.*}}(%arg{{.*}})[%{{.*}}]
// CHECK-NEXT:    affine.apply #map{{.*}}(%arg{{.*}})[%{{.*}}]
// CHECK-NEXT:    affine.parallel (%arg{{.*}}, %arg{{.*}}) = (%arg{{.*}}, %arg{{.*}}) to (%{{.*}}, %{{.*}}) {
// CHECK-NEXT:      affine.for %arg{{.*}} = #map{{.*}}(%arg{{.*}}) to min #map{{.*}}(%arg{{.*}})[%{{.*}}] {
