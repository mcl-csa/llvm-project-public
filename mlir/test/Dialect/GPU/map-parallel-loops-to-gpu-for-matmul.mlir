// RUN: mlir-opt %s --test-gpu-matmul-parallel-loop-mapping | FileCheck %s

func @matmul() {
  %c0 = constant 0 : index
  %c64 = constant 64 : index
  %c1 = constant 1 : index
  %c-1 = constant -1 : index
  %c32 = constant 32 : index
  %c1024 = constant 1024 : index
  %c16 = constant 16 : index
  %0 = alloc() : memref<1024x1024xf16>
  %1 = alloc() : memref<1024x1024xf16>
  %2 = alloc() : memref<1024x1024xf16>
  scf.parallel (%arg0, %arg1) = (%c0, %c0) to (%c1024, %c1024) step (%c64, %c64) {
    %3 = alloca() : memref<64x64xf16, 3>
    %4 = alloca() : memref<32x64xf16, 3>
    %5 = alloca() : memref<64x32xf16, 3>
    %6 = addi %arg0, %c64 : index
    %7 = cmpi slt, %6, %c1024 : index
    %8 = select %7, %6, %c1024 : index
    %9 = addi %arg1, %c64 : index
    %10 = cmpi slt, %9, %c1024 : index
    %11 = select %10, %9, %c1024 : index
    %12 = addi %arg0, %c64 : index
    %13 = cmpi slt, %12, %c1024 : index
    %14 = select %13, %12, %c1024 : index
    %15 = addi %arg1, %c64 : index
    %16 = cmpi slt, %15, %c1024 : index
    %17 = select %16, %15, %c1024 : index
    scf.parallel (%arg2, %arg3) = (%arg0, %arg1) to (%14, %17) step (%c32, %c32) {
      %18 = muli %arg0, %c-1 : index
      %19 = addi %18, %arg2 : index
      %20 = addi %19, %c16 : index
      %21 = muli %arg1, %c-1 : index
      %22 = addi %21, %arg3 : index
      %23 = addi %22, %c16 : index
      %24 = gpu.subgroup_mma_load_matrix %3[%19, %22] {leadDimension = 64 : index, operand = "COp"} : memref<64x64xf16, 3> -> !gpu.mmafragment<4, vector<2xf16>>
      %25 = gpu.subgroup_mma_load_matrix %3[%20, %22] {leadDimension = 64 : index, operand = "COp"} : memref<64x64xf16, 3> -> !gpu.mmafragment<4, vector<2xf16>>
      %26 = gpu.subgroup_mma_load_matrix %3[%19, %23] {leadDimension = 64 : index, operand = "COp"} : memref<64x64xf16, 3> -> !gpu.mmafragment<4, vector<2xf16>>
      %27 = gpu.subgroup_mma_load_matrix %3[%20, %23] {leadDimension = 64 : index, operand = "COp"} : memref<64x64xf16, 3> -> !gpu.mmafragment<4, vector<2xf16>>
      %28:4 = scf.for %arg4 = %c0 to %c1024 step %c32 iter_args(%arg5 = %24, %arg6 = %25, %arg7 = %26, %arg8 = %27) -> (!gpu.mmafragment<4, vector<2xf16>>, !gpu.mmafragment<4, vector<2xf16>>, !gpu.mmafragment<4, vector<2xf16>>, !gpu.mmafragment<4, vector<2xf16>>) {
        %31 = addi %arg4, %c32 : index
        %32 = cmpi slt, %31, %c1024 : index
        %33 = select %32, %31, %c1024 : index
        %34 = addi %arg1, %c64 : index
        %35 = cmpi slt, %34, %c1024 : index
        %36 = select %35, %34, %c1024 : index
        %37 = addi %arg0, %c64 : index
        %38 = cmpi slt, %37, %c1024 : index
        %39 = select %38, %37, %c1024 : index
        %40 = addi %arg4, %c32 : index
        %41 = cmpi slt, %40, %c1024 : index
        %42 = select %41, %40, %c1024 : index
        scf.parallel (%arg9, %arg10) = (%arg4, %arg1) to (%33, %36) step (%c1, %c1) {
          %47 = load %1[%arg9, %arg10] : memref<1024x1024xf16>
          %48 = muli %arg4, %c-1 : index
          %49 = addi %48, %arg9 : index
          %50 = muli %arg1, %c-1 : index
          %51 = addi %50, %arg10 : index
          store %47, %4[%49, %51] : memref<32x64xf16, 3>
          scf.yield
        }
        scf.parallel (%arg9, %arg10) = (%arg0, %arg4) to (%39, %42) step (%c1, %c1) {
          %47 = load %0[%arg9, %arg10] : memref<1024x1024xf16>
          %48 = muli %arg0, %c-1 : index
          %49 = addi %48, %arg9 : index
          %50 = muli %arg4, %c-1 : index
          %51 = addi %50, %arg10 : index
          store %47, %5[%49, %51] : memref<64x32xf16, 3>
          scf.yield
        }
        %43 = addi %arg4, %c32 : index
        %44 = cmpi slt, %43, %c1024 : index
        %45 = select %44, %43, %c1024 : index
        %46:4 = scf.for %arg9 = %arg4 to %45 step %c16 iter_args(%arg10 = %24, %arg11 = %25, %arg12 = %26, %arg13 = %27) -> (!gpu.mmafragment<4, vector<2xf16>>, !gpu.mmafragment<4, vector<2xf16>>, !gpu.mmafragment<4, vector<2xf16>>, !gpu.mmafragment<4, vector<2xf16>>) {
          %47 = muli %arg4, %c-1 : index
          %48 = addi %47, %arg9 : index
          %49 = addi %18, %arg2 : index
          %50 = addi %19, %c16 : index
          %51 = gpu.subgroup_mma_load_matrix %5[%49, %48] {leadDimension = 32 : index, operand = "AOp"} : memref<64x32xf16, 3> -> !gpu.mmafragment<8, vector<2xf16>>
          %52 = gpu.subgroup_mma_load_matrix %5[%50, %48] {leadDimension = 32 : index, operand = "AOp"} : memref<64x32xf16, 3> -> !gpu.mmafragment<8, vector<2xf16>>
          %53 = addi %21, %arg3 : index
          %54 = addi %22, %c16 : index
          %55 = gpu.subgroup_mma_load_matrix %4[%49, %48] {operand = "BOp", leadDimension = 64 : index} : memref<32x64xf16, 3> -> !gpu.mma_matrix<16x16xf16, "BOp">
          %56 = gpu.subgroup_mma_compute %51, %55, %24 :!gpu.mma_matrix<16x16xf16, "AOp">, !gpu.mma_matrix<16x16xf16, "BOp"> -> !gpu.mma_matrix<16x16xf16, "COp">
          %57 = gpu.subgroup_mma_compute %52, %55, %25 : !gpu.mma_matrix<16x16xf16, "AOp">, !gpu.mma_matrix<16x16xf16, "BOp"> -> !gpu.mma_matrix<16x16xf16, "COp">
          %58 = gpu.subgroup_mma_load_matrix %4[%48, %54] {operand = "BOp", leadDimension = 64 : index} : memref<32x64xf16, 3> -> !gpu.mma_matrix<16x16xf16, "BOp">
          %59 = gpu.subgroup_mma_compute %51, %58, %26 : !gpu.mma_matrix<16x16xf16, "AOp">, !gpu.mma_matrix<16x16xf16, "BOp"> -> !gpu.mma_matrix<16x16xf16, "COp">
          %60 = gpu.subgroup_mma_compute %52, %58, %27 : !gpu.mma_matrix<16x16xf16, "AOp">, !gpu.mma_matrix<16x16xf16, "BOp"> -> !gpu.mma_matrix<16x16xf16, "COp">
          scf.yield %56, %57, %59, %60 : !gpu.mma_matrix<16x16xf16, "COp">, !gpu.mma_matrix<16x16xf16, "COp">, !gpu.mma_matrix<16x16xf16, "COp">, !gpu.mma_matrix<16x16xf16, "COp">
        }
        scf.yield %46#0, %46#1, %46#2, %46#3 : !gpu.mmafragment<4, vector<2xf16>>, !gpu.mmafragment<4, vector<2xf16>>, !gpu.mmafragment<4, vector<2xf16>>, !gpu.mmafragment<4, vector<2xf16>>
      }
      %29 = addi %arg0, %c16 : index
      %30 = addi %arg1, %c16 : index
      gpu.subgroup_mma_store_matrix %28#0, %2[%arg0, %arg1] {leadDimension = 1024 : index} : !gpu.mmafragment<4, vector<2xf16>>, memref<1024x1024xf16>
      gpu.subgroup_mma_store_matrix %28#1, %2[%29, %arg1] {leadDimension = 1024 : index} : !gpu.mmafragment<4, vector<2xf16>>, memref<1024x1024xf16>
      gpu.subgroup_mma_store_matrix %28#2, %2[%arg0, %30] {leadDimension = 1024 : index} : !gpu.mmafragment<4, vector<2xf16>>, memref<1024x1024xf16>
      gpu.subgroup_mma_store_matrix %28#3, %2[%29, %30] {leadDimension = 1024 : index} : !gpu.mmafragment<4, vector<2xf16>>, memref<1024x1024xf16>
      scf.yield
    }
    scf.yield
  }
  return
}

// CHECK-DAG: #map = affine_map<(d0) -> (d0)>

// CHECK-LABEL: func @matmul()
// CHECK:   scf.parallel (%arg{{.*}}, %arg{{.*}}) = (%c{{.*}}, %c{{.*}}) to (%c{{.*}}, %c{{.*}}) step (%c{{.*}}, %c{{.*}}) {
// CHECK:     scf.parallel (%arg{{.*}}, %arg{{.*}}) = (%arg{{.*}}, %arg{{.*}}) to (%{{.*}}, %{{.*}}) step (%c{{.*}}, %c{{.*}}) {
// CHECK:       %{{.*}} = scf.for %arg{{.*}} = %c{{.*}} to %c{{.*}} step %c{{.*}} iter_args(%arg{{.*}} = %{{.*}}, %arg{{.*}} = %{{.*}}, %arg{{.*}} = %{{.*}}, %arg{{.*}} = %{{.*}}) -> (!gpu.mmafragment<4, vector<2xf16>>, !gpu.mmafragment<4, vector<2xf16>>, !gpu.mmafragment<4, vector<2xf16>>, !gpu.mmafragment<4, vector<2xf16>>) {
// CHECK:         scf.parallel (%arg{{.*}}) = (%c{{.*}}) to (%{{.*}}) step (%{{.*}}) {
// CHECK:         } {mapping = [{bound = #map, map = #map, processor = 0 : i64}]}
// CHECK:         scf.parallel (%arg{{.*}}) = (%c{{.*}}) to (%{{.*}}) step (%c{{.*}}) {
// CHECK:         } {mapping = [{bound = #map, map = #map, processor = 0 : i64}]}
// CHECK:         %{{.*}} = scf.for %arg{{.*}} = %arg{{.*}} to %{{.*}} step %c{{.*}} iter_args(%arg{{.*}} = %{{.*}}, %arg{{.*}} = %{{.*}}, %arg{{.*}} = %{{.*}}, %arg{{.*}} = %{{.*}}) -> (!gpu.mmafragment<4, vector<2xf16>>, !gpu.mmafragment<4, vector<2xf16>>, !gpu.mmafragment<4, vector<2xf16>>, !gpu.mmafragment<4, vector<2xf16>>) {
// CHECK:           scf.yield %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}} : !gpu.mmafragment<4, vector<2xf16>>, !gpu.mmafragment<4, vector<2xf16>>, !gpu.mmafragment<4, vector<2xf16>>, !gpu.mmafragment<4, vector<2xf16>>
// CHECK-NEXT:         }
// CHECK-NEXT:         scf.yield %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}} : !gpu.mmafragment<4, vector<2xf16>>, !gpu.mmafragment<4, vector<2xf16>>, !gpu.mmafragment<4, vector<2xf16>>, !gpu.mmafragment<4, vector<2xf16>>
// CHECK-NEXT:       }
// CHECK:       scf.yield
// CHECK-NEXT:     } {mapping = [{bound = #map, map = #map, processor = 7 : i64}, {bound = #map, map = #map, processor = 6 : i64}]}
// CHECK-NEXT:     scf.yield
// CHECK-NEXT:   } {mapping = [{bound = #map, map = #map, processor = 1 : i64}, {bound = #map, map = #map, processor = 0 : i64}]}
