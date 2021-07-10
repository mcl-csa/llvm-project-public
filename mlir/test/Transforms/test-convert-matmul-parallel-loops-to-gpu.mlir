// RUN: mlir-opt %s --canonicalize --test-convert-matmul-parallel-loops-to-gpu --canonicalize --cse | FileCheck %s

#map = affine_map<(d0) -> (d0)>
module  {
  memref.global "public" @frag_A : memref<128x64xf16, 3>
  memref.global "public" @frag_B : memref<64x64xf16, 3>
  memref.global "public" @frag_B_padded : memref<64x72xf16, 3>
  memref.global "public" @frag_A_padded : memref<128x72xf16, 3>
  func @main() {
    %c960 = constant 960 : index
    %c-1 = constant -1 : index
    %c512 = constant 512 : index
    %c8 = constant 8 : index
    %c48 = constant 48 : index
    %c32 = constant 32 : index
    %c64 = constant 64 : index
    %c128 = constant 128 : index
    %c0 = constant 0 : index
    %c16 = constant 16 : index
    %cst = constant 0.000000e+00 : f32
    %c1 = constant 1 : index
    %c1024 = constant 1024 : index
    %c2147483648_i64 = constant 2147483648 : i64
    %0 = memref.alloc() : memref<1024x1024xf16>
    %1 = memref.alloc() : memref<1024x1024xf16>
    %2 = memref.alloc() : memref<1024x1024xf32>
    scf.for %arg0 = %c0 to %c1024 step %c1 {
      scf.for %arg1 = %c0 to %c1024 step %c1 {
        %16 = remi_signed %arg0, %c16 : index
        %17 = remi_signed %arg1, %c16 : index
        %18 = addi %16, %17 : index
        %19 = remi_signed %18, %c16 : index
        %20 = index_cast %19 : index to i16
        %21 = sitofp %20 : i16 to f16
        memref.store %21, %0[%arg0, %arg1] : memref<1024x1024xf16>
      }
    }
    scf.for %arg0 = %c0 to %c1024 step %c1 {
      scf.for %arg1 = %c0 to %c1024 step %c1 {
        %16 = remi_signed %arg0, %c16 : index
        %17 = remi_signed %arg1, %c16 : index
        %18 = addi %16, %17 : index
        %19 = remi_signed %18, %c16 : index
        %20 = index_cast %19 : index to i16
        %21 = sitofp %20 : i16 to f16
        memref.store %21, %1[%arg0, %arg1] : memref<1024x1024xf16>
      }
    }
    scf.for %arg0 = %c0 to %c1024 step %c1 {
      scf.for %arg1 = %c0 to %c1024 step %c1 {
        memref.store %cst, %2[%arg0, %arg1] : memref<1024x1024xf32>
      }
    }
    %3 = gpu.wait async 
    %memref, %asyncToken = gpu.alloc async [%3] () : memref<1024x1024xf16>
    %4 = memref_vector_cast %memref : memref<1024x1024xf16> to memref<1024x128xvector<8xf16>>
    %memref_0, %asyncToken_1 = gpu.alloc async [%3] () : memref<1024x1024xf16>
    %5 = memref_vector_cast %memref_0 : memref<1024x1024xf16> to memref<1024x128xvector<8xf16>>
    %memref_2, %asyncToken_3 = gpu.alloc async [%3] () : memref<1024x1024xf32>
    %6 = gpu.memcpy async [%3] %memref, %0 : memref<1024x1024xf16>, memref<1024x1024xf16>
    %7 = gpu.memcpy async [%3] %memref_0, %1 : memref<1024x1024xf16>, memref<1024x1024xf16>
    %8 = gpu.memcpy async [%3] %memref_2, %2 : memref<1024x1024xf32>, memref<1024x1024xf32>
    gpu.wait [%3]
    %9 = call @rtclock() : () -> f64
    scf.parallel (%arg0, %arg1) = (%c0, %c0) to (%c1024, %c1024) step (%c128, %c64) {
      %16 = memref.get_global @frag_B_padded : memref<64x72xf16, 3>
      %17 = memref_vector_cast %16 : memref<64x72xf16, 3> to memref<64x9xvector<8xf16>, 3>
      %18 = memref.get_global @frag_A_padded : memref<128x72xf16, 3>
      %19 = memref_vector_cast %18 : memref<128x72xf16, 3> to memref<128x9xvector<8xf16>, 3>
      scf.parallel (%arg2, %arg3) = (%c0, %c0) to (%c128, %c64) step (%c64, %c32) {
        %20 = addi %arg0, %arg2 : index
        %21 = addi %arg1, %arg3 : index
        %22 = gpu.subgroup_mma_load_matrix %memref_2[%20, %21] {leadDimension = 1024 : index} : memref<1024x1024xf32> -> !gpu.mma_matrix<16x16xf32, "COp">
        %23 = addi %20, %c16 : index
        %24 = gpu.subgroup_mma_load_matrix %memref_2[%23, %21] {leadDimension = 1024 : index} : memref<1024x1024xf32> -> !gpu.mma_matrix<16x16xf32, "COp">
        %25 = addi %20, %c32 : index
        %26 = gpu.subgroup_mma_load_matrix %memref_2[%25, %21] {leadDimension = 1024 : index} : memref<1024x1024xf32> -> !gpu.mma_matrix<16x16xf32, "COp">
        %27 = addi %20, %c48 : index
        %28 = gpu.subgroup_mma_load_matrix %memref_2[%27, %21] {leadDimension = 1024 : index} : memref<1024x1024xf32> -> !gpu.mma_matrix<16x16xf32, "COp">
        %29 = addi %21, %c16 : index
        %30 = gpu.subgroup_mma_load_matrix %memref_2[%20, %29] {leadDimension = 1024 : index} : memref<1024x1024xf32> -> !gpu.mma_matrix<16x16xf32, "COp">
        %31 = gpu.subgroup_mma_load_matrix %memref_2[%23, %29] {leadDimension = 1024 : index} : memref<1024x1024xf32> -> !gpu.mma_matrix<16x16xf32, "COp">
        %32 = gpu.subgroup_mma_load_matrix %memref_2[%25, %29] {leadDimension = 1024 : index} : memref<1024x1024xf32> -> !gpu.mma_matrix<16x16xf32, "COp">
        %33 = gpu.subgroup_mma_load_matrix %memref_2[%27, %29] {leadDimension = 1024 : index} : memref<1024x1024xf32> -> !gpu.mma_matrix<16x16xf32, "COp">
        scf.parallel (%arg4) = (%c0) to (%c512) step (%c1) {
          %36 = remi_signed %arg4, %c8 : index
          %37 = divi_signed %arg4, %c8 : index
          %38 = muli %36, %c8 : index
          %39 = addi %arg1, %38 : index
          %40 = cmpi slt, %39, %c0 : index
          %41 = subi %c-1, %39 : index
          %42 = select %40, %41, %39 : index
          %43 = divi_signed %42, %c8 : index
          %44 = subi %c-1, %43 : index
          %45 = select %40, %44, %43 : index
          %46 = memref.load %5[%37, %45] : memref<1024x128xvector<8xf16>>
          %47 = cmpi slt, %38, %c0 : index
          %48 = subi %c-1, %38 : index
          %49 = select %47, %48, %38 : index
          %50 = divi_signed %49, %c8 : index
          %51 = subi %c-1, %50 : index
          %52 = select %47, %51, %50 : index
          memref.store %46, %17[%37, %52] : memref<64x9xvector<8xf16>, 3>
          scf.yield
        } {mapping = [{bound = #map, map = #map, processor = 0 : i64}]}
        scf.parallel (%arg4) = (%c0) to (%c1024) step (%c1) {
          %36 = remi_signed %arg4, %c8 : index
          %37 = divi_signed %arg4, %c8 : index
          %38 = muli %36, %c8 : index
          %39 = addi %arg0, %37 : index
          %40 = cmpi slt, %38, %c0 : index
          %41 = subi %c-1, %38 : index
          %42 = select %40, %41, %38 : index
          %43 = divi_signed %42, %c8 : index
          %44 = subi %c-1, %43 : index
          %45 = select %40, %44, %43 : index
          %46 = memref.load %4[%39, %45] : memref<1024x128xvector<8xf16>>
          memref.store %46, %19[%37, %45] : memref<128x9xvector<8xf16>, 3>
          scf.yield
        } {mapping = [{bound = #map, map = #map, processor = 0 : i64}]}
        gpu.barrier
        %34:8 = scf.for %arg4 = %c0 to %c960 step %c64 iter_args(%arg5 = %22, %arg6 = %24, %arg7 = %26, %arg8 = %28, %arg9 = %30, %arg10 = %31, %arg11 = %32, %arg12 = %33) -> (!gpu.mma_matrix<16x16xf32, "COp">, !gpu.mma_matrix<16x16xf32, "COp">, !gpu.mma_matrix<16x16xf32, "COp">, !gpu.mma_matrix<16x16xf32, "COp">, !gpu.mma_matrix<16x16xf32, "COp">, !gpu.mma_matrix<16x16xf32, "COp">, !gpu.mma_matrix<16x16xf32, "COp">, !gpu.mma_matrix<16x16xf32, "COp">) {
          scf.parallel (%arg13) = (%c0) to (%c512) step (%c1) {
            %37 = remi_signed %arg13, %c8 : index
            %38 = divi_signed %arg13, %c8 : index
            %39 = muli %37, %c8 : index
            %40 = addi %arg4, %38 : index
            %41 = addi %40, %c64 : index
            %42 = addi %arg1, %39 : index
            %43 = cmpi slt, %42, %c0 : index
            %44 = subi %c-1, %42 : index
            %45 = select %43, %44, %42 : index
            %46 = divi_signed %45, %c8 : index
            %47 = subi %c-1, %46 : index
            %48 = select %43, %47, %46 : index
            %49 = memref.load %5[%41, %48] : memref<1024x128xvector<8xf16>>
            %50 = cmpi slt, %39, %c0 : index
            %51 = subi %c-1, %39 : index
            %52 = select %50, %51, %39 : index
            %53 = divi_signed %52, %c8 : index
            %54 = subi %c-1, %53 : index
            %55 = select %50, %54, %53 : index
            memref.store %49, %17[%38, %55] : memref<64x9xvector<8xf16>, 3>
            scf.yield
          } {mapping = [{bound = #map, map = #map, processor = 0 : i64}]}
          scf.parallel (%arg13) = (%c0) to (%c1024) step (%c1) {
            %37 = remi_signed %arg13, %c8 : index
            %38 = divi_signed %arg13, %c8 : index
            %39 = muli %37, %c8 : index
            %40 = addi %arg0, %38 : index
            %41 = addi %arg4, %39 : index
            %42 = cmpi slt, %41, %c0 : index
            %43 = subi %c-1, %41 : index
            %44 = select %42, %43, %41 : index
            %45 = divi_signed %44, %c8 : index
            %46 = subi %c-1, %45 : index
            %47 = select %42, %46, %45 : index
            %48 = addi %47, %c8 : index
            %49 = memref.load %4[%40, %48] : memref<1024x128xvector<8xf16>>
            %50 = cmpi slt, %39, %c0 : index
            %51 = subi %c-1, %39 : index
            %52 = select %50, %51, %39 : index
            %53 = divi_signed %52, %c8 : index
            %54 = subi %c-1, %53 : index
            %55 = select %50, %54, %53 : index
            memref.store %49, %19[%38, %55] : memref<128x9xvector<8xf16>, 3>
            scf.yield
          } {mapping = [{bound = #map, map = #map, processor = 0 : i64}]}
          %36:8 = scf.for %arg13 = %c0 to %c64 step %c16 iter_args(%arg14 = %arg5, %arg15 = %arg6, %arg16 = %arg7, %arg17 = %arg8, %arg18 = %arg9, %arg19 = %arg10, %arg20 = %arg11, %arg21 = %arg12) -> (!gpu.mma_matrix<16x16xf32, "COp">, !gpu.mma_matrix<16x16xf32, "COp">, !gpu.mma_matrix<16x16xf32, "COp">, !gpu.mma_matrix<16x16xf32, "COp">, !gpu.mma_matrix<16x16xf32, "COp">, !gpu.mma_matrix<16x16xf32, "COp">, !gpu.mma_matrix<16x16xf32, "COp">, !gpu.mma_matrix<16x16xf32, "COp">) {
            %37 = gpu.subgroup_mma_load_matrix %18[%arg2, %arg13] {leadDimension = 72 : index} : memref<128x72xf16, 3> -> !gpu.mma_matrix<16x16xf16, "AOp">
            %38 = gpu.subgroup_mma_load_matrix %16[%arg13, %arg3] {leadDimension = 72 : index} : memref<64x72xf16, 3> -> !gpu.mma_matrix<16x16xf16, "BOp">
            %39 = gpu.subgroup_mma_compute %37, %38, %arg14 : !gpu.mma_matrix<16x16xf16, "AOp">, !gpu.mma_matrix<16x16xf16, "BOp"> -> !gpu.mma_matrix<16x16xf32, "COp">
            %40 = addi %arg2, %c16 : index
            %41 = gpu.subgroup_mma_load_matrix %18[%40, %arg13] {leadDimension = 72 : index} : memref<128x72xf16, 3> -> !gpu.mma_matrix<16x16xf16, "AOp">
            %42 = gpu.subgroup_mma_load_matrix %16[%arg13, %arg3] {leadDimension = 72 : index} : memref<64x72xf16, 3> -> !gpu.mma_matrix<16x16xf16, "BOp">
            %43 = gpu.subgroup_mma_compute %41, %42, %arg15 : !gpu.mma_matrix<16x16xf16, "AOp">, !gpu.mma_matrix<16x16xf16, "BOp"> -> !gpu.mma_matrix<16x16xf32, "COp">
            %44 = addi %arg2, %c32 : index
            %45 = gpu.subgroup_mma_load_matrix %18[%44, %arg13] {leadDimension = 72 : index} : memref<128x72xf16, 3> -> !gpu.mma_matrix<16x16xf16, "AOp">
            %46 = gpu.subgroup_mma_load_matrix %16[%arg13, %arg3] {leadDimension = 72 : index} : memref<64x72xf16, 3> -> !gpu.mma_matrix<16x16xf16, "BOp">
            %47 = gpu.subgroup_mma_compute %45, %46, %arg16 : !gpu.mma_matrix<16x16xf16, "AOp">, !gpu.mma_matrix<16x16xf16, "BOp"> -> !gpu.mma_matrix<16x16xf32, "COp">
            %48 = addi %arg2, %c48 : index
            %49 = gpu.subgroup_mma_load_matrix %18[%48, %arg13] {leadDimension = 72 : index} : memref<128x72xf16, 3> -> !gpu.mma_matrix<16x16xf16, "AOp">
            %50 = gpu.subgroup_mma_load_matrix %16[%arg13, %arg3] {leadDimension = 72 : index} : memref<64x72xf16, 3> -> !gpu.mma_matrix<16x16xf16, "BOp">
            %51 = gpu.subgroup_mma_compute %49, %50, %arg17 : !gpu.mma_matrix<16x16xf16, "AOp">, !gpu.mma_matrix<16x16xf16, "BOp"> -> !gpu.mma_matrix<16x16xf32, "COp">
            %52 = gpu.subgroup_mma_load_matrix %18[%arg2, %arg13] {leadDimension = 72 : index} : memref<128x72xf16, 3> -> !gpu.mma_matrix<16x16xf16, "AOp">
            %53 = addi %arg3, %c16 : index
            %54 = gpu.subgroup_mma_load_matrix %16[%arg13, %53] {leadDimension = 72 : index} : memref<64x72xf16, 3> -> !gpu.mma_matrix<16x16xf16, "BOp">
            %55 = gpu.subgroup_mma_compute %52, %54, %arg18 : !gpu.mma_matrix<16x16xf16, "AOp">, !gpu.mma_matrix<16x16xf16, "BOp"> -> !gpu.mma_matrix<16x16xf32, "COp">
            %56 = gpu.subgroup_mma_load_matrix %18[%40, %arg13] {leadDimension = 72 : index} : memref<128x72xf16, 3> -> !gpu.mma_matrix<16x16xf16, "AOp">
            %57 = gpu.subgroup_mma_load_matrix %16[%arg13, %53] {leadDimension = 72 : index} : memref<64x72xf16, 3> -> !gpu.mma_matrix<16x16xf16, "BOp">
            %58 = gpu.subgroup_mma_compute %56, %57, %arg19 : !gpu.mma_matrix<16x16xf16, "AOp">, !gpu.mma_matrix<16x16xf16, "BOp"> -> !gpu.mma_matrix<16x16xf32, "COp">
            %59 = gpu.subgroup_mma_load_matrix %18[%44, %arg13] {leadDimension = 72 : index} : memref<128x72xf16, 3> -> !gpu.mma_matrix<16x16xf16, "AOp">
            %60 = gpu.subgroup_mma_load_matrix %16[%arg13, %53] {leadDimension = 72 : index} : memref<64x72xf16, 3> -> !gpu.mma_matrix<16x16xf16, "BOp">
            %61 = gpu.subgroup_mma_compute %59, %60, %arg20 : !gpu.mma_matrix<16x16xf16, "AOp">, !gpu.mma_matrix<16x16xf16, "BOp"> -> !gpu.mma_matrix<16x16xf32, "COp">
            %62 = gpu.subgroup_mma_load_matrix %18[%48, %arg13] {leadDimension = 72 : index} : memref<128x72xf16, 3> -> !gpu.mma_matrix<16x16xf16, "AOp">
            %63 = gpu.subgroup_mma_load_matrix %16[%arg13, %53] {leadDimension = 72 : index} : memref<64x72xf16, 3> -> !gpu.mma_matrix<16x16xf16, "BOp">
            %64 = gpu.subgroup_mma_compute %62, %63, %arg21 : !gpu.mma_matrix<16x16xf16, "AOp">, !gpu.mma_matrix<16x16xf16, "BOp"> -> !gpu.mma_matrix<16x16xf32, "COp">
            scf.yield %39, %43, %47, %51, %55, %58, %61, %64 : !gpu.mma_matrix<16x16xf32, "COp">, !gpu.mma_matrix<16x16xf32, "COp">, !gpu.mma_matrix<16x16xf32, "COp">, !gpu.mma_matrix<16x16xf32, "COp">, !gpu.mma_matrix<16x16xf32, "COp">, !gpu.mma_matrix<16x16xf32, "COp">, !gpu.mma_matrix<16x16xf32, "COp">, !gpu.mma_matrix<16x16xf32, "COp">
          }
          scf.yield %36#0, %36#1, %36#2, %36#3, %36#4, %36#5, %36#6, %36#7 : !gpu.mma_matrix<16x16xf32, "COp">, !gpu.mma_matrix<16x16xf32, "COp">, !gpu.mma_matrix<16x16xf32, "COp">, !gpu.mma_matrix<16x16xf32, "COp">, !gpu.mma_matrix<16x16xf32, "COp">, !gpu.mma_matrix<16x16xf32, "COp">, !gpu.mma_matrix<16x16xf32, "COp">, !gpu.mma_matrix<16x16xf32, "COp">
        }
        gpu.barrier
        %35:8 = scf.for %arg4 = %c0 to %c64 step %c16 iter_args(%arg5 = %34#0, %arg6 = %34#1, %arg7 = %34#2, %arg8 = %34#3, %arg9 = %34#4, %arg10 = %34#5, %arg11 = %34#6, %arg12 = %34#7) -> (!gpu.mma_matrix<16x16xf32, "COp">, !gpu.mma_matrix<16x16xf32, "COp">, !gpu.mma_matrix<16x16xf32, "COp">, !gpu.mma_matrix<16x16xf32, "COp">, !gpu.mma_matrix<16x16xf32, "COp">, !gpu.mma_matrix<16x16xf32, "COp">, !gpu.mma_matrix<16x16xf32, "COp">, !gpu.mma_matrix<16x16xf32, "COp">) {
          %36 = gpu.subgroup_mma_load_matrix %18[%arg2, %arg4] {leadDimension = 72 : index} : memref<128x72xf16, 3> -> !gpu.mma_matrix<16x16xf16, "AOp">
          %37 = gpu.subgroup_mma_load_matrix %16[%arg4, %arg3] {leadDimension = 72 : index} : memref<64x72xf16, 3> -> !gpu.mma_matrix<16x16xf16, "BOp">
          %38 = gpu.subgroup_mma_compute %36, %37, %arg5 : !gpu.mma_matrix<16x16xf16, "AOp">, !gpu.mma_matrix<16x16xf16, "BOp"> -> !gpu.mma_matrix<16x16xf32, "COp">
          %39 = addi %arg2, %c16 : index
          %40 = gpu.subgroup_mma_load_matrix %18[%39, %arg4] {leadDimension = 72 : index} : memref<128x72xf16, 3> -> !gpu.mma_matrix<16x16xf16, "AOp">
          %41 = gpu.subgroup_mma_load_matrix %16[%arg4, %arg3] {leadDimension = 72 : index} : memref<64x72xf16, 3> -> !gpu.mma_matrix<16x16xf16, "BOp">
          %42 = gpu.subgroup_mma_compute %40, %41, %arg6 : !gpu.mma_matrix<16x16xf16, "AOp">, !gpu.mma_matrix<16x16xf16, "BOp"> -> !gpu.mma_matrix<16x16xf32, "COp">
          %43 = addi %arg2, %c32 : index
          %44 = gpu.subgroup_mma_load_matrix %18[%43, %arg4] {leadDimension = 72 : index} : memref<128x72xf16, 3> -> !gpu.mma_matrix<16x16xf16, "AOp">
          %45 = gpu.subgroup_mma_load_matrix %16[%arg4, %arg3] {leadDimension = 72 : index} : memref<64x72xf16, 3> -> !gpu.mma_matrix<16x16xf16, "BOp">
          %46 = gpu.subgroup_mma_compute %44, %45, %arg7 : !gpu.mma_matrix<16x16xf16, "AOp">, !gpu.mma_matrix<16x16xf16, "BOp"> -> !gpu.mma_matrix<16x16xf32, "COp">
          %47 = addi %arg2, %c48 : index
          %48 = gpu.subgroup_mma_load_matrix %18[%47, %arg4] {leadDimension = 72 : index} : memref<128x72xf16, 3> -> !gpu.mma_matrix<16x16xf16, "AOp">
          %49 = gpu.subgroup_mma_load_matrix %16[%arg4, %arg3] {leadDimension = 72 : index} : memref<64x72xf16, 3> -> !gpu.mma_matrix<16x16xf16, "BOp">
          %50 = gpu.subgroup_mma_compute %48, %49, %arg8 : !gpu.mma_matrix<16x16xf16, "AOp">, !gpu.mma_matrix<16x16xf16, "BOp"> -> !gpu.mma_matrix<16x16xf32, "COp">
          %51 = gpu.subgroup_mma_load_matrix %18[%arg2, %arg4] {leadDimension = 72 : index} : memref<128x72xf16, 3> -> !gpu.mma_matrix<16x16xf16, "AOp">
          %52 = addi %arg3, %c16 : index
          %53 = gpu.subgroup_mma_load_matrix %16[%arg4, %52] {leadDimension = 72 : index} : memref<64x72xf16, 3> -> !gpu.mma_matrix<16x16xf16, "BOp">
          %54 = gpu.subgroup_mma_compute %51, %53, %arg9 : !gpu.mma_matrix<16x16xf16, "AOp">, !gpu.mma_matrix<16x16xf16, "BOp"> -> !gpu.mma_matrix<16x16xf32, "COp">
          %55 = gpu.subgroup_mma_load_matrix %18[%39, %arg4] {leadDimension = 72 : index} : memref<128x72xf16, 3> -> !gpu.mma_matrix<16x16xf16, "AOp">
          %56 = gpu.subgroup_mma_load_matrix %16[%arg4, %52] {leadDimension = 72 : index} : memref<64x72xf16, 3> -> !gpu.mma_matrix<16x16xf16, "BOp">
          %57 = gpu.subgroup_mma_compute %55, %56, %arg10 : !gpu.mma_matrix<16x16xf16, "AOp">, !gpu.mma_matrix<16x16xf16, "BOp"> -> !gpu.mma_matrix<16x16xf32, "COp">
          %58 = gpu.subgroup_mma_load_matrix %18[%43, %arg4] {leadDimension = 72 : index} : memref<128x72xf16, 3> -> !gpu.mma_matrix<16x16xf16, "AOp">
          %59 = gpu.subgroup_mma_load_matrix %16[%arg4, %52] {leadDimension = 72 : index} : memref<64x72xf16, 3> -> !gpu.mma_matrix<16x16xf16, "BOp">
          %60 = gpu.subgroup_mma_compute %58, %59, %arg11 : !gpu.mma_matrix<16x16xf16, "AOp">, !gpu.mma_matrix<16x16xf16, "BOp"> -> !gpu.mma_matrix<16x16xf32, "COp">
          %61 = gpu.subgroup_mma_load_matrix %18[%47, %arg4] {leadDimension = 72 : index} : memref<128x72xf16, 3> -> !gpu.mma_matrix<16x16xf16, "AOp">
          %62 = gpu.subgroup_mma_load_matrix %16[%arg4, %52] {leadDimension = 72 : index} : memref<64x72xf16, 3> -> !gpu.mma_matrix<16x16xf16, "BOp">
          %63 = gpu.subgroup_mma_compute %61, %62, %arg12 : !gpu.mma_matrix<16x16xf16, "AOp">, !gpu.mma_matrix<16x16xf16, "BOp"> -> !gpu.mma_matrix<16x16xf32, "COp">
          scf.yield %38, %42, %46, %50, %54, %57, %60, %63 : !gpu.mma_matrix<16x16xf32, "COp">, !gpu.mma_matrix<16x16xf32, "COp">, !gpu.mma_matrix<16x16xf32, "COp">, !gpu.mma_matrix<16x16xf32, "COp">, !gpu.mma_matrix<16x16xf32, "COp">, !gpu.mma_matrix<16x16xf32, "COp">, !gpu.mma_matrix<16x16xf32, "COp">, !gpu.mma_matrix<16x16xf32, "COp">
        }
        gpu.subgroup_mma_store_matrix %35#0, %memref_2[%20, %21] {leadDimension = 1024 : index} : !gpu.mma_matrix<16x16xf32, "COp">, memref<1024x1024xf32>
        gpu.subgroup_mma_store_matrix %35#1, %memref_2[%23, %21] {leadDimension = 1024 : index} : !gpu.mma_matrix<16x16xf32, "COp">, memref<1024x1024xf32>
        gpu.subgroup_mma_store_matrix %35#2, %memref_2[%25, %21] {leadDimension = 1024 : index} : !gpu.mma_matrix<16x16xf32, "COp">, memref<1024x1024xf32>
        gpu.subgroup_mma_store_matrix %35#3, %memref_2[%27, %21] {leadDimension = 1024 : index} : !gpu.mma_matrix<16x16xf32, "COp">, memref<1024x1024xf32>
        gpu.subgroup_mma_store_matrix %35#4, %memref_2[%20, %29] {leadDimension = 1024 : index} : !gpu.mma_matrix<16x16xf32, "COp">, memref<1024x1024xf32>
        gpu.subgroup_mma_store_matrix %35#5, %memref_2[%23, %29] {leadDimension = 1024 : index} : !gpu.mma_matrix<16x16xf32, "COp">, memref<1024x1024xf32>
        gpu.subgroup_mma_store_matrix %35#6, %memref_2[%25, %29] {leadDimension = 1024 : index} : !gpu.mma_matrix<16x16xf32, "COp">, memref<1024x1024xf32>
        gpu.subgroup_mma_store_matrix %35#7, %memref_2[%27, %29] {leadDimension = 1024 : index} : !gpu.mma_matrix<16x16xf32, "COp">, memref<1024x1024xf32>
        scf.yield
      } {mapping = [{bound = #map, map = #map, processor = 7 : i64}, {bound = #map, map = #map, processor = 6 : i64}]}
      scf.yield
    } {mapping = [{bound = #map, map = #map, processor = 1 : i64}, {bound = #map, map = #map, processor = 0 : i64}]}
    %10 = call @rtclock() : () -> f64
    %11 = gpu.wait async 
    %12 = gpu.memcpy async [%11] %2, %memref_2 : memref<1024x1024xf32>, memref<1024x1024xf32>
    gpu.wait [%12]
    %13 = subf %10, %9 : f64
    %14 = sitofp %c2147483648_i64 : i64 to f64
    %15 = divf %14, %13 : f64
    call @print_flops(%15) : (f64) -> ()
    return
  }
  func private @print_memref_f32(memref<*xf32>)
  func private @print_flops(f64)
  func private @rtclock() -> f64
}

// CHECK:     gpu.launch blocks({{.*}}) in ({{.*}}) threads({{.*}}) in ({{.*}}) {
// CHECK:       scf.for %arg12 = %{{.*}} to %c128 step %c128 {
// CHECK:         scf.for %arg13 = %{{.*}} to %c64 step %c64 {
// CHECK:           scf.for %arg14 = %c0 to %c4 step %c1 {
// CHECK:           } {isCopyLoopNest = true}
// CHECK:           scf.for %arg14 = %c0 to %c8 step %c1 {
// CHECK:           } {isCopyLoopNest = true}
// CHECK:           gpu.barrier
// CHECK:           %45:8 = scf.for %arg14 = %c0 to %c960 step %c64 {{.*}} {
// CHECK:             scf.for %arg23 = %c0 to %c4 step %c1 {
// CHECK:             } {isCopyLoopNest = true}
// CHECK:             scf.for %arg23 = %c0 to %c8 step %c1 {
// CHECK:             } {isCopyLoopNest = true}
// CHECK:             %47:8 = scf.for %arg23 = %c0 to %c64 step %c16 {{.*}} {
// CHECK:           }
// CHECK:           gpu.barrier
// CHECK:           %46:8 = scf.for %arg14 = %c0 to %c64 step %c16 {{.*}} {
// CHECK:           }
// CHECK:         }
// CHECK:       }
// CHECK:       gpu.terminator
// CHECK:     }
// CHECK:     return
// CHECK:   }
