// RUN: mlir-opt %s --test-collapse-affine-parallel --canonicalize

#map0 = affine_map<(d0, d1) -> (d0 + d1)>
#map1 = affine_map<(d0, d1) -> (d0 + d1 + 16)>
#map2 = affine_map<(d0, d1) -> (d0 + d1 + 32)>
#map3 = affine_map<(d0, d1) -> (d0 + d1 + 48)>
#map4 = affine_map<() -> (0)>
#map5 = affine_map<() -> (128)>
#map6 = affine_map<() -> (32)>
#map7 = affine_map<(d0) -> (d0 + 16)>
#map8 = affine_map<(d0) -> (d0 + 32)>
#map9 = affine_map<(d0) -> (d0 + 48)>
#map10 = affine_map<() -> (1024)>
module  {
  memref.global "public" @frag_A : memref<128x32xf16, 3>
  memref.global "public" @frag_B : memref<32x128xf16, 3>
  memref.global "public" @frag_B_padded : memref<32x136xf16, 3>
  memref.global "public" @frag_A_padded : memref<128x40xf16, 3>
  func @main() {
    %c0 = constant 0 : index
    %c1024 = constant 1024 : index
    %c1 = constant 1 : index
    %cst = constant 0.000000e+00 : f32
    %c16 = constant 16 : index
    %0 = memref.alloc() : memref<1024x1024xf16>
    %1 = memref.alloc() : memref<1024x1024xf16>
    %2 = memref.alloc() : memref<1024x1024xf32>
    scf.for %arg0 = %c0 to %c1024 step %c1 {
      scf.for %arg1 = %c0 to %c1024 step %c1 {
        %13 = remi_signed %arg0, %c16 : index
        %14 = remi_signed %arg1, %c16 : index
        %15 = addi %13, %14 : index
        %16 = remi_signed %15, %c16 : index
        %17 = index_cast %16 : index to i16
        %18 = sitofp %17 : i16 to f16
        memref.store %18, %0[%arg0, %arg1] : memref<1024x1024xf16>
      }
    }
    scf.for %arg0 = %c0 to %c1024 step %c1 {
      scf.for %arg1 = %c0 to %c1024 step %c1 {
        %13 = remi_signed %arg0, %c16 : index
        %14 = remi_signed %arg1, %c16 : index
        %15 = addi %13, %14 : index
        %16 = remi_signed %15, %c16 : index
        %17 = index_cast %16 : index to i16
        %18 = sitofp %17 : i16 to f16
        memref.store %18, %1[%arg0, %arg1] : memref<1024x1024xf16>
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
    %5 = memref_vector_cast %memref : memref<1024x1024xf16> to memref<1024x128xvector<8xf16>>
    %memref_0, %asyncToken_1 = gpu.alloc async [%3] () : memref<1024x1024xf16>
    %6 = memref_vector_cast %memref_0 : memref<1024x1024xf16> to memref<1024x128xvector<8xf16>>
    %7 = memref_vector_cast %memref_0 : memref<1024x1024xf16> to memref<1024x128xvector<8xf16>>
    %memref_2, %asyncToken_3 = gpu.alloc async [%3] () : memref<1024x1024xf32>
    %8 = gpu.memcpy async [%3] %memref, %0 : memref<1024x1024xf16>, memref<1024x1024xf16>
    %9 = gpu.memcpy async [%3] %memref_0, %1 : memref<1024x1024xf16>, memref<1024x1024xf16>
    %10 = gpu.memcpy async [%3] %memref_2, %2 : memref<1024x1024xf32>, memref<1024x1024xf32>
    gpu.wait [%3]
    affine.parallel (%arg0) = (0) to (1024) step (128) {
      affine.parallel (%arg1) = (0) to (1024) step (128) {
        %13 = memref.get_global @frag_B_padded : memref<32x136xf16, 3>
        %14 = memref_vector_cast %13 : memref<32x136xf16, 3> to memref<32x17xvector<8xf16>, 3>
        %15 = memref_vector_cast %13 : memref<32x136xf16, 3> to memref<32x17xvector<8xf16>, 3>
        %16 = memref.get_global @frag_A_padded : memref<128x40xf16, 3>
        %17 = memref_vector_cast %16 : memref<128x40xf16, 3> to memref<128x5xvector<8xf16>, 3>
        %18 = memref_vector_cast %16 : memref<128x40xf16, 3> to memref<128x5xvector<8xf16>, 3>
        affine.parallel (%arg2) = (0) to (128) step (32) {
          affine.parallel (%arg3) = (0) to (128) step (64) {
            %19 = affine.apply #map0(%arg0, %arg2)
            %20 = affine.apply #map0(%arg1, %arg3)
            %21 = gpu.subgroup_mma_load_matrix %memref_2[%19, %20] {leadDimension = 1024 : index} : memref<1024x1024xf32> -> !gpu.mma_matrix<16x16xf32, "COp">
            %22 = affine.apply #map1(%arg0, %arg2)
            %23 = gpu.subgroup_mma_load_matrix %memref_2[%22, %20] {leadDimension = 1024 : index} : memref<1024x1024xf32> -> !gpu.mma_matrix<16x16xf32, "COp">
            %24 = affine.apply #map1(%arg1, %arg3)
            %25 = gpu.subgroup_mma_load_matrix %memref_2[%19, %24] {leadDimension = 1024 : index} : memref<1024x1024xf32> -> !gpu.mma_matrix<16x16xf32, "COp">
            %26 = gpu.subgroup_mma_load_matrix %memref_2[%22, %24] {leadDimension = 1024 : index} : memref<1024x1024xf32> -> !gpu.mma_matrix<16x16xf32, "COp">
            %27 = affine.apply #map2(%arg1, %arg3)
            %28 = gpu.subgroup_mma_load_matrix %memref_2[%19, %27] {leadDimension = 1024 : index} : memref<1024x1024xf32> -> !gpu.mma_matrix<16x16xf32, "COp">
            %29 = gpu.subgroup_mma_load_matrix %memref_2[%22, %27] {leadDimension = 1024 : index} : memref<1024x1024xf32> -> !gpu.mma_matrix<16x16xf32, "COp">
            %30 = affine.apply #map3(%arg1, %arg3)
            %31 = gpu.subgroup_mma_load_matrix %memref_2[%19, %30] {leadDimension = 1024 : index} : memref<1024x1024xf32> -> !gpu.mma_matrix<16x16xf32, "COp">
            %32 = gpu.subgroup_mma_load_matrix %memref_2[%22, %30] {leadDimension = 1024 : index} : memref<1024x1024xf32> -> !gpu.mma_matrix<16x16xf32, "COp">
            affine.parallel (%arg4) = (0) to (32) {
              affine.parallel (%arg5) = (0) to (128) step (8) {
                %35 = affine.load %6[%arg4, (%arg1 + %arg5) floordiv 8] : memref<1024x128xvector<8xf16>>
                affine.store %35, %14[%arg4, %arg5 floordiv 8] : memref<32x17xvector<8xf16>, 3>
              } {isParallel = true, lower_bound = #map4, step = 8 : index, upper_bound = #map5}
            } {isCopyLoopNest = true, isParallel = true, lower_bound = #map4, step = 1 : index, upper_bound = #map6}
            affine.parallel (%arg4) = (0) to (128) {
              affine.parallel (%arg5) = (0) to (32) step (8) {
                %35 = affine.load %4[%arg0 + %arg4, %arg5 floordiv 8] : memref<1024x128xvector<8xf16>>
                affine.store %35, %17[%arg4, %arg5 floordiv 8] : memref<128x5xvector<8xf16>, 3>
              } {isParallel = true, lower_bound = #map4, step = 8 : index, upper_bound = #map6}
            } {isCopyLoopNest = true, isParallel = true, lower_bound = #map4, step = 1 : index, upper_bound = #map5}
            gpu.barrier
            %33:8 = affine.for %arg4 = 0 to 992 step 32 iter_args(%arg5 = %21, %arg6 = %23, %arg7 = %25, %arg8 = %26, %arg9 = %28, %arg10 = %29, %arg11 = %31, %arg12 = %32) -> (!gpu.mma_matrix<16x16xf32, "COp">, !gpu.mma_matrix<16x16xf32, "COp">, !gpu.mma_matrix<16x16xf32, "COp">, !gpu.mma_matrix<16x16xf32, "COp">, !gpu.mma_matrix<16x16xf32, "COp">, !gpu.mma_matrix<16x16xf32, "COp">, !gpu.mma_matrix<16x16xf32, "COp">, !gpu.mma_matrix<16x16xf32, "COp">) {
              affine.parallel (%arg13) = (0) to (32) {
                affine.parallel (%arg14) = (0) to (128) step (8) {
                  %36 = affine.load %7[%arg4 + %arg13 + 32, (%arg1 + %arg14) floordiv 8] : memref<1024x128xvector<8xf16>>
                  affine.store %36, %15[%arg13, %arg14 floordiv 8] : memref<32x17xvector<8xf16>, 3>
                } {isParallel = true, lower_bound = #map4, step = 8 : index, upper_bound = #map5}
              } {isCopyLoopNest = true, isParallel = true, lower_bound = #map4, step = 1 : index, upper_bound = #map6}
              affine.parallel (%arg13) = (0) to (128) {
                affine.parallel (%arg14) = (0) to (32) step (8) {
                  %36 = affine.load %5[%arg0 + %arg13, (%arg4 + %arg14) floordiv 8 + 4] : memref<1024x128xvector<8xf16>>
                  affine.store %36, %18[%arg13, %arg14 floordiv 8] : memref<128x5xvector<8xf16>, 3>
                } {isParallel = true, lower_bound = #map4, step = 8 : index, upper_bound = #map6}
              } {isCopyLoopNest = true, isParallel = true, lower_bound = #map4, step = 1 : index, upper_bound = #map5}
              %35:8 = affine.for %arg13 = 0 to 32 step 16 iter_args(%arg14 = %arg5, %arg15 = %arg6, %arg16 = %arg7, %arg17 = %arg8, %arg18 = %arg9, %arg19 = %arg10, %arg20 = %arg11, %arg21 = %arg12) -> (!gpu.mma_matrix<16x16xf32, "COp">, !gpu.mma_matrix<16x16xf32, "COp">, !gpu.mma_matrix<16x16xf32, "COp">, !gpu.mma_matrix<16x16xf32, "COp">, !gpu.mma_matrix<16x16xf32, "COp">, !gpu.mma_matrix<16x16xf32, "COp">, !gpu.mma_matrix<16x16xf32, "COp">, !gpu.mma_matrix<16x16xf32, "COp">) {
                %36 = gpu.subgroup_mma_load_matrix %16[%arg2, %arg13] {leadDimension = 40 : index} : memref<128x40xf16, 3> -> !gpu.mma_matrix<16x16xf16, "AOp">
                %37 = gpu.subgroup_mma_load_matrix %13[%arg13, %arg3] {leadDimension = 136 : index} : memref<32x136xf16, 3> -> !gpu.mma_matrix<16x16xf16, "BOp">
                %38 = gpu.subgroup_mma_compute %36, %37, %arg14 : !gpu.mma_matrix<16x16xf16, "AOp">, !gpu.mma_matrix<16x16xf16, "BOp"> -> !gpu.mma_matrix<16x16xf32, "COp">
                %39 = affine.apply #map7(%arg2)
                %40 = gpu.subgroup_mma_load_matrix %16[%39, %arg13] {leadDimension = 40 : index} : memref<128x40xf16, 3> -> !gpu.mma_matrix<16x16xf16, "AOp">
                %41 = gpu.subgroup_mma_load_matrix %13[%arg13, %arg3] {leadDimension = 136 : index} : memref<32x136xf16, 3> -> !gpu.mma_matrix<16x16xf16, "BOp">
                %42 = gpu.subgroup_mma_compute %40, %41, %arg15 : !gpu.mma_matrix<16x16xf16, "AOp">, !gpu.mma_matrix<16x16xf16, "BOp"> -> !gpu.mma_matrix<16x16xf32, "COp">
                %43 = gpu.subgroup_mma_load_matrix %16[%arg2, %arg13] {leadDimension = 40 : index} : memref<128x40xf16, 3> -> !gpu.mma_matrix<16x16xf16, "AOp">
                %44 = affine.apply #map7(%arg3)
                %45 = gpu.subgroup_mma_load_matrix %13[%arg13, %44] {leadDimension = 136 : index} : memref<32x136xf16, 3> -> !gpu.mma_matrix<16x16xf16, "BOp">
                %46 = gpu.subgroup_mma_compute %43, %45, %arg16 : !gpu.mma_matrix<16x16xf16, "AOp">, !gpu.mma_matrix<16x16xf16, "BOp"> -> !gpu.mma_matrix<16x16xf32, "COp">
                %47 = gpu.subgroup_mma_load_matrix %16[%39, %arg13] {leadDimension = 40 : index} : memref<128x40xf16, 3> -> !gpu.mma_matrix<16x16xf16, "AOp">
                %48 = gpu.subgroup_mma_load_matrix %13[%arg13, %44] {leadDimension = 136 : index} : memref<32x136xf16, 3> -> !gpu.mma_matrix<16x16xf16, "BOp">
                %49 = gpu.subgroup_mma_compute %47, %48, %arg17 : !gpu.mma_matrix<16x16xf16, "AOp">, !gpu.mma_matrix<16x16xf16, "BOp"> -> !gpu.mma_matrix<16x16xf32, "COp">
                %50 = gpu.subgroup_mma_load_matrix %16[%arg2, %arg13] {leadDimension = 40 : index} : memref<128x40xf16, 3> -> !gpu.mma_matrix<16x16xf16, "AOp">
                %51 = affine.apply #map8(%arg3)
                %52 = gpu.subgroup_mma_load_matrix %13[%arg13, %51] {leadDimension = 136 : index} : memref<32x136xf16, 3> -> !gpu.mma_matrix<16x16xf16, "BOp">
                %53 = gpu.subgroup_mma_compute %50, %52, %arg18 : !gpu.mma_matrix<16x16xf16, "AOp">, !gpu.mma_matrix<16x16xf16, "BOp"> -> !gpu.mma_matrix<16x16xf32, "COp">
                %54 = gpu.subgroup_mma_load_matrix %16[%39, %arg13] {leadDimension = 40 : index} : memref<128x40xf16, 3> -> !gpu.mma_matrix<16x16xf16, "AOp">
                %55 = gpu.subgroup_mma_load_matrix %13[%arg13, %51] {leadDimension = 136 : index} : memref<32x136xf16, 3> -> !gpu.mma_matrix<16x16xf16, "BOp">
                %56 = gpu.subgroup_mma_compute %54, %55, %arg19 : !gpu.mma_matrix<16x16xf16, "AOp">, !gpu.mma_matrix<16x16xf16, "BOp"> -> !gpu.mma_matrix<16x16xf32, "COp">
                %57 = gpu.subgroup_mma_load_matrix %16[%arg2, %arg13] {leadDimension = 40 : index} : memref<128x40xf16, 3> -> !gpu.mma_matrix<16x16xf16, "AOp">
                %58 = affine.apply #map9(%arg3)
                %59 = gpu.subgroup_mma_load_matrix %13[%arg13, %58] {leadDimension = 136 : index} : memref<32x136xf16, 3> -> !gpu.mma_matrix<16x16xf16, "BOp">
                %60 = gpu.subgroup_mma_compute %57, %59, %arg20 : !gpu.mma_matrix<16x16xf16, "AOp">, !gpu.mma_matrix<16x16xf16, "BOp"> -> !gpu.mma_matrix<16x16xf32, "COp">
                %61 = gpu.subgroup_mma_load_matrix %16[%39, %arg13] {leadDimension = 40 : index} : memref<128x40xf16, 3> -> !gpu.mma_matrix<16x16xf16, "AOp">
                %62 = gpu.subgroup_mma_load_matrix %13[%arg13, %58] {leadDimension = 136 : index} : memref<32x136xf16, 3> -> !gpu.mma_matrix<16x16xf16, "BOp">
                %63 = gpu.subgroup_mma_compute %61, %62, %arg21 : !gpu.mma_matrix<16x16xf16, "AOp">, !gpu.mma_matrix<16x16xf16, "BOp"> -> !gpu.mma_matrix<16x16xf32, "COp">
                affine.yield %38, %42, %46, %49, %53, %56, %60, %63 : !gpu.mma_matrix<16x16xf32, "COp">, !gpu.mma_matrix<16x16xf32, "COp">, !gpu.mma_matrix<16x16xf32, "COp">, !gpu.mma_matrix<16x16xf32, "COp">, !gpu.mma_matrix<16x16xf32, "COp">, !gpu.mma_matrix<16x16xf32, "COp">, !gpu.mma_matrix<16x16xf32, "COp">, !gpu.mma_matrix<16x16xf32, "COp">
              }
              affine.yield %35#0, %35#1, %35#2, %35#3, %35#4, %35#5, %35#6, %35#7 : !gpu.mma_matrix<16x16xf32, "COp">, !gpu.mma_matrix<16x16xf32, "COp">, !gpu.mma_matrix<16x16xf32, "COp">, !gpu.mma_matrix<16x16xf32, "COp">, !gpu.mma_matrix<16x16xf32, "COp">, !gpu.mma_matrix<16x16xf32, "COp">, !gpu.mma_matrix<16x16xf32, "COp">, !gpu.mma_matrix<16x16xf32, "COp">
            } {isComputeLoopNest = true}
            gpu.barrier
            %34:8 = affine.for %arg4 = 992 to 1024 step 32 iter_args(%arg5 = %33#0, %arg6 = %33#1, %arg7 = %33#2, %arg8 = %33#3, %arg9 = %33#4, %arg10 = %33#5, %arg11 = %33#6, %arg12 = %33#7) -> (!gpu.mma_matrix<16x16xf32, "COp">, !gpu.mma_matrix<16x16xf32, "COp">, !gpu.mma_matrix<16x16xf32, "COp">, !gpu.mma_matrix<16x16xf32, "COp">, !gpu.mma_matrix<16x16xf32, "COp">, !gpu.mma_matrix<16x16xf32, "COp">, !gpu.mma_matrix<16x16xf32, "COp">, !gpu.mma_matrix<16x16xf32, "COp">) {
              %35:8 = affine.for %arg13 = 0 to 32 step 16 iter_args(%arg14 = %arg5, %arg15 = %arg6, %arg16 = %arg7, %arg17 = %arg8, %arg18 = %arg9, %arg19 = %arg10, %arg20 = %arg11, %arg21 = %arg12) -> (!gpu.mma_matrix<16x16xf32, "COp">, !gpu.mma_matrix<16x16xf32, "COp">, !gpu.mma_matrix<16x16xf32, "COp">, !gpu.mma_matrix<16x16xf32, "COp">, !gpu.mma_matrix<16x16xf32, "COp">, !gpu.mma_matrix<16x16xf32, "COp">, !gpu.mma_matrix<16x16xf32, "COp">, !gpu.mma_matrix<16x16xf32, "COp">) {
                %36 = gpu.subgroup_mma_load_matrix %16[%arg2, %arg13] {leadDimension = 40 : index} : memref<128x40xf16, 3> -> !gpu.mma_matrix<16x16xf16, "AOp">
                %37 = gpu.subgroup_mma_load_matrix %13[%arg13, %arg3] {leadDimension = 136 : index} : memref<32x136xf16, 3> -> !gpu.mma_matrix<16x16xf16, "BOp">
                %38 = gpu.subgroup_mma_compute %36, %37, %arg14 : !gpu.mma_matrix<16x16xf16, "AOp">, !gpu.mma_matrix<16x16xf16, "BOp"> -> !gpu.mma_matrix<16x16xf32, "COp">
                %39 = affine.apply #map7(%arg2)
                %40 = gpu.subgroup_mma_load_matrix %16[%39, %arg13] {leadDimension = 40 : index} : memref<128x40xf16, 3> -> !gpu.mma_matrix<16x16xf16, "AOp">
                %41 = gpu.subgroup_mma_load_matrix %13[%arg13, %arg3] {leadDimension = 136 : index} : memref<32x136xf16, 3> -> !gpu.mma_matrix<16x16xf16, "BOp">
                %42 = gpu.subgroup_mma_compute %40, %41, %arg15 : !gpu.mma_matrix<16x16xf16, "AOp">, !gpu.mma_matrix<16x16xf16, "BOp"> -> !gpu.mma_matrix<16x16xf32, "COp">
                %43 = gpu.subgroup_mma_load_matrix %16[%arg2, %arg13] {leadDimension = 40 : index} : memref<128x40xf16, 3> -> !gpu.mma_matrix<16x16xf16, "AOp">
                %44 = affine.apply #map7(%arg3)
                %45 = gpu.subgroup_mma_load_matrix %13[%arg13, %44] {leadDimension = 136 : index} : memref<32x136xf16, 3> -> !gpu.mma_matrix<16x16xf16, "BOp">
                %46 = gpu.subgroup_mma_compute %43, %45, %arg16 : !gpu.mma_matrix<16x16xf16, "AOp">, !gpu.mma_matrix<16x16xf16, "BOp"> -> !gpu.mma_matrix<16x16xf32, "COp">
                %47 = gpu.subgroup_mma_load_matrix %16[%39, %arg13] {leadDimension = 40 : index} : memref<128x40xf16, 3> -> !gpu.mma_matrix<16x16xf16, "AOp">
                %48 = gpu.subgroup_mma_load_matrix %13[%arg13, %44] {leadDimension = 136 : index} : memref<32x136xf16, 3> -> !gpu.mma_matrix<16x16xf16, "BOp">
                %49 = gpu.subgroup_mma_compute %47, %48, %arg17 : !gpu.mma_matrix<16x16xf16, "AOp">, !gpu.mma_matrix<16x16xf16, "BOp"> -> !gpu.mma_matrix<16x16xf32, "COp">
                %50 = gpu.subgroup_mma_load_matrix %16[%arg2, %arg13] {leadDimension = 40 : index} : memref<128x40xf16, 3> -> !gpu.mma_matrix<16x16xf16, "AOp">
                %51 = affine.apply #map8(%arg3)
                %52 = gpu.subgroup_mma_load_matrix %13[%arg13, %51] {leadDimension = 136 : index} : memref<32x136xf16, 3> -> !gpu.mma_matrix<16x16xf16, "BOp">
                %53 = gpu.subgroup_mma_compute %50, %52, %arg18 : !gpu.mma_matrix<16x16xf16, "AOp">, !gpu.mma_matrix<16x16xf16, "BOp"> -> !gpu.mma_matrix<16x16xf32, "COp">
                %54 = gpu.subgroup_mma_load_matrix %16[%39, %arg13] {leadDimension = 40 : index} : memref<128x40xf16, 3> -> !gpu.mma_matrix<16x16xf16, "AOp">
                %55 = gpu.subgroup_mma_load_matrix %13[%arg13, %51] {leadDimension = 136 : index} : memref<32x136xf16, 3> -> !gpu.mma_matrix<16x16xf16, "BOp">
                %56 = gpu.subgroup_mma_compute %54, %55, %arg19 : !gpu.mma_matrix<16x16xf16, "AOp">, !gpu.mma_matrix<16x16xf16, "BOp"> -> !gpu.mma_matrix<16x16xf32, "COp">
                %57 = gpu.subgroup_mma_load_matrix %16[%arg2, %arg13] {leadDimension = 40 : index} : memref<128x40xf16, 3> -> !gpu.mma_matrix<16x16xf16, "AOp">
                %58 = affine.apply #map9(%arg3)
                %59 = gpu.subgroup_mma_load_matrix %13[%arg13, %58] {leadDimension = 136 : index} : memref<32x136xf16, 3> -> !gpu.mma_matrix<16x16xf16, "BOp">
                %60 = gpu.subgroup_mma_compute %57, %59, %arg20 : !gpu.mma_matrix<16x16xf16, "AOp">, !gpu.mma_matrix<16x16xf16, "BOp"> -> !gpu.mma_matrix<16x16xf32, "COp">
                %61 = gpu.subgroup_mma_load_matrix %16[%39, %arg13] {leadDimension = 40 : index} : memref<128x40xf16, 3> -> !gpu.mma_matrix<16x16xf16, "AOp">
                %62 = gpu.subgroup_mma_load_matrix %13[%arg13, %58] {leadDimension = 136 : index} : memref<32x136xf16, 3> -> !gpu.mma_matrix<16x16xf16, "BOp">
                %63 = gpu.subgroup_mma_compute %61, %62, %arg21 : !gpu.mma_matrix<16x16xf16, "AOp">, !gpu.mma_matrix<16x16xf16, "BOp"> -> !gpu.mma_matrix<16x16xf32, "COp">
                affine.yield %38, %42, %46, %49, %53, %56, %60, %63 : !gpu.mma_matrix<16x16xf32, "COp">, !gpu.mma_matrix<16x16xf32, "COp">, !gpu.mma_matrix<16x16xf32, "COp">, !gpu.mma_matrix<16x16xf32, "COp">, !gpu.mma_matrix<16x16xf32, "COp">, !gpu.mma_matrix<16x16xf32, "COp">, !gpu.mma_matrix<16x16xf32, "COp">, !gpu.mma_matrix<16x16xf32, "COp">
              }
              affine.yield %35#0, %35#1, %35#2, %35#3, %35#4, %35#5, %35#6, %35#7 : !gpu.mma_matrix<16x16xf32, "COp">, !gpu.mma_matrix<16x16xf32, "COp">, !gpu.mma_matrix<16x16xf32, "COp">, !gpu.mma_matrix<16x16xf32, "COp">, !gpu.mma_matrix<16x16xf32, "COp">, !gpu.mma_matrix<16x16xf32, "COp">, !gpu.mma_matrix<16x16xf32, "COp">, !gpu.mma_matrix<16x16xf32, "COp">
            } {isComputeLoopNest = true}
            gpu.subgroup_mma_store_matrix %34#0, %memref_2[%19, %20] {leadDimension = 1024 : index} : !gpu.mma_matrix<16x16xf32, "COp">, memref<1024x1024xf32>
            gpu.subgroup_mma_store_matrix %34#1, %memref_2[%22, %20] {leadDimension = 1024 : index} : !gpu.mma_matrix<16x16xf32, "COp">, memref<1024x1024xf32>
            gpu.subgroup_mma_store_matrix %34#2, %memref_2[%19, %24] {leadDimension = 1024 : index} : !gpu.mma_matrix<16x16xf32, "COp">, memref<1024x1024xf32>
            gpu.subgroup_mma_store_matrix %34#3, %memref_2[%22, %24] {leadDimension = 1024 : index} : !gpu.mma_matrix<16x16xf32, "COp">, memref<1024x1024xf32>
            gpu.subgroup_mma_store_matrix %34#4, %memref_2[%19, %27] {leadDimension = 1024 : index} : !gpu.mma_matrix<16x16xf32, "COp">, memref<1024x1024xf32>
            gpu.subgroup_mma_store_matrix %34#5, %memref_2[%22, %27] {leadDimension = 1024 : index} : !gpu.mma_matrix<16x16xf32, "COp">, memref<1024x1024xf32>
            gpu.subgroup_mma_store_matrix %34#6, %memref_2[%19, %30] {leadDimension = 1024 : index} : !gpu.mma_matrix<16x16xf32, "COp">, memref<1024x1024xf32>
            gpu.subgroup_mma_store_matrix %34#7, %memref_2[%22, %30] {leadDimension = 1024 : index} : !gpu.mma_matrix<16x16xf32, "COp">, memref<1024x1024xf32>
          } {isParallel = true, lower_bound = #map4, step = 64 : index, upper_bound = #map5}
        } {isParallel = true, lower_bound = #map4, step = 32 : index, upper_bound = #map5}
      } {isParallel = true, lower_bound = #map4, step = 128 : index, upper_bound = #map10}
    } {isParallel = true, lower_bound = #map4, step = 128 : index, upper_bound = #map10}
    %11 = gpu.wait async 
    %12 = gpu.memcpy async [%11] %2, %memref_2 : memref<1024x1024xf32>, memref<1024x1024xf32>
    gpu.wait [%12]
    return
  }
  gpu.module @initC_kernel {
    gpu.func @initC_kernel(%arg0: memref<1024x1024xf32>) kernel {
      %cst = constant 0.000000e+00 : f32
      %c1024 = constant 1024 : index
      %c0 = constant 0 : index
      %c1 = constant 1 : index
      scf.for %arg1 = %c0 to %c1024 step %c1 {
        scf.for %arg2 = %c0 to %c1024 step %c1 {
          memref.store %cst, %arg0[%arg1, %arg2] : memref<1024x1024xf32>
        }
      }
      gpu.return
    }
  }
  func private @print_memref_f32(memref<*xf32>)
  func private @print_flops(f64)
  func private @rtclock() -> f64
}

