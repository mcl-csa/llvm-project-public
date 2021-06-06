// RUN: mlir-opt %s --test-gpu-matmul-parallel-loop-mapping --canonicalize

module  {
  memref.global "public" @frag_A : memref<128x32xf16, 3>
  memref.global "public" @frag_B : memref<32x128xf16, 3>
  memref.global "public" @frag_B_padded : memref<32x136xf16, 3>
  memref.global "public" @frag_A_padded : memref<128x40xf16, 3>
  func @main() {
    %c16 = constant 16 : index
    %cst = constant 0.000000e+00 : f32
    %c1 = constant 1 : index
    %c1024 = constant 1024 : index
    %c0 = constant 0 : index
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
    %c0_4 = constant 0 : index
    %c0_5 = constant 0 : index
    %c1024_6 = constant 1024 : index
    %c1024_7 = constant 1024 : index
    %c128 = constant 128 : index
    %c128_8 = constant 128 : index
    scf.parallel (%arg0, %arg1) = (%c0_4, %c0_5) to (%c1024_6, %c1024_7) step (%c128, %c128_8) {
      %13 = memref.get_global @frag_B_padded : memref<32x136xf16, 3>
      %14 = memref_vector_cast %13 : memref<32x136xf16, 3> to memref<32x17xvector<8xf16>, 3>
      %15 = memref_vector_cast %13 : memref<32x136xf16, 3> to memref<32x17xvector<8xf16>, 3>
      %16 = memref.get_global @frag_A_padded : memref<128x40xf16, 3>
      %17 = memref_vector_cast %16 : memref<128x40xf16, 3> to memref<128x5xvector<8xf16>, 3>
      %18 = memref_vector_cast %16 : memref<128x40xf16, 3> to memref<128x5xvector<8xf16>, 3>
      %c0_9 = constant 0 : index
      %c0_10 = constant 0 : index
      %c128_11 = constant 128 : index
      %c128_12 = constant 128 : index
      %c32 = constant 32 : index
      %c64 = constant 64 : index
      scf.parallel (%arg2, %arg3) = (%c0_9, %c0_10) to (%c128_11, %c128_12) step (%c32, %c64) {
        %19 = addi %arg0, %arg2 : index
        %20 = addi %arg1, %arg3 : index
        %21 = gpu.subgroup_mma_load_matrix %memref_2[%19, %20] {leadDimension = 1024 : index} : memref<1024x1024xf32> -> !gpu.mma_matrix<16x16xf32, "COp">
        %22 = addi %arg0, %arg2 : index
        %c16_13 = constant 16 : index
        %23 = addi %22, %c16_13 : index
        %24 = gpu.subgroup_mma_load_matrix %memref_2[%23, %20] {leadDimension = 1024 : index} : memref<1024x1024xf32> -> !gpu.mma_matrix<16x16xf32, "COp">
        %25 = addi %arg1, %arg3 : index
        %c16_14 = constant 16 : index
        %26 = addi %25, %c16_14 : index
        %27 = gpu.subgroup_mma_load_matrix %memref_2[%19, %26] {leadDimension = 1024 : index} : memref<1024x1024xf32> -> !gpu.mma_matrix<16x16xf32, "COp">
        %28 = gpu.subgroup_mma_load_matrix %memref_2[%23, %26] {leadDimension = 1024 : index} : memref<1024x1024xf32> -> !gpu.mma_matrix<16x16xf32, "COp">
        %29 = addi %arg1, %arg3 : index
        %c32_15 = constant 32 : index
        %30 = addi %29, %c32_15 : index
        %31 = gpu.subgroup_mma_load_matrix %memref_2[%19, %30] {leadDimension = 1024 : index} : memref<1024x1024xf32> -> !gpu.mma_matrix<16x16xf32, "COp">
        %32 = gpu.subgroup_mma_load_matrix %memref_2[%23, %30] {leadDimension = 1024 : index} : memref<1024x1024xf32> -> !gpu.mma_matrix<16x16xf32, "COp">
        %33 = addi %arg1, %arg3 : index
        %c48 = constant 48 : index
        %34 = addi %33, %c48 : index
        %35 = gpu.subgroup_mma_load_matrix %memref_2[%19, %34] {leadDimension = 1024 : index} : memref<1024x1024xf32> -> !gpu.mma_matrix<16x16xf32, "COp">
        %36 = gpu.subgroup_mma_load_matrix %memref_2[%23, %34] {leadDimension = 1024 : index} : memref<1024x1024xf32> -> !gpu.mma_matrix<16x16xf32, "COp">
        %c0_16 = constant 0 : index
        %c0_17 = constant 0 : index
        %c32_18 = constant 32 : index
        %c128_19 = constant 128 : index
        %c1_20 = constant 1 : index
        %c8 = constant 8 : index
        scf.parallel (%arg4, %arg5) = (%c0_16, %c0_17) to (%c32_18, %c128_19) step (%c1_20, %c8) {
          %39 = addi %arg1, %arg5 : index
          %c8_32 = constant 8 : index
          %c0_33 = constant 0 : index
          %c-1 = constant -1 : index
          %40 = cmpi slt, %39, %c0_33 : index
          %41 = subi %c-1, %39 : index
          %42 = select %40, %41, %39 : index
          %43 = divi_signed %42, %c8_32 : index
          %44 = subi %c-1, %43 : index
          %45 = select %40, %44, %43 : index
          %46 = memref.load %6[%arg4, %45] : memref<1024x128xvector<8xf16>>
          %c8_34 = constant 8 : index
          %c0_35 = constant 0 : index
          %c-1_36 = constant -1 : index
          %47 = cmpi slt, %arg5, %c0_35 : index
          %48 = subi %c-1_36, %arg5 : index
          %49 = select %47, %48, %arg5 : index
          %50 = divi_signed %49, %c8_34 : index
          %51 = subi %c-1_36, %50 : index
          %52 = select %47, %51, %50 : index
          memref.store %46, %14[%arg4, %52] : memref<32x17xvector<8xf16>, 3>
          scf.yield
        }
        %c0_21 = constant 0 : index
        %c0_22 = constant 0 : index
        %c128_23 = constant 128 : index
        %c32_24 = constant 32 : index
        %c1_25 = constant 1 : index
        %c8_26 = constant 8 : index
        scf.parallel (%arg4, %arg5) = (%c0_21, %c0_22) to (%c128_23, %c32_24) step (%c1_25, %c8_26) {
          %39 = addi %arg0, %arg4 : index
          %c8_32 = constant 8 : index
          %c0_33 = constant 0 : index
          %c-1 = constant -1 : index
          %40 = cmpi slt, %arg5, %c0_33 : index
          %41 = subi %c-1, %arg5 : index
          %42 = select %40, %41, %arg5 : index
          %43 = divi_signed %42, %c8_32 : index
          %44 = subi %c-1, %43 : index
          %45 = select %40, %44, %43 : index
          %46 = memref.load %4[%39, %45] : memref<1024x128xvector<8xf16>>
          %c8_34 = constant 8 : index
          %c0_35 = constant 0 : index
          %c-1_36 = constant -1 : index
          %47 = cmpi slt, %arg5, %c0_35 : index
          %48 = subi %c-1_36, %arg5 : index
          %49 = select %47, %48, %arg5 : index
          %50 = divi_signed %49, %c8_34 : index
          %51 = subi %c-1_36, %50 : index
          %52 = select %47, %51, %50 : index
          memref.store %46, %17[%arg4, %52] : memref<128x5xvector<8xf16>, 3>
          scf.yield
        }
        gpu.barrier
        %c0_27 = constant 0 : index
        %c992 = constant 992 : index
        %c32_28 = constant 32 : index
        %37:8 = scf.for %arg4 = %c0_27 to %c992 step %c32_28 iter_args(%arg5 = %21, %arg6 = %24, %arg7 = %27, %arg8 = %28, %arg9 = %31, %arg10 = %32, %arg11 = %35, %arg12 = %36) -> (!gpu.mma_matrix<16x16xf32, "COp">, !gpu.mma_matrix<16x16xf32, "COp">, !gpu.mma_matrix<16x16xf32, "COp">, !gpu.mma_matrix<16x16xf32, "COp">, !gpu.mma_matrix<16x16xf32, "COp">, !gpu.mma_matrix<16x16xf32, "COp">, !gpu.mma_matrix<16x16xf32, "COp">, !gpu.mma_matrix<16x16xf32, "COp">) {
          %c0_32 = constant 0 : index
          %c0_33 = constant 0 : index
          %c32_34 = constant 32 : index
          %c128_35 = constant 128 : index
          %c1_36 = constant 1 : index
          %c8_37 = constant 8 : index
          scf.parallel (%arg13, %arg14) = (%c0_32, %c0_33) to (%c32_34, %c128_35) step (%c1_36, %c8_37) {
            %40 = addi %arg4, %arg13 : index
            %c32_47 = constant 32 : index
            %41 = addi %40, %c32_47 : index
            %42 = addi %arg1, %arg14 : index
            %c8_48 = constant 8 : index
            %c0_49 = constant 0 : index
            %c-1 = constant -1 : index
            %43 = cmpi slt, %42, %c0_49 : index
            %44 = subi %c-1, %42 : index
            %45 = select %43, %44, %42 : index
            %46 = divi_signed %45, %c8_48 : index
            %47 = subi %c-1, %46 : index
            %48 = select %43, %47, %46 : index
            %49 = memref.load %7[%41, %48] : memref<1024x128xvector<8xf16>>
            %c8_50 = constant 8 : index
            %c0_51 = constant 0 : index
            %c-1_52 = constant -1 : index
            %50 = cmpi slt, %arg14, %c0_51 : index
            %51 = subi %c-1_52, %arg14 : index
            %52 = select %50, %51, %arg14 : index
            %53 = divi_signed %52, %c8_50 : index
            %54 = subi %c-1_52, %53 : index
            %55 = select %50, %54, %53 : index
            memref.store %49, %15[%arg13, %55] : memref<32x17xvector<8xf16>, 3>
            scf.yield
          }
          %c0_38 = constant 0 : index
          %c0_39 = constant 0 : index
          %c128_40 = constant 128 : index
          %c32_41 = constant 32 : index
          %c1_42 = constant 1 : index
          %c8_43 = constant 8 : index
          scf.parallel (%arg13, %arg14) = (%c0_38, %c0_39) to (%c128_40, %c32_41) step (%c1_42, %c8_43) {
            %40 = addi %arg0, %arg13 : index
            %41 = addi %arg4, %arg14 : index
            %c8_47 = constant 8 : index
            %c0_48 = constant 0 : index
            %c-1 = constant -1 : index
            %42 = cmpi slt, %41, %c0_48 : index
            %43 = subi %c-1, %41 : index
            %44 = select %42, %43, %41 : index
            %45 = divi_signed %44, %c8_47 : index
            %46 = subi %c-1, %45 : index
            %47 = select %42, %46, %45 : index
            %c4 = constant 4 : index
            %48 = addi %47, %c4 : index
            %49 = memref.load %5[%40, %48] : memref<1024x128xvector<8xf16>>
            %c8_49 = constant 8 : index
            %c0_50 = constant 0 : index
            %c-1_51 = constant -1 : index
            %50 = cmpi slt, %arg14, %c0_50 : index
            %51 = subi %c-1_51, %arg14 : index
            %52 = select %50, %51, %arg14 : index
            %53 = divi_signed %52, %c8_49 : index
            %54 = subi %c-1_51, %53 : index
            %55 = select %50, %54, %53 : index
            memref.store %49, %18[%arg13, %55] : memref<128x5xvector<8xf16>, 3>
            scf.yield
          }
          %c0_44 = constant 0 : index
          %c32_45 = constant 32 : index
          %c16_46 = constant 16 : index
          %39:8 = scf.for %arg13 = %c0_44 to %c32_45 step %c16_46 iter_args(%arg14 = %arg5, %arg15 = %arg6, %arg16 = %arg7, %arg17 = %arg8, %arg18 = %arg9, %arg19 = %arg10, %arg20 = %arg11, %arg21 = %arg12) -> (!gpu.mma_matrix<16x16xf32, "COp">, !gpu.mma_matrix<16x16xf32, "COp">, !gpu.mma_matrix<16x16xf32, "COp">, !gpu.mma_matrix<16x16xf32, "COp">, !gpu.mma_matrix<16x16xf32, "COp">, !gpu.mma_matrix<16x16xf32, "COp">, !gpu.mma_matrix<16x16xf32, "COp">, !gpu.mma_matrix<16x16xf32, "COp">) {
            %40 = gpu.subgroup_mma_load_matrix %16[%arg2, %arg13] {leadDimension = 40 : index} : memref<128x40xf16, 3> -> !gpu.mma_matrix<16x16xf16, "AOp">
            %41 = gpu.subgroup_mma_load_matrix %13[%arg13, %arg3] {leadDimension = 136 : index} : memref<32x136xf16, 3> -> !gpu.mma_matrix<16x16xf16, "BOp">
            %42 = gpu.subgroup_mma_compute %40, %41, %arg14 : !gpu.mma_matrix<16x16xf16, "AOp">, !gpu.mma_matrix<16x16xf16, "BOp"> -> !gpu.mma_matrix<16x16xf32, "COp">
            %c16_47 = constant 16 : index
            %43 = addi %arg2, %c16_47 : index
            %44 = gpu.subgroup_mma_load_matrix %16[%43, %arg13] {leadDimension = 40 : index} : memref<128x40xf16, 3> -> !gpu.mma_matrix<16x16xf16, "AOp">
            %45 = gpu.subgroup_mma_load_matrix %13[%arg13, %arg3] {leadDimension = 136 : index} : memref<32x136xf16, 3> -> !gpu.mma_matrix<16x16xf16, "BOp">
            %46 = gpu.subgroup_mma_compute %44, %45, %arg15 : !gpu.mma_matrix<16x16xf16, "AOp">, !gpu.mma_matrix<16x16xf16, "BOp"> -> !gpu.mma_matrix<16x16xf32, "COp">
            %47 = gpu.subgroup_mma_load_matrix %16[%arg2, %arg13] {leadDimension = 40 : index} : memref<128x40xf16, 3> -> !gpu.mma_matrix<16x16xf16, "AOp">
            %c16_48 = constant 16 : index
            %48 = addi %arg3, %c16_48 : index
            %49 = gpu.subgroup_mma_load_matrix %13[%arg13, %48] {leadDimension = 136 : index} : memref<32x136xf16, 3> -> !gpu.mma_matrix<16x16xf16, "BOp">
            %50 = gpu.subgroup_mma_compute %47, %49, %arg16 : !gpu.mma_matrix<16x16xf16, "AOp">, !gpu.mma_matrix<16x16xf16, "BOp"> -> !gpu.mma_matrix<16x16xf32, "COp">
            %51 = gpu.subgroup_mma_load_matrix %16[%43, %arg13] {leadDimension = 40 : index} : memref<128x40xf16, 3> -> !gpu.mma_matrix<16x16xf16, "AOp">
            %52 = gpu.subgroup_mma_load_matrix %13[%arg13, %48] {leadDimension = 136 : index} : memref<32x136xf16, 3> -> !gpu.mma_matrix<16x16xf16, "BOp">
            %53 = gpu.subgroup_mma_compute %51, %52, %arg17 : !gpu.mma_matrix<16x16xf16, "AOp">, !gpu.mma_matrix<16x16xf16, "BOp"> -> !gpu.mma_matrix<16x16xf32, "COp">
            %54 = gpu.subgroup_mma_load_matrix %16[%arg2, %arg13] {leadDimension = 40 : index} : memref<128x40xf16, 3> -> !gpu.mma_matrix<16x16xf16, "AOp">
            %c32_49 = constant 32 : index
            %55 = addi %arg3, %c32_49 : index
            %56 = gpu.subgroup_mma_load_matrix %13[%arg13, %55] {leadDimension = 136 : index} : memref<32x136xf16, 3> -> !gpu.mma_matrix<16x16xf16, "BOp">
            %57 = gpu.subgroup_mma_compute %54, %56, %arg18 : !gpu.mma_matrix<16x16xf16, "AOp">, !gpu.mma_matrix<16x16xf16, "BOp"> -> !gpu.mma_matrix<16x16xf32, "COp">
            %58 = gpu.subgroup_mma_load_matrix %16[%43, %arg13] {leadDimension = 40 : index} : memref<128x40xf16, 3> -> !gpu.mma_matrix<16x16xf16, "AOp">
            %59 = gpu.subgroup_mma_load_matrix %13[%arg13, %55] {leadDimension = 136 : index} : memref<32x136xf16, 3> -> !gpu.mma_matrix<16x16xf16, "BOp">
            %60 = gpu.subgroup_mma_compute %58, %59, %arg19 : !gpu.mma_matrix<16x16xf16, "AOp">, !gpu.mma_matrix<16x16xf16, "BOp"> -> !gpu.mma_matrix<16x16xf32, "COp">
            %61 = gpu.subgroup_mma_load_matrix %16[%arg2, %arg13] {leadDimension = 40 : index} : memref<128x40xf16, 3> -> !gpu.mma_matrix<16x16xf16, "AOp">
            %c48_50 = constant 48 : index
            %62 = addi %arg3, %c48_50 : index
            %63 = gpu.subgroup_mma_load_matrix %13[%arg13, %62] {leadDimension = 136 : index} : memref<32x136xf16, 3> -> !gpu.mma_matrix<16x16xf16, "BOp">
            %64 = gpu.subgroup_mma_compute %61, %63, %arg20 : !gpu.mma_matrix<16x16xf16, "AOp">, !gpu.mma_matrix<16x16xf16, "BOp"> -> !gpu.mma_matrix<16x16xf32, "COp">
            %65 = gpu.subgroup_mma_load_matrix %16[%43, %arg13] {leadDimension = 40 : index} : memref<128x40xf16, 3> -> !gpu.mma_matrix<16x16xf16, "AOp">
            %66 = gpu.subgroup_mma_load_matrix %13[%arg13, %62] {leadDimension = 136 : index} : memref<32x136xf16, 3> -> !gpu.mma_matrix<16x16xf16, "BOp">
            %67 = gpu.subgroup_mma_compute %65, %66, %arg21 : !gpu.mma_matrix<16x16xf16, "AOp">, !gpu.mma_matrix<16x16xf16, "BOp"> -> !gpu.mma_matrix<16x16xf32, "COp">
            scf.yield %42, %46, %50, %53, %57, %60, %64, %67 : !gpu.mma_matrix<16x16xf32, "COp">, !gpu.mma_matrix<16x16xf32, "COp">, !gpu.mma_matrix<16x16xf32, "COp">, !gpu.mma_matrix<16x16xf32, "COp">, !gpu.mma_matrix<16x16xf32, "COp">, !gpu.mma_matrix<16x16xf32, "COp">, !gpu.mma_matrix<16x16xf32, "COp">, !gpu.mma_matrix<16x16xf32, "COp">
          }
          scf.yield %39#0, %39#1, %39#2, %39#3, %39#4, %39#5, %39#6, %39#7 : !gpu.mma_matrix<16x16xf32, "COp">, !gpu.mma_matrix<16x16xf32, "COp">, !gpu.mma_matrix<16x16xf32, "COp">, !gpu.mma_matrix<16x16xf32, "COp">, !gpu.mma_matrix<16x16xf32, "COp">, !gpu.mma_matrix<16x16xf32, "COp">, !gpu.mma_matrix<16x16xf32, "COp">, !gpu.mma_matrix<16x16xf32, "COp">
        }
        gpu.barrier
        %c992_29 = constant 992 : index
        %c1024_30 = constant 1024 : index
        %c32_31 = constant 32 : index
        %38:8 = scf.for %arg4 = %c992_29 to %c1024_30 step %c32_31 iter_args(%arg5 = %37#0, %arg6 = %37#1, %arg7 = %37#2, %arg8 = %37#3, %arg9 = %37#4, %arg10 = %37#5, %arg11 = %37#6, %arg12 = %37#7) -> (!gpu.mma_matrix<16x16xf32, "COp">, !gpu.mma_matrix<16x16xf32, "COp">, !gpu.mma_matrix<16x16xf32, "COp">, !gpu.mma_matrix<16x16xf32, "COp">, !gpu.mma_matrix<16x16xf32, "COp">, !gpu.mma_matrix<16x16xf32, "COp">, !gpu.mma_matrix<16x16xf32, "COp">, !gpu.mma_matrix<16x16xf32, "COp">) {
          %c0_32 = constant 0 : index
          %c32_33 = constant 32 : index
          %c16_34 = constant 16 : index
          %39:8 = scf.for %arg13 = %c0_32 to %c32_33 step %c16_34 iter_args(%arg14 = %arg5, %arg15 = %arg6, %arg16 = %arg7, %arg17 = %arg8, %arg18 = %arg9, %arg19 = %arg10, %arg20 = %arg11, %arg21 = %arg12) -> (!gpu.mma_matrix<16x16xf32, "COp">, !gpu.mma_matrix<16x16xf32, "COp">, !gpu.mma_matrix<16x16xf32, "COp">, !gpu.mma_matrix<16x16xf32, "COp">, !gpu.mma_matrix<16x16xf32, "COp">, !gpu.mma_matrix<16x16xf32, "COp">, !gpu.mma_matrix<16x16xf32, "COp">, !gpu.mma_matrix<16x16xf32, "COp">) {
            %40 = gpu.subgroup_mma_load_matrix %16[%arg2, %arg13] {leadDimension = 40 : index} : memref<128x40xf16, 3> -> !gpu.mma_matrix<16x16xf16, "AOp">
            %41 = gpu.subgroup_mma_load_matrix %13[%arg13, %arg3] {leadDimension = 136 : index} : memref<32x136xf16, 3> -> !gpu.mma_matrix<16x16xf16, "BOp">
            %42 = gpu.subgroup_mma_compute %40, %41, %arg14 : !gpu.mma_matrix<16x16xf16, "AOp">, !gpu.mma_matrix<16x16xf16, "BOp"> -> !gpu.mma_matrix<16x16xf32, "COp">
            %c16_35 = constant 16 : index
            %43 = addi %arg2, %c16_35 : index
            %44 = gpu.subgroup_mma_load_matrix %16[%43, %arg13] {leadDimension = 40 : index} : memref<128x40xf16, 3> -> !gpu.mma_matrix<16x16xf16, "AOp">
            %45 = gpu.subgroup_mma_load_matrix %13[%arg13, %arg3] {leadDimension = 136 : index} : memref<32x136xf16, 3> -> !gpu.mma_matrix<16x16xf16, "BOp">
            %46 = gpu.subgroup_mma_compute %44, %45, %arg15 : !gpu.mma_matrix<16x16xf16, "AOp">, !gpu.mma_matrix<16x16xf16, "BOp"> -> !gpu.mma_matrix<16x16xf32, "COp">
            %47 = gpu.subgroup_mma_load_matrix %16[%arg2, %arg13] {leadDimension = 40 : index} : memref<128x40xf16, 3> -> !gpu.mma_matrix<16x16xf16, "AOp">
            %c16_36 = constant 16 : index
            %48 = addi %arg3, %c16_36 : index
            %49 = gpu.subgroup_mma_load_matrix %13[%arg13, %48] {leadDimension = 136 : index} : memref<32x136xf16, 3> -> !gpu.mma_matrix<16x16xf16, "BOp">
            %50 = gpu.subgroup_mma_compute %47, %49, %arg16 : !gpu.mma_matrix<16x16xf16, "AOp">, !gpu.mma_matrix<16x16xf16, "BOp"> -> !gpu.mma_matrix<16x16xf32, "COp">
            %51 = gpu.subgroup_mma_load_matrix %16[%43, %arg13] {leadDimension = 40 : index} : memref<128x40xf16, 3> -> !gpu.mma_matrix<16x16xf16, "AOp">
            %52 = gpu.subgroup_mma_load_matrix %13[%arg13, %48] {leadDimension = 136 : index} : memref<32x136xf16, 3> -> !gpu.mma_matrix<16x16xf16, "BOp">
            %53 = gpu.subgroup_mma_compute %51, %52, %arg17 : !gpu.mma_matrix<16x16xf16, "AOp">, !gpu.mma_matrix<16x16xf16, "BOp"> -> !gpu.mma_matrix<16x16xf32, "COp">
            %54 = gpu.subgroup_mma_load_matrix %16[%arg2, %arg13] {leadDimension = 40 : index} : memref<128x40xf16, 3> -> !gpu.mma_matrix<16x16xf16, "AOp">
            %c32_37 = constant 32 : index
            %55 = addi %arg3, %c32_37 : index
            %56 = gpu.subgroup_mma_load_matrix %13[%arg13, %55] {leadDimension = 136 : index} : memref<32x136xf16, 3> -> !gpu.mma_matrix<16x16xf16, "BOp">
            %57 = gpu.subgroup_mma_compute %54, %56, %arg18 : !gpu.mma_matrix<16x16xf16, "AOp">, !gpu.mma_matrix<16x16xf16, "BOp"> -> !gpu.mma_matrix<16x16xf32, "COp">
            %58 = gpu.subgroup_mma_load_matrix %16[%43, %arg13] {leadDimension = 40 : index} : memref<128x40xf16, 3> -> !gpu.mma_matrix<16x16xf16, "AOp">
            %59 = gpu.subgroup_mma_load_matrix %13[%arg13, %55] {leadDimension = 136 : index} : memref<32x136xf16, 3> -> !gpu.mma_matrix<16x16xf16, "BOp">
            %60 = gpu.subgroup_mma_compute %58, %59, %arg19 : !gpu.mma_matrix<16x16xf16, "AOp">, !gpu.mma_matrix<16x16xf16, "BOp"> -> !gpu.mma_matrix<16x16xf32, "COp">
            %61 = gpu.subgroup_mma_load_matrix %16[%arg2, %arg13] {leadDimension = 40 : index} : memref<128x40xf16, 3> -> !gpu.mma_matrix<16x16xf16, "AOp">
            %c48_38 = constant 48 : index
            %62 = addi %arg3, %c48_38 : index
            %63 = gpu.subgroup_mma_load_matrix %13[%arg13, %62] {leadDimension = 136 : index} : memref<32x136xf16, 3> -> !gpu.mma_matrix<16x16xf16, "BOp">
            %64 = gpu.subgroup_mma_compute %61, %63, %arg20 : !gpu.mma_matrix<16x16xf16, "AOp">, !gpu.mma_matrix<16x16xf16, "BOp"> -> !gpu.mma_matrix<16x16xf32, "COp">
            %65 = gpu.subgroup_mma_load_matrix %16[%43, %arg13] {leadDimension = 40 : index} : memref<128x40xf16, 3> -> !gpu.mma_matrix<16x16xf16, "AOp">
            %66 = gpu.subgroup_mma_load_matrix %13[%arg13, %62] {leadDimension = 136 : index} : memref<32x136xf16, 3> -> !gpu.mma_matrix<16x16xf16, "BOp">
            %67 = gpu.subgroup_mma_compute %65, %66, %arg21 : !gpu.mma_matrix<16x16xf16, "AOp">, !gpu.mma_matrix<16x16xf16, "BOp"> -> !gpu.mma_matrix<16x16xf32, "COp">
            scf.yield %42, %46, %50, %53, %57, %60, %64, %67 : !gpu.mma_matrix<16x16xf32, "COp">, !gpu.mma_matrix<16x16xf32, "COp">, !gpu.mma_matrix<16x16xf32, "COp">, !gpu.mma_matrix<16x16xf32, "COp">, !gpu.mma_matrix<16x16xf32, "COp">, !gpu.mma_matrix<16x16xf32, "COp">, !gpu.mma_matrix<16x16xf32, "COp">, !gpu.mma_matrix<16x16xf32, "COp">
          }
          scf.yield %39#0, %39#1, %39#2, %39#3, %39#4, %39#5, %39#6, %39#7 : !gpu.mma_matrix<16x16xf32, "COp">, !gpu.mma_matrix<16x16xf32, "COp">, !gpu.mma_matrix<16x16xf32, "COp">, !gpu.mma_matrix<16x16xf32, "COp">, !gpu.mma_matrix<16x16xf32, "COp">, !gpu.mma_matrix<16x16xf32, "COp">, !gpu.mma_matrix<16x16xf32, "COp">, !gpu.mma_matrix<16x16xf32, "COp">
        }
        gpu.subgroup_mma_store_matrix %38#0, %memref_2[%19, %20] {leadDimension = 1024 : index} : !gpu.mma_matrix<16x16xf32, "COp">, memref<1024x1024xf32>
        gpu.subgroup_mma_store_matrix %38#1, %memref_2[%23, %20] {leadDimension = 1024 : index} : !gpu.mma_matrix<16x16xf32, "COp">, memref<1024x1024xf32>
        gpu.subgroup_mma_store_matrix %38#2, %memref_2[%19, %26] {leadDimension = 1024 : index} : !gpu.mma_matrix<16x16xf32, "COp">, memref<1024x1024xf32>
        gpu.subgroup_mma_store_matrix %38#3, %memref_2[%23, %26] {leadDimension = 1024 : index} : !gpu.mma_matrix<16x16xf32, "COp">, memref<1024x1024xf32>
        gpu.subgroup_mma_store_matrix %38#4, %memref_2[%19, %30] {leadDimension = 1024 : index} : !gpu.mma_matrix<16x16xf32, "COp">, memref<1024x1024xf32>
        gpu.subgroup_mma_store_matrix %38#5, %memref_2[%23, %30] {leadDimension = 1024 : index} : !gpu.mma_matrix<16x16xf32, "COp">, memref<1024x1024xf32>
        gpu.subgroup_mma_store_matrix %38#6, %memref_2[%19, %34] {leadDimension = 1024 : index} : !gpu.mma_matrix<16x16xf32, "COp">, memref<1024x1024xf32>
        gpu.subgroup_mma_store_matrix %38#7, %memref_2[%23, %34] {leadDimension = 1024 : index} : !gpu.mma_matrix<16x16xf32, "COp">, memref<1024x1024xf32>
        scf.yield
      }
      scf.yield
    }
    %11 = gpu.wait async 
    %12 = gpu.memcpy async [%11] %2, %memref_2 : memref<1024x1024xf32>, memref<1024x1024xf32>
    gpu.wait [%12]
    return
  }
  gpu.module @initC_kernel {
    gpu.func @initC_kernel(%arg0: memref<1024x1024xf32>) kernel {
      %c1 = constant 1 : index
      %c0 = constant 0 : index
      %c1024 = constant 1024 : index
      %cst = constant 0.000000e+00 : f32
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

