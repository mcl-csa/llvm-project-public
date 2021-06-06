// RUN: mlir-opt %s --canonicalize --test-convert-matmul-parallel-loops-to-gpu --canonicalize

#map = affine_map<(d0) -> (d0)>
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
        %37 = subi %c128_19, %c0_17 : index
        %c1_21 = constant 1 : index
        %38 = subi %c8, %c1_21 : index
        %39 = addi %37, %38 : index
        %40 = divi_signed %39, %c8 : index
        %c1_22 = constant 1 : index
        %c0_23 = constant 0 : index
        %c1_24 = constant 1 : index
        %c1_25 = constant 1 : index
        %41 = muli %c1_25, %c32_18 : index
        %42 = muli %41, %40 : index
        scf.parallel (%arg4) = (%c0_23) to (%42) step (%c1_24) {
          %51 = remi_signed %arg4, %40 : index
          %52 = divi_signed %arg4, %40 : index
          %53 = muli %51, %c8 : index
          %54 = addi %arg1, %53 : index
          %c8_42 = constant 8 : index
          %c0_43 = constant 0 : index
          %c-1 = constant -1 : index
          %55 = cmpi slt, %54, %c0_43 : index
          %56 = subi %c-1, %54 : index
          %57 = select %55, %56, %54 : index
          %58 = divi_signed %57, %c8_42 : index
          %59 = subi %c-1, %58 : index
          %60 = select %55, %59, %58 : index
          %61 = memref.load %6[%52, %60] : memref<1024x128xvector<8xf16>>
          %c8_44 = constant 8 : index
          %c0_45 = constant 0 : index
          %c-1_46 = constant -1 : index
          %62 = cmpi slt, %53, %c0_45 : index
          %63 = subi %c-1_46, %53 : index
          %64 = select %62, %63, %53 : index
          %65 = divi_signed %64, %c8_44 : index
          %66 = subi %c-1_46, %65 : index
          %67 = select %62, %66, %65 : index
          memref.store %61, %14[%52, %67] : memref<32x17xvector<8xf16>, 3>
          scf.yield
        } {mapping = [{bound = #map, map = #map, processor = 0 : i64}]}
        %c0_26 = constant 0 : index
        %c0_27 = constant 0 : index
        %c128_28 = constant 128 : index
        %c32_29 = constant 32 : index
        %c1_30 = constant 1 : index
        %c8_31 = constant 8 : index
        %43 = subi %c32_29, %c0_27 : index
        %c1_32 = constant 1 : index
        %44 = subi %c8_31, %c1_32 : index
        %45 = addi %43, %44 : index
        %46 = divi_signed %45, %c8_31 : index
        %c1_33 = constant 1 : index
        %c0_34 = constant 0 : index
        %c1_35 = constant 1 : index
        %c1_36 = constant 1 : index
        %47 = muli %c1_36, %c128_28 : index
        %48 = muli %47, %46 : index
        scf.parallel (%arg4) = (%c0_34) to (%48) step (%c1_35) {
          %51 = remi_signed %arg4, %46 : index
          %52 = divi_signed %arg4, %46 : index
          %53 = muli %51, %c8_31 : index
          %54 = addi %arg0, %52 : index
          %c8_42 = constant 8 : index
          %c0_43 = constant 0 : index
          %c-1 = constant -1 : index
          %55 = cmpi slt, %53, %c0_43 : index
          %56 = subi %c-1, %53 : index
          %57 = select %55, %56, %53 : index
          %58 = divi_signed %57, %c8_42 : index
          %59 = subi %c-1, %58 : index
          %60 = select %55, %59, %58 : index
          %61 = memref.load %4[%54, %60] : memref<1024x128xvector<8xf16>>
          %c8_44 = constant 8 : index
          %c0_45 = constant 0 : index
          %c-1_46 = constant -1 : index
          %62 = cmpi slt, %53, %c0_45 : index
          %63 = subi %c-1_46, %53 : index
          %64 = select %62, %63, %53 : index
          %65 = divi_signed %64, %c8_44 : index
          %66 = subi %c-1_46, %65 : index
          %67 = select %62, %66, %65 : index
          memref.store %61, %17[%52, %67] : memref<128x5xvector<8xf16>, 3>
          scf.yield
        } {mapping = [{bound = #map, map = #map, processor = 0 : i64}]}
        gpu.barrier
        %c0_37 = constant 0 : index
        %c992 = constant 992 : index
        %c32_38 = constant 32 : index
        %49:8 = scf.for %arg4 = %c0_37 to %c992 step %c32_38 iter_args(%arg5 = %21, %arg6 = %24, %arg7 = %27, %arg8 = %28, %arg9 = %31, %arg10 = %32, %arg11 = %35, %arg12 = %36) -> (!gpu.mma_matrix<16x16xf32, "COp">, !gpu.mma_matrix<16x16xf32, "COp">, !gpu.mma_matrix<16x16xf32, "COp">, !gpu.mma_matrix<16x16xf32, "COp">, !gpu.mma_matrix<16x16xf32, "COp">, !gpu.mma_matrix<16x16xf32, "COp">, !gpu.mma_matrix<16x16xf32, "COp">, !gpu.mma_matrix<16x16xf32, "COp">) {
          %c0_42 = constant 0 : index
          %c0_43 = constant 0 : index
          %c32_44 = constant 32 : index
          %c128_45 = constant 128 : index
          %c1_46 = constant 1 : index
          %c8_47 = constant 8 : index
          %51 = subi %c128_45, %c0_43 : index
          %c1_48 = constant 1 : index
          %52 = subi %c8_47, %c1_48 : index
          %53 = addi %51, %52 : index
          %54 = divi_signed %53, %c8_47 : index
          %c1_49 = constant 1 : index
          %c0_50 = constant 0 : index
          %c1_51 = constant 1 : index
          %c1_52 = constant 1 : index
          %55 = muli %c1_52, %c32_44 : index
          %56 = muli %55, %54 : index
          scf.parallel (%arg13) = (%c0_50) to (%56) step (%c1_51) {
            %64 = remi_signed %arg13, %54 : index
            %65 = divi_signed %arg13, %54 : index
            %66 = muli %64, %c8_47 : index
            %67 = addi %arg4, %65 : index
            %c32_67 = constant 32 : index
            %68 = addi %67, %c32_67 : index
            %69 = addi %arg1, %66 : index
            %c8_68 = constant 8 : index
            %c0_69 = constant 0 : index
            %c-1 = constant -1 : index
            %70 = cmpi slt, %69, %c0_69 : index
            %71 = subi %c-1, %69 : index
            %72 = select %70, %71, %69 : index
            %73 = divi_signed %72, %c8_68 : index
            %74 = subi %c-1, %73 : index
            %75 = select %70, %74, %73 : index
            %76 = memref.load %7[%68, %75] : memref<1024x128xvector<8xf16>>
            %c8_70 = constant 8 : index
            %c0_71 = constant 0 : index
            %c-1_72 = constant -1 : index
            %77 = cmpi slt, %66, %c0_71 : index
            %78 = subi %c-1_72, %66 : index
            %79 = select %77, %78, %66 : index
            %80 = divi_signed %79, %c8_70 : index
            %81 = subi %c-1_72, %80 : index
            %82 = select %77, %81, %80 : index
            memref.store %76, %15[%65, %82] : memref<32x17xvector<8xf16>, 3>
            scf.yield
          } {mapping = [{bound = #map, map = #map, processor = 0 : i64}]}
          %c0_53 = constant 0 : index
          %c0_54 = constant 0 : index
          %c128_55 = constant 128 : index
          %c32_56 = constant 32 : index
          %c1_57 = constant 1 : index
          %c8_58 = constant 8 : index
          %57 = subi %c32_56, %c0_54 : index
          %c1_59 = constant 1 : index
          %58 = subi %c8_58, %c1_59 : index
          %59 = addi %57, %58 : index
          %60 = divi_signed %59, %c8_58 : index
          %c1_60 = constant 1 : index
          %c0_61 = constant 0 : index
          %c1_62 = constant 1 : index
          %c1_63 = constant 1 : index
          %61 = muli %c1_63, %c128_55 : index
          %62 = muli %61, %60 : index
          scf.parallel (%arg13) = (%c0_61) to (%62) step (%c1_62) {
            %64 = remi_signed %arg13, %60 : index
            %65 = divi_signed %arg13, %60 : index
            %66 = muli %64, %c8_58 : index
            %67 = addi %arg0, %65 : index
            %68 = addi %arg4, %66 : index
            %c8_67 = constant 8 : index
            %c0_68 = constant 0 : index
            %c-1 = constant -1 : index
            %69 = cmpi slt, %68, %c0_68 : index
            %70 = subi %c-1, %68 : index
            %71 = select %69, %70, %68 : index
            %72 = divi_signed %71, %c8_67 : index
            %73 = subi %c-1, %72 : index
            %74 = select %69, %73, %72 : index
            %c4 = constant 4 : index
            %75 = addi %74, %c4 : index
            %76 = memref.load %5[%67, %75] : memref<1024x128xvector<8xf16>>
            %c8_69 = constant 8 : index
            %c0_70 = constant 0 : index
            %c-1_71 = constant -1 : index
            %77 = cmpi slt, %66, %c0_70 : index
            %78 = subi %c-1_71, %66 : index
            %79 = select %77, %78, %66 : index
            %80 = divi_signed %79, %c8_69 : index
            %81 = subi %c-1_71, %80 : index
            %82 = select %77, %81, %80 : index
            memref.store %76, %18[%65, %82] : memref<128x5xvector<8xf16>, 3>
            scf.yield
          } {mapping = [{bound = #map, map = #map, processor = 0 : i64}]}
          %c0_64 = constant 0 : index
          %c32_65 = constant 32 : index
          %c16_66 = constant 16 : index
          %63:8 = scf.for %arg13 = %c0_64 to %c32_65 step %c16_66 iter_args(%arg14 = %arg5, %arg15 = %arg6, %arg16 = %arg7, %arg17 = %arg8, %arg18 = %arg9, %arg19 = %arg10, %arg20 = %arg11, %arg21 = %arg12) -> (!gpu.mma_matrix<16x16xf32, "COp">, !gpu.mma_matrix<16x16xf32, "COp">, !gpu.mma_matrix<16x16xf32, "COp">, !gpu.mma_matrix<16x16xf32, "COp">, !gpu.mma_matrix<16x16xf32, "COp">, !gpu.mma_matrix<16x16xf32, "COp">, !gpu.mma_matrix<16x16xf32, "COp">, !gpu.mma_matrix<16x16xf32, "COp">) {
            %64 = gpu.subgroup_mma_load_matrix %16[%arg2, %arg13] {leadDimension = 40 : index} : memref<128x40xf16, 3> -> !gpu.mma_matrix<16x16xf16, "AOp">
            %65 = gpu.subgroup_mma_load_matrix %13[%arg13, %arg3] {leadDimension = 136 : index} : memref<32x136xf16, 3> -> !gpu.mma_matrix<16x16xf16, "BOp">
            %66 = gpu.subgroup_mma_compute %64, %65, %arg14 : !gpu.mma_matrix<16x16xf16, "AOp">, !gpu.mma_matrix<16x16xf16, "BOp"> -> !gpu.mma_matrix<16x16xf32, "COp">
            %c16_67 = constant 16 : index
            %67 = addi %arg2, %c16_67 : index
            %68 = gpu.subgroup_mma_load_matrix %16[%67, %arg13] {leadDimension = 40 : index} : memref<128x40xf16, 3> -> !gpu.mma_matrix<16x16xf16, "AOp">
            %69 = gpu.subgroup_mma_load_matrix %13[%arg13, %arg3] {leadDimension = 136 : index} : memref<32x136xf16, 3> -> !gpu.mma_matrix<16x16xf16, "BOp">
            %70 = gpu.subgroup_mma_compute %68, %69, %arg15 : !gpu.mma_matrix<16x16xf16, "AOp">, !gpu.mma_matrix<16x16xf16, "BOp"> -> !gpu.mma_matrix<16x16xf32, "COp">
            %71 = gpu.subgroup_mma_load_matrix %16[%arg2, %arg13] {leadDimension = 40 : index} : memref<128x40xf16, 3> -> !gpu.mma_matrix<16x16xf16, "AOp">
            %c16_68 = constant 16 : index
            %72 = addi %arg3, %c16_68 : index
            %73 = gpu.subgroup_mma_load_matrix %13[%arg13, %72] {leadDimension = 136 : index} : memref<32x136xf16, 3> -> !gpu.mma_matrix<16x16xf16, "BOp">
            %74 = gpu.subgroup_mma_compute %71, %73, %arg16 : !gpu.mma_matrix<16x16xf16, "AOp">, !gpu.mma_matrix<16x16xf16, "BOp"> -> !gpu.mma_matrix<16x16xf32, "COp">
            %75 = gpu.subgroup_mma_load_matrix %16[%67, %arg13] {leadDimension = 40 : index} : memref<128x40xf16, 3> -> !gpu.mma_matrix<16x16xf16, "AOp">
            %76 = gpu.subgroup_mma_load_matrix %13[%arg13, %72] {leadDimension = 136 : index} : memref<32x136xf16, 3> -> !gpu.mma_matrix<16x16xf16, "BOp">
            %77 = gpu.subgroup_mma_compute %75, %76, %arg17 : !gpu.mma_matrix<16x16xf16, "AOp">, !gpu.mma_matrix<16x16xf16, "BOp"> -> !gpu.mma_matrix<16x16xf32, "COp">
            %78 = gpu.subgroup_mma_load_matrix %16[%arg2, %arg13] {leadDimension = 40 : index} : memref<128x40xf16, 3> -> !gpu.mma_matrix<16x16xf16, "AOp">
            %c32_69 = constant 32 : index
            %79 = addi %arg3, %c32_69 : index
            %80 = gpu.subgroup_mma_load_matrix %13[%arg13, %79] {leadDimension = 136 : index} : memref<32x136xf16, 3> -> !gpu.mma_matrix<16x16xf16, "BOp">
            %81 = gpu.subgroup_mma_compute %78, %80, %arg18 : !gpu.mma_matrix<16x16xf16, "AOp">, !gpu.mma_matrix<16x16xf16, "BOp"> -> !gpu.mma_matrix<16x16xf32, "COp">
            %82 = gpu.subgroup_mma_load_matrix %16[%67, %arg13] {leadDimension = 40 : index} : memref<128x40xf16, 3> -> !gpu.mma_matrix<16x16xf16, "AOp">
            %83 = gpu.subgroup_mma_load_matrix %13[%arg13, %79] {leadDimension = 136 : index} : memref<32x136xf16, 3> -> !gpu.mma_matrix<16x16xf16, "BOp">
            %84 = gpu.subgroup_mma_compute %82, %83, %arg19 : !gpu.mma_matrix<16x16xf16, "AOp">, !gpu.mma_matrix<16x16xf16, "BOp"> -> !gpu.mma_matrix<16x16xf32, "COp">
            %85 = gpu.subgroup_mma_load_matrix %16[%arg2, %arg13] {leadDimension = 40 : index} : memref<128x40xf16, 3> -> !gpu.mma_matrix<16x16xf16, "AOp">
            %c48_70 = constant 48 : index
            %86 = addi %arg3, %c48_70 : index
            %87 = gpu.subgroup_mma_load_matrix %13[%arg13, %86] {leadDimension = 136 : index} : memref<32x136xf16, 3> -> !gpu.mma_matrix<16x16xf16, "BOp">
            %88 = gpu.subgroup_mma_compute %85, %87, %arg20 : !gpu.mma_matrix<16x16xf16, "AOp">, !gpu.mma_matrix<16x16xf16, "BOp"> -> !gpu.mma_matrix<16x16xf32, "COp">
            %89 = gpu.subgroup_mma_load_matrix %16[%67, %arg13] {leadDimension = 40 : index} : memref<128x40xf16, 3> -> !gpu.mma_matrix<16x16xf16, "AOp">
            %90 = gpu.subgroup_mma_load_matrix %13[%arg13, %86] {leadDimension = 136 : index} : memref<32x136xf16, 3> -> !gpu.mma_matrix<16x16xf16, "BOp">
            %91 = gpu.subgroup_mma_compute %89, %90, %arg21 : !gpu.mma_matrix<16x16xf16, "AOp">, !gpu.mma_matrix<16x16xf16, "BOp"> -> !gpu.mma_matrix<16x16xf32, "COp">
            scf.yield %66, %70, %74, %77, %81, %84, %88, %91 : !gpu.mma_matrix<16x16xf32, "COp">, !gpu.mma_matrix<16x16xf32, "COp">, !gpu.mma_matrix<16x16xf32, "COp">, !gpu.mma_matrix<16x16xf32, "COp">, !gpu.mma_matrix<16x16xf32, "COp">, !gpu.mma_matrix<16x16xf32, "COp">, !gpu.mma_matrix<16x16xf32, "COp">, !gpu.mma_matrix<16x16xf32, "COp">
          }
          scf.yield %63#0, %63#1, %63#2, %63#3, %63#4, %63#5, %63#6, %63#7 : !gpu.mma_matrix<16x16xf32, "COp">, !gpu.mma_matrix<16x16xf32, "COp">, !gpu.mma_matrix<16x16xf32, "COp">, !gpu.mma_matrix<16x16xf32, "COp">, !gpu.mma_matrix<16x16xf32, "COp">, !gpu.mma_matrix<16x16xf32, "COp">, !gpu.mma_matrix<16x16xf32, "COp">, !gpu.mma_matrix<16x16xf32, "COp">
        }
        gpu.barrier
        %c992_39 = constant 992 : index
        %c1024_40 = constant 1024 : index
        %c32_41 = constant 32 : index
        %50:8 = scf.for %arg4 = %c992_39 to %c1024_40 step %c32_41 iter_args(%arg5 = %49#0, %arg6 = %49#1, %arg7 = %49#2, %arg8 = %49#3, %arg9 = %49#4, %arg10 = %49#5, %arg11 = %49#6, %arg12 = %49#7) -> (!gpu.mma_matrix<16x16xf32, "COp">, !gpu.mma_matrix<16x16xf32, "COp">, !gpu.mma_matrix<16x16xf32, "COp">, !gpu.mma_matrix<16x16xf32, "COp">, !gpu.mma_matrix<16x16xf32, "COp">, !gpu.mma_matrix<16x16xf32, "COp">, !gpu.mma_matrix<16x16xf32, "COp">, !gpu.mma_matrix<16x16xf32, "COp">) {
          %c0_42 = constant 0 : index
          %c32_43 = constant 32 : index
          %c16_44 = constant 16 : index
          %51:8 = scf.for %arg13 = %c0_42 to %c32_43 step %c16_44 iter_args(%arg14 = %arg5, %arg15 = %arg6, %arg16 = %arg7, %arg17 = %arg8, %arg18 = %arg9, %arg19 = %arg10, %arg20 = %arg11, %arg21 = %arg12) -> (!gpu.mma_matrix<16x16xf32, "COp">, !gpu.mma_matrix<16x16xf32, "COp">, !gpu.mma_matrix<16x16xf32, "COp">, !gpu.mma_matrix<16x16xf32, "COp">, !gpu.mma_matrix<16x16xf32, "COp">, !gpu.mma_matrix<16x16xf32, "COp">, !gpu.mma_matrix<16x16xf32, "COp">, !gpu.mma_matrix<16x16xf32, "COp">) {
            %52 = gpu.subgroup_mma_load_matrix %16[%arg2, %arg13] {leadDimension = 40 : index} : memref<128x40xf16, 3> -> !gpu.mma_matrix<16x16xf16, "AOp">
            %53 = gpu.subgroup_mma_load_matrix %13[%arg13, %arg3] {leadDimension = 136 : index} : memref<32x136xf16, 3> -> !gpu.mma_matrix<16x16xf16, "BOp">
            %54 = gpu.subgroup_mma_compute %52, %53, %arg14 : !gpu.mma_matrix<16x16xf16, "AOp">, !gpu.mma_matrix<16x16xf16, "BOp"> -> !gpu.mma_matrix<16x16xf32, "COp">
            %c16_45 = constant 16 : index
            %55 = addi %arg2, %c16_45 : index
            %56 = gpu.subgroup_mma_load_matrix %16[%55, %arg13] {leadDimension = 40 : index} : memref<128x40xf16, 3> -> !gpu.mma_matrix<16x16xf16, "AOp">
            %57 = gpu.subgroup_mma_load_matrix %13[%arg13, %arg3] {leadDimension = 136 : index} : memref<32x136xf16, 3> -> !gpu.mma_matrix<16x16xf16, "BOp">
            %58 = gpu.subgroup_mma_compute %56, %57, %arg15 : !gpu.mma_matrix<16x16xf16, "AOp">, !gpu.mma_matrix<16x16xf16, "BOp"> -> !gpu.mma_matrix<16x16xf32, "COp">
            %59 = gpu.subgroup_mma_load_matrix %16[%arg2, %arg13] {leadDimension = 40 : index} : memref<128x40xf16, 3> -> !gpu.mma_matrix<16x16xf16, "AOp">
            %c16_46 = constant 16 : index
            %60 = addi %arg3, %c16_46 : index
            %61 = gpu.subgroup_mma_load_matrix %13[%arg13, %60] {leadDimension = 136 : index} : memref<32x136xf16, 3> -> !gpu.mma_matrix<16x16xf16, "BOp">
            %62 = gpu.subgroup_mma_compute %59, %61, %arg16 : !gpu.mma_matrix<16x16xf16, "AOp">, !gpu.mma_matrix<16x16xf16, "BOp"> -> !gpu.mma_matrix<16x16xf32, "COp">
            %63 = gpu.subgroup_mma_load_matrix %16[%55, %arg13] {leadDimension = 40 : index} : memref<128x40xf16, 3> -> !gpu.mma_matrix<16x16xf16, "AOp">
            %64 = gpu.subgroup_mma_load_matrix %13[%arg13, %60] {leadDimension = 136 : index} : memref<32x136xf16, 3> -> !gpu.mma_matrix<16x16xf16, "BOp">
            %65 = gpu.subgroup_mma_compute %63, %64, %arg17 : !gpu.mma_matrix<16x16xf16, "AOp">, !gpu.mma_matrix<16x16xf16, "BOp"> -> !gpu.mma_matrix<16x16xf32, "COp">
            %66 = gpu.subgroup_mma_load_matrix %16[%arg2, %arg13] {leadDimension = 40 : index} : memref<128x40xf16, 3> -> !gpu.mma_matrix<16x16xf16, "AOp">
            %c32_47 = constant 32 : index
            %67 = addi %arg3, %c32_47 : index
            %68 = gpu.subgroup_mma_load_matrix %13[%arg13, %67] {leadDimension = 136 : index} : memref<32x136xf16, 3> -> !gpu.mma_matrix<16x16xf16, "BOp">
            %69 = gpu.subgroup_mma_compute %66, %68, %arg18 : !gpu.mma_matrix<16x16xf16, "AOp">, !gpu.mma_matrix<16x16xf16, "BOp"> -> !gpu.mma_matrix<16x16xf32, "COp">
            %70 = gpu.subgroup_mma_load_matrix %16[%55, %arg13] {leadDimension = 40 : index} : memref<128x40xf16, 3> -> !gpu.mma_matrix<16x16xf16, "AOp">
            %71 = gpu.subgroup_mma_load_matrix %13[%arg13, %67] {leadDimension = 136 : index} : memref<32x136xf16, 3> -> !gpu.mma_matrix<16x16xf16, "BOp">
            %72 = gpu.subgroup_mma_compute %70, %71, %arg19 : !gpu.mma_matrix<16x16xf16, "AOp">, !gpu.mma_matrix<16x16xf16, "BOp"> -> !gpu.mma_matrix<16x16xf32, "COp">
            %73 = gpu.subgroup_mma_load_matrix %16[%arg2, %arg13] {leadDimension = 40 : index} : memref<128x40xf16, 3> -> !gpu.mma_matrix<16x16xf16, "AOp">
            %c48_48 = constant 48 : index
            %74 = addi %arg3, %c48_48 : index
            %75 = gpu.subgroup_mma_load_matrix %13[%arg13, %74] {leadDimension = 136 : index} : memref<32x136xf16, 3> -> !gpu.mma_matrix<16x16xf16, "BOp">
            %76 = gpu.subgroup_mma_compute %73, %75, %arg20 : !gpu.mma_matrix<16x16xf16, "AOp">, !gpu.mma_matrix<16x16xf16, "BOp"> -> !gpu.mma_matrix<16x16xf32, "COp">
            %77 = gpu.subgroup_mma_load_matrix %16[%55, %arg13] {leadDimension = 40 : index} : memref<128x40xf16, 3> -> !gpu.mma_matrix<16x16xf16, "AOp">
            %78 = gpu.subgroup_mma_load_matrix %13[%arg13, %74] {leadDimension = 136 : index} : memref<32x136xf16, 3> -> !gpu.mma_matrix<16x16xf16, "BOp">
            %79 = gpu.subgroup_mma_compute %77, %78, %arg21 : !gpu.mma_matrix<16x16xf16, "AOp">, !gpu.mma_matrix<16x16xf16, "BOp"> -> !gpu.mma_matrix<16x16xf32, "COp">
            scf.yield %54, %58, %62, %65, %69, %72, %76, %79 : !gpu.mma_matrix<16x16xf32, "COp">, !gpu.mma_matrix<16x16xf32, "COp">, !gpu.mma_matrix<16x16xf32, "COp">, !gpu.mma_matrix<16x16xf32, "COp">, !gpu.mma_matrix<16x16xf32, "COp">, !gpu.mma_matrix<16x16xf32, "COp">, !gpu.mma_matrix<16x16xf32, "COp">, !gpu.mma_matrix<16x16xf32, "COp">
          }
          scf.yield %51#0, %51#1, %51#2, %51#3, %51#4, %51#5, %51#6, %51#7 : !gpu.mma_matrix<16x16xf32, "COp">, !gpu.mma_matrix<16x16xf32, "COp">, !gpu.mma_matrix<16x16xf32, "COp">, !gpu.mma_matrix<16x16xf32, "COp">, !gpu.mma_matrix<16x16xf32, "COp">, !gpu.mma_matrix<16x16xf32, "COp">, !gpu.mma_matrix<16x16xf32, "COp">, !gpu.mma_matrix<16x16xf32, "COp">
        }
        gpu.subgroup_mma_store_matrix %50#0, %memref_2[%19, %20] {leadDimension = 1024 : index} : !gpu.mma_matrix<16x16xf32, "COp">, memref<1024x1024xf32>
        gpu.subgroup_mma_store_matrix %50#1, %memref_2[%23, %20] {leadDimension = 1024 : index} : !gpu.mma_matrix<16x16xf32, "COp">, memref<1024x1024xf32>
        gpu.subgroup_mma_store_matrix %50#2, %memref_2[%19, %26] {leadDimension = 1024 : index} : !gpu.mma_matrix<16x16xf32, "COp">, memref<1024x1024xf32>
        gpu.subgroup_mma_store_matrix %50#3, %memref_2[%23, %26] {leadDimension = 1024 : index} : !gpu.mma_matrix<16x16xf32, "COp">, memref<1024x1024xf32>
        gpu.subgroup_mma_store_matrix %50#4, %memref_2[%19, %30] {leadDimension = 1024 : index} : !gpu.mma_matrix<16x16xf32, "COp">, memref<1024x1024xf32>
        gpu.subgroup_mma_store_matrix %50#5, %memref_2[%23, %30] {leadDimension = 1024 : index} : !gpu.mma_matrix<16x16xf32, "COp">, memref<1024x1024xf32>
        gpu.subgroup_mma_store_matrix %50#6, %memref_2[%19, %34] {leadDimension = 1024 : index} : !gpu.mma_matrix<16x16xf32, "COp">, memref<1024x1024xf32>
        gpu.subgroup_mma_store_matrix %50#7, %memref_2[%23, %34] {leadDimension = 1024 : index} : !gpu.mma_matrix<16x16xf32, "COp">, memref<1024x1024xf32>
        scf.yield
      } {mapping = [{bound = #map, map = #map, processor = 7 : i64}, {bound = #map, map = #map, processor = 6 : i64}]}
      scf.yield
    } {mapping = [{bound = #map, map = #map, processor = 1 : i64}, {bound = #map, map = #map, processor = 0 : i64}]}
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

