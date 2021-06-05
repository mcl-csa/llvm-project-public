// RUN: mlir-opt %s --canonicalize --test-convert-matmul-parallel-loops-to-gpu --canonicalize

#map = affine_map<(d0) -> (d0)>
module  {
  memref.global "public" @frag_A : memref<128x64xf16, 3>
  memref.global "public" @frag_B : memref<64x64xf16, 3>
  memref.global "public" @frag_B_padded : memref<64x72xf16, 3>
  memref.global "public" @frag_A_padded : memref<128x72xf16, 3>
  func @main() {
    %c16 = constant 16 : index
    %cst = constant 0.000000e+00 : f32
    %c0 = constant 0 : index
    %c1 = constant 1 : index
    %c137438953472_i64 = constant 137438953472 : i64
    %c4096 = constant 4096 : index
    %0 = memref.alloc() : memref<4096x4096xf16>
    %1 = memref.alloc() : memref<4096x4096xf16>
    %2 = memref.alloc() : memref<4096x4096xf32>
    scf.for %arg0 = %c0 to %c4096 step %c1 {
      scf.for %arg1 = %c0 to %c4096 step %c1 {
        %18 = remi_signed %arg0, %c16 : index
        %19 = remi_signed %arg1, %c16 : index
        %20 = addi %18, %19 : index
        %21 = remi_signed %20, %c16 : index
        %22 = index_cast %21 : index to i16
        %23 = sitofp %22 : i16 to f16
        memref.store %23, %0[%arg0, %arg1] : memref<4096x4096xf16>
      }
    }
    scf.for %arg0 = %c0 to %c4096 step %c1 {
      scf.for %arg1 = %c0 to %c4096 step %c1 {
        %18 = remi_signed %arg0, %c16 : index
        %19 = remi_signed %arg1, %c16 : index
        %20 = addi %18, %19 : index
        %21 = remi_signed %20, %c16 : index
        %22 = index_cast %21 : index to i16
        %23 = sitofp %22 : i16 to f16
        memref.store %23, %1[%arg0, %arg1] : memref<4096x4096xf16>
      }
    }
    scf.for %arg0 = %c0 to %c4096 step %c1 {
      scf.for %arg1 = %c0 to %c4096 step %c1 {
        memref.store %cst, %2[%arg0, %arg1] : memref<4096x4096xf32>
      }
    }
    %3 = gpu.wait async
    %memref, %asyncToken = gpu.alloc async [%3] () : memref<4096x4096xf16>
    %4 = memref_vector_cast %memref : memref<4096x4096xf16> to memref<4096x512xvector<8xf16>>
    %5 = memref_vector_cast %memref : memref<4096x4096xf16> to memref<4096x512xvector<8xf16>>
    %memref_0, %asyncToken_1 = gpu.alloc async [%3] () : memref<4096x4096xf16>
    %6 = memref_vector_cast %memref_0 : memref<4096x4096xf16> to memref<4096x512xvector<8xf16>>
    %7 = memref_vector_cast %memref_0 : memref<4096x4096xf16> to memref<4096x512xvector<8xf16>>
    %memref_2, %asyncToken_3 = gpu.alloc async [%3] () : memref<4096x4096xf32>
    %8 = gpu.memcpy async [%3] %memref, %0 : memref<4096x4096xf16>, memref<4096x4096xf16>
    %9 = gpu.memcpy async [%3] %memref_0, %1 : memref<4096x4096xf16>, memref<4096x4096xf16>
    %10 = gpu.memcpy async [%3] %memref_2, %2 : memref<4096x4096xf32>, memref<4096x4096xf32>
    gpu.wait [%3]
    %11 = call @rtclock() : () -> f64
    %c0_4 = constant 0 : index
    %c0_5 = constant 0 : index
    %c4096_6 = constant 4096 : index
    %c4096_7 = constant 4096 : index
    %c128 = constant 128 : index
    %c64 = constant 64 : index
    scf.parallel (%arg0, %arg1) = (%c0_4, %c0_5) to (%c4096_6, %c4096_7) step (%c128, %c64) {
      %18 = memref.get_global @frag_B_padded : memref<64x72xf16, 3>
      %19 = memref_vector_cast %18 : memref<64x72xf16, 3> to memref<64x9xvector<8xf16>, 3>
      %20 = memref_vector_cast %18 : memref<64x72xf16, 3> to memref<64x9xvector<8xf16>, 3>
      %21 = memref.get_global @frag_A_padded : memref<128x72xf16, 3>
      %22 = memref_vector_cast %21 : memref<128x72xf16, 3> to memref<128x9xvector<8xf16>, 3>
      %23 = memref_vector_cast %21 : memref<128x72xf16, 3> to memref<128x9xvector<8xf16>, 3>
      %c0_8 = constant 0 : index
      %c0_9 = constant 0 : index
      %c128_10 = constant 128 : index
      %c64_11 = constant 64 : index
      %c64_12 = constant 64 : index
      %c32 = constant 32 : index
      scf.parallel (%arg2, %arg3) = (%c0_8, %c0_9) to (%c128_10, %c64_11) step (%c64_12, %c32) {
        %24 = addi %arg0, %arg2 : index
        %25 = addi %arg1, %arg3 : index
        %26 = gpu.subgroup_mma_load_matrix %memref_2[%24, %25] {leadDimension = 4096 : index, operand = "COp"} : memref<4096x4096xf32> -> !gpu.mmafragment<8, f32>
        %27 = addi %arg0, %arg2 : index
        %c16_13 = constant 16 : index
        %28 = addi %27, %c16_13 : index
        %29 = gpu.subgroup_mma_load_matrix %memref_2[%28, %25] {leadDimension = 4096 : index, operand = "COp"} : memref<4096x4096xf32> -> !gpu.mmafragment<8, f32>
        %30 = addi %arg0, %arg2 : index
        %c32_14 = constant 32 : index
        %31 = addi %30, %c32_14 : index
        %32 = gpu.subgroup_mma_load_matrix %memref_2[%31, %25] {leadDimension = 4096 : index, operand = "COp"} : memref<4096x4096xf32> -> !gpu.mmafragment<8, f32>
        %33 = addi %arg0, %arg2 : index
        %c48 = constant 48 : index
        %34 = addi %33, %c48 : index
        %35 = gpu.subgroup_mma_load_matrix %memref_2[%34, %25] {leadDimension = 4096 : index, operand = "COp"} : memref<4096x4096xf32> -> !gpu.mmafragment<8, f32>
        %36 = addi %arg1, %arg3 : index
        %c16_15 = constant 16 : index
        %37 = addi %36, %c16_15 : index
        %38 = gpu.subgroup_mma_load_matrix %memref_2[%24, %37] {leadDimension = 4096 : index, operand = "COp"} : memref<4096x4096xf32> -> !gpu.mmafragment<8, f32>
        %39 = gpu.subgroup_mma_load_matrix %memref_2[%28, %37] {leadDimension = 4096 : index, operand = "COp"} : memref<4096x4096xf32> -> !gpu.mmafragment<8, f32>
        %40 = gpu.subgroup_mma_load_matrix %memref_2[%31, %37] {leadDimension = 4096 : index, operand = "COp"} : memref<4096x4096xf32> -> !gpu.mmafragment<8, f32>
        %41 = gpu.subgroup_mma_load_matrix %memref_2[%34, %37] {leadDimension = 4096 : index, operand = "COp"} : memref<4096x4096xf32> -> !gpu.mmafragment<8, f32>
        %c0_16 = constant 0 : index
        %c0_17 = constant 0 : index
        %c64_18 = constant 64 : index
        %c64_19 = constant 64 : index
        %c1_20 = constant 1 : index
        %c8 = constant 8 : index
        %42 = subi %c64_19, %c0_17 : index
        %c1_21 = constant 1 : index
        %43 = subi %c8, %c1_21 : index
        %44 = addi %42, %43 : index
        %45 = divi_signed %44, %c8 : index
        %c1_22 = constant 1 : index
        %c0_23 = constant 0 : index
        %c1_24 = constant 1 : index
        %c1_25 = constant 1 : index
        %46 = muli %c1_25, %c64_18 : index
        %47 = muli %46, %45 : index
        scf.parallel (%arg4) = (%c0_23) to (%47) step (%c1_24) {
          %56 = remi_signed %arg4, %45 : index
          %57 = divi_signed %arg4, %45 : index
          %58 = muli %56, %c8 : index
          %59 = addi %arg1, %58 : index
          %c8_42 = constant 8 : index
          %c0_43 = constant 0 : index
          %c-1 = constant -1 : index
          %60 = cmpi slt, %59, %c0_43 : index
          %61 = subi %c-1, %59 : index
          %62 = select %60, %61, %59 : index
          %63 = divi_signed %62, %c8_42 : index
          %64 = subi %c-1, %63 : index
          %65 = select %60, %64, %63 : index
          %66 = memref.load %7[%57, %65] : memref<4096x512xvector<8xf16>>
          %c8_44 = constant 8 : index
          %c0_45 = constant 0 : index
          %c-1_46 = constant -1 : index
          %67 = cmpi slt, %58, %c0_45 : index
          %68 = subi %c-1_46, %58 : index
          %69 = select %67, %68, %58 : index
          %70 = divi_signed %69, %c8_44 : index
          %71 = subi %c-1_46, %70 : index
          %72 = select %67, %71, %70 : index
          memref.store %66, %20[%57, %72] : memref<64x9xvector<8xf16>, 3>
          scf.yield
        } {mapping = [{bound = #map, map = #map, processor = 0 : i64}]}
        %c0_26 = constant 0 : index
        %c0_27 = constant 0 : index
        %c128_28 = constant 128 : index
        %c64_29 = constant 64 : index
        %c1_30 = constant 1 : index
        %c8_31 = constant 8 : index
        %48 = subi %c64_29, %c0_27 : index
        %c1_32 = constant 1 : index
        %49 = subi %c8_31, %c1_32 : index
        %50 = addi %48, %49 : index
        %51 = divi_signed %50, %c8_31 : index
        %c1_33 = constant 1 : index
        %c0_34 = constant 0 : index
        %c1_35 = constant 1 : index
        %c1_36 = constant 1 : index
        %52 = muli %c1_36, %c128_28 : index
        %53 = muli %52, %51 : index
        scf.parallel (%arg4) = (%c0_34) to (%53) step (%c1_35) {
          %56 = remi_signed %arg4, %51 : index
          %57 = divi_signed %arg4, %51 : index
          %58 = muli %56, %c8_31 : index
          %59 = addi %arg0, %57 : index
          %c8_42 = constant 8 : index
          %c0_43 = constant 0 : index
          %c-1 = constant -1 : index
          %60 = cmpi slt, %58, %c0_43 : index
          %61 = subi %c-1, %58 : index
          %62 = select %60, %61, %58 : index
          %63 = divi_signed %62, %c8_42 : index
          %64 = subi %c-1, %63 : index
          %65 = select %60, %64, %63 : index
          %66 = memref.load %4[%59, %65] : memref<4096x512xvector<8xf16>>
          %c8_44 = constant 8 : index
          %c0_45 = constant 0 : index
          %c-1_46 = constant -1 : index
          %67 = cmpi slt, %58, %c0_45 : index
          %68 = subi %c-1_46, %58 : index
          %69 = select %67, %68, %58 : index
          %70 = divi_signed %69, %c8_44 : index
          %71 = subi %c-1_46, %70 : index
          %72 = select %67, %71, %70 : index
          memref.store %66, %22[%57, %72] : memref<128x9xvector<8xf16>, 3>
          scf.yield
        } {mapping = [{bound = #map, map = #map, processor = 0 : i64}]}
        gpu.barrier
        %c0_37 = constant 0 : index
        %c4032 = constant 4032 : index
        %c64_38 = constant 64 : index
        %54:8 = scf.for %arg4 = %c0_37 to %c4032 step %c64_38 iter_args(%arg5 = %26, %arg6 = %29, %arg7 = %32, %arg8 = %35, %arg9 = %38, %arg10 = %39, %arg11 = %40, %arg12 = %41) -> (!gpu.mmafragment<8, f32>, !gpu.mmafragment<8, f32>, !gpu.mmafragment<8, f32>, !gpu.mmafragment<8, f32>, !gpu.mmafragment<8, f32>, !gpu.mmafragment<8, f32>, !gpu.mmafragment<8, f32>, !gpu.mmafragment<8, f32>) {
          %c0_42 = constant 0 : index
          %c0_43 = constant 0 : index
          %c64_44 = constant 64 : index
          %c64_45 = constant 64 : index
          %c1_46 = constant 1 : index
          %c8_47 = constant 8 : index
          %56 = subi %c64_45, %c0_43 : index
          %c1_48 = constant 1 : index
          %57 = subi %c8_47, %c1_48 : index
          %58 = addi %56, %57 : index
          %59 = divi_signed %58, %c8_47 : index
          %c1_49 = constant 1 : index
          %c0_50 = constant 0 : index
          %c1_51 = constant 1 : index
          %c1_52 = constant 1 : index
          %60 = muli %c1_52, %c64_44 : index
          %61 = muli %60, %59 : index
          scf.parallel (%arg13) = (%c0_50) to (%61) step (%c1_51) {
            %69 = remi_signed %arg13, %59 : index
            %70 = divi_signed %arg13, %59 : index
            %71 = muli %69, %c8_47 : index
            %72 = addi %70, %arg4 : index
            %c64_67 = constant 64 : index
            %73 = addi %72, %c64_67 : index
            %74 = addi %arg1, %71 : index
            %c8_68 = constant 8 : index
            %c0_69 = constant 0 : index
            %c-1 = constant -1 : index
            %75 = cmpi slt, %74, %c0_69 : index
            %76 = subi %c-1, %74 : index
            %77 = select %75, %76, %74 : index
            %78 = divi_signed %77, %c8_68 : index
            %79 = subi %c-1, %78 : index
            %80 = select %75, %79, %78 : index
            %81 = memref.load %6[%73, %80] : memref<4096x512xvector<8xf16>>
            %c8_70 = constant 8 : index
            %c0_71 = constant 0 : index
            %c-1_72 = constant -1 : index
            %82 = cmpi slt, %71, %c0_71 : index
            %83 = subi %c-1_72, %71 : index
            %84 = select %82, %83, %71 : index
            %85 = divi_signed %84, %c8_70 : index
            %86 = subi %c-1_72, %85 : index
            %87 = select %82, %86, %85 : index
            memref.store %81, %19[%70, %87] : memref<64x9xvector<8xf16>, 3>
            scf.yield
          } {mapping = [{bound = #map, map = #map, processor = 0 : i64}]}
          %c0_53 = constant 0 : index
          %c0_54 = constant 0 : index
          %c128_55 = constant 128 : index
          %c64_56 = constant 64 : index
          %c1_57 = constant 1 : index
          %c8_58 = constant 8 : index
          %62 = subi %c64_56, %c0_54 : index
          %c1_59 = constant 1 : index
          %63 = subi %c8_58, %c1_59 : index
          %64 = addi %62, %63 : index
          %65 = divi_signed %64, %c8_58 : index
          %c1_60 = constant 1 : index
          %c0_61 = constant 0 : index
          %c1_62 = constant 1 : index
          %c1_63 = constant 1 : index
          %66 = muli %c1_63, %c128_55 : index
          %67 = muli %66, %65 : index
          scf.parallel (%arg13) = (%c0_61) to (%67) step (%c1_62) {
            %69 = remi_signed %arg13, %65 : index
            %70 = divi_signed %arg13, %65 : index
            %71 = muli %69, %c8_58 : index
            %72 = addi %arg0, %70 : index
            %73 = addi %71, %arg4 : index
            %c8_67 = constant 8 : index
            %c0_68 = constant 0 : index
            %c-1 = constant -1 : index
            %74 = cmpi slt, %73, %c0_68 : index
            %75 = subi %c-1, %73 : index
            %76 = select %74, %75, %73 : index
            %77 = divi_signed %76, %c8_67 : index
            %78 = subi %c-1, %77 : index
            %79 = select %74, %78, %77 : index
            %c8_69 = constant 8 : index
            %80 = addi %79, %c8_69 : index
            %81 = memref.load %5[%72, %80] : memref<4096x512xvector<8xf16>>
            %c8_70 = constant 8 : index
            %c0_71 = constant 0 : index
            %c-1_72 = constant -1 : index
            %82 = cmpi slt, %71, %c0_71 : index
            %83 = subi %c-1_72, %71 : index
            %84 = select %82, %83, %71 : index
            %85 = divi_signed %84, %c8_70 : index
            %86 = subi %c-1_72, %85 : index
            %87 = select %82, %86, %85 : index
            memref.store %81, %23[%70, %87] : memref<128x9xvector<8xf16>, 3>
            scf.yield
          } {mapping = [{bound = #map, map = #map, processor = 0 : i64}]}
          %c0_64 = constant 0 : index
          %c64_65 = constant 64 : index
          %c16_66 = constant 16 : index
          %68:8 = scf.for %arg13 = %c0_64 to %c64_65 step %c16_66 iter_args(%arg14 = %arg5, %arg15 = %arg6, %arg16 = %arg7, %arg17 = %arg8, %arg18 = %arg9, %arg19 = %arg10, %arg20 = %arg11, %arg21 = %arg12) -> (!gpu.mmafragment<8, f32>, !gpu.mmafragment<8, f32>, !gpu.mmafragment<8, f32>, !gpu.mmafragment<8, f32>, !gpu.mmafragment<8, f32>, !gpu.mmafragment<8, f32>, !gpu.mmafragment<8, f32>, !gpu.mmafragment<8, f32>) {
            %69 = gpu.subgroup_mma_load_matrix %21[%arg2, %arg13] {leadDimension = 72 : index, operand = "AOp"} : memref<128x72xf16, 3> -> !gpu.mmafragment<8, vector<2xf16>>
            %70 = gpu.subgroup_mma_load_matrix %18[%arg13, %arg3] {leadDimension = 72 : index, operand = "BOp"} : memref<64x72xf16, 3> -> !gpu.mmafragment<8, vector<2xf16>>
            %71 = gpu.subgroup_mma_compute %69, %70, %arg14 : !gpu.mmafragment<8, vector<2xf16>>, !gpu.mmafragment<8, vector<2xf16>>, !gpu.mmafragment<8, f32> -> !gpu.mmafragment<8, f32>
            %c16_67 = constant 16 : index
            %72 = addi %arg2, %c16_67 : index
            %73 = gpu.subgroup_mma_load_matrix %21[%72, %arg13] {leadDimension = 72 : index, operand = "AOp"} : memref<128x72xf16, 3> -> !gpu.mmafragment<8, vector<2xf16>>
            %74 = gpu.subgroup_mma_load_matrix %18[%arg13, %arg3] {leadDimension = 72 : index, operand = "BOp"} : memref<64x72xf16, 3> -> !gpu.mmafragment<8, vector<2xf16>>
            %75 = gpu.subgroup_mma_compute %73, %74, %arg15 : !gpu.mmafragment<8, vector<2xf16>>, !gpu.mmafragment<8, vector<2xf16>>, !gpu.mmafragment<8, f32> -> !gpu.mmafragment<8, f32>
            %c32_68 = constant 32 : index
            %76 = addi %arg2, %c32_68 : index
            %77 = gpu.subgroup_mma_load_matrix %21[%76, %arg13] {leadDimension = 72 : index, operand = "AOp"} : memref<128x72xf16, 3> -> !gpu.mmafragment<8, vector<2xf16>>
            %78 = gpu.subgroup_mma_load_matrix %18[%arg13, %arg3] {leadDimension = 72 : index, operand = "BOp"} : memref<64x72xf16, 3> -> !gpu.mmafragment<8, vector<2xf16>>
            %79 = gpu.subgroup_mma_compute %77, %78, %arg16 : !gpu.mmafragment<8, vector<2xf16>>, !gpu.mmafragment<8, vector<2xf16>>, !gpu.mmafragment<8, f32> -> !gpu.mmafragment<8, f32>
            %c48_69 = constant 48 : index
            %80 = addi %arg2, %c48_69 : index
            %81 = gpu.subgroup_mma_load_matrix %21[%80, %arg13] {leadDimension = 72 : index, operand = "AOp"} : memref<128x72xf16, 3> -> !gpu.mmafragment<8, vector<2xf16>>
            %82 = gpu.subgroup_mma_load_matrix %18[%arg13, %arg3] {leadDimension = 72 : index, operand = "BOp"} : memref<64x72xf16, 3> -> !gpu.mmafragment<8, vector<2xf16>>
            %83 = gpu.subgroup_mma_compute %81, %82, %arg17 : !gpu.mmafragment<8, vector<2xf16>>, !gpu.mmafragment<8, vector<2xf16>>, !gpu.mmafragment<8, f32> -> !gpu.mmafragment<8, f32>
            %84 = gpu.subgroup_mma_load_matrix %21[%arg2, %arg13] {leadDimension = 72 : index, operand = "AOp"} : memref<128x72xf16, 3> -> !gpu.mmafragment<8, vector<2xf16>>
            %c16_70 = constant 16 : index
            %85 = addi %arg3, %c16_70 : index
            %86 = gpu.subgroup_mma_load_matrix %18[%arg13, %85] {leadDimension = 72 : index, operand = "BOp"} : memref<64x72xf16, 3> -> !gpu.mmafragment<8, vector<2xf16>>
            %87 = gpu.subgroup_mma_compute %84, %86, %arg18 : !gpu.mmafragment<8, vector<2xf16>>, !gpu.mmafragment<8, vector<2xf16>>, !gpu.mmafragment<8, f32> -> !gpu.mmafragment<8, f32>
            %88 = gpu.subgroup_mma_load_matrix %21[%72, %arg13] {leadDimension = 72 : index, operand = "AOp"} : memref<128x72xf16, 3> -> !gpu.mmafragment<8, vector<2xf16>>
            %89 = gpu.subgroup_mma_load_matrix %18[%arg13, %85] {leadDimension = 72 : index, operand = "BOp"} : memref<64x72xf16, 3> -> !gpu.mmafragment<8, vector<2xf16>>
            %90 = gpu.subgroup_mma_compute %88, %89, %arg19 : !gpu.mmafragment<8, vector<2xf16>>, !gpu.mmafragment<8, vector<2xf16>>, !gpu.mmafragment<8, f32> -> !gpu.mmafragment<8, f32>
            %91 = gpu.subgroup_mma_load_matrix %21[%76, %arg13] {leadDimension = 72 : index, operand = "AOp"} : memref<128x72xf16, 3> -> !gpu.mmafragment<8, vector<2xf16>>
            %92 = gpu.subgroup_mma_load_matrix %18[%arg13, %85] {leadDimension = 72 : index, operand = "BOp"} : memref<64x72xf16, 3> -> !gpu.mmafragment<8, vector<2xf16>>
            %93 = gpu.subgroup_mma_compute %91, %92, %arg20 : !gpu.mmafragment<8, vector<2xf16>>, !gpu.mmafragment<8, vector<2xf16>>, !gpu.mmafragment<8, f32> -> !gpu.mmafragment<8, f32>
            %94 = gpu.subgroup_mma_load_matrix %21[%80, %arg13] {leadDimension = 72 : index, operand = "AOp"} : memref<128x72xf16, 3> -> !gpu.mmafragment<8, vector<2xf16>>
            %95 = gpu.subgroup_mma_load_matrix %18[%arg13, %85] {leadDimension = 72 : index, operand = "BOp"} : memref<64x72xf16, 3> -> !gpu.mmafragment<8, vector<2xf16>>
            %96 = gpu.subgroup_mma_compute %94, %95, %arg21 : !gpu.mmafragment<8, vector<2xf16>>, !gpu.mmafragment<8, vector<2xf16>>, !gpu.mmafragment<8, f32> -> !gpu.mmafragment<8, f32>
            scf.yield %71, %75, %79, %83, %87, %90, %93, %96 : !gpu.mmafragment<8, f32>, !gpu.mmafragment<8, f32>, !gpu.mmafragment<8, f32>, !gpu.mmafragment<8, f32>, !gpu.mmafragment<8, f32>, !gpu.mmafragment<8, f32>, !gpu.mmafragment<8, f32>, !gpu.mmafragment<8, f32>
          }
          scf.yield %68#0, %68#1, %68#2, %68#3, %68#4, %68#5, %68#6, %68#7 : !gpu.mmafragment<8, f32>, !gpu.mmafragment<8, f32>, !gpu.mmafragment<8, f32>, !gpu.mmafragment<8, f32>, !gpu.mmafragment<8, f32>, !gpu.mmafragment<8, f32>, !gpu.mmafragment<8, f32>, !gpu.mmafragment<8, f32>
        }
        gpu.barrier
        %c4032_39 = constant 4032 : index
        %c4096_40 = constant 4096 : index
        %c64_41 = constant 64 : index
        %55:8 = scf.for %arg4 = %c4032_39 to %c4096_40 step %c64_41 iter_args(%arg5 = %54#0, %arg6 = %54#1, %arg7 = %54#2, %arg8 = %54#3, %arg9 = %54#4, %arg10 = %54#5, %arg11 = %54#6, %arg12 = %54#7) -> (!gpu.mmafragment<8, f32>, !gpu.mmafragment<8, f32>, !gpu.mmafragment<8, f32>, !gpu.mmafragment<8, f32>, !gpu.mmafragment<8, f32>, !gpu.mmafragment<8, f32>, !gpu.mmafragment<8, f32>, !gpu.mmafragment<8, f32>) {
          %c0_42 = constant 0 : index
          %c64_43 = constant 64 : index
          %c16_44 = constant 16 : index
          %56:8 = scf.for %arg13 = %c0_42 to %c64_43 step %c16_44 iter_args(%arg14 = %arg5, %arg15 = %arg6, %arg16 = %arg7, %arg17 = %arg8, %arg18 = %arg9, %arg19 = %arg10, %arg20 = %arg11, %arg21 = %arg12) -> (!gpu.mmafragment<8, f32>, !gpu.mmafragment<8, f32>, !gpu.mmafragment<8, f32>, !gpu.mmafragment<8, f32>, !gpu.mmafragment<8, f32>, !gpu.mmafragment<8, f32>, !gpu.mmafragment<8, f32>, !gpu.mmafragment<8, f32>) {
            %57 = gpu.subgroup_mma_load_matrix %21[%arg2, %arg13] {leadDimension = 72 : index, operand = "AOp"} : memref<128x72xf16, 3> -> !gpu.mmafragment<8, vector<2xf16>>
            %58 = gpu.subgroup_mma_load_matrix %18[%arg13, %arg3] {leadDimension = 72 : index, operand = "BOp"} : memref<64x72xf16, 3> -> !gpu.mmafragment<8, vector<2xf16>>
            %59 = gpu.subgroup_mma_compute %57, %58, %arg14 : !gpu.mmafragment<8, vector<2xf16>>, !gpu.mmafragment<8, vector<2xf16>>, !gpu.mmafragment<8, f32> -> !gpu.mmafragment<8, f32>
            %c16_45 = constant 16 : index
            %60 = addi %arg2, %c16_45 : index
            %61 = gpu.subgroup_mma_load_matrix %21[%60, %arg13] {leadDimension = 72 : index, operand = "AOp"} : memref<128x72xf16, 3> -> !gpu.mmafragment<8, vector<2xf16>>
            %62 = gpu.subgroup_mma_load_matrix %18[%arg13, %arg3] {leadDimension = 72 : index, operand = "BOp"} : memref<64x72xf16, 3> -> !gpu.mmafragment<8, vector<2xf16>>
            %63 = gpu.subgroup_mma_compute %61, %62, %arg15 : !gpu.mmafragment<8, vector<2xf16>>, !gpu.mmafragment<8, vector<2xf16>>, !gpu.mmafragment<8, f32> -> !gpu.mmafragment<8, f32>
            %c32_46 = constant 32 : index
            %64 = addi %arg2, %c32_46 : index
            %65 = gpu.subgroup_mma_load_matrix %21[%64, %arg13] {leadDimension = 72 : index, operand = "AOp"} : memref<128x72xf16, 3> -> !gpu.mmafragment<8, vector<2xf16>>
            %66 = gpu.subgroup_mma_load_matrix %18[%arg13, %arg3] {leadDimension = 72 : index, operand = "BOp"} : memref<64x72xf16, 3> -> !gpu.mmafragment<8, vector<2xf16>>
            %67 = gpu.subgroup_mma_compute %65, %66, %arg16 : !gpu.mmafragment<8, vector<2xf16>>, !gpu.mmafragment<8, vector<2xf16>>, !gpu.mmafragment<8, f32> -> !gpu.mmafragment<8, f32>
            %c48_47 = constant 48 : index
            %68 = addi %arg2, %c48_47 : index
            %69 = gpu.subgroup_mma_load_matrix %21[%68, %arg13] {leadDimension = 72 : index, operand = "AOp"} : memref<128x72xf16, 3> -> !gpu.mmafragment<8, vector<2xf16>>
            %70 = gpu.subgroup_mma_load_matrix %18[%arg13, %arg3] {leadDimension = 72 : index, operand = "BOp"} : memref<64x72xf16, 3> -> !gpu.mmafragment<8, vector<2xf16>>
            %71 = gpu.subgroup_mma_compute %69, %70, %arg17 : !gpu.mmafragment<8, vector<2xf16>>, !gpu.mmafragment<8, vector<2xf16>>, !gpu.mmafragment<8, f32> -> !gpu.mmafragment<8, f32>
            %72 = gpu.subgroup_mma_load_matrix %21[%arg2, %arg13] {leadDimension = 72 : index, operand = "AOp"} : memref<128x72xf16, 3> -> !gpu.mmafragment<8, vector<2xf16>>
            %c16_48 = constant 16 : index
            %73 = addi %arg3, %c16_48 : index
            %74 = gpu.subgroup_mma_load_matrix %18[%arg13, %73] {leadDimension = 72 : index, operand = "BOp"} : memref<64x72xf16, 3> -> !gpu.mmafragment<8, vector<2xf16>>
            %75 = gpu.subgroup_mma_compute %72, %74, %arg18 : !gpu.mmafragment<8, vector<2xf16>>, !gpu.mmafragment<8, vector<2xf16>>, !gpu.mmafragment<8, f32> -> !gpu.mmafragment<8, f32>
            %76 = gpu.subgroup_mma_load_matrix %21[%60, %arg13] {leadDimension = 72 : index, operand = "AOp"} : memref<128x72xf16, 3> -> !gpu.mmafragment<8, vector<2xf16>>
            %77 = gpu.subgroup_mma_load_matrix %18[%arg13, %73] {leadDimension = 72 : index, operand = "BOp"} : memref<64x72xf16, 3> -> !gpu.mmafragment<8, vector<2xf16>>
            %78 = gpu.subgroup_mma_compute %76, %77, %arg19 : !gpu.mmafragment<8, vector<2xf16>>, !gpu.mmafragment<8, vector<2xf16>>, !gpu.mmafragment<8, f32> -> !gpu.mmafragment<8, f32>
            %79 = gpu.subgroup_mma_load_matrix %21[%64, %arg13] {leadDimension = 72 : index, operand = "AOp"} : memref<128x72xf16, 3> -> !gpu.mmafragment<8, vector<2xf16>>
            %80 = gpu.subgroup_mma_load_matrix %18[%arg13, %73] {leadDimension = 72 : index, operand = "BOp"} : memref<64x72xf16, 3> -> !gpu.mmafragment<8, vector<2xf16>>
            %81 = gpu.subgroup_mma_compute %79, %80, %arg20 : !gpu.mmafragment<8, vector<2xf16>>, !gpu.mmafragment<8, vector<2xf16>>, !gpu.mmafragment<8, f32> -> !gpu.mmafragment<8, f32>
            %82 = gpu.subgroup_mma_load_matrix %21[%68, %arg13] {leadDimension = 72 : index, operand = "AOp"} : memref<128x72xf16, 3> -> !gpu.mmafragment<8, vector<2xf16>>
            %83 = gpu.subgroup_mma_load_matrix %18[%arg13, %73] {leadDimension = 72 : index, operand = "BOp"} : memref<64x72xf16, 3> -> !gpu.mmafragment<8, vector<2xf16>>
            %84 = gpu.subgroup_mma_compute %82, %83, %arg21 : !gpu.mmafragment<8, vector<2xf16>>, !gpu.mmafragment<8, vector<2xf16>>, !gpu.mmafragment<8, f32> -> !gpu.mmafragment<8, f32>
            scf.yield %59, %63, %67, %71, %75, %78, %81, %84 : !gpu.mmafragment<8, f32>, !gpu.mmafragment<8, f32>, !gpu.mmafragment<8, f32>, !gpu.mmafragment<8, f32>, !gpu.mmafragment<8, f32>, !gpu.mmafragment<8, f32>, !gpu.mmafragment<8, f32>, !gpu.mmafragment<8, f32>
          }
          scf.yield %56#0, %56#1, %56#2, %56#3, %56#4, %56#5, %56#6, %56#7 : !gpu.mmafragment<8, f32>, !gpu.mmafragment<8, f32>, !gpu.mmafragment<8, f32>, !gpu.mmafragment<8, f32>, !gpu.mmafragment<8, f32>, !gpu.mmafragment<8, f32>, !gpu.mmafragment<8, f32>, !gpu.mmafragment<8, f32>
        }
        gpu.subgroup_mma_store_matrix %55#0, %memref_2[%24, %25] {leadDimension = 4096 : index} : !gpu.mmafragment<8, f32>, memref<4096x4096xf32>
        gpu.subgroup_mma_store_matrix %55#1, %memref_2[%28, %25] {leadDimension = 4096 : index} : !gpu.mmafragment<8, f32>, memref<4096x4096xf32>
        gpu.subgroup_mma_store_matrix %55#2, %memref_2[%31, %25] {leadDimension = 4096 : index} : !gpu.mmafragment<8, f32>, memref<4096x4096xf32>
        gpu.subgroup_mma_store_matrix %55#3, %memref_2[%34, %25] {leadDimension = 4096 : index} : !gpu.mmafragment<8, f32>, memref<4096x4096xf32>
        gpu.subgroup_mma_store_matrix %55#4, %memref_2[%24, %37] {leadDimension = 4096 : index} : !gpu.mmafragment<8, f32>, memref<4096x4096xf32>
        gpu.subgroup_mma_store_matrix %55#5, %memref_2[%28, %37] {leadDimension = 4096 : index} : !gpu.mmafragment<8, f32>, memref<4096x4096xf32>
        gpu.subgroup_mma_store_matrix %55#6, %memref_2[%31, %37] {leadDimension = 4096 : index} : !gpu.mmafragment<8, f32>, memref<4096x4096xf32>
        gpu.subgroup_mma_store_matrix %55#7, %memref_2[%34, %37] {leadDimension = 4096 : index} : !gpu.mmafragment<8, f32>, memref<4096x4096xf32>
        scf.yield
      } {mapping = [{bound = #map, map = #map, processor = 7 : i64}, {bound = #map, map = #map, processor = 6 : i64}]}
      scf.yield
    } {mapping = [{bound = #map, map = #map, processor = 1 : i64}, {bound = #map, map = #map, processor = 0 : i64}]}
    %12 = call @rtclock() : () -> f64
    %13 = gpu.wait async
    %14 = gpu.memcpy async [%13] %2, %memref_2 : memref<4096x4096xf32>, memref<4096x4096xf32>
    gpu.wait [%14]
    %15 = subf %12, %11 : f64
    %16 = sitofp %c137438953472_i64 : i64 to f64
    %17 = divf %16, %15 : f64
    call @print_flops(%17) : (f64) -> ()
    return
  }
  func private @print_memref_f32(memref<*xf32>)
  func private @print_flops(f64)
  func private @rtclock() -> f64
}
