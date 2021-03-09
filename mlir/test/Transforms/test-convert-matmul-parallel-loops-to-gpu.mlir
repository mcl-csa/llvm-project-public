// RUN: mlir-opt %s --canonicalize --test-convert-matmul-parallel-loops-to-gpu | FileCheck %s

#map = affine_map<(d0) -> (d0)>

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
        %43 = subi %33, %arg4 : index
        %c1_0 = constant 1 : index
        %44 = subi %c1, %c1_0 : index
        %45 = addi %43, %44 : index
        %46 = divi_signed %45, %c1 : index
        %c0_1 = constant 0 : index
        %47 = subi %36, %arg1 : index
        %c1_2 = constant 1 : index
        %48 = subi %c1, %c1_2 : index
        %49 = addi %47, %48 : index
        %50 = divi_signed %49, %c1 : index
        %c0_3 = constant 0 : index
        %c0_4 = constant 0 : index
        %c1_5 = constant 1 : index
        %c1_6 = constant 1 : index
        %51 = muli %c1_6, %46 : index
        %52 = muli %51, %50 : index
        scf.parallel (%arg9) = (%c0_4) to (%52) step (%c1_5) {
          %67 = remi_signed %arg9, %50 : index
          %68 = divi_signed %arg9, %50 : index
          %69 = addi %67, %arg1 : index
          %70 = addi %68, %arg4 : index
          %71 = load %1[%70, %69] : memref<1024x1024xf16>
          %72 = muli %arg4, %c-1 : index
          %73 = addi %72, %70 : index
          %74 = muli %arg1, %c-1 : index
          %75 = addi %74, %69 : index
          store %71, %4[%73, %75] : memref<32x64xf16, 3>
          scf.yield
        } {mapping = [{bound = #map, map = #map, processor = 0 : i64}]}
        %53 = subi %39, %arg0 : index
        %c1_7 = constant 1 : index
        %54 = subi %c1, %c1_7 : index
        %55 = addi %53, %54 : index
        %56 = divi_signed %55, %c1 : index
        %c0_8 = constant 0 : index
        %57 = subi %42, %arg4 : index
        %c1_9 = constant 1 : index
        %58 = subi %c1, %c1_9 : index
        %59 = addi %57, %58 : index
        %60 = divi_signed %59, %c1 : index
        %c0_10 = constant 0 : index
        %c0_11 = constant 0 : index
        %c1_12 = constant 1 : index
        %c1_13 = constant 1 : index
        %61 = muli %c1_13, %56 : index
        %62 = muli %61, %60 : index
        scf.parallel (%arg9) = (%c0_11) to (%62) step (%c1_12) {
          %67 = remi_signed %arg9, %60 : index
          %68 = divi_signed %arg9, %60 : index
          %69 = addi %67, %arg4 : index
          %70 = addi %68, %arg0 : index
          %71 = load %0[%70, %69] : memref<1024x1024xf16>
          %72 = muli %arg0, %c-1 : index
          %73 = addi %72, %70 : index
          %74 = muli %arg4, %c-1 : index
          %75 = addi %74, %69 : index
          store %71, %5[%73, %75] : memref<64x32xf16, 3>
          scf.yield
        } {mapping = [{bound = #map, map = #map, processor = 0 : i64}]}
        %63 = addi %arg4, %c32 : index
        %64 = cmpi slt, %63, %c1024 : index
        %65 = select %64, %63, %c1024 : index
        %66:4 = scf.for %arg9 = %arg4 to %65 step %c16 iter_args(%arg10 = %24, %arg11 = %25, %arg12 = %26, %arg13 = %27) -> (!gpu.mmafragment<4, vector<2xf16>>, !gpu.mmafragment<4, vector<2xf16>>, !gpu.mmafragment<4, vector<2xf16>>, !gpu.mmafragment<4, vector<2xf16>>) {
          %67 = muli %arg4, %c-1 : index
          %68 = addi %67, %arg9 : index
          %69 = addi %18, %arg2 : index
          %70 = addi %19, %c16 : index
          %71 = gpu.subgroup_mma_load_matrix %5[%69, %68] {leadDimension = 32 : index, operand = "AOp"} : memref<64x32xf16, 3> -> !gpu.mmafragment<8, vector<2xf16>>
          %72 = gpu.subgroup_mma_load_matrix %5[%70, %68] {leadDimension = 32 : index, operand = "AOp"} : memref<64x32xf16, 3> -> !gpu.mmafragment<8, vector<2xf16>>
          %73 = addi %21, %arg3 : index
          %74 = addi %22, %c16 : index
          %75 = gpu.subgroup_mma_load_matrix %4[%68, %73] {leadDimension = 64 : index, operand = "BOp"} : memref<32x64xf16, 3> -> !gpu.mmafragment<8, vector<2xf16>>
          %76 = gpu.subgroup_mma_compute %71, %75, %24 : !gpu.mmafragment<8, vector<2xf16>>, !gpu.mmafragment<8, vector<2xf16>>, !gpu.mmafragment<4, vector<2xf16>> -> !gpu.mmafragment<4, vector<2xf16>>
          %77 = gpu.subgroup_mma_compute %72, %75, %25 : !gpu.mmafragment<8, vector<2xf16>>, !gpu.mmafragment<8, vector<2xf16>>, !gpu.mmafragment<4, vector<2xf16>> -> !gpu.mmafragment<4, vector<2xf16>>
          %78 = gpu.subgroup_mma_load_matrix %4[%68, %74] {leadDimension = 64 : index, operand = "BOp"} : memref<32x64xf16, 3> -> !gpu.mmafragment<8, vector<2xf16>>
          %79 = gpu.subgroup_mma_compute %71, %78, %26 : !gpu.mmafragment<8, vector<2xf16>>, !gpu.mmafragment<8, vector<2xf16>>, !gpu.mmafragment<4, vector<2xf16>> -> !gpu.mmafragment<4, vector<2xf16>>
          %80 = gpu.subgroup_mma_compute %72, %78, %27 : !gpu.mmafragment<8, vector<2xf16>>, !gpu.mmafragment<8, vector<2xf16>>, !gpu.mmafragment<4, vector<2xf16>> -> !gpu.mmafragment<4, vector<2xf16>>
          scf.yield %76, %77, %79, %80 : !gpu.mmafragment<4, vector<2xf16>>, !gpu.mmafragment<4, vector<2xf16>>, !gpu.mmafragment<4, vector<2xf16>>, !gpu.mmafragment<4, vector<2xf16>>
        }
        scf.yield %66#0, %66#1, %66#2, %66#3 : !gpu.mmafragment<4, vector<2xf16>>, !gpu.mmafragment<4, vector<2xf16>>, !gpu.mmafragment<4, vector<2xf16>>, !gpu.mmafragment<4, vector<2xf16>>
      }
      %29 = addi %arg0, %c16 : index
      %30 = addi %arg1, %c16 : index
      gpu.subgroup_mma_store_matrix %28#0, %2[%arg0, %arg1] {leadDimension = 1024 : index} : !gpu.mmafragment<4, vector<2xf16>>, memref<1024x1024xf16>
      gpu.subgroup_mma_store_matrix %28#1, %2[%29, %arg1] {leadDimension = 1024 : index} : !gpu.mmafragment<4, vector<2xf16>>, memref<1024x1024xf16>
      gpu.subgroup_mma_store_matrix %28#2, %2[%arg0, %30] {leadDimension = 1024 : index} : !gpu.mmafragment<4, vector<2xf16>>, memref<1024x1024xf16>
      gpu.subgroup_mma_store_matrix %28#3, %2[%29, %30] {leadDimension = 1024 : index} : !gpu.mmafragment<4, vector<2xf16>>, memref<1024x1024xf16>
      scf.yield
    } {mapping = [{bound = #map, map = #map, processor = 7 : i64}, {bound = #map, map = #map, processor = 6 : i64}]}
    scf.yield
  } {mapping = [{bound = #map, map = #map, processor = 1 : i64}, {bound = #map, map = #map, processor = 0 : i64}]}
  return
}

// CHECK-LABEL: func @matmul() {
// CHECK:   %3 = addi %c1024, %c64 : index
// CHECK-NEXT:   %4 = subi %3, %c1_4 : index
// CHECK-NEXT:   %5 = divi_unsigned %4, %c64 : index
// CHECK-NEXT:   %6 = addi %c1024, %c64 : index
// CHECK-NEXT:   %7 = subi %6, %c1_4 : index
// CHECK-NEXT:   %8 = divi_unsigned %7, %c64 : index
// CHECK-NEXT:   %9 = divi_unsigned %c64, %c32 : index
// CHECK-NEXT:   %10 = divi_unsigned %c64, %c32 : index
// CHECK-NEXT:   %11 = muli %9, %10 : index
// CHECK-NEXT:   %c32_5 = constant 32 : index
// CHECK-NEXT:   %12 = muli %11, %c32_5 : index
// CHECK-NEXT:   gpu.launch blocks(%arg0, %arg1, %arg2) in (%arg6 = %8, %arg7 = %5, %arg8 = %c1_0) threads(%arg3, %arg4, %arg5) in (%arg9 = %12, %arg10 = %c1_2, %arg11 = %c1_3) {
// CHECK-NEXT:     %13 = muli %12, %c1_2 : index
// CHECK-NEXT:     %14 = muli %arg5, %13 : index
// CHECK-NEXT:     %15 = muli %arg4, %12 : index
// CHECK-NEXT:     %16 = addi %14, %15 : index
// CHECK-NEXT:     %17 = addi %16, %arg3 : index
// CHECK-NEXT:     %c32_6 = constant 32 : index
// CHECK-NEXT:     %18 = divi_unsigned %17, %c32_6 : index
// CHECK-NEXT:     %19 = divi_unsigned %12, %c32_6 : index
// CHECK-NEXT:     %20 = muli %arg1, %c64 : index
// CHECK-NEXT:     %21 = addi %c0, %20 : index
// CHECK-NEXT:     %22 = muli %arg0, %c64 : index
// CHECK-NEXT:     %23 = addi %c0, %22 : index
// CHECK-NEXT:     %24 = alloca() : memref<64x64xf16, 3>
// CHECK-NEXT:     %25 = alloca() : memref<32x64xf16, 3>
// CHECK-NEXT:     %26 = alloca() : memref<64x32xf16, 3>
// CHECK-NEXT:     %27 = addi %21, %c64 : index
// CHECK-NEXT:     %28 = cmpi slt, %27, %c1024 : index
// CHECK-NEXT:     %29 = select %28, %27, %c1024 : index
// CHECK-NEXT:     %30 = addi %23, %c64 : index
// CHECK-NEXT:     %31 = cmpi slt, %30, %c1024 : index
// CHECK-NEXT:     %32 = select %31, %30, %c1024 : index
// CHECK-NEXT:     %33 = divi_unsigned %c64, %c32 : index
// CHECK-NEXT:     %34 = cmpi ule, %19, %33 : index
// CHECK-NEXT:     %35 = select %34, %19, %33 : index
// CHECK-NEXT:     %36 = divi_unsigned %19, %35 : index
// CHECK-NEXT:     %37 = remi_unsigned %18, %35 : index
// CHECK-NEXT:     %38 = divi_unsigned %18, %35 : index
// CHECK-NEXT:     %39 = muli %38, %c32 : index
// CHECK-NEXT:     %40 = muli %c32, %36 : index
// CHECK-NEXT:     scf.for %arg12 = %39 to %c64 step %40 {
// CHECK-NEXT:       %41 = muli %37, %c32 : index
// CHECK-NEXT:       %42 = muli %c32, %35 : index
// CHECK-NEXT:       scf.for %arg13 = %41 to %c64 step %42 {
// CHECK-NEXT:         %43 = muli %21, %c-1 : index
// CHECK-NEXT:         %44 = addi %43, %arg12 : index
// CHECK-NEXT:         %45 = addi %44, %c16 : index
// CHECK-NEXT:         %46 = muli %23, %c-1 : index
// CHECK-NEXT:         %47 = addi %46, %arg13 : index
// CHECK-NEXT:         %48 = addi %47, %c16 : index
// CHECK-NEXT:         %49 = gpu.subgroup_mma_load_matrix %24[%44, %47] {leadDimension = 64 : index, operand = "COp"} : memref<64x64xf16, 3> -> !gpu.mmafragment<4, vector<2xf16>>
// CHECK-NEXT:         %50 = gpu.subgroup_mma_load_matrix %24[%45, %47] {leadDimension = 64 : index, operand = "COp"} : memref<64x64xf16, 3> -> !gpu.mmafragment<4, vector<2xf16>>
// CHECK-NEXT:         %51 = gpu.subgroup_mma_load_matrix %24[%44, %48] {leadDimension = 64 : index, operand = "COp"} : memref<64x64xf16, 3> -> !gpu.mmafragment<4, vector<2xf16>>
// CHECK-NEXT:         %52 = gpu.subgroup_mma_load_matrix %24[%45, %48] {leadDimension = 64 : index, operand = "COp"} : memref<64x64xf16, 3> -> !gpu.mmafragment<4, vector<2xf16>>
// CHECK-NEXT:         %53:4 = scf.for %arg14 = %c0 to %c1024 step %c32 iter_args(%arg15 = %49, %arg16 = %50, %arg17 = %51, %arg18 = %52) -> (!gpu.mmafragment<4, vector<2xf16>>, !gpu.mmafragment<4, vector<2xf16>>, !gpu.mmafragment<4, vector<2xf16>>, !gpu.mmafragment<4, vector<2xf16>>) {
// CHECK-NEXT:           %56 = addi %arg14, %c32 : index
// CHECK-NEXT:           %57 = cmpi slt, %56, %c1024 : index
// CHECK-NEXT:           %58 = select %57, %56, %c1024 : index
// CHECK-NEXT:           %59 = addi %23, %c64 : index
// CHECK-NEXT:           %60 = cmpi slt, %59, %c1024 : index
// CHECK-NEXT:           %61 = select %60, %59, %c1024 : index
// CHECK-NEXT:           %62 = addi %21, %c64 : index
// CHECK-NEXT:           %63 = cmpi slt, %62, %c1024 : index
// CHECK-NEXT:           %64 = select %63, %62, %c1024 : index
// CHECK-NEXT:           %65 = addi %arg14, %c32 : index
// CHECK-NEXT:           %66 = cmpi slt, %65, %c1024 : index
// CHECK-NEXT:           %67 = select %66, %65, %c1024 : index
// CHECK-NEXT:           %68 = subi %58, %arg14 : index
// CHECK-NEXT:           %69 = subi %61, %23 : index
// CHECK-NEXT:           %70 = muli %68, %69 : index
// CHECK-NEXT:           scf.for %arg19 = %17 to %70 step %12 {
// CHECK-NEXT:             %78 = remi_signed %arg19, %69 : index
// CHECK-NEXT:             %79 = divi_signed %arg19, %69 : index
// CHECK-NEXT:             %80 = addi %78, %23 : index
// CHECK-NEXT:             %81 = addi %79, %arg14 : index
// CHECK-NEXT:             %82 = load %1[%81, %80] : memref<1024x1024xf16>
// CHECK-NEXT:             %83 = muli %arg14, %c-1 : index
// CHECK-NEXT:             %84 = addi %83, %81 : index
// CHECK-NEXT:             %85 = muli %23, %c-1 : index
// CHECK-NEXT:             %86 = addi %85, %80 : index
// CHECK-NEXT:             store %82, %25[%84, %86] : memref<32x64xf16, 3>
// CHECK-NEXT:           }
// CHECK-NEXT:           %71 = subi %64, %21 : index
// CHECK-NEXT:           %72 = subi %67, %arg14 : index
// CHECK-NEXT:           %73 = muli %71, %72 : index
// CHECK-NEXT:           scf.for %arg19 = %17 to %73 step %12 {
// CHECK-NEXT:             %78 = remi_signed %arg19, %72 : index
// CHECK-NEXT:             %79 = divi_signed %arg19, %72 : index
// CHECK-NEXT:             %80 = addi %78, %arg14 : index
// CHECK-NEXT:             %81 = addi %79, %21 : index
// CHECK-NEXT:             %82 = load %0[%81, %80] : memref<1024x1024xf16>
// CHECK-NEXT:             %83 = muli %21, %c-1 : index
// CHECK-NEXT:             %84 = addi %83, %81 : index
// CHECK-NEXT:             %85 = muli %arg14, %c-1 : index
// CHECK-NEXT:             %86 = addi %85, %80 : index
// CHECK-NEXT:             store %82, %26[%84, %86] : memref<64x32xf16, 3>
// CHECK-NEXT:           }
// CHECK-NEXT:           %74 = addi %arg14, %c32 : index
// CHECK-NEXT:           %75 = cmpi slt, %74, %c1024 : index
// CHECK-NEXT:           %76 = select %75, %74, %c1024 : index
// CHECK-NEXT:           %77:4 = scf.for %arg19 = %arg14 to %76 step %c16 iter_args(%arg20 = %49, %arg21 = %50, %arg22 = %51, %arg23 = %52) -> (!gpu.mmafragment<4, vector<2xf16>>, !gpu.mmafragment<4, vector<2xf16>>, !gpu.mmafragment<4, vector<2xf16>>, !gpu.mmafragment<4, vector<2xf16>>) {
// CHECK-NEXT:             %78 = muli %arg14, %c-1 : index
// CHECK-NEXT:             %79 = addi %78, %arg19 : index
// CHECK-NEXT:             %80 = addi %43, %arg12 : index
// CHECK-NEXT:             %81 = addi %44, %c16 : index
// CHECK-NEXT:             %82 = gpu.subgroup_mma_load_matrix %26[%80, %79] {leadDimension = 32 : index, operand = "AOp"} : memref<64x32xf16, 3> -> !gpu.mmafragment<8, vector<2xf16>>
// CHECK-NEXT:             %83 = gpu.subgroup_mma_load_matrix %26[%81, %79] {leadDimension = 32 : index, operand = "AOp"} : memref<64x32xf16, 3> -> !gpu.mmafragment<8, vector<2xf16>>
// CHECK-NEXT:             %84 = addi %46, %arg13 : index
// CHECK-NEXT:             %85 = addi %47, %c16 : index
// CHECK-NEXT:             %86 = gpu.subgroup_mma_load_matrix %25[%79, %84] {leadDimension = 64 : index, operand = "BOp"} : memref<32x64xf16, 3> -> !gpu.mmafragment<8, vector<2xf16>>
// CHECK-NEXT:             %87 = gpu.subgroup_mma_compute %82, %86, %49 : !gpu.mmafragment<8, vector<2xf16>>, !gpu.mmafragment<8, vector<2xf16>>, !gpu.mmafragment<4, vector<2xf16>> -> !gpu.mmafragment<4, vector<2xf16>>
// CHECK-NEXT:             %88 = gpu.subgroup_mma_compute %83, %86, %50 : !gpu.mmafragment<8, vector<2xf16>>, !gpu.mmafragment<8, vector<2xf16>>, !gpu.mmafragment<4, vector<2xf16>> -> !gpu.mmafragment<4, vector<2xf16>>
// CHECK-NEXT:             %89 = gpu.subgroup_mma_load_matrix %25[%79, %85] {leadDimension = 64 : index, operand = "BOp"} : memref<32x64xf16, 3> -> !gpu.mmafragment<8, vector<2xf16>>
// CHECK-NEXT:             %90 = gpu.subgroup_mma_compute %82, %89, %51 : !gpu.mmafragment<8, vector<2xf16>>, !gpu.mmafragment<8, vector<2xf16>>, !gpu.mmafragment<4, vector<2xf16>> -> !gpu.mmafragment<4, vector<2xf16>>
// CHECK-NEXT:             %91 = gpu.subgroup_mma_compute %83, %89, %52 : !gpu.mmafragment<8, vector<2xf16>>, !gpu.mmafragment<8, vector<2xf16>>, !gpu.mmafragment<4, vector<2xf16>> -> !gpu.mmafragment<4, vector<2xf16>>
// CHECK-NEXT:             scf.yield %87, %88, %90, %91 : !gpu.mmafragment<4, vector<2xf16>>, !gpu.mmafragment<4, vector<2xf16>>, !gpu.mmafragment<4, vector<2xf16>>, !gpu.mmafragment<4, vector<2xf16>>
// CHECK-NEXT:           }
// CHECK-NEXT:           scf.yield %77#0, %77#1, %77#2, %77#3 : !gpu.mmafragment<4, vector<2xf16>>, !gpu.mmafragment<4, vector<2xf16>>, !gpu.mmafragment<4, vector<2xf16>>, !gpu.mmafragment<4, vector<2xf16>>
// CHECK-NEXT:         }
// CHECK-NEXT:         %54 = addi %21, %c16 : index
// CHECK-NEXT:         %55 = addi %23, %c16 : index
// CHECK-NEXT:         gpu.subgroup_mma_store_matrix %53#0, %2[%21, %23] {leadDimension = 1024 : index} : !gpu.mmafragment<4, vector<2xf16>>, memref<1024x1024xf16>
// CHECK-NEXT:         gpu.subgroup_mma_store_matrix %53#1, %2[%54, %23] {leadDimension = 1024 : index} : !gpu.mmafragment<4, vector<2xf16>>, memref<1024x1024xf16>
// CHECK-NEXT:         gpu.subgroup_mma_store_matrix %53#2, %2[%21, %55] {leadDimension = 1024 : index} : !gpu.mmafragment<4, vector<2xf16>>, memref<1024x1024xf16>
// CHECK-NEXT:         gpu.subgroup_mma_store_matrix %53#3, %2[%54, %55] {leadDimension = 1024 : index} : !gpu.mmafragment<4, vector<2xf16>>, memref<1024x1024xf16>
// CHECK-NEXT:       }
// CHECK-NEXT:     }
// CHECK-NEXT:     gpu.terminator
// CHECK-NEXT:   }
// CHECK-NEXT:   return
// CHECK-NEXT: }
