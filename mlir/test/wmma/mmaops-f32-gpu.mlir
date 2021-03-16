// RUN: mlir-opt %s --gpu-kernel-outlining --canonicalize

module  {
  global_memref @asmem : memref<64x64xf16, 3>
  global_memref @bsmem : memref<64x64xf16, 3>
  func @main() {
    %c1024 = constant 1024 : index
    %c-1 = constant -1 : index
    %c0 = constant 0 : index
    %c1 = constant 1 : index
    %c16 = constant 16 : index
    %c64 = constant 64 : index
    %c128 = constant 128 : index
    %c32 = constant 32 : index
    %c2 = constant 2 : index
    %0 = alloc() : memref<1024x1024xf16>
    %1 = alloc() : memref<1024x1024xf16>
    %2 = alloc() : memref<1024x1024xf16>
    %3 = memref_cast %0 : memref<1024x1024xf16> to memref<*xf16>
    gpu.host_register %3 : memref<*xf16>
    %4 = memref_cast %1 : memref<1024x1024xf16> to memref<*xf16>
    gpu.host_register %4 : memref<*xf16>
    %5 = memref_cast %2 : memref<1024x1024xf16> to memref<*xf16>
    gpu.host_register %5 : memref<*xf16>
    gpu.launch blocks(%arg0, %arg1, %arg2) in (%arg6 = %c16, %arg7 = %c16, %arg8 = %c1) threads(%arg3, %arg4, %arg5) in (%arg9 = %c128, %arg10 = %c1, %arg11 = %c1) {
      %6 = muli %arg5, %c128 : index
      %7 = muli %arg4, %c128 : index
      %8 = addi %6, %7 : index
      %9 = addi %8, %arg3 : index
      %10 = divi_unsigned %9, %c32 : index
      %11 = muli %arg1, %c64 : index
      %12 = muli %arg0, %c64 : index
      %13 = get_global_memref @asmem : memref<64x64xf16, 3>
      %14 = get_global_memref @bsmem : memref<64x64xf16, 3>
      %15 = remi_unsigned %10, %c2 : index
      %16 = divi_unsigned %10, %c2 : index
      %17 = muli %16, %c32 : index
      scf.for %arg12 = %17 to %c64 step %c64 {
        %18 = muli %15, %c32 : index
        scf.for %arg13 = %18 to %c64 step %c64 {
          %19 = addi %11, %arg12 : index
          %20 = addi %12, %arg13 : index
          %21 = gpu.subgroup_mma_load_matrix %2[%19, %20] {leadDimension = 1024 : index, operand = "COp"} : memref<1024x1024xf16> -> !gpu.mmafragment<8, f32>
          %22 = addi %11, %arg12 : index
          %23 = addi %22, %c16 : index
          %24 = addi %12, %arg13 : index
          %25 = gpu.subgroup_mma_load_matrix %2[%23, %24] {leadDimension = 1024 : index, operand = "COp"} : memref<1024x1024xf16> -> !gpu.mmafragment<8, f32>
          %26 = addi %11, %arg12 : index
          %27 = addi %12, %arg13 : index
          %28 = addi %27, %c16 : index
          %29 = gpu.subgroup_mma_load_matrix %2[%26, %28] {leadDimension = 1024 : index, operand = "COp"} : memref<1024x1024xf16> -> !gpu.mmafragment<8, f32>
          %30 = addi %11, %arg12 : index
          %31 = addi %30, %c16 : index
          %32 = addi %12, %arg13 : index
          %33 = addi %32, %c16 : index
          %34 = gpu.subgroup_mma_load_matrix %2[%31, %33] {leadDimension = 1024 : index, operand = "COp"} : memref<1024x1024xf16> -> !gpu.mmafragment<8, f32>
          %35:4 = scf.for %arg14 = %c0 to %c1024 step %c64 iter_args(%arg15 = %21, %arg16 = %25, %arg17 = %29, %arg18 = %34) -> (!gpu.mmafragment<8, f32>, !gpu.mmafragment<8, f32>, !gpu.mmafragment<8, f32>, !gpu.mmafragment<8, f32>) {
            %36 = addi %arg14, %c64 : index
            %37 = addi %12, %c64 : index
            %38 = subi %36, %arg14 : index
            %39 = subi %37, %12 : index
            %40 = muli %38, %39 : index
            scf.for %arg19 = %9 to %40 step %c128 {
              %47 = remi_signed %arg19, %39 : index
              %48 = divi_signed %arg19, %39 : index
              %49 = addi %47, %12 : index
              %50 = addi %48, %arg14 : index
              %51 = load %1[%50, %49] : memref<1024x1024xf16>
              %52 = muli %arg14, %c-1 : index
              %53 = addi %52, %50 : index
              %54 = muli %12, %c-1 : index
              %55 = addi %54, %49 : index
              store %51, %13[%53, %55] : memref<64x64xf16, 3>
            }
            %41 = addi %11, %c64 : index
            %42 = addi %arg14, %c64 : index
            %43 = subi %41, %11 : index
            %44 = subi %42, %arg14 : index
            %45 = muli %43, %44 : index
            scf.for %arg19 = %9 to %45 step %c128 {
              %47 = remi_signed %arg19, %44 : index
              %48 = divi_signed %arg19, %44 : index
              %49 = addi %47, %arg14 : index
              %50 = addi %48, %11 : index
              %51 = load %0[%50, %49] : memref<1024x1024xf16>
              %52 = muli %11, %c-1 : index
              %53 = addi %52, %50 : index
              %54 = muli %arg14, %c-1 : index
              %55 = addi %54, %49 : index
              store %51, %14[%53, %55] : memref<64x64xf16, 3>
            }
            gpu.barrier
            %46:4 = scf.for %arg19 = %c0 to %c64 step %c16 iter_args(%arg20 = %arg15, %arg21 = %arg16, %arg22 = %arg17, %arg23 = %arg18) -> (!gpu.mmafragment<8, f32>, !gpu.mmafragment<8, f32>, !gpu.mmafragment<8, f32>, !gpu.mmafragment<8, f32>) {
              %47 = gpu.subgroup_mma_load_matrix %14[%arg12, %arg19] {leadDimension = 64 : index, operand = "AOp"} : memref<64x64xf16, 3> -> !gpu.mmafragment<8, vector<2xf16>>
              %48 = gpu.subgroup_mma_load_matrix %13[%arg19, %arg13] {leadDimension = 64 : index, operand = "BOp"} : memref<64x64xf16, 3> -> !gpu.mmafragment<8, vector<2xf16>>
              %49 = gpu.subgroup_mma_compute %47, %48, %arg20 : !gpu.mmafragment<8, vector<2xf16>>, !gpu.mmafragment<8, vector<2xf16>>, !gpu.mmafragment<8, f32> -> !gpu.mmafragment<8, f32>
              %50 = addi %arg12, %c16 : index
              %51 = gpu.subgroup_mma_load_matrix %14[%50, %arg19] {leadDimension = 64 : index, operand = "AOp"} : memref<64x64xf16, 3> -> !gpu.mmafragment<8, vector<2xf16>>
              %52 = gpu.subgroup_mma_load_matrix %13[%arg19, %arg13] {leadDimension = 64 : index, operand = "BOp"} : memref<64x64xf16, 3> -> !gpu.mmafragment<8, vector<2xf16>>
              %53 = gpu.subgroup_mma_compute %51, %52, %arg21 : !gpu.mmafragment<8, vector<2xf16>>, !gpu.mmafragment<8, vector<2xf16>>, !gpu.mmafragment<8, f32> -> !gpu.mmafragment<8, f32>
              %54 = gpu.subgroup_mma_load_matrix %14[%arg12, %arg19] {leadDimension = 64 : index, operand = "AOp"} : memref<64x64xf16, 3> -> !gpu.mmafragment<8, vector<2xf16>>
              %55 = addi %arg13, %c16 : index
              %56 = gpu.subgroup_mma_load_matrix %13[%arg19, %55] {leadDimension = 64 : index, operand = "BOp"} : memref<64x64xf16, 3> -> !gpu.mmafragment<8, vector<2xf16>>
              %57 = gpu.subgroup_mma_compute %54, %56, %arg22 : !gpu.mmafragment<8, vector<2xf16>>, !gpu.mmafragment<8, vector<2xf16>>, !gpu.mmafragment<8, f32> -> !gpu.mmafragment<8, f32>
              %58 = addi %arg12, %c16 : index
              %59 = gpu.subgroup_mma_load_matrix %14[%58, %arg19] {leadDimension = 64 : index, operand = "AOp"} : memref<64x64xf16, 3> -> !gpu.mmafragment<8, vector<2xf16>>
              %60 = addi %arg13, %c16 : index
              %61 = gpu.subgroup_mma_load_matrix %13[%arg19, %60] {leadDimension = 64 : index, operand = "BOp"} : memref<64x64xf16, 3> -> !gpu.mmafragment<8, vector<2xf16>>
              %62 = gpu.subgroup_mma_compute %59, %61, %arg23 : !gpu.mmafragment<8, vector<2xf16>>, !gpu.mmafragment<8, vector<2xf16>>, !gpu.mmafragment<8, f32> -> !gpu.mmafragment<8, f32>
              scf.yield %49, %53, %57, %62 : !gpu.mmafragment<8, f32>, !gpu.mmafragment<8, f32>, !gpu.mmafragment<8, f32>, !gpu.mmafragment<8, f32>
            }
            gpu.barrier
            scf.yield %46#0, %46#1, %46#2, %46#3 : !gpu.mmafragment<8, f32>, !gpu.mmafragment<8, f32>, !gpu.mmafragment<8, f32>, !gpu.mmafragment<8, f32>
          }
          gpu.subgroup_mma_store_matrix %35#0, %2[%19, %20] {leadDimension = 1024 : index} : !gpu.mmafragment<8, f32>, memref<1024x1024xf16>
          gpu.subgroup_mma_store_matrix %35#1, %2[%23, %24] {leadDimension = 1024 : index} : !gpu.mmafragment<8, f32>, memref<1024x1024xf16>
          gpu.subgroup_mma_store_matrix %35#2, %2[%26, %28] {leadDimension = 1024 : index} : !gpu.mmafragment<8, f32>, memref<1024x1024xf16>
          gpu.subgroup_mma_store_matrix %35#3, %2[%31, %33] {leadDimension = 1024 : index} : !gpu.mmafragment<8, f32>, memref<1024x1024xf16>
        }
      }
      gpu.terminator
    }
    return
  }
}

