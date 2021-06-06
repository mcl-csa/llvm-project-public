// RUN: mlir-opt %s --convert-scf-to-std -pass-pipeline='gpu.module(strip-debuginfo,convert-gpu-to-nvvm{index-bitwidth=32},gpu-to-cubin{chip=sm_75 max-reg-per-thread=255 cu-jit-opt-level=4})' -gpu-to-llvm | mlir-cpu-runner -O3 --shared-libs=%linalg_test_lib_dir/libmlir_cuda_runtime%shlibext --shared-libs=%linalg_test_lib_dir/libmlir_runner_utils%shlibext --shared-libs=%linalg_test_lib_dir/libmlir_c_runner_utils%shlibext --entry-point-result=void

module attributes {gpu.container_module}  {
  func @main() {
    %c2 = constant 2 : index
    %c256 = constant 256 : index
    %c16 = constant 16 : index
    %cst = constant 0.000000e+00 : f32
    %c1 = constant 1 : index
    %c1024 = constant 1024 : index
    %c0 = constant 0 : index
    %c128 = constant 128 : index
    %c32 = constant 32 : index
    %c64 = constant 64 : index
    %c48 = constant 48 : index
    %c8 = constant 8 : index
    %c-1 = constant -1 : index
    %c4 = constant 4 : index
    %c992 = constant 992 : index
    %0 = memref.alloc() : memref<1024x1024xf16>
    %1 = memref.alloc() : memref<1024x1024xf16>
    %2 = memref.alloc() : memref<1024x1024xf32>
    scf.for %arg0 = %c0 to %c1024 step %c1 {
      scf.for %arg1 = %c0 to %c1024 step %c1 {
        %11 = remi_signed %arg0, %c16 : index
        %12 = remi_signed %arg1, %c16 : index
        %13 = addi %11, %12 : index
        %14 = remi_signed %13, %c16 : index
        %15 = index_cast %14 : index to i16
        %16 = sitofp %15 : i16 to f16
        memref.store %16, %0[%arg0, %arg1] : memref<1024x1024xf16>
      }
    }
    scf.for %arg0 = %c0 to %c1024 step %c1 {
      scf.for %arg1 = %c0 to %c1024 step %c1 {
        %11 = remi_signed %arg0, %c16 : index
        %12 = remi_signed %arg1, %c16 : index
        %13 = addi %11, %12 : index
        %14 = remi_signed %13, %c16 : index
        %15 = index_cast %14 : index to i16
        %16 = sitofp %15 : i16 to f16
        memref.store %16, %1[%arg0, %arg1] : memref<1024x1024xf16>
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
    gpu.launch_func  @main_kernel::@main_kernel blocks in (%c8, %c8, %c1) threads in (%c256, %c1, %c1) args(%memref_2 : memref<1024x1024xf32>, %5 : memref<1024x128xvector<8xf16>>, %4 : memref<1024x128xvector<8xf16>>)
    %9 = gpu.wait async 
    %10 = gpu.memcpy async [%9] %2, %memref_2 : memref<1024x1024xf32>, memref<1024x1024xf32>
    gpu.wait [%10]
    return
  }
  gpu.module @main_kernel {
    gpu.func @main_kernel(%arg0: memref<1024x1024xf32>, %arg1: memref<1024x128xvector<8xf16>>, %arg2: memref<1024x128xvector<8xf16>>) workgroup(%arg3 : memref<32x136xf16, 3>, %arg4 : memref<128x40xf16, 3>) kernel {
      %0 = "gpu.block_id"() {dimension = "x"} : () -> index
      %1 = "gpu.block_id"() {dimension = "y"} : () -> index
      %2 = "gpu.block_id"() {dimension = "z"} : () -> index
      %3 = "gpu.thread_id"() {dimension = "x"} : () -> index
      %4 = "gpu.thread_id"() {dimension = "y"} : () -> index
      %5 = "gpu.thread_id"() {dimension = "z"} : () -> index
      %6 = "gpu.grid_dim"() {dimension = "x"} : () -> index
      %7 = "gpu.grid_dim"() {dimension = "y"} : () -> index
      %8 = "gpu.grid_dim"() {dimension = "z"} : () -> index
      %9 = "gpu.block_dim"() {dimension = "x"} : () -> index
      %10 = "gpu.block_dim"() {dimension = "y"} : () -> index
      %11 = "gpu.block_dim"() {dimension = "z"} : () -> index
      br ^bb1
    ^bb1:  // pred: ^bb0
      %c256 = constant 256 : index
      %c32 = constant 32 : index
      %c128 = constant 128 : index
      %c2 = constant 2 : index
      %c64 = constant 64 : index
      %c16 = constant 16 : index
      %c48 = constant 48 : index
      %c8 = constant 8 : index
      %c0 = constant 0 : index
      %c-1 = constant -1 : index
      %c1 = constant 1 : index
      %c4 = constant 4 : index
      %c992 = constant 992 : index
      %12 = muli %5, %c256 : index
      %13 = muli %4, %c256 : index
      %14 = addi %12, %13 : index
      %15 = addi %14, %3 : index
      %16 = divi_unsigned %15, %c32 : index
      %17 = muli %1, %c128 : index
      %18 = muli %0, %c128 : index
      %19 = memref_vector_cast %arg3 : memref<32x136xf16, 3> to memref<32x17xvector<8xf16>, 3>
      %20 = memref_vector_cast %arg4 : memref<128x40xf16, 3> to memref<128x5xvector<8xf16>, 3>
      %21 = remi_unsigned %16, %c2 : index
      %22 = divi_unsigned %16, %c2 : index
      %23 = muli %22, %c32 : index
      scf.for %arg5 = %23 to %c128 step %c128 {
        %24 = muli %21, %c64 : index
        scf.for %arg6 = %24 to %c128 step %c128 {
          %25 = addi %17, %arg5 : index
          %26 = addi %18, %arg6 : index
          %27 = gpu.subgroup_mma_load_matrix %arg0[%25, %26] {leadDimension = 1024 : index} : memref<1024x1024xf32> -> !gpu.mma_matrix<16x16xf32, "COp">
          %28 = addi %25, %c16 : index
          %29 = gpu.subgroup_mma_load_matrix %arg0[%28, %26] {leadDimension = 1024 : index} : memref<1024x1024xf32> -> !gpu.mma_matrix<16x16xf32, "COp">
          %30 = addi %26, %c16 : index
          %31 = gpu.subgroup_mma_load_matrix %arg0[%25, %30] {leadDimension = 1024 : index} : memref<1024x1024xf32> -> !gpu.mma_matrix<16x16xf32, "COp">
          %32 = gpu.subgroup_mma_load_matrix %arg0[%28, %30] {leadDimension = 1024 : index} : memref<1024x1024xf32> -> !gpu.mma_matrix<16x16xf32, "COp">
          %33 = addi %26, %c32 : index
          %34 = gpu.subgroup_mma_load_matrix %arg0[%25, %33] {leadDimension = 1024 : index} : memref<1024x1024xf32> -> !gpu.mma_matrix<16x16xf32, "COp">
          %35 = gpu.subgroup_mma_load_matrix %arg0[%28, %33] {leadDimension = 1024 : index} : memref<1024x1024xf32> -> !gpu.mma_matrix<16x16xf32, "COp">
          %36 = addi %26, %c48 : index
          %37 = gpu.subgroup_mma_load_matrix %arg0[%25, %36] {leadDimension = 1024 : index} : memref<1024x1024xf32> -> !gpu.mma_matrix<16x16xf32, "COp">
          %38 = gpu.subgroup_mma_load_matrix %arg0[%28, %36] {leadDimension = 1024 : index} : memref<1024x1024xf32> -> !gpu.mma_matrix<16x16xf32, "COp">
          scf.for %arg7 = %c0 to %c2 step %c1 {
            %41 = muli %arg7, %c256 : index
            %42 = addi %15, %41 : index
            %43 = remi_signed %42, %c16 : index
            %44 = divi_signed %42, %c16 : index
            %45 = muli %43, %c8 : index
            %46 = addi %18, %45 : index
            %47 = cmpi slt, %46, %c0 : index
            %48 = subi %c-1, %46 : index
            %49 = select %47, %48, %46 : index
            %50 = divi_signed %49, %c8 : index
            %51 = subi %c-1, %50 : index
            %52 = select %47, %51, %50 : index
            %53 = memref.load %arg1[%44, %52] : memref<1024x128xvector<8xf16>>
            %54 = cmpi slt, %45, %c0 : index
            %55 = subi %c-1, %45 : index
            %56 = select %54, %55, %45 : index
            %57 = divi_signed %56, %c8 : index
            %58 = subi %c-1, %57 : index
            %59 = select %54, %58, %57 : index
            memref.store %53, %19[%44, %59] : memref<32x17xvector<8xf16>, 3>
          } {isCopyLoopNest = true}
          scf.for %arg7 = %c0 to %c2 step %c1 {
            %41 = muli %arg7, %c256 : index
            %42 = addi %15, %41 : index
            %43 = remi_signed %42, %c4 : index
            %44 = divi_signed %42, %c4 : index
            %45 = muli %43, %c8 : index
            %46 = addi %17, %44 : index
            %47 = cmpi slt, %45, %c0 : index
            %48 = subi %c-1, %45 : index
            %49 = select %47, %48, %45 : index
            %50 = divi_signed %49, %c8 : index
            %51 = subi %c-1, %50 : index
            %52 = select %47, %51, %50 : index
            %53 = memref.load %arg2[%46, %52] : memref<1024x128xvector<8xf16>>
            memref.store %53, %20[%44, %52] : memref<128x5xvector<8xf16>, 3>
          } {isCopyLoopNest = true}
          gpu.barrier
          %39:8 = scf.for %arg7 = %c0 to %c992 step %c32 iter_args(%arg8 = %27, %arg9 = %29, %arg10 = %31, %arg11 = %32, %arg12 = %34, %arg13 = %35, %arg14 = %37, %arg15 = %38) -> (!gpu.mma_matrix<16x16xf32, "COp">, !gpu.mma_matrix<16x16xf32, "COp">, !gpu.mma_matrix<16x16xf32, "COp">, !gpu.mma_matrix<16x16xf32, "COp">, !gpu.mma_matrix<16x16xf32, "COp">, !gpu.mma_matrix<16x16xf32, "COp">, !gpu.mma_matrix<16x16xf32, "COp">, !gpu.mma_matrix<16x16xf32, "COp">) {
            gpu.barrier
            %41 = remi_signed %15, %c16 : index
            %42 = divi_signed %15, %c16 : index
            %43 = muli %41, %c8 : index
            %44 = addi %arg7, %42 : index
            %45 = addi %44, %c32 : index
            %46 = addi %18, %43 : index
            %47 = cmpi slt, %46, %c0 : index
            %48 = subi %c-1, %46 : index
            %49 = select %47, %48, %46 : index
            %50 = divi_signed %49, %c8 : index
            %51 = subi %c-1, %50 : index
            %52 = select %47, %51, %50 : index
            %53 = memref.load %arg1[%45, %52] : memref<1024x128xvector<8xf16>>
            %54 = cmpi slt, %43, %c0 : index
            %55 = subi %c-1, %43 : index
            %56 = select %54, %55, %43 : index
            %57 = divi_signed %56, %c8 : index
            %58 = subi %c-1, %57 : index
            %59 = select %54, %58, %57 : index
            %60 = addi %15, %c256 : index
            %61 = remi_signed %60, %c16 : index
            %62 = divi_signed %60, %c16 : index
            %63 = muli %61, %c8 : index
            %64 = addi %arg7, %62 : index
            %65 = addi %64, %c32 : index
            %66 = addi %18, %63 : index
            %67 = cmpi slt, %66, %c0 : index
            %68 = subi %c-1, %66 : index
            %69 = select %67, %68, %66 : index
            %70 = divi_signed %69, %c8 : index
            %71 = subi %c-1, %70 : index
            %72 = select %67, %71, %70 : index
            %73 = memref.load %arg1[%65, %72] : memref<1024x128xvector<8xf16>>
            %74 = cmpi slt, %63, %c0 : index
            %75 = subi %c-1, %63 : index
            %76 = select %74, %75, %63 : index
            %77 = divi_signed %76, %c8 : index
            %78 = subi %c-1, %77 : index
            %79 = select %74, %78, %77 : index
            %80 = remi_signed %15, %c4 : index
            %81 = divi_signed %15, %c4 : index
            %82 = muli %80, %c8 : index
            %83 = addi %17, %81 : index
            %84 = addi %arg7, %82 : index
            %85 = cmpi slt, %84, %c0 : index
            %86 = subi %c-1, %84 : index
            %87 = select %85, %86, %84 : index
            %88 = divi_signed %87, %c8 : index
            %89 = subi %c-1, %88 : index
            %90 = select %85, %89, %88 : index
            %91 = addi %90, %c4 : index
            %92 = memref.load %arg2[%83, %91] : memref<1024x128xvector<8xf16>>
            %93 = cmpi slt, %82, %c0 : index
            %94 = subi %c-1, %82 : index
            %95 = select %93, %94, %82 : index
            %96 = divi_signed %95, %c8 : index
            %97 = subi %c-1, %96 : index
            %98 = select %93, %97, %96 : index
            %99 = addi %15, %c256 : index
            %100 = remi_signed %99, %c4 : index
            %101 = divi_signed %99, %c4 : index
            %102 = muli %100, %c8 : index
            %103 = addi %17, %101 : index
            %104 = addi %arg7, %102 : index
            %105 = cmpi slt, %104, %c0 : index
            %106 = subi %c-1, %104 : index
            %107 = select %105, %106, %104 : index
            %108 = divi_signed %107, %c8 : index
            %109 = subi %c-1, %108 : index
            %110 = select %105, %109, %108 : index
            %111 = addi %110, %c4 : index
            %112 = memref.load %arg2[%103, %111] : memref<1024x128xvector<8xf16>>
            %113 = cmpi slt, %102, %c0 : index
            %114 = subi %c-1, %102 : index
            %115 = select %113, %114, %102 : index
            %116 = divi_signed %115, %c8 : index
            %117 = subi %c-1, %116 : index
            %118 = select %113, %117, %116 : index
            %119:8 = scf.for %arg16 = %c0 to %c32 step %c16 iter_args(%arg17 = %arg8, %arg18 = %arg9, %arg19 = %arg10, %arg20 = %arg11, %arg21 = %arg12, %arg22 = %arg13, %arg23 = %arg14, %arg24 = %arg15) -> (!gpu.mma_matrix<16x16xf32, "COp">, !gpu.mma_matrix<16x16xf32, "COp">, !gpu.mma_matrix<16x16xf32, "COp">, !gpu.mma_matrix<16x16xf32, "COp">, !gpu.mma_matrix<16x16xf32, "COp">, !gpu.mma_matrix<16x16xf32, "COp">, !gpu.mma_matrix<16x16xf32, "COp">, !gpu.mma_matrix<16x16xf32, "COp">) {
              %120 = gpu.subgroup_mma_load_matrix %arg4[%arg5, %arg16] {leadDimension = 40 : index} : memref<128x40xf16, 3> -> !gpu.mma_matrix<16x16xf16, "AOp">
              %121 = gpu.subgroup_mma_load_matrix %arg3[%arg16, %arg6] {leadDimension = 136 : index} : memref<32x136xf16, 3> -> !gpu.mma_matrix<16x16xf16, "BOp">
              %122 = gpu.subgroup_mma_compute %120, %121, %arg17 : !gpu.mma_matrix<16x16xf16, "AOp">, !gpu.mma_matrix<16x16xf16, "BOp"> -> !gpu.mma_matrix<16x16xf32, "COp">
              %123 = addi %arg5, %c16 : index
              %124 = gpu.subgroup_mma_load_matrix %arg4[%123, %arg16] {leadDimension = 40 : index} : memref<128x40xf16, 3> -> !gpu.mma_matrix<16x16xf16, "AOp">
              %125 = gpu.subgroup_mma_load_matrix %arg3[%arg16, %arg6] {leadDimension = 136 : index} : memref<32x136xf16, 3> -> !gpu.mma_matrix<16x16xf16, "BOp">
              %126 = gpu.subgroup_mma_compute %124, %125, %arg18 : !gpu.mma_matrix<16x16xf16, "AOp">, !gpu.mma_matrix<16x16xf16, "BOp"> -> !gpu.mma_matrix<16x16xf32, "COp">
              %127 = gpu.subgroup_mma_load_matrix %arg4[%arg5, %arg16] {leadDimension = 40 : index} : memref<128x40xf16, 3> -> !gpu.mma_matrix<16x16xf16, "AOp">
              %128 = addi %arg6, %c16 : index
              %129 = gpu.subgroup_mma_load_matrix %arg3[%arg16, %128] {leadDimension = 136 : index} : memref<32x136xf16, 3> -> !gpu.mma_matrix<16x16xf16, "BOp">
              %130 = gpu.subgroup_mma_compute %127, %129, %arg19 : !gpu.mma_matrix<16x16xf16, "AOp">, !gpu.mma_matrix<16x16xf16, "BOp"> -> !gpu.mma_matrix<16x16xf32, "COp">
              %131 = gpu.subgroup_mma_load_matrix %arg4[%123, %arg16] {leadDimension = 40 : index} : memref<128x40xf16, 3> -> !gpu.mma_matrix<16x16xf16, "AOp">
              %132 = gpu.subgroup_mma_load_matrix %arg3[%arg16, %128] {leadDimension = 136 : index} : memref<32x136xf16, 3> -> !gpu.mma_matrix<16x16xf16, "BOp">
              %133 = gpu.subgroup_mma_compute %131, %132, %arg20 : !gpu.mma_matrix<16x16xf16, "AOp">, !gpu.mma_matrix<16x16xf16, "BOp"> -> !gpu.mma_matrix<16x16xf32, "COp">
              %134 = gpu.subgroup_mma_load_matrix %arg4[%arg5, %arg16] {leadDimension = 40 : index} : memref<128x40xf16, 3> -> !gpu.mma_matrix<16x16xf16, "AOp">
              %135 = addi %arg6, %c32 : index
              %136 = gpu.subgroup_mma_load_matrix %arg3[%arg16, %135] {leadDimension = 136 : index} : memref<32x136xf16, 3> -> !gpu.mma_matrix<16x16xf16, "BOp">
              %137 = gpu.subgroup_mma_compute %134, %136, %arg21 : !gpu.mma_matrix<16x16xf16, "AOp">, !gpu.mma_matrix<16x16xf16, "BOp"> -> !gpu.mma_matrix<16x16xf32, "COp">
              %138 = gpu.subgroup_mma_load_matrix %arg4[%123, %arg16] {leadDimension = 40 : index} : memref<128x40xf16, 3> -> !gpu.mma_matrix<16x16xf16, "AOp">
              %139 = gpu.subgroup_mma_load_matrix %arg3[%arg16, %135] {leadDimension = 136 : index} : memref<32x136xf16, 3> -> !gpu.mma_matrix<16x16xf16, "BOp">
              %140 = gpu.subgroup_mma_compute %138, %139, %arg22 : !gpu.mma_matrix<16x16xf16, "AOp">, !gpu.mma_matrix<16x16xf16, "BOp"> -> !gpu.mma_matrix<16x16xf32, "COp">
              %141 = gpu.subgroup_mma_load_matrix %arg4[%arg5, %arg16] {leadDimension = 40 : index} : memref<128x40xf16, 3> -> !gpu.mma_matrix<16x16xf16, "AOp">
              %142 = addi %arg6, %c48 : index
              %143 = gpu.subgroup_mma_load_matrix %arg3[%arg16, %142] {leadDimension = 136 : index} : memref<32x136xf16, 3> -> !gpu.mma_matrix<16x16xf16, "BOp">
              %144 = gpu.subgroup_mma_compute %141, %143, %arg23 : !gpu.mma_matrix<16x16xf16, "AOp">, !gpu.mma_matrix<16x16xf16, "BOp"> -> !gpu.mma_matrix<16x16xf32, "COp">
              %145 = gpu.subgroup_mma_load_matrix %arg4[%123, %arg16] {leadDimension = 40 : index} : memref<128x40xf16, 3> -> !gpu.mma_matrix<16x16xf16, "AOp">
              %146 = gpu.subgroup_mma_load_matrix %arg3[%arg16, %142] {leadDimension = 136 : index} : memref<32x136xf16, 3> -> !gpu.mma_matrix<16x16xf16, "BOp">
              %147 = gpu.subgroup_mma_compute %145, %146, %arg24 : !gpu.mma_matrix<16x16xf16, "AOp">, !gpu.mma_matrix<16x16xf16, "BOp"> -> !gpu.mma_matrix<16x16xf32, "COp">
              scf.yield %122, %126, %130, %133, %137, %140, %144, %147 : !gpu.mma_matrix<16x16xf32, "COp">, !gpu.mma_matrix<16x16xf32, "COp">, !gpu.mma_matrix<16x16xf32, "COp">, !gpu.mma_matrix<16x16xf32, "COp">, !gpu.mma_matrix<16x16xf32, "COp">, !gpu.mma_matrix<16x16xf32, "COp">, !gpu.mma_matrix<16x16xf32, "COp">, !gpu.mma_matrix<16x16xf32, "COp">
            }
            gpu.barrier
            memref.store %53, %19[%42, %59] : memref<32x17xvector<8xf16>, 3>
            memref.store %73, %19[%62, %79] : memref<32x17xvector<8xf16>, 3>
            memref.store %92, %20[%81, %98] : memref<128x5xvector<8xf16>, 3>
            memref.store %112, %20[%101, %118] : memref<128x5xvector<8xf16>, 3>
            scf.yield %119#0, %119#1, %119#2, %119#3, %119#4, %119#5, %119#6, %119#7 : !gpu.mma_matrix<16x16xf32, "COp">, !gpu.mma_matrix<16x16xf32, "COp">, !gpu.mma_matrix<16x16xf32, "COp">, !gpu.mma_matrix<16x16xf32, "COp">, !gpu.mma_matrix<16x16xf32, "COp">, !gpu.mma_matrix<16x16xf32, "COp">, !gpu.mma_matrix<16x16xf32, "COp">, !gpu.mma_matrix<16x16xf32, "COp">
          }
          gpu.barrier
          %40:8 = scf.for %arg7 = %c0 to %c32 step %c16 iter_args(%arg8 = %39#0, %arg9 = %39#1, %arg10 = %39#2, %arg11 = %39#3, %arg12 = %39#4, %arg13 = %39#5, %arg14 = %39#6, %arg15 = %39#7) -> (!gpu.mma_matrix<16x16xf32, "COp">, !gpu.mma_matrix<16x16xf32, "COp">, !gpu.mma_matrix<16x16xf32, "COp">, !gpu.mma_matrix<16x16xf32, "COp">, !gpu.mma_matrix<16x16xf32, "COp">, !gpu.mma_matrix<16x16xf32, "COp">, !gpu.mma_matrix<16x16xf32, "COp">, !gpu.mma_matrix<16x16xf32, "COp">) {
            %41 = gpu.subgroup_mma_load_matrix %arg4[%arg5, %arg7] {leadDimension = 40 : index} : memref<128x40xf16, 3> -> !gpu.mma_matrix<16x16xf16, "AOp">
            %42 = gpu.subgroup_mma_load_matrix %arg3[%arg7, %arg6] {leadDimension = 136 : index} : memref<32x136xf16, 3> -> !gpu.mma_matrix<16x16xf16, "BOp">
            %43 = gpu.subgroup_mma_compute %41, %42, %arg8 : !gpu.mma_matrix<16x16xf16, "AOp">, !gpu.mma_matrix<16x16xf16, "BOp"> -> !gpu.mma_matrix<16x16xf32, "COp">
            %44 = addi %arg5, %c16 : index
            %45 = gpu.subgroup_mma_load_matrix %arg4[%44, %arg7] {leadDimension = 40 : index} : memref<128x40xf16, 3> -> !gpu.mma_matrix<16x16xf16, "AOp">
            %46 = gpu.subgroup_mma_load_matrix %arg3[%arg7, %arg6] {leadDimension = 136 : index} : memref<32x136xf16, 3> -> !gpu.mma_matrix<16x16xf16, "BOp">
            %47 = gpu.subgroup_mma_compute %45, %46, %arg9 : !gpu.mma_matrix<16x16xf16, "AOp">, !gpu.mma_matrix<16x16xf16, "BOp"> -> !gpu.mma_matrix<16x16xf32, "COp">
            %48 = gpu.subgroup_mma_load_matrix %arg4[%arg5, %arg7] {leadDimension = 40 : index} : memref<128x40xf16, 3> -> !gpu.mma_matrix<16x16xf16, "AOp">
            %49 = addi %arg6, %c16 : index
            %50 = gpu.subgroup_mma_load_matrix %arg3[%arg7, %49] {leadDimension = 136 : index} : memref<32x136xf16, 3> -> !gpu.mma_matrix<16x16xf16, "BOp">
            %51 = gpu.subgroup_mma_compute %48, %50, %arg10 : !gpu.mma_matrix<16x16xf16, "AOp">, !gpu.mma_matrix<16x16xf16, "BOp"> -> !gpu.mma_matrix<16x16xf32, "COp">
            %52 = gpu.subgroup_mma_load_matrix %arg4[%44, %arg7] {leadDimension = 40 : index} : memref<128x40xf16, 3> -> !gpu.mma_matrix<16x16xf16, "AOp">
            %53 = gpu.subgroup_mma_load_matrix %arg3[%arg7, %49] {leadDimension = 136 : index} : memref<32x136xf16, 3> -> !gpu.mma_matrix<16x16xf16, "BOp">
            %54 = gpu.subgroup_mma_compute %52, %53, %arg11 : !gpu.mma_matrix<16x16xf16, "AOp">, !gpu.mma_matrix<16x16xf16, "BOp"> -> !gpu.mma_matrix<16x16xf32, "COp">
            %55 = gpu.subgroup_mma_load_matrix %arg4[%arg5, %arg7] {leadDimension = 40 : index} : memref<128x40xf16, 3> -> !gpu.mma_matrix<16x16xf16, "AOp">
            %56 = addi %arg6, %c32 : index
            %57 = gpu.subgroup_mma_load_matrix %arg3[%arg7, %56] {leadDimension = 136 : index} : memref<32x136xf16, 3> -> !gpu.mma_matrix<16x16xf16, "BOp">
            %58 = gpu.subgroup_mma_compute %55, %57, %arg12 : !gpu.mma_matrix<16x16xf16, "AOp">, !gpu.mma_matrix<16x16xf16, "BOp"> -> !gpu.mma_matrix<16x16xf32, "COp">
            %59 = gpu.subgroup_mma_load_matrix %arg4[%44, %arg7] {leadDimension = 40 : index} : memref<128x40xf16, 3> -> !gpu.mma_matrix<16x16xf16, "AOp">
            %60 = gpu.subgroup_mma_load_matrix %arg3[%arg7, %56] {leadDimension = 136 : index} : memref<32x136xf16, 3> -> !gpu.mma_matrix<16x16xf16, "BOp">
            %61 = gpu.subgroup_mma_compute %59, %60, %arg13 : !gpu.mma_matrix<16x16xf16, "AOp">, !gpu.mma_matrix<16x16xf16, "BOp"> -> !gpu.mma_matrix<16x16xf32, "COp">
            %62 = gpu.subgroup_mma_load_matrix %arg4[%arg5, %arg7] {leadDimension = 40 : index} : memref<128x40xf16, 3> -> !gpu.mma_matrix<16x16xf16, "AOp">
            %63 = addi %arg6, %c48 : index
            %64 = gpu.subgroup_mma_load_matrix %arg3[%arg7, %63] {leadDimension = 136 : index} : memref<32x136xf16, 3> -> !gpu.mma_matrix<16x16xf16, "BOp">
            %65 = gpu.subgroup_mma_compute %62, %64, %arg14 : !gpu.mma_matrix<16x16xf16, "AOp">, !gpu.mma_matrix<16x16xf16, "BOp"> -> !gpu.mma_matrix<16x16xf32, "COp">
            %66 = gpu.subgroup_mma_load_matrix %arg4[%44, %arg7] {leadDimension = 40 : index} : memref<128x40xf16, 3> -> !gpu.mma_matrix<16x16xf16, "AOp">
            %67 = gpu.subgroup_mma_load_matrix %arg3[%arg7, %63] {leadDimension = 136 : index} : memref<32x136xf16, 3> -> !gpu.mma_matrix<16x16xf16, "BOp">
            %68 = gpu.subgroup_mma_compute %66, %67, %arg15 : !gpu.mma_matrix<16x16xf16, "AOp">, !gpu.mma_matrix<16x16xf16, "BOp"> -> !gpu.mma_matrix<16x16xf32, "COp">
            scf.yield %43, %47, %51, %54, %58, %61, %65, %68 : !gpu.mma_matrix<16x16xf32, "COp">, !gpu.mma_matrix<16x16xf32, "COp">, !gpu.mma_matrix<16x16xf32, "COp">, !gpu.mma_matrix<16x16xf32, "COp">, !gpu.mma_matrix<16x16xf32, "COp">, !gpu.mma_matrix<16x16xf32, "COp">, !gpu.mma_matrix<16x16xf32, "COp">, !gpu.mma_matrix<16x16xf32, "COp">
          }
          gpu.subgroup_mma_store_matrix %40#0, %arg0[%25, %26] {leadDimension = 1024 : index} : !gpu.mma_matrix<16x16xf32, "COp">, memref<1024x1024xf32>
          gpu.subgroup_mma_store_matrix %40#1, %arg0[%28, %26] {leadDimension = 1024 : index} : !gpu.mma_matrix<16x16xf32, "COp">, memref<1024x1024xf32>
          gpu.subgroup_mma_store_matrix %40#2, %arg0[%25, %30] {leadDimension = 1024 : index} : !gpu.mma_matrix<16x16xf32, "COp">, memref<1024x1024xf32>
          gpu.subgroup_mma_store_matrix %40#3, %arg0[%28, %30] {leadDimension = 1024 : index} : !gpu.mma_matrix<16x16xf32, "COp">, memref<1024x1024xf32>
          gpu.subgroup_mma_store_matrix %40#4, %arg0[%25, %33] {leadDimension = 1024 : index} : !gpu.mma_matrix<16x16xf32, "COp">, memref<1024x1024xf32>
          gpu.subgroup_mma_store_matrix %40#5, %arg0[%28, %33] {leadDimension = 1024 : index} : !gpu.mma_matrix<16x16xf32, "COp">, memref<1024x1024xf32>
          gpu.subgroup_mma_store_matrix %40#6, %arg0[%25, %36] {leadDimension = 1024 : index} : !gpu.mma_matrix<16x16xf32, "COp">, memref<1024x1024xf32>
          gpu.subgroup_mma_store_matrix %40#7, %arg0[%28, %36] {leadDimension = 1024 : index} : !gpu.mma_matrix<16x16xf32, "COp">, memref<1024x1024xf32>
        }
      }
      gpu.return
    }
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
