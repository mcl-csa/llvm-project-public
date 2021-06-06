// RUN: mlir-opt %s --convert-scf-to-std -pass-pipeline='gpu.module(strip-debuginfo,convert-gpu-to-nvvm{index-bitwidth=32},gpu-to-cubin{chip=sm_75 max-reg-per-thread=255 cu-jit-opt-level=4})' -gpu-to-llvm | mlir-cpu-runner -O3 --shared-libs=%linalg_test_lib_dir/libmlir_cuda_runtime%shlibext --shared-libs=%linalg_test_lib_dir/libmlir_runner_utils%shlibext --shared-libs=%linalg_test_lib_dir/libmlir_c_runner_utils%shlibext --entry-point-result=void

module attributes {gpu.container_module}  {
  func @main() {
    %c768 = constant 768 : index
    %c512 = constant 512 : index
    %c4 = constant 4 : index
    %c2 = constant 2 : index
    %c256 = constant 256 : index
    %c16 = constant 16 : index
    %cst = constant 0.000000e+00 : f16
    %c1 = constant 1 : index
    %c1024 = constant 1024 : index
    %c0 = constant 0 : index
    %c128 = constant 128 : index
    %c32 = constant 32 : index
    %c64 = constant 64 : index
    %c48 = constant 48 : index
    %c8 = constant 8 : index
    %c-1 = constant -1 : index
    %c960 = constant 960 : index
    %0 = memref.alloc() : memref<1024x1024xf16>
    %1 = memref.alloc() : memref<1024x1024xf16>
    %2 = memref.alloc() : memref<1024x1024xf16>
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
        memref.store %cst, %2[%arg0, %arg1] : memref<1024x1024xf16>
      }
    }
    %3 = gpu.wait async 
    %memref, %asyncToken = gpu.alloc async [%3] () : memref<1024x1024xf16>
    %4 = memref_vector_cast %memref : memref<1024x1024xf16> to memref<1024x128xvector<8xf16>>
    %memref_0, %asyncToken_1 = gpu.alloc async [%3] () : memref<1024x1024xf16>
    %5 = memref_vector_cast %memref_0 : memref<1024x1024xf16> to memref<1024x128xvector<8xf16>>
    %memref_2, %asyncToken_3 = gpu.alloc async [%3] () : memref<1024x1024xf16>
    %6 = gpu.memcpy async [%3] %memref, %0 : memref<1024x1024xf16>, memref<1024x1024xf16>
    %7 = gpu.memcpy async [%3] %memref_0, %1 : memref<1024x1024xf16>, memref<1024x1024xf16>
    %8 = gpu.memcpy async [%3] %memref_2, %2 : memref<1024x1024xf16>, memref<1024x1024xf16>
    gpu.wait [%3]
    gpu.launch_func  @main_kernel::@main_kernel blocks in (%c8, %c8, %c1) threads in (%c256, %c1, %c1) args(%memref_2 : memref<1024x1024xf16>, %5 : memref<1024x128xvector<8xf16>>, %4 : memref<1024x128xvector<8xf16>>)
    %9 = gpu.wait async 
    %10 = gpu.memcpy async [%9] %2, %memref_2 : memref<1024x1024xf16>, memref<1024x1024xf16>
    gpu.wait [%10]
    return
  }
  gpu.module @main_kernel {
    gpu.func @main_kernel(%arg0: memref<1024x1024xf16>, %arg1: memref<1024x128xvector<8xf16>>, %arg2: memref<1024x128xvector<8xf16>>) workgroup(%arg3 : memref<64x136xf16, 3>, %arg4 : memref<128x72xf16, 3>) kernel {
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
      %c4 = constant 4 : index
      %c1 = constant 1 : index
      %c512 = constant 512 : index
      %c768 = constant 768 : index
      %c960 = constant 960 : index
      %12 = muli %5, %c256 : index
      %13 = muli %4, %c256 : index
      %14 = addi %12, %13 : index
      %15 = addi %14, %3 : index
      %16 = divi_unsigned %15, %c32 : index
      %17 = muli %1, %c128 : index
      %18 = muli %0, %c128 : index
      %19 = memref_vector_cast %arg3 : memref<64x136xf16, 3> to memref<64x17xvector<8xf16>, 3>
      %20 = memref_vector_cast %arg4 : memref<128x72xf16, 3> to memref<128x9xvector<8xf16>, 3>
      %21 = remi_unsigned %16, %c2 : index
      %22 = divi_unsigned %16, %c2 : index
      %23 = muli %22, %c32 : index
      scf.for %arg5 = %23 to %c128 step %c128 {
        %24 = muli %21, %c64 : index
        scf.for %arg6 = %24 to %c128 step %c128 {
          %25 = addi %17, %arg5 : index
          %26 = addi %18, %arg6 : index
          %27 = gpu.subgroup_mma_load_matrix %arg0[%25, %26] {leadDimension = 1024 : index} : memref<1024x1024xf16> -> !gpu.mma_matrix<16x16xf16, "COp">
          %28 = addi %25, %c16 : index
          %29 = gpu.subgroup_mma_load_matrix %arg0[%28, %26] {leadDimension = 1024 : index} : memref<1024x1024xf16> -> !gpu.mma_matrix<16x16xf16, "COp">
          %30 = addi %26, %c16 : index
          %31 = gpu.subgroup_mma_load_matrix %arg0[%25, %30] {leadDimension = 1024 : index} : memref<1024x1024xf16> -> !gpu.mma_matrix<16x16xf16, "COp">
          %32 = gpu.subgroup_mma_load_matrix %arg0[%28, %30] {leadDimension = 1024 : index} : memref<1024x1024xf16> -> !gpu.mma_matrix<16x16xf16, "COp">
          %33 = addi %26, %c32 : index
          %34 = gpu.subgroup_mma_load_matrix %arg0[%25, %33] {leadDimension = 1024 : index} : memref<1024x1024xf16> -> !gpu.mma_matrix<16x16xf16, "COp">
          %35 = gpu.subgroup_mma_load_matrix %arg0[%28, %33] {leadDimension = 1024 : index} : memref<1024x1024xf16> -> !gpu.mma_matrix<16x16xf16, "COp">
          %36 = addi %26, %c48 : index
          %37 = gpu.subgroup_mma_load_matrix %arg0[%25, %36] {leadDimension = 1024 : index} : memref<1024x1024xf16> -> !gpu.mma_matrix<16x16xf16, "COp">
          %38 = gpu.subgroup_mma_load_matrix %arg0[%28, %36] {leadDimension = 1024 : index} : memref<1024x1024xf16> -> !gpu.mma_matrix<16x16xf16, "COp">
          scf.for %arg7 = %c0 to %c4 step %c1 {
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
            memref.store %53, %19[%44, %59] : memref<64x17xvector<8xf16>, 3>
          } {isCopyLoopNest = true}
          scf.for %arg7 = %c0 to %c4 step %c1 {
            %41 = muli %arg7, %c256 : index
            %42 = addi %15, %41 : index
            %43 = remi_signed %42, %c8 : index
            %44 = divi_signed %42, %c8 : index
            %45 = muli %43, %c8 : index
            %46 = addi %17, %44 : index
            %47 = cmpi slt, %45, %c0 : index
            %48 = subi %c-1, %45 : index
            %49 = select %47, %48, %45 : index
            %50 = divi_signed %49, %c8 : index
            %51 = subi %c-1, %50 : index
            %52 = select %47, %51, %50 : index
            %53 = memref.load %arg2[%46, %52] : memref<1024x128xvector<8xf16>>
            memref.store %53, %20[%44, %52] : memref<128x9xvector<8xf16>, 3>
          } {isCopyLoopNest = true}
          gpu.barrier
          %39:8 = scf.for %arg7 = %c0 to %c960 step %c64 iter_args(%arg8 = %27, %arg9 = %29, %arg10 = %31, %arg11 = %32, %arg12 = %34, %arg13 = %35, %arg14 = %37, %arg15 = %38) -> (!gpu.mma_matrix<16x16xf16, "COp">, !gpu.mma_matrix<16x16xf16, "COp">, !gpu.mma_matrix<16x16xf16, "COp">, !gpu.mma_matrix<16x16xf16, "COp">, !gpu.mma_matrix<16x16xf16, "COp">, !gpu.mma_matrix<16x16xf16, "COp">, !gpu.mma_matrix<16x16xf16, "COp">, !gpu.mma_matrix<16x16xf16, "COp">) {
            gpu.barrier
            %41 = remi_signed %15, %c16 : index
            %42 = divi_signed %15, %c16 : index
            %43 = muli %41, %c8 : index
            %44 = addi %arg7, %42 : index
            %45 = addi %44, %c64 : index
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
            %65 = addi %64, %c64 : index
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
            %80 = addi %15, %c512 : index
            %81 = remi_signed %80, %c16 : index
            %82 = divi_signed %80, %c16 : index
            %83 = muli %81, %c8 : index
            %84 = addi %arg7, %82 : index
            %85 = addi %84, %c64 : index
            %86 = addi %18, %83 : index
            %87 = cmpi slt, %86, %c0 : index
            %88 = subi %c-1, %86 : index
            %89 = select %87, %88, %86 : index
            %90 = divi_signed %89, %c8 : index
            %91 = subi %c-1, %90 : index
            %92 = select %87, %91, %90 : index
            %93 = memref.load %arg1[%85, %92] : memref<1024x128xvector<8xf16>>
            %94 = cmpi slt, %83, %c0 : index
            %95 = subi %c-1, %83 : index
            %96 = select %94, %95, %83 : index
            %97 = divi_signed %96, %c8 : index
            %98 = subi %c-1, %97 : index
            %99 = select %94, %98, %97 : index
            %100 = addi %15, %c768 : index
            %101 = remi_signed %100, %c16 : index
            %102 = divi_signed %100, %c16 : index
            %103 = muli %101, %c8 : index
            %104 = addi %arg7, %102 : index
            %105 = addi %104, %c64 : index
            %106 = addi %18, %103 : index
            %107 = cmpi slt, %106, %c0 : index
            %108 = subi %c-1, %106 : index
            %109 = select %107, %108, %106 : index
            %110 = divi_signed %109, %c8 : index
            %111 = subi %c-1, %110 : index
            %112 = select %107, %111, %110 : index
            %113 = memref.load %arg1[%105, %112] : memref<1024x128xvector<8xf16>>
            %114 = cmpi slt, %103, %c0 : index
            %115 = subi %c-1, %103 : index
            %116 = select %114, %115, %103 : index
            %117 = divi_signed %116, %c8 : index
            %118 = subi %c-1, %117 : index
            %119 = select %114, %118, %117 : index
            %120 = remi_signed %15, %c8 : index
            %121 = divi_signed %15, %c8 : index
            %122 = muli %120, %c8 : index
            %123 = addi %17, %121 : index
            %124 = addi %arg7, %122 : index
            %125 = cmpi slt, %124, %c0 : index
            %126 = subi %c-1, %124 : index
            %127 = select %125, %126, %124 : index
            %128 = divi_signed %127, %c8 : index
            %129 = subi %c-1, %128 : index
            %130 = select %125, %129, %128 : index
            %131 = addi %130, %c8 : index
            %132 = memref.load %arg2[%123, %131] : memref<1024x128xvector<8xf16>>
            %133 = cmpi slt, %122, %c0 : index
            %134 = subi %c-1, %122 : index
            %135 = select %133, %134, %122 : index
            %136 = divi_signed %135, %c8 : index
            %137 = subi %c-1, %136 : index
            %138 = select %133, %137, %136 : index
            %139 = addi %15, %c256 : index
            %140 = remi_signed %139, %c8 : index
            %141 = divi_signed %139, %c8 : index
            %142 = muli %140, %c8 : index
            %143 = addi %17, %141 : index
            %144 = addi %arg7, %142 : index
            %145 = cmpi slt, %144, %c0 : index
            %146 = subi %c-1, %144 : index
            %147 = select %145, %146, %144 : index
            %148 = divi_signed %147, %c8 : index
            %149 = subi %c-1, %148 : index
            %150 = select %145, %149, %148 : index
            %151 = addi %150, %c8 : index
            %152 = memref.load %arg2[%143, %151] : memref<1024x128xvector<8xf16>>
            %153 = cmpi slt, %142, %c0 : index
            %154 = subi %c-1, %142 : index
            %155 = select %153, %154, %142 : index
            %156 = divi_signed %155, %c8 : index
            %157 = subi %c-1, %156 : index
            %158 = select %153, %157, %156 : index
            %159 = addi %15, %c512 : index
            %160 = remi_signed %159, %c8 : index
            %161 = divi_signed %159, %c8 : index
            %162 = muli %160, %c8 : index
            %163 = addi %17, %161 : index
            %164 = addi %arg7, %162 : index
            %165 = cmpi slt, %164, %c0 : index
            %166 = subi %c-1, %164 : index
            %167 = select %165, %166, %164 : index
            %168 = divi_signed %167, %c8 : index
            %169 = subi %c-1, %168 : index
            %170 = select %165, %169, %168 : index
            %171 = addi %170, %c8 : index
            %172 = memref.load %arg2[%163, %171] : memref<1024x128xvector<8xf16>>
            %173 = cmpi slt, %162, %c0 : index
            %174 = subi %c-1, %162 : index
            %175 = select %173, %174, %162 : index
            %176 = divi_signed %175, %c8 : index
            %177 = subi %c-1, %176 : index
            %178 = select %173, %177, %176 : index
            %179 = addi %15, %c768 : index
            %180 = remi_signed %179, %c8 : index
            %181 = divi_signed %179, %c8 : index
            %182 = muli %180, %c8 : index
            %183 = addi %17, %181 : index
            %184 = addi %arg7, %182 : index
            %185 = cmpi slt, %184, %c0 : index
            %186 = subi %c-1, %184 : index
            %187 = select %185, %186, %184 : index
            %188 = divi_signed %187, %c8 : index
            %189 = subi %c-1, %188 : index
            %190 = select %185, %189, %188 : index
            %191 = addi %190, %c8 : index
            %192 = memref.load %arg2[%183, %191] : memref<1024x128xvector<8xf16>>
            %193 = cmpi slt, %182, %c0 : index
            %194 = subi %c-1, %182 : index
            %195 = select %193, %194, %182 : index
            %196 = divi_signed %195, %c8 : index
            %197 = subi %c-1, %196 : index
            %198 = select %193, %197, %196 : index
            %199:8 = scf.for %arg16 = %c0 to %c64 step %c16 iter_args(%arg17 = %arg8, %arg18 = %arg9, %arg19 = %arg10, %arg20 = %arg11, %arg21 = %arg12, %arg22 = %arg13, %arg23 = %arg14, %arg24 = %arg15) -> (!gpu.mma_matrix<16x16xf16, "COp">, !gpu.mma_matrix<16x16xf16, "COp">, !gpu.mma_matrix<16x16xf16, "COp">, !gpu.mma_matrix<16x16xf16, "COp">, !gpu.mma_matrix<16x16xf16, "COp">, !gpu.mma_matrix<16x16xf16, "COp">, !gpu.mma_matrix<16x16xf16, "COp">, !gpu.mma_matrix<16x16xf16, "COp">) {
              %200 = gpu.subgroup_mma_load_matrix %arg4[%arg5, %arg16] {leadDimension = 72 : index} : memref<128x72xf16, 3> -> !gpu.mma_matrix<16x16xf16, "AOp">
              %201 = gpu.subgroup_mma_load_matrix %arg3[%arg16, %arg6] {leadDimension = 136 : index} : memref<64x136xf16, 3> -> !gpu.mma_matrix<16x16xf16, "BOp">
              %202 = gpu.subgroup_mma_compute %200, %201, %arg17 : !gpu.mma_matrix<16x16xf16, "AOp">, !gpu.mma_matrix<16x16xf16, "BOp"> -> !gpu.mma_matrix<16x16xf16, "COp">
              %203 = addi %arg5, %c16 : index
              %204 = gpu.subgroup_mma_load_matrix %arg4[%203, %arg16] {leadDimension = 72 : index} : memref<128x72xf16, 3> -> !gpu.mma_matrix<16x16xf16, "AOp">
              %205 = gpu.subgroup_mma_load_matrix %arg3[%arg16, %arg6] {leadDimension = 136 : index} : memref<64x136xf16, 3> -> !gpu.mma_matrix<16x16xf16, "BOp">
              %206 = gpu.subgroup_mma_compute %204, %205, %arg18 : !gpu.mma_matrix<16x16xf16, "AOp">, !gpu.mma_matrix<16x16xf16, "BOp"> -> !gpu.mma_matrix<16x16xf16, "COp">
              %207 = gpu.subgroup_mma_load_matrix %arg4[%arg5, %arg16] {leadDimension = 72 : index} : memref<128x72xf16, 3> -> !gpu.mma_matrix<16x16xf16, "AOp">
              %208 = addi %arg6, %c16 : index
              %209 = gpu.subgroup_mma_load_matrix %arg3[%arg16, %208] {leadDimension = 136 : index} : memref<64x136xf16, 3> -> !gpu.mma_matrix<16x16xf16, "BOp">
              %210 = gpu.subgroup_mma_compute %207, %209, %arg19 : !gpu.mma_matrix<16x16xf16, "AOp">, !gpu.mma_matrix<16x16xf16, "BOp"> -> !gpu.mma_matrix<16x16xf16, "COp">
              %211 = gpu.subgroup_mma_load_matrix %arg4[%203, %arg16] {leadDimension = 72 : index} : memref<128x72xf16, 3> -> !gpu.mma_matrix<16x16xf16, "AOp">
              %212 = gpu.subgroup_mma_load_matrix %arg3[%arg16, %208] {leadDimension = 136 : index} : memref<64x136xf16, 3> -> !gpu.mma_matrix<16x16xf16, "BOp">
              %213 = gpu.subgroup_mma_compute %211, %212, %arg20 : !gpu.mma_matrix<16x16xf16, "AOp">, !gpu.mma_matrix<16x16xf16, "BOp"> -> !gpu.mma_matrix<16x16xf16, "COp">
              %214 = gpu.subgroup_mma_load_matrix %arg4[%arg5, %arg16] {leadDimension = 72 : index} : memref<128x72xf16, 3> -> !gpu.mma_matrix<16x16xf16, "AOp">
              %215 = addi %arg6, %c32 : index
              %216 = gpu.subgroup_mma_load_matrix %arg3[%arg16, %215] {leadDimension = 136 : index} : memref<64x136xf16, 3> -> !gpu.mma_matrix<16x16xf16, "BOp">
              %217 = gpu.subgroup_mma_compute %214, %216, %arg21 : !gpu.mma_matrix<16x16xf16, "AOp">, !gpu.mma_matrix<16x16xf16, "BOp"> -> !gpu.mma_matrix<16x16xf16, "COp">
              %218 = gpu.subgroup_mma_load_matrix %arg4[%203, %arg16] {leadDimension = 72 : index} : memref<128x72xf16, 3> -> !gpu.mma_matrix<16x16xf16, "AOp">
              %219 = gpu.subgroup_mma_load_matrix %arg3[%arg16, %215] {leadDimension = 136 : index} : memref<64x136xf16, 3> -> !gpu.mma_matrix<16x16xf16, "BOp">
              %220 = gpu.subgroup_mma_compute %218, %219, %arg22 : !gpu.mma_matrix<16x16xf16, "AOp">, !gpu.mma_matrix<16x16xf16, "BOp"> -> !gpu.mma_matrix<16x16xf16, "COp">
              %221 = gpu.subgroup_mma_load_matrix %arg4[%arg5, %arg16] {leadDimension = 72 : index} : memref<128x72xf16, 3> -> !gpu.mma_matrix<16x16xf16, "AOp">
              %222 = addi %arg6, %c48 : index
              %223 = gpu.subgroup_mma_load_matrix %arg3[%arg16, %222] {leadDimension = 136 : index} : memref<64x136xf16, 3> -> !gpu.mma_matrix<16x16xf16, "BOp">
              %224 = gpu.subgroup_mma_compute %221, %223, %arg23 : !gpu.mma_matrix<16x16xf16, "AOp">, !gpu.mma_matrix<16x16xf16, "BOp"> -> !gpu.mma_matrix<16x16xf16, "COp">
              %225 = gpu.subgroup_mma_load_matrix %arg4[%203, %arg16] {leadDimension = 72 : index} : memref<128x72xf16, 3> -> !gpu.mma_matrix<16x16xf16, "AOp">
              %226 = gpu.subgroup_mma_load_matrix %arg3[%arg16, %222] {leadDimension = 136 : index} : memref<64x136xf16, 3> -> !gpu.mma_matrix<16x16xf16, "BOp">
              %227 = gpu.subgroup_mma_compute %225, %226, %arg24 : !gpu.mma_matrix<16x16xf16, "AOp">, !gpu.mma_matrix<16x16xf16, "BOp"> -> !gpu.mma_matrix<16x16xf16, "COp">
              scf.yield %202, %206, %210, %213, %217, %220, %224, %227 : !gpu.mma_matrix<16x16xf16, "COp">, !gpu.mma_matrix<16x16xf16, "COp">, !gpu.mma_matrix<16x16xf16, "COp">, !gpu.mma_matrix<16x16xf16, "COp">, !gpu.mma_matrix<16x16xf16, "COp">, !gpu.mma_matrix<16x16xf16, "COp">, !gpu.mma_matrix<16x16xf16, "COp">, !gpu.mma_matrix<16x16xf16, "COp">
            }
            gpu.barrier
            memref.store %53, %19[%42, %59] : memref<64x17xvector<8xf16>, 3>
            memref.store %73, %19[%62, %79] : memref<64x17xvector<8xf16>, 3>
            memref.store %93, %19[%82, %99] : memref<64x17xvector<8xf16>, 3>
            memref.store %113, %19[%102, %119] : memref<64x17xvector<8xf16>, 3>
            memref.store %132, %20[%121, %138] : memref<128x9xvector<8xf16>, 3>
            memref.store %152, %20[%141, %158] : memref<128x9xvector<8xf16>, 3>
            memref.store %172, %20[%161, %178] : memref<128x9xvector<8xf16>, 3>
            memref.store %192, %20[%181, %198] : memref<128x9xvector<8xf16>, 3>
            scf.yield %199#0, %199#1, %199#2, %199#3, %199#4, %199#5, %199#6, %199#7 : !gpu.mma_matrix<16x16xf16, "COp">, !gpu.mma_matrix<16x16xf16, "COp">, !gpu.mma_matrix<16x16xf16, "COp">, !gpu.mma_matrix<16x16xf16, "COp">, !gpu.mma_matrix<16x16xf16, "COp">, !gpu.mma_matrix<16x16xf16, "COp">, !gpu.mma_matrix<16x16xf16, "COp">, !gpu.mma_matrix<16x16xf16, "COp">
          }
          gpu.barrier
          %40:8 = scf.for %arg7 = %c0 to %c64 step %c16 iter_args(%arg8 = %39#0, %arg9 = %39#1, %arg10 = %39#2, %arg11 = %39#3, %arg12 = %39#4, %arg13 = %39#5, %arg14 = %39#6, %arg15 = %39#7) -> (!gpu.mma_matrix<16x16xf16, "COp">, !gpu.mma_matrix<16x16xf16, "COp">, !gpu.mma_matrix<16x16xf16, "COp">, !gpu.mma_matrix<16x16xf16, "COp">, !gpu.mma_matrix<16x16xf16, "COp">, !gpu.mma_matrix<16x16xf16, "COp">, !gpu.mma_matrix<16x16xf16, "COp">, !gpu.mma_matrix<16x16xf16, "COp">) {
            %41 = gpu.subgroup_mma_load_matrix %arg4[%arg5, %arg7] {leadDimension = 72 : index} : memref<128x72xf16, 3> -> !gpu.mma_matrix<16x16xf16, "AOp">
            %42 = gpu.subgroup_mma_load_matrix %arg3[%arg7, %arg6] {leadDimension = 136 : index} : memref<64x136xf16, 3> -> !gpu.mma_matrix<16x16xf16, "BOp">
            %43 = gpu.subgroup_mma_compute %41, %42, %arg8 : !gpu.mma_matrix<16x16xf16, "AOp">, !gpu.mma_matrix<16x16xf16, "BOp"> -> !gpu.mma_matrix<16x16xf16, "COp">
            %44 = addi %arg5, %c16 : index
            %45 = gpu.subgroup_mma_load_matrix %arg4[%44, %arg7] {leadDimension = 72 : index} : memref<128x72xf16, 3> -> !gpu.mma_matrix<16x16xf16, "AOp">
            %46 = gpu.subgroup_mma_load_matrix %arg3[%arg7, %arg6] {leadDimension = 136 : index} : memref<64x136xf16, 3> -> !gpu.mma_matrix<16x16xf16, "BOp">
            %47 = gpu.subgroup_mma_compute %45, %46, %arg9 : !gpu.mma_matrix<16x16xf16, "AOp">, !gpu.mma_matrix<16x16xf16, "BOp"> -> !gpu.mma_matrix<16x16xf16, "COp">
            %48 = gpu.subgroup_mma_load_matrix %arg4[%arg5, %arg7] {leadDimension = 72 : index} : memref<128x72xf16, 3> -> !gpu.mma_matrix<16x16xf16, "AOp">
            %49 = addi %arg6, %c16 : index
            %50 = gpu.subgroup_mma_load_matrix %arg3[%arg7, %49] {leadDimension = 136 : index} : memref<64x136xf16, 3> -> !gpu.mma_matrix<16x16xf16, "BOp">
            %51 = gpu.subgroup_mma_compute %48, %50, %arg10 : !gpu.mma_matrix<16x16xf16, "AOp">, !gpu.mma_matrix<16x16xf16, "BOp"> -> !gpu.mma_matrix<16x16xf16, "COp">
            %52 = gpu.subgroup_mma_load_matrix %arg4[%44, %arg7] {leadDimension = 72 : index} : memref<128x72xf16, 3> -> !gpu.mma_matrix<16x16xf16, "AOp">
            %53 = gpu.subgroup_mma_load_matrix %arg3[%arg7, %49] {leadDimension = 136 : index} : memref<64x136xf16, 3> -> !gpu.mma_matrix<16x16xf16, "BOp">
            %54 = gpu.subgroup_mma_compute %52, %53, %arg11 : !gpu.mma_matrix<16x16xf16, "AOp">, !gpu.mma_matrix<16x16xf16, "BOp"> -> !gpu.mma_matrix<16x16xf16, "COp">
            %55 = gpu.subgroup_mma_load_matrix %arg4[%arg5, %arg7] {leadDimension = 72 : index} : memref<128x72xf16, 3> -> !gpu.mma_matrix<16x16xf16, "AOp">
            %56 = addi %arg6, %c32 : index
            %57 = gpu.subgroup_mma_load_matrix %arg3[%arg7, %56] {leadDimension = 136 : index} : memref<64x136xf16, 3> -> !gpu.mma_matrix<16x16xf16, "BOp">
            %58 = gpu.subgroup_mma_compute %55, %57, %arg12 : !gpu.mma_matrix<16x16xf16, "AOp">, !gpu.mma_matrix<16x16xf16, "BOp"> -> !gpu.mma_matrix<16x16xf16, "COp">
            %59 = gpu.subgroup_mma_load_matrix %arg4[%44, %arg7] {leadDimension = 72 : index} : memref<128x72xf16, 3> -> !gpu.mma_matrix<16x16xf16, "AOp">
            %60 = gpu.subgroup_mma_load_matrix %arg3[%arg7, %56] {leadDimension = 136 : index} : memref<64x136xf16, 3> -> !gpu.mma_matrix<16x16xf16, "BOp">
            %61 = gpu.subgroup_mma_compute %59, %60, %arg13 : !gpu.mma_matrix<16x16xf16, "AOp">, !gpu.mma_matrix<16x16xf16, "BOp"> -> !gpu.mma_matrix<16x16xf16, "COp">
            %62 = gpu.subgroup_mma_load_matrix %arg4[%arg5, %arg7] {leadDimension = 72 : index} : memref<128x72xf16, 3> -> !gpu.mma_matrix<16x16xf16, "AOp">
            %63 = addi %arg6, %c48 : index
            %64 = gpu.subgroup_mma_load_matrix %arg3[%arg7, %63] {leadDimension = 136 : index} : memref<64x136xf16, 3> -> !gpu.mma_matrix<16x16xf16, "BOp">
            %65 = gpu.subgroup_mma_compute %62, %64, %arg14 : !gpu.mma_matrix<16x16xf16, "AOp">, !gpu.mma_matrix<16x16xf16, "BOp"> -> !gpu.mma_matrix<16x16xf16, "COp">
            %66 = gpu.subgroup_mma_load_matrix %arg4[%44, %arg7] {leadDimension = 72 : index} : memref<128x72xf16, 3> -> !gpu.mma_matrix<16x16xf16, "AOp">
            %67 = gpu.subgroup_mma_load_matrix %arg3[%arg7, %63] {leadDimension = 136 : index} : memref<64x136xf16, 3> -> !gpu.mma_matrix<16x16xf16, "BOp">
            %68 = gpu.subgroup_mma_compute %66, %67, %arg15 : !gpu.mma_matrix<16x16xf16, "AOp">, !gpu.mma_matrix<16x16xf16, "BOp"> -> !gpu.mma_matrix<16x16xf16, "COp">
            scf.yield %43, %47, %51, %54, %58, %61, %65, %68 : !gpu.mma_matrix<16x16xf16, "COp">, !gpu.mma_matrix<16x16xf16, "COp">, !gpu.mma_matrix<16x16xf16, "COp">, !gpu.mma_matrix<16x16xf16, "COp">, !gpu.mma_matrix<16x16xf16, "COp">, !gpu.mma_matrix<16x16xf16, "COp">, !gpu.mma_matrix<16x16xf16, "COp">, !gpu.mma_matrix<16x16xf16, "COp">
          }
          gpu.subgroup_mma_store_matrix %40#0, %arg0[%25, %26] {leadDimension = 1024 : index} : !gpu.mma_matrix<16x16xf16, "COp">, memref<1024x1024xf16>
          gpu.subgroup_mma_store_matrix %40#1, %arg0[%28, %26] {leadDimension = 1024 : index} : !gpu.mma_matrix<16x16xf16, "COp">, memref<1024x1024xf16>
          gpu.subgroup_mma_store_matrix %40#2, %arg0[%25, %30] {leadDimension = 1024 : index} : !gpu.mma_matrix<16x16xf16, "COp">, memref<1024x1024xf16>
          gpu.subgroup_mma_store_matrix %40#3, %arg0[%28, %30] {leadDimension = 1024 : index} : !gpu.mma_matrix<16x16xf16, "COp">, memref<1024x1024xf16>
          gpu.subgroup_mma_store_matrix %40#4, %arg0[%25, %33] {leadDimension = 1024 : index} : !gpu.mma_matrix<16x16xf16, "COp">, memref<1024x1024xf16>
          gpu.subgroup_mma_store_matrix %40#5, %arg0[%28, %33] {leadDimension = 1024 : index} : !gpu.mma_matrix<16x16xf16, "COp">, memref<1024x1024xf16>
          gpu.subgroup_mma_store_matrix %40#6, %arg0[%25, %36] {leadDimension = 1024 : index} : !gpu.mma_matrix<16x16xf16, "COp">, memref<1024x1024xf16>
          gpu.subgroup_mma_store_matrix %40#7, %arg0[%28, %36] {leadDimension = 1024 : index} : !gpu.mma_matrix<16x16xf16, "COp">, memref<1024x1024xf16>
        }
      }
      gpu.return
    }
  }
  gpu.module @initC_kernel {
    gpu.func @initC_kernel(%arg0: memref<1024x1024xf16>) kernel {
      %c1 = constant 1 : index
      %c0 = constant 0 : index
      %c1024 = constant 1024 : index
      %cst = constant 0.000000e+00 : f16
      scf.for %arg1 = %c0 to %c1024 step %c1 {
        scf.for %arg2 = %c0 to %c1024 step %c1 {
          memref.store %cst, %arg0[%arg1, %arg2] : memref<1024x1024xf16>
        }
      }
      gpu.return
    }
  }
  func private @print_memref_f32(memref<*xf32>)
  func private @print_flops(f64)
  func private @rtclock() -> f64
}

