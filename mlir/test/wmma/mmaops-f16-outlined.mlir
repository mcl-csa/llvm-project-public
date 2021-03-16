// RUN: mlir-opt --convert-scf-to-std %s | mlir-cuda-runner --index-bitwidth=32 \
// RUN:   --sm=sm_75 -gpu-to-cubin="gpu-binary-annotation=nvvm.cubin" \
// RUN:   -gpu-to-llvm="gpu-binary-annotation=nvvm.cubin" \
// RUN:   --shared-libs=%linalg_test_lib_dir/libmlir_cuda_runtime%shlibext \
// RUN:   --shared-libs=%linalg_test_lib_dir/libmlir_runner_utils%shlibext \
// RUN:   --shared-libs=%linalg_test_lib_dir/libmlir_c_runner_utils%shlibext \
// RUN:   --entry-point-result=void \

module attributes {gpu.container_module}  {
  func @main() {
    %c1 = constant 1 : index
    %c0 = constant 0 : index
    %c1024 = constant 1024 : index
    %cst_0 = constant 0.000000e+00 : f16
    %c4_f = constant 4.0e+00 : f16
    %c16 = constant 16 : index
    %c128 = constant 128 : index
    %A = alloc() : memref<1024x1024xf16>
    %B = alloc() : memref<1024x1024xf16>
    %C = alloc() : memref<1024x1024xf16>
    %out = alloc() : memref<1024x1024xf32>

    // Initalize the input matrix A.
    scf.for %arg0 = %c0 to %c1024 step %c1 {
      scf.for %arg1 = %c0 to %c1024 step %c1 {
        %add = addi %arg0, %arg1 : index
        %add_int = index_cast %add : index to i16
        %add_float = sitofp %add_int : i16 to f16
        %rem = remf %add_float, %c4_f : f16 
        store %rem, %A[%arg0, %arg1] : memref<1024x1024xf16>
      }
    }

    // Initalize the input matrix B.
    scf.for %arg0 = %c0 to %c1024 step %c1 {
      scf.for %arg1 = %c0 to %c1024 step %c1 {
        %add = addi %arg0, %arg1 : index
        %add_int = index_cast %add : index to i16
        %add_float = sitofp %add_int : i16 to f16
        %rem = remf %add_float, %c4_f : f16 
        store %rem, %B[%arg0, %arg1] : memref<1024x1024xf16>
      }
    }

    // Intialize C matrix with zeros.
    scf.for %arg0 = %c0 to %c1024 step %c1 {
      scf.for %arg1 = %c0 to %c1024 step %c1 {
        store %cst_0, %C[%arg0, %arg1] : memref<1024x1024xf16>
      }
    }

    %3 = memref_cast %A : memref<1024x1024xf16> to memref<*xf16>
    gpu.host_register %3 : memref<*xf16>
    %4 = memref_cast %B : memref<1024x1024xf16> to memref<*xf16>
    gpu.host_register %4 : memref<*xf16>
    %5 = memref_cast %C : memref<1024x1024xf16> to memref<*xf16>
    gpu.host_register %5 : memref<*xf16>
    %6 = memref_cast %out : memref<1024x1024xf32> to memref<*xf32>
    gpu.host_register %6 : memref<*xf32>

    %t_start = call @rtclock() : () -> (f64)
    gpu.launch_func  @main_kernel::@main_kernel blocks in (%c16, %c16, %c1) threads in (%c128, %c1, %c1) args(%C : memref<1024x1024xf16>, %B : memref<1024x1024xf16>, %A : memref<1024x1024xf16>)
    %t_end = call @rtclock() : () -> (f64)

    %M = dim %C, %c0 : memref<1024x1024xf16>
    %N = dim %C, %c1 : memref<1024x1024xf16>
    %K = dim %A, %c1 : memref<1024x1024xf16>

    %t = subf %t_end, %t_start : f64
    %f1 = muli %M, %N : index
    %f2 = muli %f1, %K : index
    // 2*M*N*K.
    %reps = constant 1 : index
    %c2 = constant 2 : index
    %f3 = muli %c2, %f2 : index
    %num_flops = muli %reps, %f3 : index
    %num_flops_i = index_cast %num_flops : index to i64
    %num_flops_f = sitofp %num_flops_i : i64 to f64
    %flops = divf %num_flops_f, %t : f64
    call @print_flops(%flops) : (f64) -> ()
    
    scf.for %arg0 = %c0 to %c1024 step %c1 {
      scf.for %arg1 = %c0 to %c1024 step %c1 {
        %18 = load %C[%arg0, %arg1] : memref<1024x1024xf16>
        %19 = fpext %18 : f16 to f32
        store %19, %out[%arg0, %arg1] : memref<1024x1024xf32>
      }
    }

    call @print_memref_f32(%6) : (memref<*xf32>) -> ()
    return
  }
  gpu.module @main_kernel {
    gpu.func @main_kernel(%C: memref<1024x1024xf16>, %B: memref<1024x1024xf16>, %A: memref<1024x1024xf16>) workgroup(%12 : memref<64x64xf16, 3>, %13 : memref<64x64xf16, 3>) kernel {
      %c128 = constant 128 : index
      %c32 = constant 32 : index
      %c64 = constant 64 : index
      %c2 = constant 2 : index
      %c16 = constant 16 : index
      %c-1 = constant -1 : index
      %c0 = constant 0 : index
      %c1024 = constant 1024 : index
      %0 = "gpu.block_id"() {dimension = "x"} : () -> index
      %1 = "gpu.block_id"() {dimension = "y"} : () -> index
      %2 = "gpu.thread_id"() {dimension = "x"} : () -> index
      %3 = "gpu.thread_id"() {dimension = "y"} : () -> index
      %4 = "gpu.thread_id"() {dimension = "z"} : () -> index
      %5 = muli %4, %c128 : index
      %6 = muli %3, %c128 : index
      %7 = addi %5, %6 : index
      %8 = addi %7, %2 : index
      %9 = divi_unsigned %8, %c32 : index
      %10 = muli %1, %c64 : index
      %11 = muli %0, %c64 : index
      %14 = remi_unsigned %9, %c2 : index
      %15 = divi_unsigned %9, %c2 : index
      %16 = muli %15, %c32 : index
      scf.for %arg3 = %16 to %c64 step %c64 {
        %17 = muli %14, %c32 : index
        scf.for %arg4 = %17 to %c64 step %c64 {
          %18 = addi %10, %arg3 : index
          %19 = addi %11, %arg4 : index
          %20 = gpu.subgroup_mma_load_matrix %C[%18, %19] {leadDimension = 1024 : index, operand = "COp"} : memref<1024x1024xf16> -> !gpu.mmafragment<4, vector<2xf16>>
          %21 = addi %10, %arg3 : index
          %22 = addi %21, %c16 : index
          %23 = addi %11, %arg4 : index
          %24 = gpu.subgroup_mma_load_matrix %C[%22, %23] {leadDimension = 1024 : index, operand = "COp"} : memref<1024x1024xf16> -> !gpu.mmafragment<4, vector<2xf16>>
          %25 = addi %10, %arg3 : index
          %26 = addi %11, %arg4 : index
          %27 = addi %26, %c16 : index
          %28 = gpu.subgroup_mma_load_matrix %C[%25, %27] {leadDimension = 1024 : index, operand = "COp"} : memref<1024x1024xf16> -> !gpu.mmafragment<4, vector<2xf16>>
          %29 = addi %10, %arg3 : index
          %30 = addi %29, %c16 : index
          %31 = addi %11, %arg4 : index
          %32 = addi %31, %c16 : index
          %33 = gpu.subgroup_mma_load_matrix %C[%30, %32] {leadDimension = 1024 : index, operand = "COp"} : memref<1024x1024xf16> -> !gpu.mmafragment<4, vector<2xf16>>
          %34:4 = scf.for %arg5 = %c0 to %c1024 step %c64 iter_args(%arg6 = %20, %arg7 = %24, %arg8 = %28, %arg9 = %33) -> (!gpu.mmafragment<4, vector<2xf16>>, !gpu.mmafragment<4, vector<2xf16>>, !gpu.mmafragment<4, vector<2xf16>>, !gpu.mmafragment<4, vector<2xf16>>) {
            %35 = addi %arg5, %c64 : index
            %36 = addi %11, %c64 : index
            %37 = subi %35, %arg5 : index
            %38 = subi %36, %11 : index
            %39 = muli %37, %38 : index
            scf.for %arg10 = %8 to %39 step %c128 {
              %46 = remi_signed %arg10, %38 : index
              %47 = divi_signed %arg10, %38 : index
              %48 = addi %46, %11 : index
              %49 = addi %47, %arg5 : index
              %50 = load %B[%49, %48] : memref<1024x1024xf16>
              %51 = muli %arg5, %c-1 : index
              %52 = addi %51, %49 : index
              %53 = muli %11, %c-1 : index
              %54 = addi %53, %48 : index
              store %50, %12[%52, %54] : memref<64x64xf16, 3>
            }
            %40 = addi %10, %c64 : index
            %41 = addi %arg5, %c64 : index
            %42 = subi %40, %10 : index
            %43 = subi %41, %arg5 : index
            %44 = muli %42, %43 : index
            scf.for %arg10 = %8 to %44 step %c128 {
              %46 = remi_signed %arg10, %43 : index
              %47 = divi_signed %arg10, %43 : index
              %48 = addi %46, %arg5 : index
              %49 = addi %47, %10 : index
              %50 = load %A[%49, %48] : memref<1024x1024xf16>
              %51 = muli %10, %c-1 : index
              %52 = addi %51, %49 : index
              %53 = muli %arg5, %c-1 : index
              %54 = addi %53, %48 : index
              store %50, %13[%52, %54] : memref<64x64xf16, 3>
            }
            gpu.barrier
            %45:4 = scf.for %arg10 = %c0 to %c64 step %c16 iter_args(%arg11 = %arg6, %arg12 = %arg7, %arg13 = %arg8, %arg14 = %arg9) -> (!gpu.mmafragment<4, vector<2xf16>>, !gpu.mmafragment<4, vector<2xf16>>, !gpu.mmafragment<4, vector<2xf16>>, !gpu.mmafragment<4, vector<2xf16>>) {
              %46 = gpu.subgroup_mma_load_matrix %13[%arg3, %arg10] {leadDimension = 64 : index, operand = "AOp"} : memref<64x64xf16, 3> -> !gpu.mmafragment<8, vector<2xf16>>
              %47 = gpu.subgroup_mma_load_matrix %12[%arg10, %arg4] {leadDimension = 64 : index, operand = "BOp"} : memref<64x64xf16, 3> -> !gpu.mmafragment<8, vector<2xf16>>
              %48 = gpu.subgroup_mma_compute %46, %47, %arg11 : !gpu.mmafragment<8, vector<2xf16>>, !gpu.mmafragment<8, vector<2xf16>>, !gpu.mmafragment<4, vector<2xf16>> -> !gpu.mmafragment<4, vector<2xf16>>
              %49 = addi %arg3, %c16 : index
              %50 = gpu.subgroup_mma_load_matrix %13[%49, %arg10] {leadDimension = 64 : index, operand = "AOp"} : memref<64x64xf16, 3> -> !gpu.mmafragment<8, vector<2xf16>>
              %51 = gpu.subgroup_mma_load_matrix %12[%arg10, %arg4] {leadDimension = 64 : index, operand = "BOp"} : memref<64x64xf16, 3> -> !gpu.mmafragment<8, vector<2xf16>>
              %52 = gpu.subgroup_mma_compute %50, %51, %arg12 : !gpu.mmafragment<8, vector<2xf16>>, !gpu.mmafragment<8, vector<2xf16>>, !gpu.mmafragment<4, vector<2xf16>> -> !gpu.mmafragment<4, vector<2xf16>>
              %53 = gpu.subgroup_mma_load_matrix %13[%arg3, %arg10] {leadDimension = 64 : index, operand = "AOp"} : memref<64x64xf16, 3> -> !gpu.mmafragment<8, vector<2xf16>>
              %54 = addi %arg4, %c16 : index
              %55 = gpu.subgroup_mma_load_matrix %12[%arg10, %54] {leadDimension = 64 : index, operand = "BOp"} : memref<64x64xf16, 3> -> !gpu.mmafragment<8, vector<2xf16>>
              %56 = gpu.subgroup_mma_compute %53, %55, %arg13 : !gpu.mmafragment<8, vector<2xf16>>, !gpu.mmafragment<8, vector<2xf16>>, !gpu.mmafragment<4, vector<2xf16>> -> !gpu.mmafragment<4, vector<2xf16>>
              %57 = addi %arg3, %c16 : index
              %58 = gpu.subgroup_mma_load_matrix %13[%57, %arg10] {leadDimension = 64 : index, operand = "AOp"} : memref<64x64xf16, 3> -> !gpu.mmafragment<8, vector<2xf16>>
              %59 = addi %arg4, %c16 : index
              %60 = gpu.subgroup_mma_load_matrix %12[%arg10, %59] {leadDimension = 64 : index, operand = "BOp"} : memref<64x64xf16, 3> -> !gpu.mmafragment<8, vector<2xf16>>
              %61 = gpu.subgroup_mma_compute %58, %60, %arg14 : !gpu.mmafragment<8, vector<2xf16>>, !gpu.mmafragment<8, vector<2xf16>>, !gpu.mmafragment<4, vector<2xf16>> -> !gpu.mmafragment<4, vector<2xf16>>
              scf.yield %48, %52, %56, %61 : !gpu.mmafragment<4, vector<2xf16>>, !gpu.mmafragment<4, vector<2xf16>>, !gpu.mmafragment<4, vector<2xf16>>, !gpu.mmafragment<4, vector<2xf16>>
            }
            gpu.barrier
            scf.yield %45#0, %45#1, %45#2, %45#3 : !gpu.mmafragment<4, vector<2xf16>>, !gpu.mmafragment<4, vector<2xf16>>, !gpu.mmafragment<4, vector<2xf16>>, !gpu.mmafragment<4, vector<2xf16>>
          }
          gpu.subgroup_mma_store_matrix %34#0, %C[%18, %19] {leadDimension = 1024 : index} : !gpu.mmafragment<4, vector<2xf16>>, memref<1024x1024xf16>
          gpu.subgroup_mma_store_matrix %34#1, %C[%22, %23] {leadDimension = 1024 : index} : !gpu.mmafragment<4, vector<2xf16>>, memref<1024x1024xf16>
          gpu.subgroup_mma_store_matrix %34#2, %C[%25, %27] {leadDimension = 1024 : index} : !gpu.mmafragment<4, vector<2xf16>>, memref<1024x1024xf16>
          gpu.subgroup_mma_store_matrix %34#3, %C[%30, %32] {leadDimension = 1024 : index} : !gpu.mmafragment<4, vector<2xf16>>, memref<1024x1024xf16>
        }
      }
      gpu.return
    }
  }
  func private @print_memref_f32(memref<*xf32>)
  func private @print_flops(f64)
  func private @rtclock() -> (f64)
}

