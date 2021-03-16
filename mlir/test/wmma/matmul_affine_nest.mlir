// RUN: mlir-opt %s --canonicalize --affine-loop-tile="num-tiling-levels=2 tile-sizes=64,64,16,32,32,16 relative-indexing=true" --canonicalize -test-gpu-matmul-fast-buffer-placement="matrices=A,B global-allocation=true" --canonicalize --test-specialize-affine-matmul-for-wmma=accum=f32 --canonicalize --test-collapse-affine-parallel --canonicalize --lower-affine --test-gpu-matmul-parallel-loop-mapping --canonicalize --test-convert-matmul-parallel-loops-to-gpu --gpu-kernel-outlining --test-gpu-mark-global-as-workgroup-memory --canonicalize --convert-scf-to-std | mlir-cuda-runner -O3 --sm=sm_75 --max-reg-per-thread=100 --index-bitwidth=32 -gpu-to-cubin="gpu-binary-annotation=nvvm.cubin" -gpu-to-llvm="gpu-binary-annotation=nvvm.cubin" --shared-libs=%linalg_test_lib_dir/libmlir_cuda_runtime%shlibext --shared-libs=%linalg_test_lib_dir/libmlir_runner_utils%shlibext --shared-libs=%linalg_test_lib_dir/libmlir_c_runner_utils%shlibext --entry-point-result=void

// nvprof could be used by using '... | nvprof --print-gpu-trace mlir-cuda-runner ..'

func @main() {
  %c16_f = constant 16.0e+00 : f16
  %f0 = constant 0.0e+00 : f32
  %A = alloc() : memref<1024x1024xf16>
  %B = alloc() : memref<1024x1024xf16>
  %C = alloc() : memref<1024x1024xf32>
  
  %c0 = constant 0 : index
  %c1 = constant 1 : index
  %M = dim %A, %c0: memref<1024x1024xf16> 
  %N = dim %B, %c1: memref<1024x1024xf16> 
  %K = dim %A, %c1: memref<1024x1024xf16> 
  
  // Intialize the Input matrix A.
  scf.for %arg0 = %c0 to %M step %c1 {
    scf.for %arg1 = %c0 to %K step %c1 {
      %add = addi %arg0, %arg1 : index
      %add_int = index_cast %add : index to i16
      %add_float = sitofp %add_int : i16 to f16
      %rem = remf %add_float, %c16_f : f16 
      store %rem, %A[%arg0, %arg1] : memref<1024x1024xf16>
    }
  }

  // Intialize the Input matrix B.
  scf.for %arg0 = %c0 to %K step %c1 {
    scf.for %arg1 = %c0 to %N step %c1 {
      %add = addi %arg0, %arg1 : index
      %add_int = index_cast %add : index to i16
      %add_float = sitofp %add_int : i16 to f16
      %rem = remf %add_float, %c16_f : f16 
      store %rem, %B[%arg0, %arg1] : memref<1024x1024xf16>
    }
  }

  // Intialize C matrix with zeros.
  scf.for %arg0 = %c0 to %M step %c1 {
    scf.for %arg1 = %c0 to %N step %c1 {
      store %f0, %C[%arg0, %arg1] : memref<1024x1024xf32>
    }
  }
  
  %00 = memref_cast %A : memref<1024x1024xf16> to memref<*xf16>
  gpu.host_register %00 : memref<*xf16>
  %11 = memref_cast %B : memref<1024x1024xf16> to memref<*xf16>
  gpu.host_register %11 : memref<*xf16>
  %22 = memref_cast %C : memref<1024x1024xf32> to memref<*xf32>
  gpu.host_register %22 : memref<*xf32>
  
  %t_start = call @rtclock() : () -> (f64)
  affine.for %i = 0 to %M {
    affine.for %j = 0 to %N {
      affine.for %l = 0 to %K {
        %a = affine.load %A[%i, %l] : memref<1024x1024xf16>
        %b = affine.load %B[%l, %j] : memref<1024x1024xf16>
        %c = affine.load %C[%i, %j] : memref<1024x1024xf32>
        %p = mulf %a, %b : f16
        %q = fpext %p : f16 to f32
        %co = addf %c, %q : f32
        affine.store %co, %C[%i, %j] : memref<1024x1024xf32>
      }
    }
  }
  %t_end = call @rtclock() : () -> (f64)
  
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
  
  call @print_memref_f32(%22) : (memref<*xf32>) -> ()
  
  return
}

func private @print_memref_f32(memref<*xf32>)
func private @print_flops(f64)
func private @rtclock() -> (f64)
