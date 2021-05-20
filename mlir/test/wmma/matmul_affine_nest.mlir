// RUN: mlir-opt %s --canonicalize --affine-loop-tile="num-tiling-levels=2 tile-sizes=128,64,64,64,32,16 relative-indexing=true" --canonicalize --canonicalize -test-gpu-matmul-fast-buffer-placement="matrices=A,B global-allocation=true" --canonicalize --test-mark-parallel-loops --test-specialize-affine-matmul-for-wmma="accum=f32 padding-a=8 padding-b=8" --canonicalize --cse --test-split-compute-loop  --canonicalize --test-gpu-matmul-barrier-insertion  --test-normalize-copy-loops --canonicalize --cse --test-vectorize-gpu-matmul-copy-loops="load-store-width=128" --canonicalize --test-collapse-affine-parallel --canonicalize  --lower-affine --test-gpu-matmul-parallel-loop-mapping --canonicalize --test-convert-matmul-parallel-loops-to-gpu --cse --test-unroll-and-delay-copies --canonicalize -gpu-kernel-outlining  --test-gpu-mark-global-as-workgroup-memory --canonicalize --cse --convert-scf-to-std

// nvprof could be used by using '... | nvprof --print-gpu-trace mlir-cuda-runner ..'

func @main() {
  %c16_f = constant 16.0e+00 : f16
  %f0 = constant 0.0e+00 : f32
  %A = memref.alloc() : memref<4096x4096xf16>
  %B = memref.alloc() : memref<4096x4096xf16>
  %C = memref.alloc() : memref<4096x4096xf32>
  
  %c0 = constant 0 : index
  %c1 = constant 1 : index
  %M = memref.dim %A, %c0: memref<4096x4096xf16> 
  %N = memref.dim %B, %c1: memref<4096x4096xf16> 
  %K = memref.dim %A, %c1: memref<4096x4096xf16> 
  
  // Intialize the Input matrix A.
  scf.for %arg0 = %c0 to %M step %c1 {
    scf.for %arg1 = %c0 to %K step %c1 {
      %add = addi %arg0, %arg1 : index
      %add_int = index_cast %add : index to i16
      %add_float = sitofp %add_int : i16 to f16
      %rem = remf %add_float, %c16_f : f16 
      memref.store %rem, %A[%arg0, %arg1] : memref<4096x4096xf16>
    }
  }

  // Intialize the Input matrix B.
  scf.for %arg0 = %c0 to %K step %c1 {
    scf.for %arg1 = %c0 to %N step %c1 {
      %add = addi %arg0, %arg1 : index
      %add_int = index_cast %add : index to i16
      %add_float = sitofp %add_int : i16 to f16
      %rem = remf %add_float, %c16_f : f16 
      memref.store %rem, %B[%arg0, %arg1] : memref<4096x4096xf16>
    }
  }

  // Intialize C matrix with zeros.
  scf.for %arg0 = %c0 to %M step %c1 {
    scf.for %arg1 = %c0 to %N step %c1 {
      memref.store %f0, %C[%arg0, %arg1] : memref<4096x4096xf32>
    }
  }
  
  %t0 = gpu.wait async

  // Allocate actual input/output arrays on device.
  %gpu_A, %t5 = gpu.alloc async [%t0] () : memref<4096x4096xf16>
  %gpu_B, %t6 = gpu.alloc async [%t0] () : memref<4096x4096xf16>
  %gpu_C, %t7 = gpu.alloc async [%t0] () : memref<4096x4096xf32>

  // Copy initialized arrays from host to device.
  %t2 = gpu.memcpy async [%t0] %gpu_A, %A : memref<4096x4096xf16>, memref<4096x4096xf16>
  %t3 = gpu.memcpy async [%t0] %gpu_B, %B : memref<4096x4096xf16>, memref<4096x4096xf16>
  %t4 = gpu.memcpy async [%t0] %gpu_C, %C : memref<4096x4096xf32>, memref<4096x4096xf32>
  
  gpu.wait [%t0]

  // Main kernel
  %t_start = call @rtclock() : () -> (f64)
  affine.for %i = 0 to %M {
    affine.for %j = 0 to %N {
      affine.for %l = 0 to %K {
        %a = affine.load %gpu_A[%i, %l] : memref<4096x4096xf16>
        %b = affine.load %gpu_B[%l, %j] : memref<4096x4096xf16>
        %c = affine.load %gpu_C[%i, %j] : memref<4096x4096xf32>
        %p = mulf %a, %b : f16
        %q = fpext %p : f16 to f32
        %co = addf %c, %q : f32
        affine.store %co, %gpu_C[%i, %j] : memref<4096x4096xf32>
      }
    }
  }
  %t_end = call @rtclock() : () -> (f64)
  
  %t1 = gpu.wait async 
  // Copy result matrix back to host for printing.
  %t8 = gpu.memcpy async [%t1] %C, %gpu_C : memref<4096x4096xf32>, memref<4096x4096xf32>
  gpu.wait[%t8]
  
  // Logic for printing perf.
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
  
  %22 = memref.cast %C : memref<4096x4096xf32> to memref<*xf32>
  
  return
}

func private @print_memref_f32(memref<*xf32>)
func private @print_flops(f64)
func private @rtclock() -> (f64)
