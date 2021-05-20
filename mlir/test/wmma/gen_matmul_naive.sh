echo "module attributes {gpu.container_module} {
  func @main() {
    %cst = constant 1.000000e+00 : f16
    %cst_0 = constant 0.000000e+00 : f16
    %c16_f = constant 16.0e+00 : f16
    %c32 = constant 32 : index
    %c128 = constant 128 : index
    %c-1 = constant -1 : index
    %c64 = constant 64 : index
    %c16 = constant 16 : index
    %c0 = constant 0 : index
    %c1 = constant 1 : index
    %0 = memref.alloc() : memref<$1x$2xf16>
    %1 = memref.alloc() : memref<$2x$3xf16>
    %2 = memref.alloc() : memref<$1x$3xf16>
    %00 = memref.alloc() : memref<$1x$2xf32>
    %22 = memref.alloc() : memref<$2x$3xf32>
    %44 = memref.alloc() : memref<$1x$3xf32>
    %c1_1 = constant 1 : index
    %c1_2 = constant 1 : index
    %c1_3 = constant 1 : index
    %c1_4 = constant 1 : index
    %c1_5 = constant 1 : index
    
    %M = memref.dim %44, %c0 : memref<$1x$3xf32>
    %N = memref.dim %44, %c1 : memref<$1x$3xf32>
    %K = memref.dim %00, %c1 : memref<$1x$2xf32>
     
    // Intialize A matrix.
    scf.for %arg0 = %c0 to %M step %c1 {
      scf.for %arg1 = %c0 to %K step %c1 {
        %a0 = remi_signed %arg0, %c16 : index
        %a1 = remi_signed %arg1, %c16 : index
        %add = addi %a0, %a1 : index
        %addm = remi_signed %add, %c16 : index
        %add_int = index_cast %addm : index to i16
        %add_float = sitofp %add_int : i16 to f16
        memref.store %add_float, %0[%arg0, %arg1] : memref<$1x$2xf16>
      }
    }
    
    // Convert fp16 to fp32
    scf.for %arg0 = %c0 to %M step %c1 {
      scf.for %arg1 = %c0 to %K step %c1 {
        %6 = memref.load %0[%arg0, %arg1] : memref<$1x$2xf16>
        %7 = fpext %6 : f16 to f32
        memref.store %7, %00[%arg0, %arg1] : memref<$1x$2xf32>
      }
    }
    
    // Intialize B matrix.
    scf.for %arg0 = %c0 to %K step %c1 {
      scf.for %arg1 = %c0 to %N step %c1 {
        %b0 = remi_signed %arg0, %c16 : index
        %b1 = remi_signed %arg1, %c16 : index
        %add = addi %b0, %b1 : index
        %addm = remi_signed %add, %c16 : index
        %add_int = index_cast %addm : index to i16
        %add_float = sitofp %add_int : i16 to f16
        memref.store %add_float, %1[%arg0, %arg1] : memref<$2x$3xf16>
      }
    }
    
    // Convert fp16 to fp32
    scf.for %arg0 = %c0 to %K step %c1 {
      scf.for %arg1 = %c0 to %N step %c1 {
        %6 = memref.load %1[%arg0, %arg1] : memref<$2x$3xf16>
        %7 = fpext %6 : f16 to f32
        memref.store %7, %22[%arg0, %arg1] : memref<$2x$3xf32>
      }
    }

    // Intialize the accumulator matrix with zeros.
    scf.for %arg0 = %c0 to %M step %c1 {
      scf.for %arg1 = %c0 to %N step %c1 {
        memref.store %cst_0, %2[%arg0, %arg1] : memref<$1x$3xf16>
      }
    }

    // Convert fp16 to fp32
    scf.for %arg0 = %c0 to %M step %c1 {
      scf.for %arg1 = %c0 to %N step %c1 {
        %6 = memref.load %2[%arg0, %arg1] : memref<$1x$3xf16>
        %7 = fpext %6 : f16 to f32
        memref.store %7, %44[%arg0, %arg1] : memref<$1x$3xf32>
      }
    }
    
    %111 = memref.cast %00 : memref<$1x$2xf32> to memref<*xf32>
    gpu.host_register %111 : memref<*xf32>
    
    %333 = memref.cast %22 : memref<$2x$3xf32> to memref<*xf32>
    gpu.host_register %333 : memref<*xf32>
    
    %555 = memref.cast %44 : memref<$1x$3xf32> to memref<*xf32>
    gpu.host_register %555 : memref<*xf32>

    %gridy = divi_unsigned %M, %c32 : index
    %gridx = divi_unsigned %N, %c32 : index

    %t_start = call @rtclock() : () -> (f64)
    gpu.launch_func  @matmul_kernel::@matmul_naive blocks in (%gridx, %gridy, %c1) threads in (%c32, %c32, %c1) args(%44 : memref<$1x$3xf32>, %22 : memref<$2x$3xf32>, %00 : memref<$1x$2xf32>)
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

    %out_naive_cast = memref.cast %44 : memref<$1x$3xf32> to memref<*xf32>
    call @print_memref_f32(%out_naive_cast) : (memref<*xf32>) -> ()
    return
  }
  gpu.module @matmul_kernel {
    gpu.func @matmul_naive(%arg0: memref<$1x$3xf32>, %arg1: memref<$2x$3xf32>, %arg2: memref<$1x$2xf32>) kernel {
      %0 = \"gpu.block_id\"() {dimension = \"x\"} : () -> index
      %1 = \"gpu.block_id\"() {dimension = \"y\"} : () -> index
      %2 = \"gpu.block_id\"() {dimension = \"z\"} : () -> index
      %3 = \"gpu.thread_id\"() {dimension = \"x\"} : () -> index
      %4 = \"gpu.thread_id\"() {dimension = \"y\"} : () -> index
      %5 = \"gpu.thread_id\"() {dimension = \"z\"} : () -> index
      %6 = \"gpu.grid_dim\"() {dimension = \"x\"} : () -> index
      %7 = \"gpu.grid_dim\"() {dimension = \"y\"} : () -> index
      %8 = \"gpu.grid_dim\"() {dimension = \"z\"} : () -> index
      %9 = \"gpu.block_dim\"() {dimension = \"x\"} : () -> index
      %10 = \"gpu.block_dim\"() {dimension = \"y\"} : () -> index
      %11 = \"gpu.block_dim\"() {dimension = \"z\"} : () -> index
      br ^bb1
    ^bb1:  // pred: ^bb0  
      %12 = muli %0, %9 : index
      %13 = addi %12, %3 : index // xdim
      %14 = muli %1, %10 : index
      %15 = addi %14, %4 : index // ydim
      
      %c$2 = constant $2 : index
      %c1 = constant 1 : index
      %c0 = constant 0 : index
      scf.for %k = %c0 to %c$2 step %c1 {
        %a = memref.load %arg2[%15,%k] : memref<$1x$2xf32>
        %b = memref.load %arg1[%k,%13] : memref<$2x$3xf32>
        %c = memref.load %arg0[%15, %13] : memref<$1x$3xf32>
        %mul = mulf %a, %b : f32
        %add = addf %c, %mul : f32
        memref.store %add, %arg0[%15,%13] : memref<$1x$3xf32>
      }

      gpu.return
    }
  }
  func private @print_memref_f32(memref<*xf32>)
  func private @print_flops(f64)
  func private @rtclock() -> (f64)
}"
