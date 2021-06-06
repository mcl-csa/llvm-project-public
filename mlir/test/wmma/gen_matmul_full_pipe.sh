echo "func @main() {
  %c16_f = constant 16.0e+00 : f16
  %c16 = constant 16 : index
  %f0 = constant 0.0e+00 : f32
  %A = memref.alloc() : memref<$1x$2xf16>
  %B = memref.alloc() : memref<$2x$3xf16>
  %C = memref.alloc() : memref<$1x$3xf32>
  
  %c0 = constant 0 : index
  %c1 = constant 1 : index
  %M = memref.dim %A, %c0: memref<$1x$2xf16> 
  %N = memref.dim %B, %c1: memref<$2x$3xf16> 
  %K = memref.dim %A, %c1: memref<$1x$2xf16> 
  
  // Intialize the Input matrix A.
  scf.for %arg0 = %c0 to %M step %c1 {
    scf.for %arg1 = %c0 to %K step %c1 {
      %a0 = remi_signed %arg0, %c16 : index
      %a1 = remi_signed %arg1, %c16 : index
      %add = addi %a0, %a1 : index
      %addm = remi_signed %add, %c16 : index
      %add_int = index_cast %addm : index to i16
      %add_float = sitofp %add_int : i16 to f16
      memref.store %add_float, %A[%arg0, %arg1] : memref<$1x$2xf16>
    }
  }

  // Intialize the Input matrix B.
  scf.for %arg0 = %c0 to %K step %c1 {
    scf.for %arg1 = %c0 to %N step %c1 {
      %b0 = remi_signed %arg0, %c16 : index
      %b1 = remi_signed %arg1, %c16 : index
      %add = addi %b0, %b1 : index
      %addm = remi_signed %add, %c16 : index
      %add_int = index_cast %addm : index to i16
      %add_float = sitofp %add_int : i16 to f16
      memref.store %add_float, %B[%arg0, %arg1] : memref<$2x$3xf16>
    }
  }

  // Intialize C matrix with zeros.
  scf.for %arg0 = %c0 to %M step %c1 {
    scf.for %arg1 = %c0 to %N step %c1 {
      memref.store %f0, %C[%arg0, %arg1] : memref<$1x$3xf32>
    }
  }
  
  %t0 = gpu.wait async

  // Allocate actual input/output arrays on device.
  %gpu_A, %t5 = gpu.alloc async [%t0] () : memref<$1x$2xf16>
  %gpu_B, %t6 = gpu.alloc async [%t0] () : memref<$2x$3xf16>
  %gpu_C, %t7 = gpu.alloc async [%t0] () : memref<$1x$3xf32>

  // Copy initialized arrays from host to device.
  %t2 = gpu.memcpy async [%t0] %gpu_A, %A : memref<$1x$2xf16>, memref<$1x$2xf16>
  %t3 = gpu.memcpy async [%t0] %gpu_B, %B : memref<$2x$3xf16>, memref<$2x$3xf16>
  %t4 = gpu.memcpy async [%t0] %gpu_C, %C : memref<$1x$3xf32>, memref<$1x$3xf32>
  
  gpu.wait [%t0]

  // Main kernel
  affine.for %i = 0 to %M {
    affine.for %j = 0 to %N {
      affine.for %l = 0 to %K {
        %a = affine.load %gpu_A[%i, %l] : memref<$1x$2xf16>
        %b = affine.load %gpu_B[%l, %j] : memref<$2x$3xf16>
        %c = affine.load %gpu_C[%i, %j] : memref<$1x$3xf32>
        %p = mulf %a, %b : f16
        %q = fpext %p : f16 to f32
        %co = addf %c, %q : f32
        affine.store %co, %gpu_C[%i, %j] : memref<$1x$3xf32>
      }
    }
  }
  
  %t1 = gpu.wait async 
  // Copy result matrix back to host for printing.
  %t8 = gpu.memcpy async [%t1] %C, %gpu_C : memref<$1x$3xf32>, memref<$1x$3xf32>
  gpu.wait[%t8]
  
  %22 = memref.cast %C : memref<$1x$3xf32> to memref<*xf32>"
  
  if [[ $4 -eq 1 ]]
  then
    echo "call @print_memref_f32(%22) : (memref<*xf32>) -> ()" 
  fi

  echo "return
}

gpu.module @initC_kernel {
  gpu.func @initC_kernel(%arg0: memref<$1x$3xf32>) kernel {
    %c1 = constant 1 : index
    %c0 = constant 0 : index
    %dim0 = memref.dim %arg0, %c0 : memref<$1x$3xf32>
    %dim1 = memref.dim %arg0, %c1 : memref<$1x$3xf32>
    %cst = constant 0.000000e+00 : f32
    %mul = muli %dim0, %dim1 : index

    scf.for %arg1 = %c0 to %dim1 step %c1{
      scf.for %arg2 = %c0 to %dim0 step %c1{
        memref.store %cst, %arg0[%arg1,%arg2] : memref<$1x$3xf32> 
      }   
    }

    gpu.return
  }
}

func private @print_memref_f32(memref<*xf32>)
func private @print_flops(f64)
func private @rtclock() -> (f64)"
