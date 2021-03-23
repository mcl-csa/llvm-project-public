#!/bin/bash

# Usage: ./run_matmul_test.sh --problem_size_m 1024 --problem_size_k 1024 --problem_size_n 1024 --thread_block_tile_m 64 --thread_block_tile_n 64 --thread_block_tile_k 16 --warp_tile_m 32 --warp_tile_n 32 --warp_tile_k 16 --verify 1

# Define the default parameters.
problem_size_m=${problem_size_m:-4096}
problem_size_k=${problem_size_k:-4096}
problem_size_n=${problem_size_n:-4096}
thread_block_tile_m=${thread_block_tile_m:-128}
thread_block_tile_n=${thread_block_tile_n:-128}
thread_block_tile_k=${thread_block_tile_k:-16}
warp_tile_m=${warp_tile_m:-64}
warp_tile_n=${warp_tile_n:-64}
warp_tile_k=${warp_tile_k:-16}
load_store_width=${load_store_width:-128}
print_output=${print_output:-0}
verify=${verify:-0}

# Get the passed parameter values if any.
while [ $# -gt 0 ]; do
  if [[ $1 == *"--"* ]]
  then
    param="${1/--/}"
    declare $param="$2"
  fi

  shift
done

# Set print_output if verfiy is enabled.
if [[ $verify -eq 1 ]]
then
  print_output=1
fi

MLIR_OPT="../../../build/bin/mlir-opt"
MLIR_CUDA_RUNNER="../../../build/bin/mlir-cuda-runner"
MLIR_RUNTIME_LIB_DIR="../../../build/lib"
MLIR_RUNTIME_LIBS="--shared-libs=$MLIR_RUNTIME_LIB_DIR/libmlir_runner_utils.so --shared-libs=$MLIR_RUNTIME_LIB_DIR/libmlir_cuda_runtime.so --shared-libs=$MLIR_RUNTIME_LIB_DIR/libmlir_c_runner_utils.so"

# Run the pipe end to end.
echo "Generating and running matmul (optimized)"
./gen_matmul_full_pipe.sh $problem_size_m $problem_size_k $problem_size_n $print_output \
  | $MLIR_OPT \
  --canonicalize \
  --affine-loop-tile="num-tiling-levels=2 tile-sizes=$thread_block_tile_m,$thread_block_tile_n,$thread_block_tile_k,$warp_tile_m,$warp_tile_n,$warp_tile_k relative-indexing=true" \
  --canonicalize \
  -test-gpu-matmul-fast-buffer-placement="matrices=A,B global-allocation=true" \
  --canonicalize \
  --test-specialize-affine-matmul-for-wmma="accum=f32 load-store-width=$load_store_width" \
  --canonicalize \
  --test-collapse-affine-parallel \
  --canonicalize \
  --lower-affine \
  --test-gpu-matmul-parallel-loop-mapping \
  --canonicalize \
  --test-convert-matmul-parallel-loops-to-gpu \
  --gpu-kernel-outlining \
  --test-gpu-mark-global-as-workgroup-memory \
  --canonicalize \
  --cse \
  --convert-scf-to-std \
  | $MLIR_CUDA_RUNNER -O3 --max-reg-per-thread=255 --sm=sm_75 --index-bitwidth=32 -gpu-to-cubin="gpu-binary-annotation=nvvm.cubin" -gpu-to-llvm="gpu-binary-annotation=nvvm.cubin" $MLIR_RUNTIME_LIBS --entry-point-result=void > full_pipe.out

if [[ $verify -eq 1 ]]
then
  echo "Generating and running matmul naive (unoptimized)"
  ./gen_matmul_naive.sh $problem_size_m $problem_size_k $problem_size_n | $MLIR_OPT --convert-scf-to-std | $MLIR_CUDA_RUNNER -O3 --max-reg-per-thread=200 --sm=sm_75 --index-bitwidth=32 -gpu-to-cubin="gpu-binary-annotation=nvvm.cubin" -gpu-to-llvm="gpu-binary-annotation=nvvm.cubin" $MLIR_RUNTIME_LIBS --entry-point-result=void > naive.out

  # Delete first line in the output which contains irrelecant memref info.
  sed '1d' full_pipe.out > tmpfile; mv tmpfile full_pipe.out
  sed '1d' naive.out > tmpfile; mv tmpfile naive.out

  # Compare the output.
  cmp full_pipe.out naive.out
fi
