#!/bin/bash

# Usage: ./run_matmul_test.sh --problem_size_m 1024 --problem_size_n 1024 --problem_size_k 1024 --thread_block_tile_m 64 --thread_block_tile_n 64 --thread_block_tile_k 16 --warp_tile_m 32 --warp_tile_n 32 --warp_tile_k 16 --padding_a 8 --padding_b 8 --verify 1

# Define the default parameters.
problem_size_m=${problem_size_m:-8192}
problem_size_n=${problem_size_n:-8192}
problem_size_k=${problem_size_k:-8192}
thread_block_tile_m=${thread_block_tile_m:-128}
thread_block_tile_n=${thread_block_tile_n:-128}
thread_block_tile_k=${thread_block_tile_k:-64}
warp_tile_m=${warp_tile_m:-32}
warp_tile_n=${warp_tile_n:-64}
warp_tile_k=${warp_tile_k:-32}
padding_a=${padding_a:-8}
padding_b=${padding_b:-8}
load_store_width=${load_store_width:-128}
reg_per_thread=${reg_per_thread:-255}
jit_opt_level=${jit_opt_level:-4}
print_output=${print_output:-0}
verify=${verify:-0}
num_iters=${num_iters:-10}
accum=${accum:-f32}

# Get the passed parameter values if any.
while [ $# -gt 0 ]; do
  if [[ $1 == *"--"* ]]
  then
    param="${1/--/}"
    declare $param="$2"
  fi

  shift
done

if [[ $verify -eq 1 ]]
then
  print_output=1
fi

# Calculate flops.
((flops = $problem_size_m * $problem_size_n * $problem_size_k * 2))

MLIR_OPT="../../../build/bin/mlir-opt"
MLIR_CPU_RUNNER="../../../build/bin/mlir-cpu-runner"
MLIR_RUNTIME_LIB_DIR="../../../build/lib"
MLIR_RUNTIME_LIBS="--shared-libs=$MLIR_RUNTIME_LIB_DIR/libmlir_runner_utils.so --shared-libs=$MLIR_RUNTIME_LIB_DIR/libmlir_cuda_runtime.so --shared-libs=$MLIR_RUNTIME_LIB_DIR/libmlir_c_runner_utils.so"

# Run the optimized version with the full pipeline.
echo "Generating and running matmul (optimized)"

if [[ "$accum" == "f32" ]]
then
  ./gen_matmul_full_pipe.sh $problem_size_m $problem_size_k $problem_size_n $print_output > matmul_opt_base.mlir
else
  ./gen_matmul_full_pipe_f16.sh $problem_size_m $problem_size_k $problem_size_n $print_output > matmul_opt_base.mlir
fi

$MLIR_OPT matmul_opt_base.mlir \
  --canonicalize \
  --affine-loop-tile="num-tiling-levels=2 tile-sizes=$thread_block_tile_m,$thread_block_tile_n,$thread_block_tile_k,$warp_tile_m,$warp_tile_n,$warp_tile_k relative-indexing=true" \
  --canonicalize \
  -test-gpu-matmul-fast-buffer-placement="matrices=A,B global-allocation=true" \
  --canonicalize \
  --test-mark-parallel-loops > tiled.mlir
  $MLIR_OPT tiled.mlir --test-specialize-affine-matmul-for-wmma="accum=$accum padding-a=$padding_a padding-b=$padding_b" \
  --canonicalize --cse > specialized.mlir
  $MLIR_OPT specialized.mlir --test-split-compute-loop \
  --canonicalize --test-gpu-matmul-barrier-insertion > pipelined.mlir
  $MLIR_OPT pipelined.mlir --test-normalize-copy-loops --canonicalize --cse > normalized.mlir
  $MLIR_OPT normalized.mlir --test-vectorize-gpu-matmul-copy-loops="load-store-width=$load_store_width" \
  --canonicalize > vectorized.mlir
  $MLIR_OPT vectorized.mlir --test-collapse-affine-parallel > parallel.mlir
  $MLIR_OPT parallel.mlir --canonicalize \
  --lower-affine > scf.mlir
  $MLIR_OPT scf.mlir --test-gpu-matmul-parallel-loop-mapping > mapped.mlir
  $MLIR_OPT mapped.mlir --canonicalize \
  --test-convert-matmul-parallel-loops-to-gpu --cse > matmul_inter.mlir
  $MLIR_OPT matmul_inter.mlir --test-unroll-and-delay-copies --canonicalize > unrolled.mlir
  $MLIR_OPT unrolled.mlir --gpu-kernel-outlining \
  --test-gpu-mark-global-as-workgroup-memory > outlined.mlir
  $MLIR_OPT outlined.mlir --canonicalize \
  --cse \
  --convert-scf-to-std > matmul_opt_final.mlir

$MLIR_OPT matmul_opt_final.mlir -pass-pipeline='gpu.module(strip-debuginfo,convert-gpu-to-nvvm{index-bitwidth=32},gpu-to-cubin{chip=sm_75 max-reg-per-thread='$reg_per_thread' cu-jit-opt-level='$jit_opt_level'})' -gpu-to-llvm | nvprof --csv $MLIR_CPU_RUNNER -O3 $MLIR_RUNTIME_LIBS --entry-point-result=void > full_pipe.out 2> nvprof.txt

# Get average execution time of `main_kernel`.
interval=$(cat nvprof.txt | (awk '/main_kernel/') | (awk -F',' '{print $5}'))

# Check if perf is reported by `nvprof`.
if [ -z "$interval" ]
then
    echo -e "\e[31merror:\e[0m" "Either codeGen failed or execTime was not given by nvprof."
    exit
fi

# Check if time is reported in millisecond, microsecond or second. If in micro or milli,
# change to second.
if grep -q ms,ms,ms nvprof.txt; then
  interval=$(echo "( $interval / 1000)" | bc -l)
else
  if grep -q us,us,us nvprof.txt; then
    interval=$(echo "( $interval / 1000000)" | bc -l)
  else
    if grep -q s,s,s nvprof.txt; then
      interval=$(echo "( $interval )" | bc -l)
    else
      echo -e "\e[31merror:\e[0m" "Unsupported time format given by nvprof."
      exit
    fi
  fi
fi
 
rm nvprof.txt

# Calculate performance.
>&2 printf '%.6f TFLOPs\n' $(echo "($flops / $interval) / 1000000000000" | bc -l)

# Delete first line in the output which is irrelevant for output verification.
sed '1d' full_pipe.out > tmpfile; mv tmpfile full_pipe.out

# Run the naive version for verification.
if [[ $verify -eq 1 ]]
then
  echo "Generating and running matmul naive (unoptimized)"
  ./gen_matmul_naive.sh $problem_size_m $problem_size_k $problem_size_n > matmul_naive.mlir
  $MLIR_OPT matmul_naive.mlir --convert-scf-to-std | $MLIR_OPT -pass-pipeline='gpu.module(strip-debuginfo,convert-gpu-to-nvvm{index-bitwidth=32},gpu-to-cubin{chip=sm_75 max-reg-per-thread=255 cu-jit-opt-level=4})' -gpu-to-llvm | $MLIR_CPU_RUNNER -O3 $MLIR_RUNTIME_LIBS --entry-point-result=void > naive.out

  echo -ne "Verifying...   "
  # Delete first line in the output which is irrelevant for output verification.
  sed '1d' naive.out > tmpfile; mv tmpfile naive.out

  # Compare the output.
  cmp full_pipe.out naive.out
  if [ $? -ne 0 ]; then
    echo -e "\e[31mFailed\e[0m" " $file"!
  else
    echo -e "\e[32mPassed\e[0m"
  fi
fi
