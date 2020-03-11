#!/bin/bash

LLC=/home/eddied/Work/llvm-project/build/bin/llc

# MLIR -> CPU
oec-opt --stencil-call-inlining ./hdiff.mlir > hdiff-inline.mlir
oec-opt --stencil-call-inlining --stencil-shape-inference --stencil-shape-shift ./hdiff.mlir > hdiff-shift.mlir
oec-opt --stencil-call-inlining --stencil-shape-inference --stencil-shape-shift --stencil-shape-shift --convert-stencil-to-standard --lower-affine ./hdiff.mlir > hdiff-affine.mlir
oec-opt --stencil-call-inlining --stencil-shape-inference --stencil-shape-shift --stencil-shape-shift --convert-stencil-to-standard --lower-affine --canonicalize --convert-loop-to-std ./hdiff.mlir > hdiff-std.mlir
oec-opt --stencil-call-inlining --stencil-shape-inference --stencil-shape-shift --stencil-shape-shift --convert-stencil-to-standard --lower-affine --canonicalize --convert-loop-to-std --convert-std-to-llvm='emit-c-wrappers=1' ./hdiff.mlir > hdiff-llvm.mlir
mlir-translate --mlir-to-llvmir hdiff-llvm.mlir -o hdiff-llvm.ll
$LLC hdiff-llvm.ll -o hdiff.s
clang++ hdiff.s -O3 -c -o hdiff.o

# MLIR -> GPU
oec-opt --stencil-call-inlining --stencil-shape-inference --stencil-shape-shift --cse --convert-stencil-to-standard --lower-affine --canonicalize --convert-loops-to-gpu --gpu-block-dims=2 --gpu-thread-dims=1 --gpu-kernel-outlining --cse --canonicalize ./hdiff.mlir > hdiff-gpu.mlir
#oec-opt --stencil-call-inlining --stencil-shape-inference --stencil-shape-shift --cse --convert-stencil-to-standard --lower-affine --canonicalize --convert-loops-to-gpu --gpu-block-dims=2 --gpu-thread-dims=1 --gpu-kernel-outlining --cse --canonicalize --stencil-gpu-to-cubin ./hdiff.mlir > hdiff-cubin.mlir
oec-opt --stencil-call-inlining --stencil-shape-inference --stencil-shape-shift --cse --convert-stencil-to-standard --lower-affine --canonicalize --convert-loops-to-gpu --gpu-block-dims=2 --gpu-thread-dims=1 --gpu-kernel-outlining --cse --canonicalize --stencil-gpu-to-cubin --stencil-gpu-to-cuda --cse --canonicalize --disable-pass-threading ./hdiff.mlir > hdiff-cuda.mlir
mlir-translate --mlir-to-llvmir hdiff-cuda.mlir -o hdiff-cuda.ll
$LLC hdiff-cuda.ll -o hdiff.s
clang++ hdiff.s -O3 -lcudart -lcuda -c -o hdiff.o
