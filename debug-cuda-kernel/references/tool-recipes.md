# Tool Recipes

## Compute Sanitizer

```bash
compute-sanitizer --tool memcheck ./app
compute-sanitizer --tool memcheck --leak-check full ./app
compute-sanitizer --tool memcheck --track-stream-ordered-races all ./app
compute-sanitizer --tool racecheck ./app
compute-sanitizer --tool synccheck ./app
compute-sanitizer --tool initcheck ./app
```

Compile for source attribution (example):

```bash
nvcc -lineinfo -Xcompiler -rdynamic -o app app.cu
```

Use coredump generation when needed:

```bash
compute-sanitizer --tool memcheck --generate-coredump yes ./app
cuda-gdb target cudacore <generated_cudacore_file>
```

## cuda-gdb

Debug build:

```bash
nvcc -g -G -o app app.cu
cuda-gdb ./app
```

Useful commands inside cuda-gdb:
- `info cuda kernels`
- `info cuda threads`
- `info cuda warps`
- `info cuda barriers`

Optional exception attach flow:

```bash
export CUDA_DEVICE_WAITS_ON_EXCEPTION=1
```

## Nsight Systems (nsys)

Quick timeline capture:

```bash
nsys profile --trace=cuda,nvtx -d 20 --sample=none --cpuctxsw=none -o nsys_report ./app
```

Capture only profiled region:

```bash
nsys profile --trace=cuda,nvtx --capture-range=cudaProfilerApi -o nsys_range ./app
```

## Nsight Compute (ncu)

Basic kernel profiling:

```bash
ncu -o ncu_basic ./app
```

Deep profiling:

```bash
ncu --set full -o ncu_full ./app
```

Filter to one kernel invocation:

```bash
ncu -k regex:my_kernel -s 10 -c 1 -o ncu_one ./app
```

For mandatory concurrent kernels in multi-process scenarios, consider lockstep launch coordination:

```bash
ncu --lockstep-kernel-launch --target-processes all -o ncu_lockstep ./app
```

## Minimal runtime error-checking macros

```cpp
#define CUDA_CHECK(call) do { \
  cudaError_t err = (call); \
  if (err != cudaSuccess) { \
    fprintf(stderr, "CUDA error %s:%d: %s\\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
    abort(); \
  } \
} while (0)

#define CUDA_KERNEL_CHECK() do { \
  cudaError_t err = cudaGetLastError(); \
  if (err != cudaSuccess) { \
    fprintf(stderr, "Kernel launch error %s:%d: %s\\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
    abort(); \
  } \
} while (0)
```
