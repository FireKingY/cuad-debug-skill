# CUDA Kernel Debug Playbook

## Correctness decision tree

1. Run with `CUDA_LAUNCH_BLOCKING=1`.
2. Run `compute-sanitizer --tool memcheck`.
3. If memcheck reports errors:
   - Fix bounds/alignment/lifetime first.
   - Re-run memcheck until clean.
4. If memcheck is clean but outputs are unstable:
   - Run `--tool racecheck`.
   - Run `--tool synccheck`.
   - Run `--tool initcheck`.
5. If still unresolved:
   - Add minimal `assert()` invariants in kernel.
   - Use targeted `printf()` sampling.
   - Escalate to `cuda-gdb` / coredump workflow.

## Bug pattern checklist

### 1) Memory errors
- Validate `i = blockIdx.x * blockDim.x + threadIdx.x` and all multidim flattening formulas.
- Confirm every global memory access is guarded by bounds checks.
- Check alignment assumptions for vectorized loads/stores.
- Confirm pointer lifetime with stream-ordered allocators (`cudaMallocAsync/cudaFreeAsync`).

### 2) Race conditions
- Shared-memory writes must be visible before reads (`__syncthreads()` as needed).
- Warp-level cooperation must use correct `__syncwarp(mask)` mask.
- Confirm atomic operation use and scope where required.

### 3) Synchronization issues
- Never place `__syncthreads()` on paths not reached by all block threads.
- Check cooperative groups barriers for full participation requirements.

### 4) Indexing errors
- Verify tile offsets, strides, pitch, and leading dimensions.
- Validate boundary tiles separately (tail processing).

### 5) Performance regressions
- Confirm coalesced global accesses.
- Check occupancy vs register/shared-memory pressure.
- Quantify branch divergence and stall reasons in hot kernels.
- Remove unnecessary synchronizations and tiny kernel launches.
