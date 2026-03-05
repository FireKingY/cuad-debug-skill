---
name: debug-cuda-kernel
description: Debug CUDA kernel correctness and performance issues using a structured workflow with Compute Sanitizer, cuda-gdb, Nsight Systems, Nsight Compute, NVTX, printf/assert, and environment controls. Use when users report CUDA errors (illegal memory access, misaligned address, launch failure), nondeterministic outputs/races, synchronization bugs, incorrect indexing, or GPU performance regressions.
---

# CUDA Kernel Debug Skill

Follow this workflow in order; stop early when you have strong evidence.

## 1) Stabilize and reproduce

1. Ask for: minimal repro command, GPU model, driver version, CUDA Toolkit version, and whether issue is correctness or performance.
2. Make execution deterministic where possible:
   - Fix random seeds.
   - Reduce data size while preserving failure.
   - Reduce streams/concurrency.
3. Force async errors to surface near the source during debugging:
   - `CUDA_LAUNCH_BLOCKING=1`
4. Require strict host-side error checks in user code:
   - Check every CUDA Runtime API return.
   - After each kernel launch, call `cudaGetLastError()`.
   - Add strategic `cudaDeviceSynchronize()` while debugging.

If user cannot share code, ask for a minimal kernel snippet plus launch config and expected vs actual behavior.

## 2) Choose the correct first tool

Default sequence for correctness:
1. `compute-sanitizer --tool memcheck` first.
2. If memcheck is clean and output still wrong:
   - `--tool racecheck` for shared-memory hazards.
   - `--tool synccheck` for barrier/mask misuse.
   - `--tool initcheck` for uninitialized reads.

For performance regressions:
1. `nsys` to find where time is spent end-to-end.
2. `ncu` to inspect one/few hot kernels deeply.

Use recipes from:
- [playbook](references/playbook.md)
- [tool-recipes](references/tool-recipes.md)

## 3) Classify findings quickly

Map symptom to likely root cause:
- `illegal memory access` / `misaligned address` -> indexing, bounds, alignment, lifetime.
- Intermittent wrong results -> race/synchronization/memory-order assumptions.
- Hangs or barrier errors -> divergent control around `__syncthreads()` / invalid `__syncwarp(mask)`.
- Stable numeric mismatch at boundaries -> index flattening/stride/tile mapping errors.
- Throughput drop -> uncoalesced accesses, occupancy limits, branch divergence, excessive sync.

## 4) Use intrusive probes carefully

Use only when sanitizer/profiler evidence is insufficient:
- `assert()` to fail fast on invariants (indices, masks, bounds).
- `printf()` only for sampled threads (e.g., one lane per warp/block).

Warn user that both can perturb execution; assert/trap can invalidate CUDA context.

## 5) Repair and regression gate

After proposing a fix:
1. Re-run the original repro.
2. Re-run the same sanitizer/profiler command that exposed the issue.
3. Verify across at least two launch sizes (small + production-like).
4. For performance fixes, compare before/after key metrics (runtime, achieved bandwidth, occupancy-related counters).

## 6) Reporting format for users

Always return:
1. **Observed symptom**
2. **Evidence** (tool output category + source location + thread/block/kernel identity)
3. **Root cause hypothesis**
4. **Patch strategy** (minimal code change)
5. **Validation commands**
6. **Risk notes** (tool overhead, perturbation, platform caveats)

## 7) Constraints and caveats

- Prefer Compute Sanitizer over legacy `cuda-memcheck` (deprecated/removed in modern toolkits).
- Explain that `racecheck` coverage is primarily shared-memory hazards; clean result does not prove global-memory race absence.
- Mention profiler overhead and replay effects (`ncu` multi-pass can perturb concurrency).
- On Windows, mention TDR risk for long sanitizer/profiler runs.

## 8) Ready-to-copy triage prompt

Use this prompt when kicking off a CUDA debug session:

"Provide: failing command, GPU/driver/CUDA versions, compile flags, minimal kernel + launch config, and expected vs actual output. I will run: (1) memcheck, then (2) race/sync/init checks if needed, then (3) nsys/ncu for performance. I will return evidence, likely root cause, and minimal patch guidance."
