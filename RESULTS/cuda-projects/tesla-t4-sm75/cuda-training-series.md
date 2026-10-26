# olcf/cuda-training-series on tesla-t4-sm75

Commit `9cd77d539d85ee3457e09a083ba299b787b7a16e`. Back to [LEDGER.md](../LEDGER.md).

Skipped before running (15): exercises/hw1/hello.cu (exercise template); exercises/hw1/matrix_mul.cu (exercise template); exercises/hw1/vector_add.cu (exercise template); exercises/hw11/test.cu (needs MPI); exercises/hw13/axpy_cublas_with_fixme.cu (exercise template); exercises/hw13/axpy_stream_capture_with_fixme.cu (exercise template); exercises/hw2/matrix_mul_shared.cu (exercise template); exercises/hw2/stencil_1d.cu (exercise template); exercises/hw3/vector_add.cu (exercise template); exercises/hw4/matrix_sums.cu (exercise template); exercises/hw7/overlap.cu (exercise template); exercises/hw8/task1/task1.cu (exercise template); exercises/hw8/task2/task2.cu (exercise template); exercises/hw9/task1.cu (exercise template); exercises/hw9/task2.cu (exercise template)

## not-a-test (5)

an unfinished or deliberately broken teaching exercise; the repo's solution file is the test

- `exercises/hw12/task1.cu` -- deliberately broken debugging exercise (off-by-one `e <= BLOCK_SIZE`, missing barrier); task1_solution.cu is the fixed program
- `exercises/hw12/task2.cu` -- deliberately broken debugging exercise (prints -inf); task2_solution.cu prints 0.693148
- `exercises/hw13/axpy_cublas_from_scratch.cu` [same result in 1 further session(s)] -- unfinished student exercise (`thread` for `threads`, missing semicolons, wrong block count); the solution is in hw13/Solutions/
- `exercises/hw5/max_reduction.cu` -- unfinished exercise: still computes the sum; max_reduction_solution.cu computes the max
- `exercises/hw6/linked_list.cu` [same result in 1 further session(s)] -- unfinished exercise: host malloc'd list handed to the GPU; linked_list_solution.cu uses managed memory

## not-tested (1)

needs hardware, libraries or a toolkit the bench host does not have (2 GPUs, OpenGL/EGL, a newer CUDA); no verdict about the program

- `exercises/hw7/multi_solution.cu` [same result in 1 further session(s)] -- needs 2 GPUs ('invalid device ordinal'); Colab gives one

## reference (16)

ran clean, two runs agree: its stdout SHA-256 is a usable reference

- `exercises/hw1/vector_add_solution.cu` `ba67edaa4d91eee7`
- `exercises/hw12/task1_solution.cu` `2b28e627f8ab4e9d`
- `exercises/hw12/task2_solution.cu` `8a057b6501dba45e`
- `exercises/hw13/axpy_stream_capture_from_scratch.cu` `43fa75e83506d053`
- `exercises/hw13/Solutions/axpy_cublas_solution.cu` `1420bd949ee4f3df`
- `exercises/hw2/stencil_1d_solution.cu` `ecbe8676c39b35df`
- `exercises/hw3/vector_add_solution.cu` `ba67edaa4d91eee7`
- `exercises/hw4/matrix_sums_solution.cu` `7b6d9b8b2bf9edc5`
- `exercises/hw5/matrix_sums.cu` `7b6d9b8b2bf9edc5`
- `exercises/hw5/matrix_sums_solution.cu` `7b6d9b8b2bf9edc5`
- `exercises/hw5/max_reduction_solution.cu` `9fac610c535e3d1b`
- `exercises/hw5/reductions.cu` `954fa1c4339ed0a9`
- `exercises/hw6/array_inc.cu` `490c307f3bb439e5`
- `exercises/hw6/array_inc_solution.cu` [PATCHED COPY (source changed to build/run; clone untouched) -- written for CUDA <= 12 -- not a bug in the program. cudaMemPrefetchAsync(ptr, bytes, device) was replaced in CUDA 13 by a form taking a cudaMemLocation and flags (nvcc 13: 'no suitable constructor exists to convert from int to cudaMemLocation'). The copy uses the CUDA 13 form.] `490c307f3bb439e5`
- `exercises/hw6/linked_list_solution.cu` `7cbc2c1f474a83da`
- `exercises/hw9/task1_solution.cu` `de6e14733386d31b`

## pass-only (14)

ran clean but output varies between runs (timings): judge by exit status only

- `exercises/hw1/hello_solution.cu` [output differs between Colab sessions, so not a reference]
- `exercises/hw1/matrix_mul_solution.cu`
- `exercises/hw10/streams.cu`
- `exercises/hw10/streams_solution.cu`
- `exercises/hw13/axpy_stream_capture_timer.cu`
- `exercises/hw13/Solutions/axpy_stream_capture_solution.cu`
- `exercises/hw2/matrix_mul_shared_solution.cu`
- `exercises/hw7/multi.cu`
- `exercises/hw7/overlap_solution.cu`
- `exercises/hw8/task1/task1_solution.cu`
- `exercises/hw8/task2/task2_solution.cu`
- `exercises/hw8/task3/task3.cu`
- `exercises/hw8/task3/task3_solution.cu`
- `exercises/hw9/task2_solution.cu`
