# siboehm/SGEMM_CUDA on tesla-t4-sm75

Commit `5a7dcc513d951ba764d51bc9d587b3163f3a894d`. Back to [LEDGER.md](../LEDGER.md).

Skipped before running (0): none

## pass-only (12)

ran clean but output varies between runs (timings): judge by exit status only

- `sgemm.cu kernel 0`
- `sgemm.cu kernel 1`
- `sgemm.cu kernel 2`
- `sgemm.cu kernel 3`
- `sgemm.cu kernel 4`
- `sgemm.cu kernel 5`
- `sgemm.cu kernel 6`
- `sgemm.cu kernel 7`
- `sgemm.cu kernel 8`
- `sgemm.cu kernel 9`
- `sgemm.cu kernel 10`
- `sgemm.cu kernel 12`

## run-fail (1)

built, non-zero exit

- `sgemm.cu kernel 11` -- `Divergence! Should 18.27, Is 11.89 (Diff  6.38) at 132` (raw: `tesla-t4-sm75/results/SGEMM_CUDA.json`)
