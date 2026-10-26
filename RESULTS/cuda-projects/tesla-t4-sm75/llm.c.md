# karpathy/llm.c on tesla-t4-sm75

Commit `f1e2ace651495b74ae22d45d1723443fd00ecd3a`. Back to [LEDGER.md](../LEDGER.md).

Skipped before running (2): dev/cuda/attention_forward.cu (needs cuDNN); dev/cuda/nccl_all_reduce.cu (needs NCCL)

## pass-only (20)

ran clean but output varies between runs (timings): judge by exit status only

- `dev/cuda/adamw.cu`
- `dev/cuda/attention_backward.cu` [timing loops cut (1 repeat, fewer sizes): correctness only] [rescued: restarted with a longer budget; attempts 300s -> rc 124, 900s -> rc 0]
- `dev/cuda/classifier_fused.cu` [timing loops cut (1 repeat, fewer sizes): correctness only]
- `dev/cuda/crossentropy_forward.cu`
- `dev/cuda/crossentropy_softmax_backward.cu`
- `dev/cuda/encoder_backward.cu`
- `dev/cuda/encoder_forward.cu`
- `dev/cuda/fused_residual_forward.cu`
- `dev/cuda/gelu_backward.cu`
- `dev/cuda/gelu_forward.cu`
- `dev/cuda/global_norm.cu`
- `dev/cuda/layernorm_backward.cu`
- `dev/cuda/layernorm_forward.cu`
- `dev/cuda/matmul_backward.cu`
- `dev/cuda/matmul_backward_bias.cu`
- `dev/cuda/matmul_forward.cu` [timing loops cut (1 repeat, fewer sizes): correctness only]
- `dev/cuda/permute.cu`
- `dev/cuda/residual_forward.cu`
- `dev/cuda/softmax_forward.cu`
- `dev/cuda/trimat_forward.cu`
