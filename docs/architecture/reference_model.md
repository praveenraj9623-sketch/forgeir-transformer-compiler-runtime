# Deterministic reference transformer

## Scope

Milestone 2 defines a small CPU PyTorch model that serves as a numerical reference. It does not
export a graph and does not depend on ForgeIR IR, compiler passes, runtime execution, GPU kernels,
MLIR, pretrained weights, or external data.

`TinyTransformerConfig` is a frozen dataclass. The default configuration is batch size 2, sequence
length 32, vocabulary size 256, hidden size 128, intermediate size 384, four attention heads,
float32 arithmetic, epsilon 1e-5, and seed 42. Configuration construction rejects nonpositive
dimensions, unsupported dtypes, invalid head partitioning, non-finite epsilon, unsafe seed values,
per-dimension limits, and total parameter or activation element budgets before model allocation.

Only float32 is supported in this milestone. Inputs are int64 token identifiers with exact configured
batch and sequence dimensions. Unsupported devices, shapes, token identifiers, and dtypes fail
explicitly.

## Model

For token identifiers `t`, the embedding table produces the initial hidden states
`X = Embedding(t)`. The block uses a pre-normalized attention residual followed by a pre-normalized
GELU MLP residual. All linear projections omit bias, dropout is absent, and the model is placed in
evaluation mode.

### RMSNorm

For a hidden vector `x` of width `d`, a learned scale vector `g`, and positive epsilon `epsilon`,
RMSNorm is

    rms(x) = sqrt((1 / d) * sum_i(x_i^2) + epsilon)
    RMSNorm(x)_i = g_i * x_i / rms(x).

The implementation computes the mean square along the final tensor dimension and multiplies by
`rsqrt(mean_square + epsilon)`.

### Causal multi-head self-attention

For normalized hidden states `N` and explicit projection matrices,

    Q = N W_Q,    K = N W_K,    V = N W_V.

The tensors are reshaped into `h` heads of width `d_h = d / h`. For query position `i` and key
position `j`, the scaled score is

    S_ij = (Q_i dot K_j) / sqrt(d_h) + M_ij,

where the causal mask is

    M_ij = 0         when j <= i,
    M_ij = -infinity when j > i.

Softmax is implemented explicitly and stabilized by subtracting each row maximum:

    P_ij = exp(S_ij - max_k S_ik) / sum_k exp(S_ik - max_k S_ik).

The per-head result is `P V`. Heads are concatenated and passed through the output projection
`W_O`. The implementation does not call PyTorch scaled-dot-product-attention helpers.

### GELU MLP

The MLP expands from hidden width `d` to intermediate width `m`, applies the exact GELU, and projects
back to `d`:

    GELU(z) = 0.5 * z * (1 + erf(z / sqrt(2))),
    MLP(x) = GELU(x W_1) W_2.

PyTorch `nn.GELU(approximate="none")` implements this exact form.

### Residual equations

With independent learned RMSNorm scales for the attention and MLP paths, the block computes

    H = X + Attention(RMSNorm_1(X)),
    Y = H + MLP(RMSNorm_2(H)).

The first equation is the attention residual connection; the second is the GELU MLP residual
connection.

## Determinism

Model construction occurs inside a forked CPU random-number-generator scope seeded from the
configuration, so it does not alter the caller's RNG state. Input identifiers use a separate CPU
generator with the same configured seed. Reference inference runs under `torch.inference_mode()`.
There are no stochastic layers. The generated model remains on CPU and in evaluation mode.

## Reference artifacts

Run the generator from a source checkout with the source package and built Milestone 1 binding on
`PYTHONPATH`:

    python -m forgeir.reference.generate

An explicit destination can be selected with `--output-dir`. One run writes:

- `configuration.json`: schema version and immutable model configuration.
- `input_tensor.npz`: deterministic int64 `input_ids`.
- `weight_tensors.npz`: every sorted model state tensor.
- `expected_output.npz`: expected float32 hidden states.
- `tensor_metadata.json`: artifact, archive key, semantic name, shape, dtype, and raw-content SHA-256
  for every serialized tensor.
- `manifest.sha256`: SHA-256 for every configuration, tensor archive, and metadata file.

The manifest intentionally does not hash itself. Verification requires the exact artifact filename
set and uses constant-time digest comparison. All JSON is emitted with sorted keys and a final
newline; NPZ member order is stable.
