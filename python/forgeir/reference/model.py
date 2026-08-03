"""Explicit deterministic PyTorch reference transformer modules."""

from __future__ import annotations

import math
from typing import cast

import torch
from torch import Tensor, nn

from forgeir.reference.config import TinyTransformerConfig


class RMSNorm(nn.Module):
    """Root-mean-square normalization with a learned elementwise scale."""

    def __init__(self, hidden_size: int, epsilon: float) -> None:
        super().__init__()
        if hidden_size <= 0:
            raise ValueError("hidden_size must be positive")
        if not math.isfinite(epsilon) or epsilon <= 0.0:
            raise ValueError("epsilon must be finite and positive")
        self.hidden_size = hidden_size
        self.epsilon = epsilon
        self.weight = nn.Parameter(torch.ones(hidden_size, dtype=torch.float32))

    def forward(self, input_tensor: Tensor) -> Tensor:
        if not input_tensor.is_floating_point():
            raise TypeError("RMSNorm input must have a floating-point dtype")
        if input_tensor.ndim == 0:
            raise ValueError("RMSNorm input must have at least one dimension")
        if input_tensor.shape[-1] != self.hidden_size:
            raise ValueError(
                f"RMSNorm expected final dimension {self.hidden_size}, "
                f"received {input_tensor.shape[-1]}"
            )
        mean_square = input_tensor.square().mean(dim=-1, keepdim=True)
        normalized = input_tensor * torch.rsqrt(mean_square + self.epsilon)
        return normalized * self.weight


class CausalSelfAttention(nn.Module):
    """Causal multi-head self-attention expressed through primitive operations."""

    causal_mask: Tensor

    def __init__(self, config: TinyTransformerConfig) -> None:
        super().__init__()
        self.config = config
        self.query_projection = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.key_projection = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.value_projection = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.output_projection = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.scale = 1.0 / math.sqrt(config.head_size)
        causal_mask = torch.tril(
            torch.ones(config.sequence_length, config.sequence_length, dtype=torch.bool)
        )
        self.register_buffer("causal_mask", causal_mask, persistent=False)

    @staticmethod
    def stable_softmax(scores: Tensor) -> Tensor:
        if not scores.is_floating_point():
            raise TypeError("attention scores must have a floating-point dtype")
        maximum = scores.amax(dim=-1, keepdim=True)
        exponentials = torch.exp(scores - maximum)
        return exponentials / exponentials.sum(dim=-1, keepdim=True)

    def _validate_hidden_states(self, hidden_states: Tensor) -> tuple[int, int]:
        if hidden_states.ndim != 3:
            raise ValueError("attention input must have shape [batch, sequence, hidden]")
        batch_size, sequence_length, hidden_size = hidden_states.shape
        if batch_size <= 0 or batch_size > self.config.batch_size:
            raise ValueError(f"attention batch dimension must be in [1, {self.config.batch_size}]")
        if sequence_length <= 0 or sequence_length > self.config.sequence_length:
            raise ValueError(
                f"attention sequence dimension must be in [1, {self.config.sequence_length}]"
            )
        if hidden_size != self.config.hidden_size:
            raise ValueError(f"attention hidden dimension must be {self.config.hidden_size}")
        if hidden_states.dtype != self.config.torch_dtype:
            raise TypeError(f"attention input dtype must be {self.config.dtype}")
        if hidden_states.device.type != "cpu":
            raise ValueError("Milestone 2 reference attention supports CPU tensors only")
        return batch_size, sequence_length

    def _split_heads(self, projected: Tensor, batch_size: int, sequence_length: int) -> Tensor:
        return projected.view(
            batch_size,
            sequence_length,
            self.config.num_heads,
            self.config.head_size,
        ).transpose(1, 2)

    def forward_with_weights(self, hidden_states: Tensor) -> tuple[Tensor, Tensor]:
        batch_size, sequence_length = self._validate_hidden_states(hidden_states)
        query = self._split_heads(
            self.query_projection(hidden_states), batch_size, sequence_length
        )
        key = self._split_heads(
            self.key_projection(hidden_states), batch_size, sequence_length
        )
        value = self._split_heads(
            self.value_projection(hidden_states), batch_size, sequence_length
        )

        scores = torch.matmul(query, key.transpose(-2, -1)) * self.scale
        mask = self.causal_mask[:sequence_length, :sequence_length].view(
            1, 1, sequence_length, sequence_length
        )
        masked_scores = scores.masked_fill(~mask, -torch.inf)
        attention_weights = self.stable_softmax(masked_scores)
        attended = torch.matmul(attention_weights, value)
        merged = attended.transpose(1, 2).contiguous().view(
            batch_size, sequence_length, self.config.hidden_size
        )
        return self.output_projection(merged), attention_weights

    def forward(self, hidden_states: Tensor) -> Tensor:
        output, _ = self.forward_with_weights(hidden_states)
        return output


class TinyTransformerBlock(nn.Module):
    """Pre-normalized attention and GELU MLP residual block."""

    def __init__(self, config: TinyTransformerConfig) -> None:
        super().__init__()
        self.attention_norm = RMSNorm(config.hidden_size, config.epsilon)
        self.attention = CausalSelfAttention(config)
        self.mlp_norm = RMSNorm(config.hidden_size, config.epsilon)
        self.mlp = nn.Sequential(
            nn.Linear(config.hidden_size, config.intermediate_size, bias=False),
            nn.GELU(approximate="none"),
            nn.Linear(config.intermediate_size, config.hidden_size, bias=False),
        )

    def forward(self, hidden_states: Tensor) -> Tensor:
        normalized = cast(Tensor, self.attention_norm(hidden_states))
        attention_output = cast(Tensor, self.attention(normalized))
        attention_residual = hidden_states + attention_output
        mlp_output = cast(Tensor, self.mlp(self.mlp_norm(attention_residual)))
        return attention_residual + mlp_output


class TinyTransformerModel(nn.Module):
    """Token embedding followed by one deterministic transformer block."""

    def __init__(self, config: TinyTransformerConfig) -> None:
        super().__init__()
        self.config = config
        self.token_embedding = nn.Embedding(config.vocabulary_size, config.hidden_size)
        self.block = TinyTransformerBlock(config)

    def forward(self, input_ids: Tensor) -> Tensor:
        expected_shape = (self.config.batch_size, self.config.sequence_length)
        if tuple(input_ids.shape) != expected_shape:
            raise ValueError(f"input_ids must have shape {expected_shape}")
        if input_ids.dtype != torch.int64:
            raise TypeError("input_ids must have dtype int64")
        if input_ids.device.type != "cpu":
            raise ValueError("Milestone 2 reference model supports CPU tensors only")
        has_negative_id = bool(torch.any(input_ids < 0))
        has_oversized_id = bool(torch.any(input_ids >= self.config.vocabulary_size))
        if has_negative_id or has_oversized_id:
            raise ValueError("input_ids contains a token outside the configured vocabulary")
        embedded = cast(Tensor, self.token_embedding(input_ids))
        return cast(Tensor, self.block(embedded))


def create_deterministic_model(
    config: TinyTransformerConfig | None = None,
) -> TinyTransformerModel:
    """Construct an evaluation-mode model without changing the caller's RNG state."""
    resolved_config = config or TinyTransformerConfig()
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(resolved_config.seed)
        model = TinyTransformerModel(resolved_config)
    model.eval()
    return model


def create_deterministic_input(
    config: TinyTransformerConfig | None = None,
) -> Tensor:
    """Create deterministic token IDs on CPU from the configuration seed."""
    resolved_config = config or TinyTransformerConfig()
    generator = torch.Generator(device="cpu")
    generator.manual_seed(resolved_config.seed)
    return torch.randint(
        low=0,
        high=resolved_config.vocabulary_size,
        size=(resolved_config.batch_size, resolved_config.sequence_length),
        dtype=torch.int64,
        generator=generator,
        device="cpu",
    )
