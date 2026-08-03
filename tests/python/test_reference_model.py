"""Behavioral tests for the deterministic PyTorch reference transformer."""

from dataclasses import FrozenInstanceError

import pytest
import torch
from torch import nn

from forgeir.reference import (
    CausalSelfAttention,
    RMSNorm,
    TinyTransformerConfig,
    create_deterministic_input,
    create_deterministic_model,
)


def _small_config() -> TinyTransformerConfig:
    return TinyTransformerConfig(
        vocabulary_size=32,
        hidden_size=8,
        intermediate_size=16,
        num_heads=2,
        sequence_length=4,
        batch_size=1,
    )


def test_default_configuration_and_immutability() -> None:
    config = TinyTransformerConfig()
    assert config.as_dict() == {
        "vocabulary_size": 256,
        "hidden_size": 128,
        "intermediate_size": 384,
        "num_heads": 4,
        "sequence_length": 32,
        "batch_size": 2,
        "epsilon": 1e-5,
        "dtype": "float32",
        "seed": 42,
    }
    with pytest.raises(FrozenInstanceError):
        config.hidden_size = 64


@pytest.mark.parametrize(
    "field",
    [
        "vocabulary_size",
        "hidden_size",
        "intermediate_size",
        "num_heads",
        "sequence_length",
        "batch_size",
    ],
)
def test_configuration_rejects_nonpositive_dimensions(field: str) -> None:
    with pytest.raises(ValueError, match="must be positive"):
        TinyTransformerConfig(**{field: 0})


def test_configuration_rejects_invalid_head_partition() -> None:
    with pytest.raises(ValueError, match="divisible"):
        TinyTransformerConfig(hidden_size=10, num_heads=4)


def test_configuration_rejects_unsupported_dtype() -> None:
    with pytest.raises(ValueError, match="unsupported dtype"):
        TinyTransformerConfig(dtype="float16")


def test_configuration_rejects_unsafe_dimensions_and_allocation_budget() -> None:
    with pytest.raises(ValueError, match="safe upper bound"):
        TinyTransformerConfig(sequence_length=4_097)
    with pytest.raises(ValueError, match="parameter element budget"):
        TinyTransformerConfig(
            vocabulary_size=65_536,
            hidden_size=1_024,
            intermediate_size=16_384,
            num_heads=16,
        )


def test_rmsnorm_matches_equation() -> None:
    norm = RMSNorm(hidden_size=4, epsilon=1e-5)
    with torch.no_grad():
        norm.weight.copy_(torch.tensor([1.0, 0.5, 1.5, 2.0]))
    input_tensor = torch.tensor([[[1.0, -2.0, 3.0, -4.0]]], dtype=torch.float32)
    mean_square = input_tensor.square().mean(dim=-1, keepdim=True)
    expected = input_tensor * torch.rsqrt(mean_square + norm.epsilon) * norm.weight
    torch.testing.assert_close(norm(input_tensor), expected, rtol=0.0, atol=1e-7)


def test_attention_shape_and_probability_normalization() -> None:
    config = _small_config()
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(config.seed)
        attention = CausalSelfAttention(config).eval()
    hidden_states = torch.arange(32, dtype=torch.float32).reshape(1, 4, 8) / 32.0
    output, weights = attention.forward_with_weights(hidden_states)
    assert output.shape == (1, 4, 8)
    assert weights.shape == (1, 2, 4, 4)
    torch.testing.assert_close(weights.sum(dim=-1), torch.ones(1, 2, 4), rtol=0.0, atol=1e-7)


def test_attention_is_strictly_causal() -> None:
    config = _small_config()
    attention = CausalSelfAttention(config).eval()
    hidden_states = torch.arange(32, dtype=torch.float32).reshape(1, 4, 8) / 32.0
    _, weights = attention.forward_with_weights(hidden_states)
    forbidden = torch.triu(torch.ones(4, 4, dtype=torch.bool), diagonal=1)
    assert torch.count_nonzero(weights[..., forbidden]) == 0

    model = create_deterministic_model(config)
    input_ids = create_deterministic_input(config)
    changed = input_ids.clone()
    changed[0, -1] = (changed[0, -1] + 1) % config.vocabulary_size
    with torch.inference_mode():
        baseline_output = model(input_ids)
        changed_output = model(changed)
    torch.testing.assert_close(baseline_output[:, :-1], changed_output[:, :-1], rtol=0.0, atol=0.0)


def test_parameters_inputs_and_repeated_outputs_are_deterministic() -> None:
    config = _small_config()
    first_model = create_deterministic_model(config)
    second_model = create_deterministic_model(config)
    first_input = create_deterministic_input(config)
    second_input = create_deterministic_input(config)
    assert torch.equal(first_input, second_input)
    for name, parameter in first_model.state_dict().items():
        assert torch.equal(parameter, second_model.state_dict()[name])
    with torch.inference_mode():
        first_output = first_model(first_input)
        repeated_output = first_model(first_input)
        second_output = second_model(second_input)
    assert torch.equal(first_output, repeated_output)
    assert torch.equal(first_output, second_output)


def test_output_is_finite_and_changes_with_input() -> None:
    config = _small_config()
    model = create_deterministic_model(config)
    input_ids = create_deterministic_input(config)
    changed = input_ids.clone()
    changed[0, 0] = (changed[0, 0] + 1) % config.vocabulary_size
    with torch.inference_mode():
        baseline = model(input_ids)
        changed_output = model(changed)
    assert torch.isfinite(baseline).all()
    assert not torch.equal(baseline, changed_output)


def test_model_is_evaluation_only_and_contains_no_dropout() -> None:
    model = create_deterministic_model(_small_config())
    assert not model.training
    assert not any(isinstance(module, nn.Dropout) for module in model.modules())
