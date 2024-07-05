import torch
import os

from typing import List
from jaxtyping import Float
from torch import Tensor
from tqdm import tqdm

from pipeline.utils.hook_utils import add_hooks
from pipeline.model_utils.model_base import ModelBase


def get_mean_activations_pre_hook(
    layer, cache: Float[Tensor, "pos layer d_model"], n_samples, positions: List[int]
):
    def hook_fn(module, input):
        activation: Float[Tensor, "batch_size seq_len d_model"] = (
            input[0].clone().to(cache)
        )

        if torch.isnan(activation).any() or torch.isinf(activation).any():
            print(f"NaN or Inf detected in activation at layer {layer}")
            print(
                f"Max value: {activation.max().item()}, Min value: {activation.min().item()}"
            )
            print(
                f"Mean value: {activation.mean().item()}, Std deviation: {activation.std().item()}"
            )
            nan_count = torch.isnan(activation).sum().item()
            inf_count = torch.isinf(activation).sum().item()
            print(f"NaN count: {nan_count}, Inf count: {inf_count}")

        activation = torch.nan_to_num(activation, nan=0.0, posinf=1e6, neginf=-1e6)
        contribution = (1.0 / n_samples) * activation[:, positions, :].sum(dim=0)
        cache[:, layer] += contribution

    return hook_fn


def get_mean_activations(
    model,
    tokenizer,
    instructions,
    tokenize_instructions_fn,
    block_modules: List[torch.nn.Module],
    batch_size=32,
    positions=[-1],
):
    torch.cuda.empty_cache()

    n_positions = len(positions)
    n_layers = model.config.num_hidden_layers
    n_samples = len(instructions)
    d_model = model.config.hidden_size

    # we store the mean activations in high-precision to avoid numerical issues
    mean_activations = torch.zeros(
        (n_positions, n_layers, d_model), dtype=torch.float64, device=model.device
    )

    fwd_pre_hooks = []
    for layer in range(n_layers):
        hook = get_mean_activations_pre_hook(
            layer, mean_activations, n_samples, positions
        )
        fwd_pre_hooks.append((block_modules[layer], hook))

    for i in tqdm(range(0, len(instructions), batch_size)):
        inputs = tokenize_instructions_fn(instructions=instructions[i : i + batch_size])

        with add_hooks(module_forward_pre_hooks=fwd_pre_hooks, module_forward_hooks=[]):
            model(
                input_ids=inputs.input_ids.to(model.device),
                attention_mask=inputs.attention_mask.to(model.device),
            )

    return mean_activations


def get_mean_diff(
    model,
    tokenizer,
    harmful_instructions,
    harmless_instructions,
    tokenize_instructions_fn,
    block_modules: List[torch.nn.Module],
    batch_size=1,
    positions=[-1],
):
    def process_instructions(instructions):
        mean_activations = None
        for i in range(0, len(instructions), batch_size):
            batch = instructions[i : i + batch_size]
            batch_mean = get_mean_activations(
                model,
                tokenizer,
                batch,
                tokenize_instructions_fn,
                block_modules,
                batch_size=batch_size,
                positions=positions,
            )
            if mean_activations is None:
                mean_activations = batch_mean
            else:
                mean_activations += batch_mean

            print(f"Batch {i//batch_size + 1} processed:")
            print(f"Max: {batch_mean.max().item()}, Min: {batch_mean.min().item()}")
            print(f"Mean: {batch_mean.mean().item()}, Std: {batch_mean.std().item()}")

        mean_activations /= len(instructions)
        return mean_activations

    print("Processing harmful instructions:")
    mean_activations_harmful = process_instructions(harmful_instructions)
    print("\nHarmful mean activations stats:")
    print(
        f"Max: {mean_activations_harmful.max().item()}, Min: {mean_activations_harmful.min().item()}"
    )
    print(
        f"Mean: {mean_activations_harmful.mean().item()}, Std: {mean_activations_harmful.std().item()}"
    )

    print("\nProcessing harmless instructions:")
    mean_activations_harmless = process_instructions(harmless_instructions)
    print("\nHarmless mean activations stats:")
    print(
        f"Max: {mean_activations_harmless.max().item()}, Min: {mean_activations_harmless.min().item()}"
    )
    print(
        f"Mean: {mean_activations_harmless.mean().item()}, Std: {mean_activations_harmless.std().item()}"
    )

    mean_diff = mean_activations_harmful - mean_activations_harmless
    print("\nMean diff stats:")
    print(f"Max: {mean_diff.max().item()}, Min: {mean_diff.min().item()}")
    print(f"Mean: {mean_diff.mean().item()}, Std: {mean_diff.std().item()}")

    return mean_diff


def generate_directions(
    model_base: ModelBase, harmful_instructions, harmless_instructions, artifact_dir
):
    if not os.path.exists(artifact_dir):
        os.makedirs(artifact_dir)

    mean_diffs = get_mean_diff(
        model_base.model,
        model_base.tokenizer,
        harmful_instructions,
        harmless_instructions,
        model_base.tokenize_instructions_fn,
        model_base.model_block_modules,
        positions=list(range(-len(model_base.eoi_toks), 0)),
    )

    print("Final mean_diffs shape:", mean_diffs.shape)
    print("Final mean_diffs contains NaN:", mean_diffs.isnan().any())
    if mean_diffs.isnan().any():
        print(
            "Positions of NaN in final mean_diffs:", torch.nonzero(mean_diffs.isnan())
        )
    print(
        "Expected shape:",
        (
            len(model_base.eoi_toks),
            model_base.model.config.num_hidden_layers,
            model_base.model.config.hidden_size,
        ),
    )

    assert mean_diffs.shape == (
        len(model_base.eoi_toks),
        model_base.model.config.num_hidden_layers,
        model_base.model.config.hidden_size,
    )
    assert not mean_diffs.isnan().any()

    torch.save(mean_diffs, f"{artifact_dir}/mean_diffs.pt")

    return mean_diffs
