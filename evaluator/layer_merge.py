"""
GeLaCo — Layer Merge Operations
================================
Implements differential weight merging from the GeLaCo paper (arXiv:2507.10059).

Merge formula (§3.1, faithful to LaCo, Yang et al., 2024b):
    θ*_base = θ_base + Σ_{k=1}^{end-base} (θ_{base+k} - θ_base)

Layer mapping ψ (§3.1) is implemented iteratively, per-operation, so overlapping
or redundant merges in a genotype collapse correctly to no-ops via the paper's
dynamic remapping (rather than the simpler cumulative-offset shortcut, which
diverges from the paper for overlapping operations).
"""

import torch
import torch.nn as nn

from config import NUM_ORIGINAL_LAYERS


def apply_differential_merge(model, base: int, end: int) -> None:
    """
    Apply differential weight merging to collapse layers [base..end] into
    the base layer.

    The paper's formula (faithful reproduction, NOT algebraically simplified):
        θ*_base = θ_base + Σ_{k=1}^{end-base} (θ_{base+k} - θ_base)

    This modifies the model IN-PLACE.

    Args:
        model: HuggingFace LlamaForCausalLM model.
        base: Base layer index (0-indexed).
        end: End layer index (inclusive, 0-indexed).
    """
    layers = model.model.layers
    num_layers = len(layers)

    assert 0 <= base < num_layers, f"base={base} out of range [0, {num_layers})"
    assert base < end < num_layers, f"end={end} must be > base={base} and < {num_layers}"

    num_merged = end - base  # Number of layers being collapsed into base
    print(f"[LayerMerge] Merging layers [{base}..{end}] into layer {base}")
    print(f"[LayerMerge] Collapsing {num_merged} layers (keeping base layer {base})")

    base_layer = layers[base]

    # Apply differential weight merging to every parameter in the base layer
    # θ*_base = θ_base + Σ_{k=1}^{num_merged} (θ_{base+k} - θ_base)
    #
    # IMPORTANT: We snapshot θ_base BEFORE the loop so that all differentials
    # are computed relative to the ORIGINAL base, not a partially-modified one.
    # This is faithful to the paper's formula and safe for multi-merge scenarios.
    with torch.no_grad():
        base_params = dict(base_layer.named_parameters())

        for param_name, base_param in base_params.items():
            # Snapshot the original base parameter values
            original_base_data = base_param.data.clone()
            differential_sum = torch.zeros_like(base_param.data)

            for k in range(1, num_merged + 1):
                layer_k = layers[base + k]
                # Get the corresponding parameter from layer base+k
                param_k = dict(layer_k.named_parameters())[param_name]

                # Compute the differential: (θ_{base+k} - θ_base)
                # Using the ORIGINAL θ_base snapshot, not the modified one
                differential = param_k.data - original_base_data
                differential_sum += differential

            # Apply: θ*_base = θ_base + Σ differentials
            base_param.data = original_base_data + differential_sum

    print(f"[LayerMerge] Differential weight merging applied to layer {base}")


def remove_collapsed_layers(model, base: int, end: int) -> None:
    """
    Remove collapsed layers (base+1 to end) from the model after merging.

    This modifies the model IN-PLACE:
    - Removes layers from model.model.layers (nn.ModuleList)
    - Updates model.config.num_hidden_layers

    Args:
        model: HuggingFace LlamaForCausalLM model.
        base: Base layer index.
        end: End layer index (inclusive).
    """
    num_to_remove = end - base
    original_count = len(model.model.layers)

    # Build new layer list without the collapsed layers
    # Keep: layers[0..base] + layers[end+1..L-1]
    remaining_layers = []
    for i in range(len(model.model.layers)):
        if i <= base or i > end:
            remaining_layers.append(model.model.layers[i])

    # Replace the ModuleList
    model.model.layers = nn.ModuleList(remaining_layers)

    # Update config
    new_count = len(model.model.layers)
    model.config.num_hidden_layers = new_count

    # Re-index remaining layers so cache updates work properly
    for idx, layer in enumerate(model.model.layers):
        layer.layer_idx = idx
        if hasattr(layer, "self_attn") and hasattr(layer.self_attn, "layer_idx"):
            layer.self_attn.layer_idx = idx

    print(f"[LayerMerge] Removed {num_to_remove} layers: "
          f"{original_count} → {new_count} layers")


def _replay_psi(
    merge_ops: list[tuple[int, int, int]],
    num_original_layers: int,
) -> tuple[list[int], list[tuple[int, int]]]:
    """
    Replay the paper's dynamic ψ mapping over the given merge operations.

    Returns:
        psi: final ψ array of length num_original_layers (original → compressed idx).
        effective_ops: list of (compressed_base, compressed_end) pairs actually applied,
                       in genotype order. Used for canonical caching.
    """
    L = num_original_layers
    psi = list(range(L))
    effective_ops: list[tuple[int, int]] = []

    for b, e, a in merge_ops:
        if a != 1 or e <= b or b < 0 or e >= L:
            continue
        cb, ce = psi[b], psi[e]
        if ce <= cb:
            # Layers already collapsed by a prior op — this op degenerates to no-op
            continue
        effective_ops.append((cb, ce))
        delta = ce - cb
        for j in range(L):
            if b <= j <= e:
                psi[j] = cb
            elif j > e:
                psi[j] = max(0, psi[j] - delta)

    return psi, effective_ops


def apply_merge_operations(
    model,
    merge_ops: list[tuple[int, int, int]],
    num_original_layers: int = NUM_ORIGINAL_LAYERS,
) -> dict[int, int]:
    """
    Apply a genotype of merge operations using the paper's iterative ψ mapping (§3.1).

    Each operation is (b, e, a):
    - b, e: base and end layer indices in ORIGINAL model indexing.
    - a: activation flag, 1 to apply, 0 to skip.

    Operations are applied in genotype order (NOT sorted). For each active op,
    the current ψ is used to map (b, e) → (cb, ce) in the *currently compressed*
    model. If the mapped interval has zero length (because a prior op already
    collapsed it), the op is silently skipped — matching the paper's behavior.

    Returns:
        layer_mapping: dict mapping original layer index → compressed layer index.
    """
    L = num_original_layers
    psi = list(range(L))
    applied = 0

    for b, e, a in merge_ops:
        if a != 1 or e <= b or b < 0 or e >= L:
            continue
        cb, ce = psi[b], psi[e]
        if ce <= cb:
            continue

        apply_differential_merge(model, cb, ce)
        remove_collapsed_layers(model, cb, ce)
        delta = ce - cb
        for j in range(L):
            if b <= j <= e:
                psi[j] = cb
            elif j > e:
                psi[j] = max(0, psi[j] - delta)
        applied += 1

    if applied == 0:
        print("[LayerMerge] No effective merge operations applied")
    else:
        print(f"[LayerMerge] Applied {applied} effective operation(s); "
              f"final model has {len(model.model.layers)} layers")

    return {j: psi[j] for j in range(L)}


def canonical_effective_ops(
    merge_ops: list[tuple[int, int, int]],
    num_original_layers: int = NUM_ORIGINAL_LAYERS,
) -> list[tuple[int, int]]:
    """
    Return the list of effective (compressed_base, compressed_end) merges that
    a given genotype would actually apply, in order. This is the canonical
    representation used for cache deduplication: many genotypes collapse to
    the same effective op list and must share a cache entry.
    """
    _, effective = _replay_psi(merge_ops, num_original_layers)
    return effective


def build_layer_mapping(
    num_original_layers: int,
    merge_ops: list[tuple[int, int, int]],
) -> dict[int, int]:
    """
    Compute ψ without touching a model. Accepts full (b, e, a) triples so the
    caller does not have to pre-filter.
    """
    psi, _ = _replay_psi(merge_ops, num_original_layers)
    return {j: psi[j] for j in range(num_original_layers)}


if __name__ == "__main__":
    # Single-merge regression: matches prior behavior.
    print("Testing ψ for merge [(5, 7, 1)]:")
    mapping = build_layer_mapping(32, [(5, 7, 1)])
    assert mapping[0] == 0
    assert mapping[4] == 4
    assert mapping[5] == 5
    assert mapping[6] == 5
    assert mapping[7] == 5
    assert mapping[8] == 6
    assert mapping[31] == 29
    print("  ok")

    # Overlapping ops: second op must degenerate to a no-op (paper §3.1).
    print("Testing ψ for overlapping [(5, 7, 1), (6, 7, 1)]:")
    psi, eff = _replay_psi([(5, 7, 1), (6, 7, 1)], 32)
    assert eff == [(5, 7)], f"expected one effective merge, got {eff}"
    assert psi[6] == 5 and psi[7] == 5
    assert psi[8] == 6
    print("  ok")

    # Disjoint ops: both apply, ψ composes correctly.
    print("Testing ψ for disjoint [(5, 7, 1), (10, 12, 1)]:")
    psi, eff = _replay_psi([(5, 7, 1), (10, 12, 1)], 32)
    assert len(eff) == 2
    assert eff[0] == (5, 7)
    assert eff[1] == (10 - 2, 12 - 2)  # second op mapped through first
    assert psi[12] == 8
    assert psi[13] == 9
    print("  ok")

    # Inactive flag: no ops applied.
    print("Testing ψ with all a=0:")
    psi, eff = _replay_psi([(5, 7, 0)], 32)
    assert eff == []
    assert psi == list(range(32))
    print("  ok")

    print("All ψ assertions passed.")
