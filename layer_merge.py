"""
GeLaCo Prototype — Layer Merge Operations
===========================================
Implements the differential weight merging from the GeLaCo paper.

The paper follows LaCo (Yang et al., 2024b) for the merge formula:
   θ*_base = θ_base + Σ_{k=1}^{end-base} (θ_{base+k} - θ_base)

After merging weights, the collapsed layers are removed from the model.
"""

import copy
import torch
import torch.nn as nn


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


def apply_merge_operations(model, merge_ops: list[tuple[int, int, int]]) -> dict[int, int]:
    """
    Apply a list of merge operations sequentially and return the layer mapping.

    Each operation is (base, end, active):
    - base: starting layer index (in original model indexing)
    - end: ending layer index (inclusive, in original model indexing)
    - active: 1 to apply, 0 to skip

    Returns:
        layer_mapping: dict mapping original layer index → compressed layer index.
    """
    # Filter active operations
    active_ops = [(b, e) for b, e, a in merge_ops if a == 1 and e > b]

    if not active_ops:
        print("[LayerMerge] No active merge operations to apply")
        num_layers = len(model.model.layers)
        return {i: i for i in range(num_layers)}

    # Sort operations by base index (process from first to last)
    active_ops.sort(key=lambda x: x[0])

    print(f"[LayerMerge] Applying {len(active_ops)} merge operations: {active_ops}")

    # Build layer mapping BEFORE modifying the model
    # This maps original layer indices → compressed layer indices
    original_num_layers = len(model.model.layers)
    layer_mapping = build_layer_mapping(original_num_layers, active_ops)

    # Apply operations sequentially with index adjustment
    # As we remove layers, indices shift, so we track cumulative offset
    cumulative_removed = 0
    for base_orig, end_orig in active_ops:
        # Adjust indices for previously removed layers
        base_adj = base_orig - cumulative_removed
        end_adj = end_orig - cumulative_removed

        # Safety check
        if base_adj < 0 or end_adj >= len(model.model.layers):
            print(f"[LayerMerge] WARNING: Skipping invalid adjusted operation "
                  f"({base_adj}, {end_adj}), model has {len(model.model.layers)} layers")
            continue

        # Apply merge + remove
        apply_differential_merge(model, base_adj, end_adj)
        remove_collapsed_layers(model, base_adj, end_adj)

        num_removed = end_orig - base_orig
        cumulative_removed += num_removed

    print(f"[LayerMerge] Final model has {len(model.model.layers)} layers")
    return layer_mapping


def build_layer_mapping(
    num_original_layers: int,
    active_ops: list[tuple[int, int]],
) -> dict[int, int]:
    """
    Build the mapping ϕ from original layer indices to compressed layer indices.

    For a merge (base, end):
    - ϕ(i) = i (for i < base, adjusted for prior merges)
    - ϕ(i) = base (for base ≤ i ≤ end, all map to base)
    - ϕ(i) = i - (end - base) (for i > end, shifted down)

    This accounts for cumulative structural changes from multiple merges,
    as described in Section 3.2 of the paper.

    Args:
        num_original_layers: Total layers in original model.
        active_ops: List of (base, end) tuples, sorted by base.

    Returns:
        Dict mapping each original layer index to its compressed layer index.
    """
    mapping = {i: i for i in range(num_original_layers)}

    cumulative_shift = 0
    for base, end in active_ops:
        num_removed = end - base
        for i in range(num_original_layers):
            if base <= i <= end:
                # All merged layers map to the base layer's position
                mapping[i] = base - cumulative_shift
            elif i > end:
                # Layers after the merge are shifted down
                mapping[i] = mapping[i] - num_removed
        cumulative_shift += num_removed

    print(f"[LayerMerge] Layer mapping (original → compressed):")
    # Print a compact summary
    for i in range(num_original_layers):
        marker = " (merged)" if any(b <= i <= e for b, e in active_ops) else ""
        print(f"  layer {i:2d} → compressed layer {mapping[i]:2d}{marker}")

    return mapping


if __name__ == "__main__":
    # Quick test of layer mapping logic (no model needed)
    print("Testing layer mapping for merge (5, 7):")
    mapping = build_layer_mapping(32, [(5, 7)])
    assert mapping[0] == 0
    assert mapping[4] == 4
    assert mapping[5] == 5
    assert mapping[6] == 5  # merged into 5
    assert mapping[7] == 5  # merged into 5
    assert mapping[8] == 6  # shifted down by 2
    assert mapping[31] == 29
    print("✓ All mapping assertions passed")
