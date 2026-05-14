"""
GeLaCo Prototype — Module-wise Similarity Fitness
====================================================
Implements the fitness function from Section 3.2 of the GeLaCo paper.

The fitness consists of three components:
1. Attention Block Similarity: cosine similarity of q/k/v/o_proj activations
2. FFN Similarity: cosine similarity of gate/up/down_proj activations
3. Hidden State Similarity: cosine similarity of final hidden states

All similarities are computed on ACTIVATIONS during inference (not raw weights),
using forward hooks to capture intermediate outputs.
"""

import torch
import torch.nn.functional as F

from config import MAX_SENTENCE_LENGTH


# =============================================================================
# Module names to hook for each similarity component
# =============================================================================
ATTENTION_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj"]
FFN_MODULES = ["gate_proj", "up_proj", "down_proj"]


def _cosine_similarity(t1: torch.Tensor, t2: torch.Tensor) -> float:
    """
    Compute cosine similarity between two tensors.
    Tensors are flattened to 1D vectors before computing similarity.

    Args:
        t1: First tensor (any shape).
        t2: Second tensor (must be same shape as t1 after flattening).

    Returns:
        Cosine similarity as a float in [-1, 1].
    """
    v1 = t1.flatten().float()
    v2 = t2.flatten().float()
    return F.cosine_similarity(v1.unsqueeze(0), v2.unsqueeze(0)).item()


class ActivationCollector:
    """
    Registers forward hooks on attention and FFN submodules of a model
    to collect their output activations during inference.

    Activations are stored keyed by (layer_index, module_name).
    """

    def __init__(self, model, model_name: str = "model"):
        """
        Args:
            model: HuggingFace LlamaForCausalLM model.
            model_name: Label for logging.
        """
        self.model = model
        self.model_name = model_name
        self.activations: dict[tuple[int, str], torch.Tensor] = {}
        self._hooks: list = []

    def register_hooks(self) -> None:
        """Register forward hooks on all attention and FFN submodules."""
        self.clear()

        for layer_idx, layer in enumerate(self.model.model.layers):
            # Attention submodules
            for module_name in ATTENTION_MODULES:
                module = getattr(layer.self_attn, module_name)
                hook = module.register_forward_hook(
                    self._make_hook(layer_idx, module_name)
                )
                self._hooks.append(hook)

            # FFN/MLP submodules
            for module_name in FFN_MODULES:
                module = getattr(layer.mlp, module_name)
                hook = module.register_forward_hook(
                    self._make_hook(layer_idx, module_name)
                )
                self._hooks.append(hook)

    def _make_hook(self, layer_idx: int, module_name: str):
        """Create a hook function that stores the output activation."""
        def hook_fn(module, input, output):
            # Store output activation (detach to avoid gradient tracking)
            self.activations[(layer_idx, module_name)] = output.detach()
        return hook_fn

    def clear(self) -> None:
        """Remove all hooks and clear stored activations."""
        for hook in self._hooks:
            hook.remove()
        self._hooks = []
        self.activations = {}

    def clear_activations(self) -> None:
        """Clear stored activations (keep hooks registered)."""
        self.activations = {}


def compute_attention_similarity(
    orig_activations: dict,
    comp_activations: dict,
    layer_mapping: dict[int, int],
    num_original_layers: int,
) -> float:
    """
    Compute attention block similarity between original and compressed models.

    For each original layer, compare its attention module activations
    (q/k/v/o_proj) against the corresponding compressed layer's activations.
    Average across all modules within each layer, then across all layers.

    Args:
        orig_activations: Activations from the original model.
        comp_activations: Activations from the compressed model.
        layer_mapping: original layer idx → compressed layer idx.
        num_original_layers: Number of layers in the original model.

    Returns:
        Average attention similarity score.
    """
    layer_similarities = []

    for orig_layer in range(num_original_layers):
        comp_layer = layer_mapping[orig_layer]
        module_sims = []

        for module_name in ATTENTION_MODULES:
            orig_key = (orig_layer, module_name)
            comp_key = (comp_layer, module_name)

            if orig_key in orig_activations and comp_key in comp_activations:
                sim = _cosine_similarity(
                    orig_activations[orig_key],
                    comp_activations[comp_key],
                )
                module_sims.append(sim)

        if module_sims:
            layer_similarities.append(sum(module_sims) / len(module_sims))

    if not layer_similarities:
        return 0.0
    return sum(layer_similarities) / len(layer_similarities)


def compute_ffn_similarity(
    orig_activations: dict,
    comp_activations: dict,
    layer_mapping: dict[int, int],
    num_original_layers: int,
) -> float:
    """
    Compute FFN (MLP) similarity between original and compressed models.

    Same approach as attention: compare gate/up/down_proj activations,
    average within layers, then across layers.

    Args:
        orig_activations: Activations from the original model.
        comp_activations: Activations from the compressed model.
        layer_mapping: original layer idx → compressed layer idx.
        num_original_layers: Number of layers in the original model.

    Returns:
        Average FFN similarity score.
    """
    layer_similarities = []

    for orig_layer in range(num_original_layers):
        comp_layer = layer_mapping[orig_layer]
        module_sims = []

        for module_name in FFN_MODULES:
            orig_key = (orig_layer, module_name)
            comp_key = (comp_layer, module_name)

            if orig_key in orig_activations and comp_key in comp_activations:
                sim = _cosine_similarity(
                    orig_activations[orig_key],
                    comp_activations[comp_key],
                )
                module_sims.append(sim)

        if module_sims:
            layer_similarities.append(sum(module_sims) / len(module_sims))

    if not layer_similarities:
        return 0.0
    return sum(layer_similarities) / len(layer_similarities)


def compute_hidden_state_similarity(
    orig_hidden: torch.Tensor,
    comp_hidden: torch.Tensor,
) -> float:
    """
    Compute cosine similarity between final hidden state representations.

    Args:
        orig_hidden: Last hidden state from original model [batch, seq, hidden].
        comp_hidden: Last hidden state from compressed model [batch, seq, hidden].

    Returns:
        Cosine similarity score.
    """
    return _cosine_similarity(orig_hidden, comp_hidden)


def compute_fitness(
    original_model,
    compressed_model,
    layer_mapping: dict[int, int],
    tokenizer,
    sentences: list[str],
    num_original_layers: int,
    device: str = "cuda",
    verbose: bool = True,
) -> dict:
    """
    Compute the GeLaCo module-wise similarity fitness.

    For each calibration sentence:
    1. Run inference on both models with hooks collecting activations
    2. Compute attention similarity (q/k/v/o_proj across all layers)
    3. Compute FFN similarity (gate/up/down_proj across all layers)
    4. Compute hidden state similarity (final hidden states)
    5. Average the three components

    The final fitness is averaged over all sentences.

    Args:
        original_model: Original (unmodified) LlamaForCausalLM.
        compressed_model: Compressed (merged) LlamaForCausalLM.
        layer_mapping: original layer idx → compressed layer idx.
        tokenizer: HuggingFace tokenizer.
        sentences: List of calibration sentences.
        num_original_layers: Original model's layer count.
        device: Torch device.

    Returns:
        Dict with fitness scores:
        {
            "attention_similarity": float,
            "ffn_similarity": float,
            "hidden_state_similarity": float,
            "fitness": float,  # average of the three
            "per_sentence": list of per-sentence dicts,
        }
    """
    if verbose:
        print(f"\n[Fitness] Computing module-wise similarity fitness")
        print(f"[Fitness] Sentences: {len(sentences)}, "
              f"Original layers: {num_original_layers}, "
              f"Compressed layers: {len(compressed_model.model.layers)}")

    # Set up activation collectors
    orig_collector = ActivationCollector(original_model, "original")
    comp_collector = ActivationCollector(compressed_model, "compressed")

    orig_collector.register_hooks()
    comp_collector.register_hooks()

    all_attn_sims = []
    all_ffn_sims = []
    all_hs_sims = []
    per_sentence_results = []

    original_model.eval()
    compressed_model.eval()

    with torch.no_grad():
        for sent_idx, sentence in enumerate(sentences):
            # Tokenize
            inputs = tokenizer(
                sentence,
                return_tensors="pt",
                max_length=MAX_SENTENCE_LENGTH,
                truncation=True,
                padding=False,
            ).to(device)

            # Clear previous activations
            orig_collector.clear_activations()
            comp_collector.clear_activations()

            # Forward pass on original model
            orig_outputs = original_model(**inputs, output_hidden_states=True)
            orig_hidden = orig_outputs.hidden_states[-1]  # Final hidden state

            # Forward pass on compressed model
            comp_outputs = compressed_model(**inputs, output_hidden_states=True)
            comp_hidden = comp_outputs.hidden_states[-1]  # Final hidden state

            # Compute three similarity components
            attn_sim = compute_attention_similarity(
                orig_collector.activations,
                comp_collector.activations,
                layer_mapping,
                num_original_layers,
            )

            ffn_sim = compute_ffn_similarity(
                orig_collector.activations,
                comp_collector.activations,
                layer_mapping,
                num_original_layers,
            )

            hs_sim = compute_hidden_state_similarity(orig_hidden, comp_hidden)

            # Average of three components
            sentence_fitness = (attn_sim + ffn_sim + hs_sim) / 3.0

            all_attn_sims.append(attn_sim)
            all_ffn_sims.append(ffn_sim)
            all_hs_sims.append(hs_sim)

            sent_result = {
                "sentence_idx": sent_idx,
                "attention_similarity": attn_sim,
                "ffn_similarity": ffn_sim,
                "hidden_state_similarity": hs_sim,
                "fitness": sentence_fitness,
            }
            per_sentence_results.append(sent_result)

            if verbose:
                preview = sentence[:60] + "..." if len(sentence) > 60 else sentence
                print(f"  [Sentence {sent_idx}] attn={attn_sim:.4f} "
                      f"ffn={ffn_sim:.4f} hs={hs_sim:.4f} "
                      f"fitness={sentence_fitness:.4f} | {preview}")

    # Clean up hooks
    orig_collector.clear()
    comp_collector.clear()

    # Average over all sentences
    avg_attn = sum(all_attn_sims) / len(all_attn_sims) if all_attn_sims else 0.0
    avg_ffn = sum(all_ffn_sims) / len(all_ffn_sims) if all_ffn_sims else 0.0
    avg_hs = sum(all_hs_sims) / len(all_hs_sims) if all_hs_sims else 0.0
    avg_fitness = (avg_attn + avg_ffn + avg_hs) / 3.0

    results = {
        "attention_similarity": avg_attn,
        "ffn_similarity": avg_ffn,
        "hidden_state_similarity": avg_hs,
        "fitness": avg_fitness,
        "per_sentence": per_sentence_results,
    }

    if verbose:
        print(f"\n[Fitness] === FINAL RESULTS ===")
        print(f"[Fitness]   Attention similarity:    {avg_attn:.6f}")
        print(f"[Fitness]   FFN similarity:          {avg_ffn:.6f}")
        print(f"[Fitness]   Hidden state similarity: {avg_hs:.6f}")
        print(f"[Fitness]   Overall fitness:         {avg_fitness:.6f}")

    return results
