import logging
import math
import types

import folder_paths
import torch
import torch.nn.functional as F
from comfy.lora import model_lora_keys_unet
from comfy.lora_convert import convert_lora
from comfy.utils import load_torch_file

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# LoRA loading helpers
# ---------------------------------------------------------------------------

def load_lora_weights(lora_name, model):
    """Load a LoRA file and return {model_key: (lora_up, lora_down, alpha)}."""
    lora_path = folder_paths.get_full_path_or_raise("loras", lora_name)
    lora_data = load_torch_file(lora_path, safe_load=True)
    lora_data = convert_lora(lora_data)

    diff_model = model.model
    key_map = model_lora_keys_unet(diff_model, {})

    from comfy.lora import load_lora, weight_adapter
    loaded = load_lora(lora_data, key_map, log_missing=False)

    deltas = {}
    for model_key, adapter in loaded.items():
        if adapter is None or not hasattr(adapter, "weights"):
            continue
        w = adapter.weights  # (lora_up, lora_down, alpha, mid, dora_scale, reshape)
        lora_up = w[0]
        lora_down = w[1]
        alpha = w[2]
        mid = w[3]
        dora_scale = w[4]

        if dora_scale is not None:
            log.warning("[PromptRelay] DoRA LoRAs not yet supported for temporal blending, skipping key: %s", model_key)
            continue

        if mid is not None:
            # LoCon: pre-compute combined down matrix
            # mid: [r, r2, k, k] style, reshape and combine
            mid_t = mid.transpose(0, 1).flatten(start_dim=1)
            down_t = lora_down.transpose(0, 1).flatten(start_dim=1)
            final_shape = [lora_down.shape[1], lora_down.shape[0], mid.shape[2], mid.shape[3]]
            lora_down = (
                torch.mm(down_t, mid_t)
                .reshape(final_shape)
                .transpose(0, 1)
            )

        alpha_scale = (alpha / lora_down.shape[0]) if alpha is not None else 1.0
        deltas[model_key] = (lora_up, lora_down, alpha_scale)

    if not deltas:
        raise ValueError(f"LoRA '{lora_name}' has no compatible weights for the current model")

    log.info("[PromptRelay] Loaded LoRA '%s': %d layer keys", lora_name, len(deltas))
    return deltas


def build_lora_assignment(num_segments, lora_slot_map, extend_last):
    """Map each segment index to a list of LoRA column indices.

    lora_slot_map: list of lists of LoRA column indices, one per slot.
                   An empty list means that slot has no LoRAs.
    extend_last: if True, the last assigned LoRA list fills any remaining segments.
    """
    if num_segments == 0:
        raise ValueError("At least one segment is required")

    assignment = []
    last_lora_indices = []

    for seg_idx in range(num_segments):
        if seg_idx < len(lora_slot_map):
            slot = lora_slot_map[seg_idx]
            if slot:
                assignment.append(list(slot))
                last_lora_indices = list(slot)
            else:
                assignment.append(
                    list(last_lora_indices) if extend_last and last_lora_indices else []
                )
        else:
            assignment.append(
                list(last_lora_indices) if extend_last and last_lora_indices else []
            )

    return assignment


def build_blend_tensor(segment_lengths, tokens_per_frame, lora_assignment, num_loras, device, epsilon=1e-3):
    """Build a [total_tokens, num_loras] blend tensor with Gaussian transitions.

    Each token receives a weight per LoRA based on Gaussian proximity to that
    LoRA's segment.  Inside the segment window the weight is 1.0; outside it
    decays as exp(-((d-window)^2) / (2*sigma^2)).  Smaller *epsilon* gives
    sharper transitions (same formula as the prompt-relay attention mask).

    All LoRAs assigned to the same segment share the same temporal weight
    profile (they stack additively in the forward pass).  When a deduplicated
    LoRA appears in adjacent segments, ``torch.max`` keeps the weight at 1.0
    across the transition instead of doubling it.
    """
    sigma = 1.0 / math.log(1.0 / epsilon) if 0 < epsilon < 1 else 0.1448

    total_frames = sum(segment_lengths)
    total_tokens = total_frames * tokens_per_frame
    blend = torch.zeros(total_tokens, num_loras, device=device)

    # Collect metadata for segments that have LoRAs
    seg_info = []  # (midpoint, window, lora_idx)
    frame_cursor = 0
    for seg_idx, seg_len in enumerate(segment_lengths):
        lora_indices = lora_assignment[seg_idx]
        if lora_indices and seg_len > 0:
            midpoint = (2 * frame_cursor + seg_len) // 2
            window = max(seg_len // 2 - 2, 0)
            for lora_idx in lora_indices:
                seg_info.append((midpoint, window, lora_idx))
        frame_cursor += seg_len

    if not seg_info:
        return blend

    frames = torch.arange(total_frames, device=device, dtype=torch.float32)
    two_sigma_sq = 2.0 * sigma * sigma

    for midpoint, window, lora_idx in seg_info:
        d = (frames - midpoint).abs()
        weights = torch.where(
            d <= window,
            torch.ones_like(d),
            torch.exp(-((d - window).pow(2)) / two_sigma_sq),
        )
        token_weights = weights.repeat_interleave(tokens_per_frame)
        # Use max so that a deduplicated LoRA shared across adjacent segments
        # stays at weight 1.0 instead of summing to 2.0 at the boundary.
        blend[:, lora_idx] = torch.max(blend[:, lora_idx], token_weights)

    return blend


# ---------------------------------------------------------------------------
# Layer patching
# ---------------------------------------------------------------------------

def _make_patched_linear(orig_module, lora_deltas_for_layer, blend_tensor):
    """Return a replacement forward() that applies temporal LoRA blending.

    lora_deltas_for_layer: list of (lora_up, lora_down, alpha_scale) per LoRA,
                           in the order they appear in blend_tensor columns.
    blend_tensor: [total_tokens, num_loras] on cpu (will be sliced per-batch).
    """

    def patched_forward(self_module, x):
        import comfy.model_management
        device = x.device
        dtype = x.dtype
        weight = comfy.model_management.cast_to_device(self_module.weight, device, dtype)
        bias = comfy.model_management.cast_to_device(self_module.bias, device, dtype) if self_module.bias is not None else None
        out = F.linear(x, weight, bias)
        num_tokens = x.shape[1]

        # Slice blend to match actual token count (may differ at boundary)
        if num_tokens <= blend_tensor.shape[0]:
            b = blend_tensor[:num_tokens].to(device=x.device, dtype=x.dtype)
        else:
            # Pad if somehow more tokens (shouldn't happen normally)
            b = F.pad(blend_tensor, (0, 0, 0, num_tokens - blend_tensor.shape[0])).to(
                device=x.device, dtype=x.dtype
            )

        for i, (lora_up, lora_down, alpha_scale) in enumerate(lora_deltas_for_layer):
            blend_col = b[:, i:i + 1]  # [tokens, 1]
            down_w = comfy.model_management.cast_to_device(lora_down, device, dtype)  # [rank, in_dim]
            up_w = comfy.model_management.cast_to_device(lora_up, device, dtype)  # [out_dim, rank]
            mid = x @ down_w.T  # [B, tokens, in_dim] @ [in_dim, rank] -> [B, tokens, rank]
            delta = mid @ up_w.T  # [B, tokens, rank] @ [rank, out_dim] -> [B, tokens, out_dim]
            out = out + (alpha_scale * blend_col) * delta

        return out

    return patched_forward


class _LinearPatch:
    """Descriptor that binds a patched forward onto a linear module."""

    def __init__(self, forward_fn):
        self.forward_fn = forward_fn

    def __get__(self, obj, objtype=None):
        fn = self.forward_fn
        return types.MethodType(fn, obj)


def apply_lora_temporal_patches(model_clone, lora_deltas_by_key, blend_tensor):
    """Patch every linear layer that has LoRA deltas with temporal blending.

    lora_deltas_by_key: {model_key: [(lora_up, lora_down, alpha_scale), ...]}
                        One entry per LoRA, for each model key.
    blend_tensor: [total_tokens, num_loras] blend weights.
    """
    # Collect unique module keys (strip .weight suffix)
    module_keys = set()
    for model_key in lora_deltas_by_key:
        for suffix in (".weight", ".bias"):
            if model_key.endswith(suffix):
                module_keys.add(model_key[: -len(suffix)])
                break
        else:
            module_keys.add(model_key)

    patched = 0
    for mod_key in module_keys:
        # Build per-layer LoRA list: [(lora_up, lora_down, alpha_scale), ...]
        weight_key = mod_key + ".weight" if mod_key + ".weight" in lora_deltas_by_key else mod_key
        per_lora = lora_deltas_by_key.get(weight_key, [])
        if not per_lora:
            continue

        patch_key = f"{mod_key}.forward"
        if patch_key in getattr(model_clone, "object_patches", {}):
            log.warning("[PromptRelay] Skipping already-patched key: %s", patch_key)
            continue

        # Get the actual module to inspect its shape
        try:
            module = model_clone.model
            for attr in mod_key.split("."):
                module = getattr(module, attr)
        except AttributeError:
            log.warning("[PromptRelay] Cannot resolve module for key: %s", mod_key)
            continue

        # Validate that LoRA dimensions are compatible with the module's weight.
        # Mismatches occur when key resolution maps a LoRA trained for one model
        # variant (e.g. LTX-V video-only) onto the wrong layer in another variant
        # (e.g. LTX-V AV which has extra audio attention layers).
        module_out, module_in = module.weight.shape
        for i, (lora_up, lora_down, _) in enumerate(per_lora):
            lora_in = lora_down.shape[1]
            lora_out = lora_up.shape[0]
            if lora_in != module_in or lora_out != module_out:
                log.warning(
                    "[PromptRelay] Skipping %s: LoRA #%d dims (%dx%d) incompatible "
                    "with module weight (%dx%d)",
                    mod_key, i, lora_out, lora_in, module_out, module_in,
                )
                continue

        patched_fn = _make_patched_linear(module, per_lora, blend_tensor)
        model_clone.add_object_patch(patch_key, _LinearPatch(patched_fn).__get__(module, type(module)))
        patched += 1

    log.info("[PromptRelay] Patched %d linear layers for temporal LoRA blending", patched)
