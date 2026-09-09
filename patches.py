import logging
import types

import comfy.ldm.modules.attention

log = logging.getLogger(__name__)


def _masked_attention(q, k, v, heads, mask, transformer_options={}, **kwargs):
    # Bypass wrap_attn (sage/etc may ignore masks) by calling attention_pytorch directly.
    return comfy.ldm.modules.attention.attention_pytorch(
        q, k, v, heads, mask=mask,
        _inside_attn_wrapper=True,
        transformer_options=transformer_options,
        **kwargs,
    )


def _wan_t2v_forward(self, mask_fn, x, context, transformer_options={}, **kwargs):
    q = self.norm_q(self.q(x))
    k = self.norm_k(self.k(context))
    v = self.v(context)

    mask = mask_fn(q.shape[1], k.shape[1], q.dtype, q.device, transformer_options)
    if mask is not None:
        x = _masked_attention(q, k, v, heads=self.num_heads, mask=mask,
                              transformer_options=transformer_options)
    else:
        x = comfy.ldm.modules.attention.optimized_attention(
            q, k, v, heads=self.num_heads, transformer_options=transformer_options,
        )
    return self.o(x)


def _wan_i2v_forward(self, mask_fn, x, context, context_img_len, transformer_options={}, **kwargs):
    context_img = context[:, :context_img_len]
    context_text = context[:, context_img_len:]

    q = self.norm_q(self.q(x))

    k_img = self.norm_k_img(self.k_img(context_img))
    v_img = self.v_img(context_img)
    img_x = comfy.ldm.modules.attention.optimized_attention(
        q, k_img, v_img, heads=self.num_heads, transformer_options=transformer_options,
    )

    k = self.norm_k(self.k(context_text))
    v = self.v(context_text)

    mask = mask_fn(q.shape[1], k.shape[1], q.dtype, q.device, transformer_options)
    if mask is not None:
        x = _masked_attention(q, k, v, heads=self.num_heads, mask=mask,
                              transformer_options=transformer_options)
    else:
        x = comfy.ldm.modules.attention.optimized_attention(
            q, k, v, heads=self.num_heads, transformer_options=transformer_options,
        )

    return self.o(x + img_x)


def _make_masked_override(prev_override):
    """transformer_options override that routes mask-bearing attention calls through
    attention_pytorch (sage/etc. drop arbitrary masks). Chains to a prior override
    when no mask is present so we don't clobber other backends."""
    def override(func, *args, **kwargs):
        if kwargs.get("mask") is not None:
            return comfy.ldm.modules.attention.attention_pytorch(*args, **kwargs)
        if prev_override is not None:
            return prev_override(func, *args, **kwargs)
        return func(*args, **kwargs)
    return override


def _make_ltx_mask_wrapper(underlying, mask_fn):
    """Wrap an existing LTX cross-attn forward (the default `CrossAttention.forward`
    or another node's patch — e.g. KJNodes NAG), injecting PromptRelay's additive
    mask via the `mask` kwarg the upstream signature already accepts.

    `underlying` must already be bound to its module — callable as
    `underlying(x, context=..., mask=..., ...)`.
    """
    def wrapped(_self, x, context=None, mask=None, pe=None, k_pe=None, transformer_options={}):
        if context is not None:
            pr_mask = mask_fn(x.shape[1], context.shape[1], x.dtype, x.device, transformer_options)
            if pr_mask is not None:
                mask = pr_mask if mask is None else mask + pr_mask

        if mask is not None:
            prev = transformer_options.get("optimized_attention_override")
            transformer_options = {
                **transformer_options,
                "optimized_attention_override": _make_masked_override(prev),
            }

        return underlying(
            x, context=context, mask=mask, pe=pe, k_pe=k_pe,
            transformer_options=transformer_options,
        )

    wrapped._promptrelay_wrapper = True
    return wrapped


class _CrossAttnPatch:
    """Descriptor that binds (impl, mask_fn) as a method onto a Wan cross-attn module."""

    def __init__(self, impl, mask_fn):
        self.impl = impl
        self.mask_fn = mask_fn

    def __get__(self, obj, objtype=None):
        impl, mask_fn = self.impl, self.mask_fn

        def wrapped(self_module, *args, **kwargs):
            return impl(self_module, mask_fn, *args, **kwargs)

        return types.MethodType(wrapped, obj)


def detect_model_type(model):
    """Return (arch, patch_size, temporal_stride) for latent geometry.

    temporal_stride is the VAE's pixel→latent temporal compression factor,
    used to convert user-facing pixel frame counts to latent frames.
    """
    diff_model = model.model.diffusion_model

    if hasattr(diff_model, "patch_size") and not hasattr(diff_model, "patchifier"):
        raw = diff_model.patch_size
        patch_size = (1, raw, raw) if isinstance(raw, int) else tuple(raw)
        return "wan", patch_size, 4

    if hasattr(diff_model, "patchifier"):
        return "ltx", (1, 1, 1), int(diff_model.vae_scale_factors[0])

    raise ValueError(
        f"Unsupported model type: {type(diff_model).__name__}. "
        f"Currently supports Wan and LTX models."
    )


def _check_unpatched(model_clone, key):
    if key in getattr(model_clone, "object_patches", {}):
        raise RuntimeError(
            f"PromptRelay: cross-attention forward at '{key}' is already patched by "
            "another node. Stacking is not supported for this architecture — remove "
            "the conflicting node."
        )


def apply_patches(model_clone, arch, mask_fn):
    diffusion_model = model_clone.get_model_object("diffusion_model")

    if arch == "wan":
        from comfy.ldm.wan.model import WanI2VCrossAttention
        for idx, block in enumerate(diffusion_model.blocks):
            key = f"diffusion_model.blocks.{idx}.cross_attn.forward"
            _check_unpatched(model_clone, key)
            cross_attn = block.cross_attn
            impl = _wan_i2v_forward if isinstance(cross_attn, WanI2VCrossAttention) else _wan_t2v_forward
            model_clone.add_object_patch(key, _CrossAttnPatch(impl, mask_fn).__get__(cross_attn, cross_attn.__class__))
        return

    if arch == "ltx":
        to = model_clone.model_options["transformer_options"]
        to["promptrelay_mask_fn"] = mask_fn

        for idx, block in enumerate(diffusion_model.transformer_blocks):
            for attr in ("attn2", "audio_attn2"):
                module = getattr(block, attr, None)
                if module is None:
                    continue
                key = f"diffusion_model.transformer_blocks.{idx}.{attr}.forward"
                # get_model_object returns the prior patch if present, else the default bound forward.
                underlying = model_clone.get_model_object(key)
                wrapper = _make_ltx_mask_wrapper(underlying, mask_fn)
                model_clone.add_object_patch(key, types.MethodType(wrapper, module))
        return

    raise ValueError(f"Unknown model arch: {arch}")
