"""
ComfyUI-CacheDiT: Boogu Specialized Node
==========================================

Dedicated node for Boogu DiT model (Turbo / Edit / Raw variants).
Boogu uses a multi-stream architecture (single_stream_layers +
double_stream_layers + noise_refiner). The transformer.forward returns
a tuple of multiple tensors, which requires explicit tuple-level caching.

Supports three presets:
  • Boogu       — Turbo variant (4-8 steps)
  • Boogu-Edit  — Edit variant (20-30 steps)  
  • Boogu-Raw   — Raw variant (30-50 steps)

All variants use the same cache wrapper — only warmup/skip parameters differ.
"""

from __future__ import annotations
import logging
import time
import traceback
import torch
import comfy.model_patcher
import comfy.patcher_extension
from typing import TYPE_CHECKING, Any, Dict, Optional

if TYPE_CHECKING:
    from comfy.model_patcher import ModelPatcher

logger = logging.getLogger("ComfyUI-CacheDiT-Boogu")


# =============================================================================
# Per-transformer cache registry (multi-instance safe, like Wan/Anima nodes)
# =============================================================================

_boogu_cache_registry: Dict[int, Dict[str, Any]] = {}


def _get_or_create_cache_state(transformer_id: int) -> Dict[str, Any]:
    if transformer_id not in _boogu_cache_registry:
        _boogu_cache_registry[transformer_id] = {
            "enabled": False,
            "transformer_id": transformer_id,
            "call_count": 0,
            "skip_count": 0,
            "compute_count": 0,
            "last_result": None,
            "config": None,
            "compute_times": [],
        }
    return _boogu_cache_registry[transformer_id]


# =============================================================================
# Cache configuration
# =============================================================================

class BooguCacheConfig:
    """Configuration for Boogu cache optimization."""
    
    def __init__(
        self,
        warmup_steps: int = 2,
        skip_interval: int = 2,
        noise_scale: float = 0.0,
        verbose: bool = False,
        print_summary: bool = True,
    ):
        self.warmup_steps = warmup_steps
        self.skip_interval = skip_interval
        self.noise_scale = noise_scale
        self.verbose = verbose
        self.print_summary = print_summary
        
        # Runtime state
        self.is_enabled = False
        self.num_inference_steps: Optional[int] = None
        self.current_step: int = 0
    
    def clone(self) -> "BooguCacheConfig":
        new_config = BooguCacheConfig(
            warmup_steps=self.warmup_steps,
            skip_interval=self.skip_interval,
            noise_scale=self.noise_scale,
            verbose=self.verbose,
            print_summary=self.print_summary,
        )
        new_config.is_enabled = self.is_enabled
        new_config.num_inference_steps = self.num_inference_steps
        return new_config
    
    def reset(self):
        self.current_step = 0


# =============================================================================
# Cache wrapper
# =============================================================================

def _enable_boogu_cache(transformer, config: BooguCacheConfig):
    """Enable lightweight cache for Boogu transformer (tuple output aware)."""
    transformer_id = id(transformer)
    state = _get_or_create_cache_state(transformer_id)
    
    # Already enabled on this transformer?
    if hasattr(transformer, '_original_forward_boogu'):
        if state.get("transformer_id") == transformer_id:
            logger.info("[Boogu-Cache] Already enabled, resetting state")
            state.update({
                "call_count": 0, "skip_count": 0,
                "compute_count": 0, "last_result": None,
                "compute_times": [],
            })
            return
    
    transformer._original_forward_boogu = transformer.forward
    
    state.update({
        "enabled": True, "transformer_id": transformer_id,
        "call_count": 0, "skip_count": 0,
        "compute_count": 0, "last_result": None,
        "config": config, "compute_times": [],
    })
    
    warmup = config.warmup_steps
    skip = config.skip_interval
    noise = config.noise_scale
    
    def cached_forward(*args, **kwargs):
        st = _get_or_create_cache_state(transformer_id)
        st["call_count"] += 1
        call_id = st["call_count"]
        
        # Warmup phase: always compute
        if call_id <= warmup:
            start = time.perf_counter()
            result = transformer._original_forward_boogu(*args, **kwargs)
            elapsed = time.perf_counter() - start
            
            st["compute_count"] += 1
            st["compute_times"].append(elapsed)
            
            # Handle tuple output (multi-stream)
            if isinstance(result, tuple):
                st["last_result"] = tuple(
                    r.detach() if isinstance(r, torch.Tensor) else r
                    for r in result
                )
            elif isinstance(result, torch.Tensor):
                st["last_result"] = result.detach()
            else:
                st["last_result"] = result
            
            return result
        
        # Cache hit decision
        steps_after_warmup = call_id - warmup
        should_skip = (steps_after_warmup % skip == 0)
        
        if should_skip and st["last_result"] is not None:
            st["skip_count"] += 1
            cached = st["last_result"]
            
            # Apply noise injection if configured
            if noise > 0:
                if isinstance(cached, tuple):
                    cached = tuple(
                        (r + torch.randn_like(r) * noise)
                        if isinstance(r, torch.Tensor) else r
                        for r in cached
                    )
                elif isinstance(cached, torch.Tensor):
                    cached = cached + torch.randn_like(cached) * noise
            
            return cached
        else:
            start = time.perf_counter()
            result = transformer._original_forward_boogu(*args, **kwargs)
            elapsed = time.perf_counter() - start
            
            st["compute_count"] += 1
            st["compute_times"].append(elapsed)
            
            if isinstance(result, tuple):
                st["last_result"] = tuple(
                    r.detach() if isinstance(r, torch.Tensor) else r
                    for r in result
                )
            elif isinstance(result, torch.Tensor):
                st["last_result"] = result.detach()
            else:
                st["last_result"] = result
            
            return result
    
    transformer.forward = cached_forward
    
    logger.info(
        f"[Boogu-Cache] Enabled: warmup={warmup}, "
        f"skip_interval={skip}, noise={noise}"
    )


# =============================================================================
# ComfyUI Node
# =============================================================================

class CacheDiT_Boogu_Optimizer:
    NAME = "⚡ Boogu Cache Optimizer"
    CATEGORY = "CacheDiT"
    
    @classmethod
    def INPUT_TYPES(cls):
        from .utils import get_all_preset_names
        preset_names = get_all_preset_names()
        boogu_presets = [n for n in preset_names if n.startswith("Boogu")]
        return {
            "required": {
                "model": ("MODEL",),
                "model_type": (boogu_presets + ["Auto"],),
                "enable": ("BOOLEAN", {"default": True}),
                "print_summary": ("BOOLEAN", {"default": True}),
            },
        }
    
    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "apply"
    
    def apply(self, model, model_type, enable, print_summary):
        if not enable:
            return (model,)
        
        # Get transformer
        model_patcher = model
        if hasattr(model_patcher, 'model'):
            inner = model_patcher.model
        else:
            inner = model_patcher
        
        dm = inner.diffusion_model
        while hasattr(dm, '_orig_mod'):
            dm = dm._orig_mod
        
        # Load preset
        from .utils import get_preset
        preset = get_preset(model_type if model_type != "Auto" else "Boogu")
        
        config = BooguCacheConfig(
            warmup_steps=preset.max_warmup_steps,
            skip_interval=preset.skip_interval,
            noise_scale=preset.noise_scale,
            verbose=False,
            print_summary=print_summary,
        )
        
        _enable_boogu_cache(dm, config)
        
        return (model,)


NODE_CLASS_MAPPINGS = {"CacheDiT_Boogu_Optimizer": CacheDiT_Boogu_Optimizer}
NODE_DISPLAY_NAME_MAPPINGS = {"CacheDiT_Boogu_Optimizer": "⚡ Boogu Cache Optimizer"}
