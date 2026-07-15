"""
ComfyUI-CacheDiT: Anima Specialized Node
=========================================

Dedicated node for Anima DiT model (anima_baseV10 / Anima Turbo).
Supports two acceleration strategies:

  • adaptive_l1     — L1-distance + per-step-position threshold (TeaCache-style)
                      Safer for quality: skips only when content is truly stable.
  • fixed_interval  — Simple warmup + skip_interval (Wan/LTX2-style)
                      Predictable, minimal overhead.

Both strategies are 100% pure PyTorch — no Triton, no CUDA kernels, XPU-ready.

Key Features:
- Per-transformer cache isolation (id-based registry)
- Memory-efficient caching (detach-only, no clone)
- Automatic state reset per sampling run (OUTER_SAMPLE wrapper)
- XPU auto-detection for cache device placement
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

logger = logging.getLogger("ComfyUI-CacheDiT-Anima")

CACHE_MODES = ["adaptive_l1", "fixed_interval"]


# =============================================================================
# XPU device helper
# =============================================================================

def _get_cache_device() -> str:
    """Auto-detect the best cache storage device. XPU > CUDA > CPU."""
    if torch.xpu.is_available():
        return "xpu"
    elif torch.cuda.is_available():
        return "cuda"
    return "cpu"


# =============================================================================
# Math helpers (pure PyTorch — XPU safe, no Triton dependency)
# =============================================================================

def _relative_l1_distance(a: torch.Tensor, b: torch.Tensor) -> float:
    """
    Relative L1 distance — measures how much the input changed since last step.
    Pure PyTorch ops, XPU compatible.
    """
    return (
        (a - b).abs().mean() / a.abs().mean().clamp(min=1e-8)
    ).to(torch.float32).item()


def _adaptive_threshold(
    step_percent: float,
    base: float,
    early_factor: float,
    late_factor: float,
) -> float:
    """
    Adjust threshold based on denoising progress.

    step_percent: 0.0 = first step (max noise), 1.0 = last step (clean)

    Early steps  (0.00 – 0.35):  structure still forming → low threshold  → rarely skip
    Late steps   (0.70 – 1.00):  details are stable    → high threshold → skip aggressively
    """
    if step_percent < 0.35:
        return base * early_factor
    elif step_percent > 0.70:
        return base * late_factor
    else:
        return base


# =============================================================================
# Per-stream cache state (TeaCache-style for adaptive_l1 mode)
# =============================================================================

class _StreamState:
    """Tracks input/output/delta for the diffusion stream."""
    __slots__ = ("prev_x", "prev_out", "prev_t", "accumulated", "skips", "max_t", "_cache_device")

    def __init__(self, cache_device: str = "xpu"):
        self.prev_x: Optional[torch.Tensor] = None
        self.prev_out: Optional[torch.Tensor] = None
        self.prev_t: float = -1.0
        self.accumulated: float = 0.0
        self.skips: int = 0
        self.max_t: float = 1000.0
        self._cache_device = cache_device

    def reset(self):
        self.prev_x = None
        self.prev_out = None
        self.accumulated = 0.0
        self.skips = 0
        self.max_t = 1000.0


# =============================================================================
# Per-transformer cache state registry
# =============================================================================

_anima_cache_registry: Dict[int, Dict[str, Any]] = {}


def _get_or_create_cache_state(transformer_id: int) -> Dict[str, Any]:
    """Get or create cache state for a specific transformer instance."""
    if transformer_id not in _anima_cache_registry:
        _anima_cache_registry[transformer_id] = {
            "enabled": False,
            "transformer_id": transformer_id,
            "call_count": 0,
            "skip_count": 0,
            "compute_count": 0,
            "last_result": None,
            "config": None,
            "compute_times": [],
            "stream": None,
        }
    return _anima_cache_registry[transformer_id]


class AnimaCacheConfig:
    """Configuration for Anima cache optimization."""

    def __init__(
        self,
        cache_mode: str = "adaptive_l1",
        warmup_steps: int = 4,
        skip_interval: int = 2,
        l1_threshold: float = 0.15,
        early_factor: float = 0.4,
        late_factor: float = 1.8,
        cache_device: str = "xpu",
        verbose: bool = False,
        print_summary: bool = True,
    ):
        self.cache_mode = cache_mode
        self.warmup_steps = warmup_steps
        self.skip_interval = skip_interval
        self.l1_threshold = l1_threshold
        self.early_factor = early_factor
        self.late_factor = late_factor
        self.cache_device = cache_device
        self.verbose = verbose
        self.print_summary = print_summary

        self.is_enabled = False
        self.num_inference_steps: Optional[int] = None
        self.current_step: int = 0

    def clone(self) -> "AnimaCacheConfig":
        new_config = AnimaCacheConfig(
            cache_mode=self.cache_mode,
            warmup_steps=self.warmup_steps,
            skip_interval=self.skip_interval,
            l1_threshold=self.l1_threshold,
            early_factor=self.early_factor,
            late_factor=self.late_factor,
            cache_device=self.cache_device,
            verbose=self.verbose,
            print_summary=self.print_summary,
        )
        new_config.is_enabled = self.is_enabled
        new_config.num_inference_steps = self.num_inference_steps
        return new_config

    def reset(self):
        self.current_step = 0


# =============================================================================
# Cache engine
# =============================================================================

def _enable_anima_cache(transformer, config: AnimaCacheConfig):
    """Enable lightweight cache for Anima transformer."""
    transformer_id = id(transformer)
    state = _get_or_create_cache_state(transformer_id)

    if hasattr(transformer, '_original_forward_anima'):
        if state.get("transformer_id") == transformer_id:
            logger.info("[Anima-Cache] Already enabled, resetting state")
            state.update({
                "call_count": 0,
                "skip_count": 0,
                "compute_count": 0,
                "last_result": None,
                "compute_times": [],
                "stream": None,
            })
            return

    transformer._original_forward_anima = transformer.forward

    state.update({
        "enabled": True,
        "transformer_id": transformer_id,
        "call_count": 0,
        "skip_count": 0,
        "compute_count": 0,
        "last_result": None,
        "config": config,
        "compute_times": [],
        "stream": _StreamState(cache_device=config.cache_device)
               if config.cache_mode == "adaptive_l1" else None,
    })

    # ── Result caching helpers ──

    def _cache_result(result):
        if isinstance(result, torch.Tensor):
            state["last_result"] = result.detach()
        elif isinstance(result, tuple):
            state["last_result"] = tuple(
                r.detach() if isinstance(r, torch.Tensor) else r
                for r in result
            )
        else:
            state["last_result"] = result

    def _return_cached(x_device, x_dtype):
        cached = state["last_result"]
        if isinstance(cached, torch.Tensor):
            return cached.to(x_device).to(x_dtype)
        elif isinstance(cached, tuple):
            return tuple(
                r.to(x_device).to(x_dtype) if isinstance(r, torch.Tensor) else r
                for r in cached
            )
        return cached

    # ── fixed_interval forward ──

    def _fixed_interval_forward(*args, **kwargs):
        s = _get_or_create_cache_state(transformer_id)
        s["call_count"] += 1
        call_id = s["call_count"]
        cfg = s.get("config")
        warmup = cfg.warmup_steps if cfg else 4
        skip_int = cfg.skip_interval if cfg else 2

        if call_id <= warmup:
            start = time.time()
            result = transformer._original_forward_anima(*args, **kwargs)
            s["compute_count"] += 1
            s["compute_times"].append(time.time() - start)
            _cache_result(result)
            return result

        if ((call_id - warmup) % skip_int != 0) and s["last_result"] is not None:
            s["skip_count"] += 1
            x_dev = args[0].device if len(args) > 0 else "cpu"
            x_dt = args[0].dtype if len(args) > 0 else torch.float32
            return _return_cached(x_dev, x_dt)

        start = time.time()
        result = transformer._original_forward_anima(*args, **kwargs)
        s["compute_count"] += 1
        s["compute_times"].append(time.time() - start)
        _cache_result(result)
        return result

    # ── adaptive_l1 forward ──

    def _adaptive_l1_forward(*args, **kwargs):
        s = _get_or_create_cache_state(transformer_id)
        s["call_count"] += 1
        call_id = s["call_count"]
        cfg = s.get("config")

        warmup = cfg.warmup_steps if cfg else 4
        base_thr = cfg.l1_threshold if cfg else 0.15
        early_f = cfg.early_factor if cfg else 0.4
        late_f = cfg.late_factor if cfg else 1.8
        cache_dev = cfg.cache_device if cfg else "xpu"

        x = args[0] if len(args) > 0 else None
        t_tensor = args[1] if len(args) > 1 else None
        t_val = float(t_tensor[0]) if t_tensor is not None and t_tensor.numel() > 0 else 500.0

        st: _StreamState = s.get("stream")
        if st is None:
            st = _StreamState(cache_device=cache_dev)
            s["stream"] = st

        # Warmup or first call
        if call_id <= warmup or st.prev_x is None:
            start = time.time()
            result = transformer._original_forward_anima(*args, **kwargs)
            s["compute_count"] += 1
            s["compute_times"].append(time.time() - start)
            if x is not None:
                st.prev_x = x.detach()
                st.max_t = max(1e-4, t_val)
            if isinstance(result, torch.Tensor):
                st.prev_out = result.detach().to(cache_dev)
            _cache_result(result)
            return result

        # Detect new generation run
        if x is not None and st.prev_x is not None:
            if x.shape != st.prev_x.shape or t_val > st.prev_t + 1e-4:
                st.reset()
                st.max_t = max(1e-4, t_val)

        st.prev_t = t_val

        # Adaptive threshold
        step_pct = max(0.0, min(1.0, 1.0 - t_val / st.max_t))
        thr = _adaptive_threshold(step_pct, base_thr, early_f, late_f)

        # L1 delta
        delta = _relative_l1_distance(st.prev_x, x) if (x is not None and st.prev_x is not None) else 999.0
        st.accumulated += delta

        if st.accumulated < thr and st.prev_out is not None:
            s["skip_count"] += 1
            st.skips += 1
            return st.prev_out.to(x.device).to(x.dtype) if isinstance(st.prev_out, torch.Tensor) else st.prev_out

        # Compute
        st.accumulated = 0.0
        start = time.time()
        result = transformer._original_forward_anima(*args, **kwargs)
        s["compute_count"] += 1
        s["compute_times"].append(time.time() - start)
        if x is not None:
            st.prev_x = x.detach()
        if isinstance(result, torch.Tensor):
            st.prev_out = result.detach().to(cache_dev)
        _cache_result(result)
        return result

    # Bind
    transformer.forward = (
        _adaptive_l1_forward if config.cache_mode == "adaptive_l1"
        else _fixed_interval_forward
    )

    logger.info(
        f"[Anima-Cache] Enabled | mode={config.cache_mode}, warmup={config.warmup_steps}"
        + (f", skip={config.skip_interval}" if config.cache_mode == "fixed_interval"
           else f", l1_thr={config.l1_threshold}, early={config.early_factor}, late={config.late_factor}")
    )


def _refresh_anima_cache(transformer, config: AnimaCacheConfig):
    """Reset all counters and stream state for new generation."""
    transformer_id = id(transformer)
    state = _get_or_create_cache_state(transformer_id)

    try:
        state["call_count"] = 0
        state["skip_count"] = 0
        state["compute_count"] = 0
        state["last_result"] = None
        state["compute_times"] = []
        state["config"] = config
        if state.get("stream") is not None:
            state["stream"].reset()
        if config.verbose:
            logger.info(
                f"[Anima-Cache] Reset | transformer {transformer_id}, "
                f"{config.num_inference_steps} steps"
            )
    except Exception as e:
        logger.error(f"[Anima-Cache] Refresh failed: {e}")
        traceback.print_exc()


def _get_anima_cache_stats(transformer_id: int):
    """Get statistics from Anima cache."""
    if transformer_id not in _anima_cache_registry:
        return None
    state = _anima_cache_registry[transformer_id]
    if not state.get("enabled"):
        return None

    total = state["call_count"]
    cached = state["skip_count"]
    computed = state["compute_count"]
    if total == 0:
        return None

    cfg = state.get("config")
    return {
        "transformer_id": transformer_id,
        "total_calls": total,
        "computed_calls": computed,
        "cached_calls": cached,
        "cache_hit_rate": (cached / total) * 100,
        "estimated_speedup": total / max(computed, 1),
        "avg_compute_time": sum(state["compute_times"]) / max(len(state["compute_times"]), 1),
        "mode": cfg.cache_mode if cfg else "unknown",
    }


# =============================================================================
# OUTER_SAMPLE wrapper
# =============================================================================

def _anima_outer_sample_wrapper(executor, *args, **kwargs):
    """
    OUTER_SAMPLE wrapper — resets cache state before each sampling run.
    """
    guider = executor.class_obj
    orig_model_options = guider.model_options
    transformer = None
    config = None

    try:
        guider.model_options = comfy.model_patcher.create_model_options_clone(orig_model_options)

        config: AnimaCacheConfig = guider.model_options.get("transformer_options", {}).get("anima_cache")
        if config is None:
            return executor(*args, **kwargs)

        config = config.clone()
        config.reset()
        guider.model_options["transformer_options"]["anima_cache"] = config

        sigmas = args[3] if len(args) > 3 else kwargs.get("sigmas")
        if sigmas is not None:
            config.num_inference_steps = len(sigmas) - 1

        model_patcher = guider.model_patcher
        if hasattr(model_patcher, 'model') and hasattr(model_patcher.model, 'diffusion_model'):
            transformer = model_patcher.model.diffusion_model
            transformer_id = id(transformer)

            if config.num_inference_steps is not None:
                if not hasattr(transformer, '_original_forward_anima'):
                    logger.info(
                        f"[Anima-Cache] Enabling | id={transformer_id}, "
                        f"steps={config.num_inference_steps}, mode={config.cache_mode}"
                    )
                    _enable_anima_cache(transformer, config)
                    config.is_enabled = True
                else:
                    logger.info(
                        f"[Anima-Cache] Refreshing | id={transformer_id}, "
                        f"steps={config.num_inference_steps}"
                    )
                    _refresh_anima_cache(transformer, config)
                    config.is_enabled = True

        result = executor(*args, **kwargs)

        if config.print_summary and transformer is not None:
            stats = _get_anima_cache_stats(id(transformer))
            if stats:
                logger.info(
                    f"\n"
                    f"[Anima-Cache] Lightweight Cache Statistics:\n"
                    f"  Mode: {stats['mode']}\n"
                    f"  Total Steps: {stats['total_calls']}\n"
                    f"  Computed: {stats['computed_calls']}\n"
                    f"  Cached: {stats['cached_calls']}\n"
                    f"  Cache Hit Rate: {stats['cache_hit_rate']:.1f}%\n"
                    f"  Estimated Speedup: {stats['estimated_speedup']:.2f}x\n"
                    f"  Avg Compute Time: {stats['avg_compute_time']:.3f}s"
                )
        return result

    except Exception as e:
        logger.error(f"[Anima-Cache] OUTER_SAMPLE failed: {e}")
        traceback.print_exc()
        return executor(*args, **kwargs)
    finally:
        guider.model_options = orig_model_options


# =============================================================================
# Node Definition
# =============================================================================

class AnimaCacheOptimizer:
    """
    Anima Cache Optimizer Node

    Accelerates Anima DiT (anima_baseV10 / Anima Turbo) via step-level caching.
    Two strategies: adaptive_l1 (TeaCache-style, quality-aware) and
    fixed_interval (simple warmup+skip). Pure PyTorch — XPU compatible.

    Place between model loader and KSampler.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL", {"tooltip": "Input model (any DiT loader)"}),
                "enable": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Enable/Disable Anima cache acceleration"
                }),
                "cache_mode": (CACHE_MODES, {
                    "default": "adaptive_l1",
                    "tooltip": (
                        "adaptive_l1: L1-distance + per-step threshold (TeaCache-style)\n"
                        "fixed_interval: warmup + skip_interval (simple & predictable)"
                    ),
                }),
                "warmup_steps": ("INT", {
                    "default": 4, "min": 1, "max": 20, "step": 1,
                    "tooltip": "Initial steps always computed to build cache baseline. Recommended: 3-6."
                }),
                # ── adaptive_l1 ──
                "l1_threshold": ("FLOAT", {
                    "default": 0.15, "min": 0.01, "max": 1.0, "step": 0.01, "display": "slider",
                    "tooltip": "[adaptive_l1] Base L1 threshold. Lower = fewer skips, higher quality."
                }),
                "early_factor": ("FLOAT", {
                    "default": 0.4, "min": 0.1, "max": 1.0, "step": 0.05, "display": "slider",
                    "tooltip": "[adaptive_l1] Multiplier for early steps (structure forming). 0.4 = very conservative."
                }),
                "late_factor": ("FLOAT", {
                    "default": 1.8, "min": 1.0, "max": 4.0, "step": 0.1, "display": "slider",
                    "tooltip": "[adaptive_l1] Multiplier for late steps (details stable). 1.8 = aggressive skipping."
                }),
                # ── fixed_interval ──
                "skip_interval": ("INT", {
                    "default": 2, "min": 2, "max": 10, "step": 1,
                    "tooltip": "[fixed_interval] Compute every Nth step after warmup. Recommended: 2-3."
                }),
                "print_summary": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Print performance statistics after generation"
                }),
            },
        }

    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("optimized_model",)
    FUNCTION = "optimize"
    CATEGORY = "⚡ CacheDiT"
    DESCRIPTION = (
        "Anima DiT Cache Accelerator\n\n"
        "Two modes: adaptive_l1 (TeaCache-style, quality-aware) and "
        "fixed_interval (simple warmup+skip). Pure PyTorch — XPU compatible."
    )

    def optimize(
        self,
        model,
        enable: bool = True,
        cache_mode: str = "adaptive_l1",
        warmup_steps: int = 4,
        l1_threshold: float = 0.15,
        early_factor: float = 0.4,
        late_factor: float = 1.8,
        skip_interval: int = 2,
        print_summary: bool = True,
    ):
        if not enable:
            logger.info("[Anima-Cache] Disabled")
            return self.disable(model)

        cache_device = _get_cache_device()

        transformer = None
        existing_config = None
        if hasattr(model.model, 'diffusion_model'):
            transformer = model.model.diffusion_model
            existing_config = getattr(transformer, '_anima_cache_config', None)

        if existing_config is not None:
            changed = (
                existing_config.get("cache_mode") != cache_mode or
                existing_config.get("warmup_steps") != warmup_steps or
                existing_config.get("l1_threshold") != l1_threshold or
                existing_config.get("early_factor") != early_factor or
                existing_config.get("late_factor") != late_factor or
                existing_config.get("skip_interval") != skip_interval or
                existing_config.get("print_summary") != print_summary
            )
            if changed:
                logger.info("[Anima-Cache] Parameters changed, reconfiguring...")
                model = self.disable(model)[0]
            else:
                logger.info("[Anima-Cache] Configuration unchanged")
                return (model,)

        model = model.clone()

        config = AnimaCacheConfig(
            cache_mode=cache_mode,
            warmup_steps=warmup_steps,
            skip_interval=skip_interval,
            l1_threshold=l1_threshold,
            early_factor=early_factor,
            late_factor=late_factor,
            cache_device=cache_device,
            print_summary=print_summary,
        )

        if "transformer_options" not in model.model_options:
            model.model_options["transformer_options"] = {}
        model.model_options["transformer_options"]["anima_cache"] = config

        if transformer is not None:
            transformer._anima_cache_config = {
                "cache_mode": cache_mode,
                "warmup_steps": warmup_steps,
                "skip_interval": skip_interval,
                "l1_threshold": l1_threshold,
                "early_factor": early_factor,
                "late_factor": late_factor,
                "print_summary": print_summary,
            }

        try:
            model.add_wrapper_with_key(
                comfy.patcher_extension.WrappersMP.OUTER_SAMPLE,
                "anima_cache",
                _anima_outer_sample_wrapper
            )
            logger.info(
                f"[Anima-Cache] Configured | mode={cache_mode}, "
                f"warmup={warmup_steps}, device={cache_device}"
            )
        except Exception as e:
            logger.error(f"[Anima-Cache] Failed to register wrapper: {e}")
            traceback.print_exc()

        return (model,)

    def disable(self, model):
        """Cleanly restore the model to its original state."""
        model = model.clone()

        if "anima_cache" in model.model_options.get("transformer_options", {}):
            del model.model_options["transformer_options"]["anima_cache"]
        if "anima_cache" in model.wrappers.get(comfy.patcher_extension.WrappersMP.OUTER_SAMPLE, {}):
            del model.wrappers[comfy.patcher_extension.WrappersMP.OUTER_SAMPLE]["anima_cache"]

        try:
            if hasattr(model.model, 'diffusion_model'):
                t = model.model.diffusion_model
                if hasattr(t, '_original_forward_anima'):
                    t.forward = t._original_forward_anima
                    delattr(t, '_original_forward_anima')
                    logger.info("[Anima-Cache] Restored original forward")
                if hasattr(t, '_anima_cache_config'):
                    delattr(t, '_anima_cache_config')
                tid = id(t)
                if tid in _anima_cache_registry:
                    del _anima_cache_registry[tid]
                    logger.info("[Anima-Cache] Cache state cleared")
        except Exception as e:
            logger.warning(f"[Anima-Cache] Restore warning: {e}")

        return (model,)


# =============================================================================
# Node Registration
# =============================================================================

NODE_CLASS_MAPPINGS = {
    "AnimaCacheOptimizer": AnimaCacheOptimizer,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "AnimaCacheOptimizer": "⚡ CacheDiT Anima Accelerator",
}
