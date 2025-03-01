"""Sampling Module for LLM Speculative Sampling

This module provides various sampling strategies for language models, including:
1. Autoregressive sampling - standard token-by-token generation
2. Speculative sampling - using a smaller model to draft tokens for a larger model
3. QSPEC sampling - quantized speculative sampling with different precision levels

The module is designed to be modular and extensible, allowing for easy integration
of new sampling methods and model types.
"""

# Core sampling methods
from sampling.autoregressive_sampling import autoregressive_sampling
from sampling.speculative_sampling import speculative_sampling, speculative_sampling_v2
from sampling.kvcache_model import KVCacheModel

# Import quantization modules
try:
    # Quantized model implementations
    from sampling.quantization import QuantizedModel, QSPECModel
    from sampling.quantization.qspec_sampling import qspec_sampling, qspec_sampling_different_models
    
    # Flag to indicate quantization modules are available
    HAS_QUANTIZATION = True
    
    # Export all public API functions
    __all__ = [
        # Core sampling methods
        "speculative_sampling", "speculative_sampling_v2", "autoregressive_sampling", "KVCacheModel",
        # Quantization modules
        "QuantizedModel", "QSPECModel", "qspec_sampling", "qspec_sampling_different_models",
        # Flags
        "HAS_QUANTIZATION"
    ]
except ImportError:
    # Quantization modules not available
    HAS_QUANTIZATION = False
    
    # Export only core API functions
    __all__ = [
        "speculative_sampling", "speculative_sampling_v2", "autoregressive_sampling", "KVCacheModel",
        "HAS_QUANTIZATION"
    ]