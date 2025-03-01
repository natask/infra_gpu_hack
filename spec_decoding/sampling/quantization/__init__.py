"""Quantization Module for LLM Speculative Sampling

This module provides quantization utilities for language models, enabling:
1. Different quantization methods (none, int8, int4, fp8, fp4)
2. QSPEC approach combining different quantization schemes for draft and verification
3. Support for both same-model and different-model speculative sampling

The quantization techniques aim to improve inference efficiency while maintaining
output quality through strategic precision allocation.
"""

# Import model classes
from .quantized_model import QuantizedModel, QSPECModel, QuantizationMethod

# Make sampling functions available at the package level
from .qspec_sampling import qspec_sampling, qspec_sampling_different_models

__all__ = [
    # Model classes
    'QuantizedModel', 'QSPECModel', 'QuantizationMethod',
    # Sampling functions
    'qspec_sampling', 'qspec_sampling_different_models'
]
