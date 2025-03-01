"""Quantized model implementations for LLM Speculative Sampling.

This module provides classes for working with quantized language models, including:
1. QuantizationMethod - Enum of supported quantization methods
2. QuantizedModel - Wrapper for loading and using quantized models
3. QSPECModel - Implementation of the QSPEC approach with different quantization schemes

The QSPEC approach combines two complementary quantization schemes:
- Low-precision activation-weight quantization for drafting tokens (faster but less accurate)
- High-precision weight-only quantization for verifying tokens (more accurate but slower)
"""

import torch
import torch.nn as nn
from typing import Optional, Dict, Any, Tuple, List, Union
from enum import Enum
import logging
from transformers import AutoModelForCausalLM, PreTrainedModel

# Import optional dependencies with fallbacks
try:
    import bitsandbytes as bnb
    HAS_BITSANDBYTES = True
except ImportError:
    HAS_BITSANDBYTES = False

try:
    from optimum.bettertransformer import BetterTransformer
    HAS_BETTERTRANSFORMER = True
except ImportError:
    HAS_BETTERTRANSFORMER = False

# Configure logging
logger = logging.getLogger(__name__)


class QuantizationMethod(Enum):
    """Supported quantization methods for language models.
    
    This enum defines the available quantization methods:
    - NONE: No quantization (full precision, FP32/FP16)
    - INT8: 8-bit weight-only quantization (reduces model size, preserves accuracy)
    - INT4: 4-bit weight-only quantization (smaller size, some accuracy loss)
    - FP8: 8-bit activation-weight quantization (faster inference)
    - FP4: 4-bit activation-weight quantization (fastest inference, lower accuracy)
    """
    NONE = "none"  # No quantization (full precision)
    INT8 = "int8"  # 8-bit weight-only quantization
    INT4 = "int4"  # 4-bit weight-only quantization
    FP8 = "fp8"    # 8-bit activation-weight quantization
    FP4 = "fp4"    # 4-bit activation-weight quantization


class QuantizedModel:
    """A wrapper for quantized models that provides a unified interface.
    
    This class supports loading and using models with different quantization methods:
    - No quantization (full precision)
    - Weight-only quantization (INT8, INT4)
    - Activation-weight quantization (FP8, FP4)
    
    It provides a consistent interface regardless of the underlying quantization,
    making it easier to swap between different precision levels for experimentation.
    """
    
    def __init__(
        self,
        model_name_or_path: str,
        quantization_method: Union[QuantizationMethod, str] = QuantizationMethod.NONE,
        device: Optional[str] = None,
        **kwargs
    ):
        """Initialize a quantized model with the specified method.
        
        Args:
            model_name_or_path (str): The name or path of the model to load from HuggingFace
            quantization_method (Union[QuantizationMethod, str]): The quantization method to use
                Can be a QuantizationMethod enum or a string (none, int8, int4, fp8, fp4)
            device (Optional[str]): The device to load the model on (cuda or cpu)
            **kwargs: Additional arguments to pass to the model loading function
        """
        # Store the model name/path for reference
        self.model_name_or_path = model_name_or_path
        
        # Convert string quantization method to enum if needed
        if isinstance(quantization_method, str):
            try:
                self.quantization_method = QuantizationMethod(quantization_method.lower())
            except ValueError:
                logger.warning(f"Unknown quantization method: {quantization_method}. Using NONE.")
                self.quantization_method = QuantizationMethod.NONE
        else:
            self.quantization_method = quantization_method
        
        # Set device, defaulting to CUDA if available
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load the model with the specified quantization
        self.model = self._load_model(**kwargs)
        
        # Flag to indicate if this is a weight-only quantized model
        self.is_weight_only = self.quantization_method in [QuantizationMethod.INT8, QuantizationMethod.INT4]
        
    def _load_model(self, **kwargs) -> PreTrainedModel:
        """Load the model with the specified quantization method.
        
        This internal method handles the details of loading models with different
        quantization schemes, including fallbacks when dependencies are missing
        or when running on CPU where some quantization methods aren't supported.
        
        Args:
            **kwargs: Additional arguments to pass to the model loading function
            
        Returns:
            PreTrainedModel: The loaded model with the specified quantization
            
        Raises:
            ImportError: If required dependencies for quantization are missing
            ValueError: If an unsupported quantization method is specified
        """
        # Case 1: No quantization (full precision)
        if self.quantization_method == QuantizationMethod.NONE:
            model = AutoModelForCausalLM.from_pretrained(
                self.model_name_or_path,
                device_map=self.device,
                trust_remote_code=True,
                **kwargs
            )
            return model
        
        # Case 2: 8-bit weight-only quantization (INT8)
        elif self.quantization_method == QuantizationMethod.INT8:
            # Check for required dependency
            if not HAS_BITSANDBYTES:
                raise ImportError("bitsandbytes is required for INT8 quantization. Install it with `pip install bitsandbytes`.")
            
            # Fall back to full precision on CPU
            if self.device == 'cpu':
                logger.warning("INT8 quantization requires GPU. Falling back to full precision on CPU.")
                model = AutoModelForCausalLM.from_pretrained(
                    self.model_name_or_path,
                    device_map=self.device,
                    trust_remote_code=True,
                    **kwargs
                )
                return model
            
            # Load 8-bit quantized model on GPU
            model = AutoModelForCausalLM.from_pretrained(
                self.model_name_or_path,
                device_map=self.device,
                load_in_8bit=True,
                trust_remote_code=True,
                **kwargs
            )
            return model
        
        # Case 3: 4-bit weight-only quantization (INT4)
        elif self.quantization_method == QuantizationMethod.INT4:
            # Check for required dependency
            if not HAS_BITSANDBYTES:
                raise ImportError("bitsandbytes is required for INT4 quantization. Install it with `pip install bitsandbytes`.")
            
            # Fall back to full precision on CPU
            if self.device == 'cpu':
                logger.warning("INT4 quantization requires GPU. Falling back to full precision on CPU.")
                model = AutoModelForCausalLM.from_pretrained(
                    self.model_name_or_path,
                    device_map=self.device,
                    trust_remote_code=True,
                    **kwargs
                )
                return model
            
            # Load 4-bit quantized model on GPU
            model = AutoModelForCausalLM.from_pretrained(
                self.model_name_or_path,
                device_map=self.device,
                load_in_4bit=True,
                trust_remote_code=True,
                **kwargs
            )
            return model
        
        # Case 4: Activation-weight quantization (FP8, FP4)
        elif self.quantization_method in [QuantizationMethod.FP8, QuantizationMethod.FP4]:
            # First load the model normally
            model = AutoModelForCausalLM.from_pretrained(
                self.model_name_or_path,
                device_map=self.device,
                trust_remote_code=True,
                **kwargs
            )
            
            # Then apply BetterTransformer for activation-weight quantization if available
            if HAS_BETTERTRANSFORMER:
                try:
                    model = BetterTransformer.transform(model)
                    logger.info(f"Applied BetterTransformer for {self.quantization_method.value} quantization")
                except Exception as e:
                    logger.warning(f"Failed to apply BetterTransformer: {e}. Using model without transformation.")
            else:
                logger.warning("BetterTransformer not available. Using model without activation-weight quantization.")
            
            # Note: For proper FP8/FP4 activation-weight quantization, additional steps would be needed
            # This is a simplified implementation for demonstration purposes
            return model
        
        # Case 5: Unsupported quantization method
        else:
            raise ValueError(f"Unsupported quantization method: {self.quantization_method}")
    
    def __call__(self, *args, **kwargs):
        """Forward pass through the model.
        
        This method allows the QuantizedModel to be called directly with the same
        interface as the underlying model.
        
        Args:
            *args: Positional arguments to pass to the model
            **kwargs: Keyword arguments to pass to the model
            
        Returns:
            The output of the model's forward pass
        """
        return self.model(*args, **kwargs)
    
    @property
    def device(self):
        """Get the device of the model."""
        return self._device
    
    @device.setter
    def device(self, device):
        """Set the device of the model."""
        self._device = device
        
    def to(self, device):
        """Move the model to the specified device.
        
        Args:
            device (str): The device to move the model to (e.g., 'cuda', 'cpu')
            
        Returns:
            QuantizedModel: Self, for method chaining
        """
        self._device = device
        if hasattr(self.model, 'to'):
            self.model = self.model.to(device)
        return self


class QSPECModel:
    """Implementation of the QSPEC approach for speculative decoding with quantized models.
    
    The QSPEC (Quantized Speculative Sampling) approach combines two complementary
    quantization schemes for efficient speculative decoding:
    
    1. A fast, low-precision activation-weight quantized model for drafting tokens
       (typically using FP8 or FP4 quantization)
    2. A more accurate, high-precision weight-only quantized model for verifying tokens
       (typically using INT8 or INT4 quantization)
    
    This combination aims to achieve the speed benefits of lower precision while
    maintaining the accuracy of higher precision models.
    """
    
    def __init__(
        self,
        model_name_or_path: str,
        draft_quantization: Union[QuantizationMethod, str] = QuantizationMethod.FP8,
        verify_quantization: Union[QuantizationMethod, str] = QuantizationMethod.INT8,
        device: Optional[str] = None,
        **kwargs
    ):
        """Initialize a QSPEC model with draft and verification components.
        
        Args:
            model_name_or_path (str): The name or path of the model to load
            draft_quantization (Union[QuantizationMethod, str]): Quantization method for the draft model
                Default is FP8 (activation-weight quantization) for faster drafting
            verify_quantization (Union[QuantizationMethod, str]): Quantization method for the verification model
                Default is INT8 (weight-only quantization) for more accurate verification
            device (Optional[str]): The device to load the models on (cuda or cpu)
            **kwargs: Additional arguments to pass to the model loading functions
        """
        # Store configuration
        self.model_name_or_path = model_name_or_path
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        
        logger.info(f"Initializing QSPEC model with {draft_quantization} draft and {verify_quantization} verification")
        
        # Create draft model with activation-weight quantization (faster but less accurate)
        self.draft_model = QuantizedModel(
            model_name_or_path,
            quantization_method=draft_quantization,
            device=self.device,
            **kwargs
        )
        
        # Create verification model with weight-only quantization (more accurate)
        self.verify_model = QuantizedModel(
            model_name_or_path,
            quantization_method=verify_quantization,
            device=self.device,
            **kwargs
        )
        
        # Ensure the models are compatible (same vocabulary size)
        if hasattr(self.draft_model.model, 'config') and hasattr(self.verify_model.model, 'config'):
            if self.draft_model.model.config.vocab_size != self.verify_model.model.config.vocab_size:
                logger.warning("Draft and verification models have different vocabulary sizes, which may cause issues.")
    
    def draft(self, *args, **kwargs):
        """Generate draft tokens using the low-precision model.
        
        This method forwards the call to the draft model, which is typically
        using a faster but less accurate quantization method.
        
        Args:
            *args: Positional arguments to pass to the model
            **kwargs: Keyword arguments to pass to the model
            
        Returns:
            The output of the draft model's forward pass
        """
        return self.draft_model(*args, **kwargs)
    
    def verify(self, *args, **kwargs):
        """Verify tokens using the high-precision model.
        
        This method forwards the call to the verification model, which is typically
        using a more accurate quantization method.
        
        Args:
            *args: Positional arguments to pass to the model
            **kwargs: Keyword arguments to pass to the model
            
        Returns:
            The output of the verification model's forward pass
        """
        return self.verify_model(*args, **kwargs)
    
    @property
    def device(self):
        """Get the device of the models."""
        return self._device
    
    @device.setter
    def device(self, device):
        """Set the device of the models.
        
        This updates the device for both the draft and verification models.
        
        Args:
            device (str): The device to set ('cuda' or 'cpu')
        """
        self._device = device
        if hasattr(self, 'draft_model'):
            self.draft_model.device = device
        if hasattr(self, 'verify_model'):
            self.verify_model.device = device
    
    def to(self, device):
        """Move the models to the specified device.
        
        Args:
            device (str): The device to move the models to (e.g., 'cuda', 'cpu')
            
        Returns:
            QSPECModel: Self, for method chaining
        """
        self.device = device
        if hasattr(self, 'draft_model'):
            self.draft_model.to(device)
        if hasattr(self, 'verify_model'):
            self.verify_model.to(device)
        return self
