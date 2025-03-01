import torch
from typing import Optional, Tuple, List, Union, Dict, Any
import logging
from tqdm import tqdm

from sampling.kvcache_model import KVCacheModel
from sampling.utils import norm_logits, sample, max_fn
from globals import Decoder
from .quantized_model import QuantizedModel, QSPECModel

logger = logging.getLogger(__name__)

@torch.no_grad()
def qspec_sampling(
    prefix: torch.Tensor,
    model_name_or_path: str,
    draft_quantization: str = "fp8",
    verify_quantization: str = "int8",
    max_len: int = 20,
    gamma: int = 4,
    temperature: float = 1,
    top_k: int = 0,
    top_p: float = 0,
    verbose: bool = False,
    random_seed: Optional[int] = None,
    device: Optional[str] = None,
    **model_kwargs
) -> torch.Tensor:
    """
    QSPEC Sampling: Speculative sampling using quantized models.
    
    This implementation combines the strengths of activation-weight quantization (for fast drafting)
    and weight-only quantization (for accurate verification) in a speculative decoding framework.
    
    Args:
        prefix: Input sequence tensor, shape (batch, prefix_seqlen).
        model_name_or_path: Model name or path to load.
        draft_quantization: Quantization method for the draft model.
        verify_quantization: Quantization method for the verification model.
        max_len: Maximum number of tokens to generate.
        gamma: Number of tokens to generate in each draft step.
        temperature: Sampling temperature.
        top_k: Top-k sampling parameter.
        top_p: Top-p sampling parameter.
        verbose: Whether to print verbose output.
        random_seed: Random seed for reproducibility.
        device: Device to run the models on.
        **model_kwargs: Additional arguments to pass to the model loading function.
        
    Returns:
        Generated sequence tensor, shape (batch, prefix_seqlen + generated_len).
    """
    seq_len = prefix.shape[1]
    T = seq_len + max_len
    
    assert prefix.shape[0] == 1, "Input batch size must be 1"
    
    device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create QSPEC model with both draft and verification models
    qspec_model = QSPECModel(
        model_name_or_path=model_name_or_path,
        draft_quantization=draft_quantization,
        verify_quantization=verify_quantization,
        device=device,
        **model_kwargs
    )
    
    # Create KV cache models for both draft and verification
    draft_model_cache = KVCacheModel(qspec_model.draft_model, temperature, top_k, top_p)
    verify_model_cache = KVCacheModel(qspec_model.verify_model, temperature, top_k, top_p)
    
    # Statistics tracking
    resample_count = 0
    target_sample_count = 0
    accepted_count = 0
    
    while prefix.shape[1] < T:
        # Current prefix length
        prefix_len = prefix.shape[1]
        
        # Generate draft tokens using the low-precision model
        x = draft_model_cache.generate(prefix, gamma)
        
        # Verify the draft tokens using the high-precision model
        _ = verify_model_cache.generate(x, 1)
        
        n = prefix_len + gamma - 1
        
        # Acceptance/rejection sampling
        for i in range(gamma):
            if random_seed:
                torch.manual_seed(random_seed)
            
            r = torch.rand(1, device=device)
            j = x[:, prefix_len + i]
            
            # Calculate acceptance probability
            draft_prob = draft_model_cache._prob_history[:, prefix_len + i - 1, j]
            verify_prob = verify_model_cache._prob_history[:, prefix_len + i - 1, j]
            
            # Accept if random value is less than the ratio of probabilities
            if r > verify_prob / draft_prob:
                # Reject
                n = prefix_len + i - 1
                break
            
            if verbose:
                print(f"draft guess accepted {j[0]}: \033[31m{Decoder().decode(torch.tensor([j]))}\033[0m")
            
            accepted_count += 1
        
        assert n >= prefix_len - 1, f"n {n}, prefix_len {prefix_len}"
        prefix = x[:, :n + 1]
        
        # Rollback KV caches
        draft_model_cache.rollback(n + 1)
        
        assert draft_model_cache._prob_history.shape[-2] <= n + 1, f"draft_model prob list shape {draft_model_cache._prob_history.shape}, n {n}"
        
        if n < prefix_len + gamma - 1:
            # Rejection occurred, sample from the verification model
            t = sample(max_fn(verify_model_cache._prob_history[:, n, :] - draft_model_cache._prob_history[:, n, :]))
            if verbose:
                print(f"verification resamples at position {n}: \033[34m{Decoder().decode(t)}\033[0m")
            resample_count += 1
            verify_model_cache.rollback(n + 1)
        else:
            # All draft tokens were accepted, sample next token from verification model
            assert n == verify_model_cache._prob_history.shape[1] - 1
            t = sample(verify_model_cache._prob_history[:, -1, :])
            if verbose:
                print(f"verification samples {n}: \033[35m{Decoder().decode(t)}\033[0m")
            target_sample_count += 1
            verify_model_cache.rollback(n + 2)
        
        # Append the new token to the prefix
        prefix = torch.cat((prefix, t), dim=1)
    
    if verbose:
        print(f"generated tokens numbers {prefix.shape[-1] - seq_len}, accepted_count {accepted_count}, target_sample_count {target_sample_count}, resample_count {resample_count}")
        
        # Calculate acceptance rate
        acceptance_rate = accepted_count / (accepted_count + resample_count) if (accepted_count + resample_count) > 0 else 0
        print(f"acceptance rate: {acceptance_rate:.2f}")
    
    return prefix


@torch.no_grad()
def qspec_sampling_different_models(
    prefix: torch.Tensor,
    draft_model_name_or_path: str,
    verify_model_name_or_path: str,
    draft_quantization: str = "fp8",
    verify_quantization: str = "int8",
    max_len: int = 20,
    gamma: int = 4,
    temperature: float = 1,
    top_k: int = 0,
    top_p: float = 0,
    verbose: bool = False,
    random_seed: Optional[int] = None,
    device: Optional[str] = None,
    **model_kwargs
) -> torch.Tensor:
    """
    QSPEC Sampling with different models for draft and verification.
    
    This implementation allows using different models for drafting and verification,
    while still leveraging quantization benefits.
    
    Args:
        prefix: Input sequence tensor, shape (batch, prefix_seqlen).
        draft_model_name_or_path: Model name or path for the draft model.
        verify_model_name_or_path: Model name or path for the verification model.
        draft_quantization: Quantization method for the draft model.
        verify_quantization: Quantization method for the verification model.
        max_len: Maximum number of tokens to generate.
        gamma: Number of tokens to generate in each draft step.
        temperature: Sampling temperature.
        top_k: Top-k sampling parameter.
        top_p: Top-p sampling parameter.
        verbose: Whether to print verbose output.
        random_seed: Random seed for reproducibility.
        device: Device to run the models on.
        **model_kwargs: Additional arguments to pass to the model loading function.
        
    Returns:
        Generated sequence tensor, shape (batch, prefix_seqlen + generated_len).
    """
    seq_len = prefix.shape[1]
    T = seq_len + max_len
    
    assert prefix.shape[0] == 1, "Input batch size must be 1"
    
    device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create draft and verification models separately
    draft_model = QuantizedModel(
        model_name_or_path=draft_model_name_or_path,
        quantization_method=draft_quantization,
        device=device,
        **model_kwargs
    )
    
    verify_model = QuantizedModel(
        model_name_or_path=verify_model_name_or_path,
        quantization_method=verify_quantization,
        device=device,
        **model_kwargs
    )
    
    # Create KV cache models
    draft_model_cache = KVCacheModel(draft_model, temperature, top_k, top_p)
    verify_model_cache = KVCacheModel(verify_model, temperature, top_k, top_p)
    
    # Statistics tracking
    resample_count = 0
    target_sample_count = 0
    accepted_count = 0
    
    while prefix.shape[1] < T:
        # Current prefix length
        prefix_len = prefix.shape[1]
        
        # Generate draft tokens using the draft model
        x = draft_model_cache.generate(prefix, gamma)
        
        # Verify the draft tokens using the verification model
        _ = verify_model_cache.generate(x, 1)
        
        n = prefix_len + gamma - 1
        
        # Acceptance/rejection sampling
        for i in range(gamma):
            if random_seed:
                torch.manual_seed(random_seed)
            
            r = torch.rand(1, device=device)
            j = x[:, prefix_len + i]
            
            # Calculate acceptance probability
            draft_prob = draft_model_cache._prob_history[:, prefix_len + i - 1, j]
            verify_prob = verify_model_cache._prob_history[:, prefix_len + i - 1, j]
            
            # Accept if random value is less than the ratio of probabilities
            if r > verify_prob / draft_prob:
                # Reject
                n = prefix_len + i - 1
                break
            
            if verbose:
                print(f"draft guess accepted {j[0]}: \033[31m{Decoder().decode(torch.tensor([j]))}\033[0m")
            
            accepted_count += 1
        
        assert n >= prefix_len - 1, f"n {n}, prefix_len {prefix_len}"
        prefix = x[:, :n + 1]
        
        # Rollback KV caches
        draft_model_cache.rollback(n + 1)
        
        assert draft_model_cache._prob_history.shape[-2] <= n + 1, f"draft_model prob list shape {draft_model_cache._prob_history.shape}, n {n}"
        
        if n < prefix_len + gamma - 1:
            # Rejection occurred, sample from the verification model
            t = sample(max_fn(verify_model_cache._prob_history[:, n, :] - draft_model_cache._prob_history[:, n, :]))
            if verbose:
                print(f"verification resamples at position {n}: \033[34m{Decoder().decode(t)}\033[0m")
            resample_count += 1
            verify_model_cache.rollback(n + 1)
        else:
            # All draft tokens were accepted, sample next token from verification model
            assert n == verify_model_cache._prob_history.shape[1] - 1
            t = sample(verify_model_cache._prob_history[:, -1, :])
            if verbose:
                print(f"verification samples {n}: \033[35m{Decoder().decode(t)}\033[0m")
            target_sample_count += 1
            verify_model_cache.rollback(n + 2)
        
        # Append the new token to the prefix
        prefix = torch.cat((prefix, t), dim=1)
    
    if verbose:
        print(f"generated tokens numbers {prefix.shape[-1] - seq_len}, accepted_count {accepted_count}, target_sample_count {target_sample_count}, resample_count {resample_count}")
        
        # Calculate acceptance rate
        acceptance_rate = accepted_count / (accepted_count + resample_count) if (accepted_count + resample_count) > 0 else 0
        print(f"acceptance rate: {acceptance_rate:.2f}")
    
    return prefix
