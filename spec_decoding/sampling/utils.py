"""Utility functions for sampling from language models.

This module provides helper functions for token sampling, including:
1. Top-k and top-p (nucleus) filtering
2. Temperature-based logits normalization
3. Robust token sampling with error handling
4. Probability distribution utilities
"""

import torch
from torch.nn import functional as F


def top_k_top_p_filter(logits: torch.Tensor, top_k: int = 0, top_p: float = 0.0):
    """Apply top-k and/or top-p (nucleus) filtering to logits.
    
    This function filters a distribution of logits using top-k and/or top-p (nucleus) filtering.
    In top-k filtering, only the k most likely tokens are kept. In top-p filtering, tokens are
    kept until their cumulative probability exceeds p.
    
    Args:
        logits (torch.Tensor): 2D tensor with shape (batch, vocab)
        top_k (int, optional): Number of highest probability tokens to keep. Defaults to 0 (disabled).
        top_p (float, optional): Cumulative probability threshold. Defaults to 0.0 (disabled).

    Returns:
        torch.Tensor: Filtered logits with low-probability tokens set to -inf
    """
    # Apply top-k filtering if specified (k > 0)
    if top_k > 0:
        # Get values of top k tokens
        filter_values = torch.topk(logits, min(top_k, logits.size(-1)))[0]
        # Set all logits below the kth token to -inf
        logits[logits < filter_values[:, [-1]]] = float('-inf')
    
    # Apply top-p (nucleus) filtering if specified (p > 0)
    if top_p > 0.0:
        # Sort logits in descending order
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        # Calculate cumulative probabilities
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
        
        # Create filter mask where cumulative probability exceeds threshold
        filter_mask = cumulative_probs > top_p
        # Shift the mask to exclude the first token that exceeds the threshold
        filter_mask[..., 1:] = filter_mask[..., :-1].clone()
        filter_mask[..., 0] = 0  # Always keep the most likely token
        
        # Map the filter back to original logits order
        indices_to_remove = filter_mask.scatter(1, sorted_indices, filter_mask)
        # Set filtered logits to -inf
        logits[indices_to_remove] = float('-inf')
    
    return logits


def norm_logits(logits: torch.Tensor, temperature: float, top_k: float, top_p: float) -> torch.Tensor:
    """Normalize logits using temperature and apply top-k/top-p filtering.
    
    This function applies temperature scaling to logits (higher temperature = more random),
    then applies top-k and top-p filtering, and finally converts to probabilities.
    
    Args:
        logits (torch.Tensor): Raw logits with shape (batch, vocab)
        temperature (float): Temperature parameter for scaling logits
        top_k (float): Number of highest probability tokens to keep
        top_p (float): Cumulative probability threshold for nucleus sampling

    Returns:
        torch.Tensor: Normalized probability distribution
    """
    assert logits.dim() == 2, "Logits must be a 2D tensor with shape (batch, vocab)"
    
    # Apply temperature scaling (higher temperature = more random)
    logits = logits / temperature
    
    # Apply top-k and top-p filtering
    logits = top_k_top_p_filter(logits, top_k=top_k, top_p=top_p)
    
    # Convert to probability distribution
    probs = F.softmax(logits, dim=1)
    
    return probs


def sample(probs: torch.Tensor, num_samples: int = 1):
    """Sample tokens from a probability distribution with robust error handling.
    
    This function samples tokens from a probability distribution, handling potential
    issues like NaN values, negative probabilities, or all-zero distributions.
    It also avoids sampling the padding token (0) when possible.
    
    Args:
        probs (torch.Tensor): Probability distribution with shape (batch, vocab)
        num_samples (int, optional): Number of samples to draw. Defaults to 1.

    Returns:
        torch.Tensor: Sampled token indices
    """
    # Handle potential issues with probabilities
    # Ensure no NaN, inf, or negative values
    probs = torch.where(torch.isnan(probs) | torch.isinf(probs) | (probs < 0), 
                       torch.zeros_like(probs), probs)
    
    # Ensure at least one non-zero probability
    if probs.sum() == 0:
        # If all probs are zero, set uniform distribution
        probs = torch.ones_like(probs) / probs.size(-1)
    
    # Renormalize to ensure valid probability distribution
    probs = probs / probs.sum(dim=-1, keepdim=True)
    
    # Sample from the distribution
    idx_next = torch.multinomial(probs, num_samples=num_samples)
    
    # Check for padding token (0) and avoid it if possible
    if (idx_next.item() == 0):
        # Zero out the probability of the padding token
        probs[0, 0] = 0
        # Renormalize the distribution
        probs = probs / probs.sum(dim=-1, keepdim=True)
        # Sample again
        idx_next = torch.multinomial(probs, num_samples=num_samples)
    
    return idx_next


def max_fn(x):
    """Normalize positive values in a tensor.
    
    This function zeroes out negative values and normalizes the result,
    effectively computing norm(max(x, 0)).
    
    Args:
        x (torch.Tensor): Input tensor
        
    Returns:
        torch.Tensor: Normalized tensor with negative values zeroed out
    """
    # Zero out negative values
    x_max = torch.where(x > 0, x, torch.zeros_like(x))
    # Compute sum for normalization
    x_max_sum = torch.sum(x_max, dim=1, keepdim=True) 
    # Normalize
    return x_max / x_max_sum
