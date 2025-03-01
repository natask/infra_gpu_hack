"""Autoregressive sampling implementation for language models.

This module provides the standard autoregressive sampling approach where tokens
are generated one at a time using the model's predictions.
"""

import torch
from tqdm import tqdm
from sampling.utils import norm_logits, sample


@torch.no_grad()
def autoregressive_sampling(x: torch.Tensor, model: torch.nn.Module, N: int,
                            temperature: float = 1, top_k: int = 0, top_p: float = 0):
    """Generate tokens using standard autoregressive sampling.
    
    This function generates tokens one at a time using the provided model,
    applying temperature, top-k, and top-p filtering to control randomness.
    It uses key-value caching for efficiency when available.
    
    Args:
        x (torch.Tensor): Input token ids with shape (batch_size, seq_len)
        model (torch.nn.Module): Language model that implements the HuggingFace interface
        N (int): Number of new tokens to generate
        temperature (float, optional): Temperature for sampling. Defaults to 1.
        top_k (int, optional): Number of highest probability tokens to keep. Defaults to 0 (disabled).
        top_p (float, optional): Cumulative probability threshold. Defaults to 0 (disabled).
    
    Returns:
        torch.Tensor: Extended sequence with generated tokens appended
    """
    # Current sequence length
    n = len(x)
    # Target sequence length after generation
    T = len(x) + N

    # Initialize KV-cache for efficient generation
    past_key_values = None
    
    # Generate tokens until we reach the target length
    while n < T:
        # Use KV-cache for efficient generation when available
        if past_key_values:
            # Only need to process the last token with KV-cache
            last_ids = x[:, -1]
            # Ensure proper dimensions
            if last_ids.dim() == 1:
                last_ids = torch.unsqueeze(last_ids, 0)
            # Forward pass with KV-cache
            outputs = model(last_ids, past_key_values=past_key_values, use_cache=True)
        else:
            # First pass processes the entire input sequence
            outputs = model(x)
        
        # Get normalized probabilities for the next token
        last_p = norm_logits(outputs.logits[::, -1, :], temperature, top_k, top_p)
        
        # Update KV-cache for next iteration
        past_key_values = outputs.past_key_values
        
        # Sample the next token
        idx_next = sample(last_p)
        
        # Append the new token to the sequence
        x = torch.cat((x, idx_next), dim=1)
        
        # Update current length
        n += 1
        
    return x

