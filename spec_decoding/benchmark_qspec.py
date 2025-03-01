import torch
import argparse
import contexttimer
import logging
from colorama import Fore, Style
from transformers import AutoTokenizer

from sampling import (
    autoregressive_sampling, 
    speculative_sampling, 
    speculative_sampling_v2,
    qspec_sampling,
    qspec_sampling_different_models,
    QuantizedModel,
    QSPECModel
)
from globals import Decoder

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def parse_arguments():
    parser = argparse.ArgumentParser(description='Benchmark for QSPEC sampling')

    parser.add_argument('--input', type=str, default="The quick brown fox jumps over the lazy ")
    parser.add_argument('--model_name', type=str, default="gpt2")
    parser.add_argument('--draft_model_name', type=str, default="gpt2")
    parser.add_argument('--verify_model_name', type=str, default="gpt2-medium")
    parser.add_argument('--draft_quantization', type=str, default="fp8", 
                        choices=["none", "int8", "int4", "fp8", "fp4"],
                        help='Quantization method for the draft model')
    parser.add_argument('--verify_quantization', type=str, default="int8", 
                        choices=["none", "int8", "int4", "fp8", "fp4"],
                        help='Quantization method for the verification model')
    parser.add_argument('--verbose', '-v', action='store_true', default=False, 
                        help='Enable verbose mode')
    parser.add_argument('--seed', '-s', type=int, default=None, 
                        help='Set a random seed for reproducibility')
    parser.add_argument('--max_tokens', '-M', type=int, default=20, 
                        help='Max token number to generate')
    parser.add_argument('--gamma', '-g', type=int, default=4, 
                        help='Number of tokens to draft in each step')
    parser.add_argument('--same_model', action='store_true', default=False,
                        help='Use the same model for both draft and verification')
    parser.add_argument('--test_all', action='store_true', default=False,
                        help='Test all sampling methods for comparison')
    
    args = parser.parse_args()
    return args


def color_print(text, color=Fore.RED):
    print(color + text + Style.RESET_ALL)


def benchmark(fn, print_prefix, iterations=5, *args, **kwargs):
    """Run a benchmark for a given function."""
    with contexttimer.Timer() as t:
        total_tokens = 0
        for _ in range(iterations):
            output = fn(*args, **kwargs)
            total_tokens += len(output[0]) - kwargs.get('input_ids', args[0]).shape[1]
    
    tokens_per_sec = total_tokens / t.elapsed
    avg_time = t.elapsed / iterations
    
    print(f"\n[benchmark] {print_prefix}")
    print(f"  - Tokens/sec: {tokens_per_sec:.2f}")
    print(f"  - Avg time per run: {avg_time:.4f} sec")
    print(f"  - Total tokens generated: {total_tokens}")
    
    return tokens_per_sec, avg_time, total_tokens


def run_benchmarks(args):
    """Run benchmarks for different sampling methods."""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Set up tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    Decoder().set_tokenizer(tokenizer)
    
    # Encode input text
    input_ids = tokenizer.encode(args.input, return_tensors='pt').to(device)
    
    # Sampling parameters
    top_k = 20
    top_p = 0.9
    
    results = {}
    
    # Test autoregressive sampling with verification model
    if args.test_all or not args.same_model:
        logger.info(f"Loading verification model: {args.verify_model_name} with {args.verify_quantization} quantization")
        verify_model = QuantizedModel(
            args.verify_model_name,
            quantization_method=args.verify_quantization,
            device=device
        )
        
        torch.manual_seed(args.seed or 123)
        output = autoregressive_sampling(input_ids, verify_model, args.max_tokens, top_k=top_k, top_p=top_p)
        generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
        color_print(f"Verification model autoregressive sampling: {generated_text}", Fore.BLUE)
        
        results['verify_autoregressive'] = benchmark(
            autoregressive_sampling,
            f"Verification model ({args.verify_quantization}) autoregressive",
            5,
            input_ids,
            verify_model,
            args.max_tokens,
            top_k=top_k,
            top_p=top_p
        )
    
    # Test autoregressive sampling with draft model
    if args.test_all:
        logger.info(f"Loading draft model: {args.draft_model_name} with {args.draft_quantization} quantization")
        draft_model = QuantizedModel(
            args.draft_model_name,
            quantization_method=args.draft_quantization,
            device=device
        )
        
        torch.manual_seed(args.seed or 123)
        output = autoregressive_sampling(input_ids, draft_model, args.max_tokens, top_k=top_k, top_p=top_p)
        generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
        color_print(f"Draft model autoregressive sampling: {generated_text}", Fore.GREEN)
        
        results['draft_autoregressive'] = benchmark(
            autoregressive_sampling,
            f"Draft model ({args.draft_quantization}) autoregressive",
            5,
            input_ids,
            draft_model,
            args.max_tokens,
            top_k=top_k,
            top_p=top_p
        )
    
    # Test standard speculative sampling
    if args.test_all and not args.same_model:
        logger.info("Testing standard speculative sampling")
        torch.manual_seed(args.seed or 123)
        output = speculative_sampling(
            input_ids, 
            draft_model, 
            verify_model, 
            args.max_tokens, 
            gamma=args.gamma, 
            top_k=top_k, 
            top_p=top_p, 
            random_seed=args.seed,
            verbose=args.verbose
        )
        generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
        color_print(f"Standard speculative sampling: {generated_text}", Fore.YELLOW)
        
        results['standard_speculative'] = benchmark(
            speculative_sampling,
            "Standard speculative sampling",
            5,
            input_ids,
            draft_model,
            verify_model,
            args.max_tokens,
            gamma=args.gamma,
            top_k=top_k,
            top_p=top_p,
            random_seed=args.seed
        )
    
    # Test QSPEC sampling
    if args.same_model:
        logger.info(f"Testing QSPEC sampling with same model: {args.model_name}")
        torch.manual_seed(args.seed or 123)
        output = qspec_sampling(
            input_ids,
            args.model_name,
            draft_quantization=args.draft_quantization,
            verify_quantization=args.verify_quantization,
            max_len=args.max_tokens,
            gamma=args.gamma,
            top_k=top_k,
            top_p=top_p,
            verbose=args.verbose,
            random_seed=args.seed,
            device=device
        )
        generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
        color_print(f"QSPEC sampling (same model): {generated_text}", Fore.MAGENTA)
        
        results['qspec_same_model'] = benchmark(
            qspec_sampling,
            f"QSPEC sampling (same model: {args.model_name})",
            5,
            input_ids,
            args.model_name,
            draft_quantization=args.draft_quantization,
            verify_quantization=args.verify_quantization,
            max_len=args.max_tokens,
            gamma=args.gamma,
            top_k=top_k,
            top_p=top_p,
            random_seed=args.seed,
            device=device
        )
    else:
        logger.info(f"Testing QSPEC sampling with different models: {args.draft_model_name} and {args.verify_model_name}")
        torch.manual_seed(args.seed or 123)
        output = qspec_sampling_different_models(
            input_ids,
            args.draft_model_name,
            args.verify_model_name,
            draft_quantization=args.draft_quantization,
            verify_quantization=args.verify_quantization,
            max_len=args.max_tokens,
            gamma=args.gamma,
            top_k=top_k,
            top_p=top_p,
            verbose=args.verbose,
            random_seed=args.seed,
            device=device
        )
        generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
        color_print(f"QSPEC sampling (different models): {generated_text}", Fore.MAGENTA)
        
        results['qspec_different_models'] = benchmark(
            qspec_sampling_different_models,
            f"QSPEC sampling (draft: {args.draft_model_name}, verify: {args.verify_model_name})",
            5,
            input_ids,
            args.draft_model_name,
            args.verify_model_name,
            draft_quantization=args.draft_quantization,
            verify_quantization=args.verify_quantization,
            max_len=args.max_tokens,
            gamma=args.gamma,
            top_k=top_k,
            top_p=top_p,
            random_seed=args.seed,
            device=device
        )
    
    # Print summary of results
    print("\n" + "="*50)
    print("BENCHMARK SUMMARY")
    print("="*50)
    
    for name, (tokens_per_sec, avg_time, total_tokens) in results.items():
        print(f"{name}:")
        print(f"  - Tokens/sec: {tokens_per_sec:.2f}")
        print(f"  - Avg time: {avg_time:.4f} sec")
        print("-"*50)
    
    # Calculate speedups if we have both verification autoregressive and QSPEC results
    if 'verify_autoregressive' in results and ('qspec_same_model' in results or 'qspec_different_models' in results):
        base_tokens_per_sec = results['verify_autoregressive'][0]
        
        if 'qspec_same_model' in results:
            qspec_tokens_per_sec = results['qspec_same_model'][0]
            speedup = qspec_tokens_per_sec / base_tokens_per_sec
            print(f"QSPEC (same model) speedup over verification model: {speedup:.2f}x")
        
        if 'qspec_different_models' in results:
            qspec_tokens_per_sec = results['qspec_different_models'][0]
            speedup = qspec_tokens_per_sec / base_tokens_per_sec
            print(f"QSPEC (different models) speedup over verification model: {speedup:.2f}x")
        
        if 'standard_speculative' in results:
            spec_tokens_per_sec = results['standard_speculative'][0]
            speedup = spec_tokens_per_sec / base_tokens_per_sec
            print(f"Standard speculative speedup over verification model: {speedup:.2f}x")
            
            if 'qspec_different_models' in results:
                qspec_speedup = qspec_tokens_per_sec / spec_tokens_per_sec
                print(f"QSPEC speedup over standard speculative: {qspec_speedup:.2f}x")


if __name__ == "__main__":
    args = parse_arguments()
    run_benchmarks(args)
