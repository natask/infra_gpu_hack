import torch
import argparse
from transformers import AutoTokenizer
from colorama import Fore, Style

from sampling import (
    autoregressive_sampling,
    qspec_sampling,
    qspec_sampling_different_models,
    QuantizedModel
)
from globals import Decoder

def parse_arguments():
    parser = argparse.ArgumentParser(description='QSPEC Sampling Example')
    
    parser.add_argument('--input', type=str, default="The quick brown fox jumps over the lazy ")
    parser.add_argument('--model_name', type=str, default="gpt2")
    parser.add_argument('--draft_model_name', type=str, default="gpt2")
    parser.add_argument('--verify_model_name', type=str, default="gpt2-medium")
    parser.add_argument('--draft_quantization', type=str, default="fp8", 
                        choices=["none", "int8", "int4", "fp8", "fp4"])
    parser.add_argument('--verify_quantization', type=str, default="int8", 
                        choices=["none", "int8", "int4", "fp8", "fp4"])
    parser.add_argument('--max_tokens', '-M', type=int, default=20)
    parser.add_argument('--gamma', '-g', type=int, default=4)
    parser.add_argument('--verbose', '-v', action='store_true', default=False)
    parser.add_argument('--seed', '-s', type=int, default=42)
    parser.add_argument('--same_model', action='store_true', default=False)
    
    return parser.parse_args()

def color_print(text, color=Fore.GREEN):
    print(color + text + Style.RESET_ALL)

def main():
    args = parse_arguments()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Set up tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    Decoder().set_tokenizer(tokenizer)
    
    # Encode input text
    input_ids = tokenizer.encode(args.input, return_tensors='pt').to(device)
    
    print(f"Input text: {args.input}")
    print(f"Device: {device}")
    
    # Sampling parameters
    top_k = 20
    top_p = 0.9
    
    # Set random seed for reproducibility
    torch.manual_seed(args.seed)
    
    # Run autoregressive sampling for comparison
    if not args.same_model:
        print("\n" + "="*50)
        color_print("Running autoregressive sampling with verification model...", Fore.BLUE)
        verify_model = QuantizedModel(
            args.verify_model_name,
            quantization_method=args.verify_quantization,
            device=device
        )
        
        auto_output = autoregressive_sampling(
            input_ids, 
            verify_model, 
            args.max_tokens, 
            top_k=top_k, 
            top_p=top_p
        )
        auto_text = tokenizer.decode(auto_output[0], skip_special_tokens=True)
        color_print(f"Autoregressive output: {auto_text}", Fore.BLUE)
    
    # Run QSPEC sampling
    print("\n" + "="*50)
    if args.same_model:
        color_print(f"Running QSPEC sampling with same model ({args.model_name})...", Fore.MAGENTA)
        torch.manual_seed(args.seed)
        qspec_output = qspec_sampling(
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
    else:
        color_print(f"Running QSPEC sampling with different models...", Fore.MAGENTA)
        color_print(f"Draft model: {args.draft_model_name} ({args.draft_quantization})", Fore.YELLOW)
        color_print(f"Verification model: {args.verify_model_name} ({args.verify_quantization})", Fore.YELLOW)
        
        torch.manual_seed(args.seed)
        qspec_output = qspec_sampling_different_models(
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
    
    qspec_text = tokenizer.decode(qspec_output[0], skip_special_tokens=True)
    color_print(f"QSPEC output: {qspec_text}", Fore.MAGENTA)
    
    print("\n" + "="*50)
    print("Configuration:")
    print(f"  - Draft quantization: {args.draft_quantization}")
    print(f"  - Verification quantization: {args.verify_quantization}")
    print(f"  - Gamma (tokens to draft): {args.gamma}")
    print(f"  - Max tokens to generate: {args.max_tokens}")
    print("="*50)

if __name__ == "__main__":
    main()
