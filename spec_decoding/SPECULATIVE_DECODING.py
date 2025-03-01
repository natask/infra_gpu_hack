#!/usr/bin/env python
"""
Speculative Inference Script

This script loads a draft model and an application model, then uses speculative decoding
to efficiently generate text. It provides an interactive interface for users to input
prompts and see the generated results, including both the generated text and tokens.

Use:
    python speculative_inference.py --draft_model gpt2 --application_model gpt2-medium
Ex:
    python SPECULATIVE_DECODING.py --draft_model gpt2 --application_model gpt2-medium --prompt "what is the difference between a lion and tiger" --output_file results.json

Metric Dectection Example:
    python run_metrics.py --draft_model gpt2 --
application_model gpt2-medium --prompt "Explain quantum computing in simple terms." --max_tokens 30 --gamma_v
alues 2,4 --runs 2

Bash: 

python speculative_inference.py \
  --draft_model gpt2 \
  --application_model gpt2-medium \
  --method google \
  --max_tokens 100 \
  --gamma 6 \
  --temperature 0.8 \
  --top_k 40 \
  --top_p 0.95 \
  --prompt "Your prompt here" \
  --output_file results.json

The script supports multiple speculative decoding methods and customizable generation parameters.
"""

import torch
import argparse
import time
from typing import Dict, List, Tuple, Optional, Union
from transformers import AutoTokenizer, AutoModelForCausalLM
from colorama import Fore, Style, init
import json

# Import sampling methods
from sampling import (
    autoregressive_sampling,
    speculative_sampling,
    speculative_sampling_v2,
    qspec_sampling_different_models
)
from globals import Decoder

# Initialize colorama
init()

class SpeculativeInferenceEngine:
    """
    Engine for running speculative inference with a draft model and an application model.
    
    This class handles loading models, tokenization, and running different speculative
    decoding methods. It provides methods for generating text from prompts and
    displaying detailed information about the generation process.
    """
    
    def __init__(
        self,
        draft_model_name: str,
        application_model_name: str,
        draft_quantization: str = "none",
        application_quantization: str = "none",
        device: Optional[str] = None,
        verbose: bool = False
    ):
        """
        Initialize the speculative inference engine.
        
        Args:
            draft_model_name: Name or path of the draft model
            application_model_name: Name or path of the application model
            draft_quantization: Quantization method for draft model
            application_quantization: Quantization method for application model
            device: Device to run models on ('cuda' or 'cpu')
            verbose: Whether to print verbose output
        """
        self.draft_model_name = draft_model_name
        self.application_model_name = application_model_name
        self.draft_quantization = draft_quantization
        self.application_quantization = application_quantization
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.verbose = verbose
        
        # Load tokenizer (using application model's tokenizer)
        self.tokenizer = AutoTokenizer.from_pretrained(application_model_name)
        Decoder().set_tokenizer(self.tokenizer)
        
        # Load models
        self._load_models()
        
        print(f"{Fore.GREEN}Speculative Inference Engine initialized:{Style.RESET_ALL}")
        print(f"  - Draft model: {draft_model_name} ({draft_quantization})")
        print(f"  - Application model: {application_model_name} ({application_quantization})")
        print(f"  - Device: {self.device}")
        print(f"  - Tokenizer vocabulary size: {len(self.tokenizer)}")
        
    def _load_models(self):
        """Load the draft and application models."""
        print(f"{Fore.YELLOW}Loading draft model: {self.draft_model_name}...{Style.RESET_ALL}")
        self.draft_model = AutoModelForCausalLM.from_pretrained(
            self.draft_model_name,
            trust_remote_code=True
        ).to(self.device)
        
        print(f"{Fore.YELLOW}Loading application model: {self.application_model_name}...{Style.RESET_ALL}")
        self.application_model = AutoModelForCausalLM.from_pretrained(
            self.application_model_name,
            trust_remote_code=True
        ).to(self.device)
        
        print(f"{Fore.GREEN}Models loaded successfully!{Style.RESET_ALL}")
    
    def generate(
        self,
        prompt: str,
        method: str = "google",
        max_tokens: int = 50,
        gamma: int = 4,
        temperature: float = 1.0,
        top_k: int = 20,
        top_p: float = 0.9,
        seed: Optional[int] = None,
        return_tokens: bool = True,
        add_system_message: bool = True
    ) -> Dict:
        """
        Generate text using speculative decoding.
        
        Args:
            prompt: Input text prompt
            method: Speculative decoding method ('google', 'deepmind', or 'qspec')
            max_tokens: Maximum number of tokens to generate
            gamma: Number of tokens to draft in each step
            temperature: Sampling temperature
            top_k: Top-k sampling parameter
            top_p: Top-p sampling parameter
            seed: Random seed for reproducibility
            return_tokens: Whether to return token IDs in the result
            
        Returns:
            Dictionary containing generation results and metadata
        """
        # Set random seed if provided
        if seed is not None:
            torch.manual_seed(seed)
        
        # Format prompt with system message if requested
        if add_system_message:
            formatted_prompt = f"<|system|>\nYou are a helpful assistant.\n<|end_of_system|>\n{prompt}"
        else:
            formatted_prompt = prompt
            
        # Encode input
        start_time = time.time()
        input_ids = self.tokenizer.encode(formatted_prompt, return_tensors='pt').to(self.device)
        input_length = input_ids.shape[1]
        
        # Select speculative decoding method
        if method == "google":
            output = speculative_sampling(
                input_ids,
                self.draft_model,
                self.application_model,
                max_len=max_tokens,
                gamma=gamma,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                verbose=self.verbose,
                random_seed=seed
            )
        elif method == "deepmind":
            output = speculative_sampling_v2(
                input_ids,
                self.draft_model,
                self.application_model,
                max_len=max_tokens,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                random_seed=seed
            )
        elif method == "qspec":
            output = qspec_sampling_different_models(
                input_ids,
                self.draft_model_name,
                self.application_model_name,
                draft_quantization=self.draft_quantization,
                verify_quantization=self.application_quantization,
                max_len=max_tokens,
                gamma=gamma,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                verbose=self.verbose,
                random_seed=seed,
                device=self.device
            )
        elif method == "autoregressive":
            # Baseline comparison using standard autoregressive sampling
            output = autoregressive_sampling(
                input_ids,
                self.application_model,
                max_tokens,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p
            )
        else:
            raise ValueError(f"Unknown method: {method}. Choose from 'google', 'deepmind', 'qspec', or 'autoregressive'.")
        
        # Calculate generation time
        generation_time = time.time() - start_time
        
        # Extract generated tokens (excluding input)
        generated_tokens = output[0, input_length:].tolist()
        
        # Decode full output and generated text
        full_text = self.tokenizer.decode(output[0], skip_special_tokens=True)
        generated_text = self.tokenizer.decode(output[0, input_length:], skip_special_tokens=True)
        
        # Calculate tokens per second
        tokens_per_second = len(generated_tokens) / generation_time if generation_time > 0 else 0
        
        # Prepare result
        result = {
            "full_text": full_text,
            "generated_text": generated_text,
            "generation_time_seconds": generation_time,
            "tokens_generated": len(generated_tokens),
            "tokens_per_second": tokens_per_second,
            "method": method,
            "parameters": {
                "max_tokens": max_tokens,
                "gamma": gamma,
                "temperature": temperature,
                "top_k": top_k,
                "top_p": top_p,
                "seed": seed,
                "add_system_message": add_system_message
            }
        }
        
        # Include token IDs if requested
        if return_tokens:
            result["input_tokens"] = input_ids[0].tolist()
            result["generated_tokens"] = generated_tokens
            
            # Add token strings for better interpretability
            result["input_token_strings"] = [self.tokenizer.decode([t]) for t in result["input_tokens"]]
            result["generated_token_strings"] = [self.tokenizer.decode([t]) for t in result["generated_tokens"]]
        
        return result
    
    def print_generation_result(self, result: Dict):
        """
        Print generation result in a formatted way.
        
        Args:
            result: Generation result dictionary from generate()
        """
        print("\n" + "="*80)
        print(f"{Fore.CYAN}Generation Result ({result['method']} method):{Style.RESET_ALL}")
        print("-"*80)
        
        # Print prompt and generated text
        print(f"{Fore.YELLOW}Prompt:{Style.RESET_ALL} {result['full_text'][:len(result['full_text'])-len(result['generated_text'])]}")
        print(f"{Fore.GREEN}Generated:{Style.RESET_ALL} {result['generated_text']}")
        print("-"*80)
        
        # Print generation stats
        print(f"Tokens generated: {result['tokens_generated']}")
        print(f"Generation time: {result['generation_time_seconds']:.3f} seconds")
        print(f"Speed: {result['tokens_per_second']:.2f} tokens/second")
        print("-"*80)
        
        # Print parameters
        params = result['parameters']
        system_msg_status = "ON" if params['add_system_message'] else "OFF"
        print(f"Parameters: max_tokens={params['max_tokens']}, gamma={params['gamma']}, " +
              f"temperature={params['temperature']}, top_k={params['top_k']}, top_p={params['top_p']}, " +
              f"system_message={system_msg_status}")
        
        # Print token details if available
        if "generated_token_strings" in result:
            print("-"*80)
            print(f"{Fore.CYAN}Generated tokens:{Style.RESET_ALL}")
            for i, (token_id, token_str) in enumerate(zip(result["generated_tokens"], result["generated_token_strings"])):
                print(f"  {i+1:3d}: {token_id:6d} → {repr(token_str)}")
        
        print("="*80 + "\n")

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Speculative Inference with Draft and Application Models')
    
    # Model configuration
    parser.add_argument('--draft_model', type=str, default="gpt2",
                        help='Name or path of the draft model')
    parser.add_argument('--application_model', type=str, default="gpt2-medium",
                        help='Name or path of the application model')
    parser.add_argument('--draft_quantization', type=str, default="none",
                        choices=["none", "int8", "int4", "fp8", "fp4"],
                        help='Quantization method for draft model')
    parser.add_argument('--application_quantization', type=str, default="none",
                        choices=["none", "int8", "int4", "fp8", "fp4"],
                        help='Quantization method for application model')
    
    # Generation parameters
    parser.add_argument('--method', type=str, default="google",
                        choices=["google", "deepmind", "qspec", "autoregressive"],
                        help='Speculative decoding method')
    parser.add_argument('--max_tokens', type=int, default=50,
                        help='Maximum number of tokens to generate')
    parser.add_argument('--gamma', type=int, default=4,
                        help='Number of tokens to draft in each step')
    parser.add_argument('--temperature', type=float, default=1.0,
                        help='Sampling temperature')
    parser.add_argument('--top_k', type=int, default=20,
                        help='Top-k sampling parameter')
    parser.add_argument('--top_p', type=float, default=0.9,
                        help='Top-p sampling parameter')
    parser.add_argument('--seed', type=int, default=None,
                        help='Random seed for reproducibility')
    
    # Other options
    parser.add_argument('--verbose', action='store_true',
                        help='Print verbose output')
    parser.add_argument('--interactive', action='store_true',
                        help='Run in interactive mode')
    parser.add_argument('--prompt', type=str, default=None,
                        help='Input prompt (for non-interactive mode)')
    parser.add_argument('--output_file', type=str, default=None,
                        help='Save results to this JSON file')
    parser.add_argument('--no-system-message', dest='add_system_message', action='store_false',
                        help='Do not add system message to prompts')
    parser.set_defaults(add_system_message=True)
    
    return parser.parse_args()

def interactive_mode(engine, args):
    """Run the engine in interactive mode."""
    print(f"\n{Fore.CYAN}=== Speculative Inference Interactive Mode ==={Style.RESET_ALL}")
    print("Type 'exit' or 'quit' to exit.")
    print("Type 'params' to view current parameters.")
    print("Type 'set <param> <value>' to change a parameter.")
    print("Type 'methods' to see available decoding methods.")
    print("Type 'save <filename>' to save the last result to a file.")
    print("Type 'toggle system' to toggle system message on/off.")
    
    # Parameters that can be changed
    params = {
        "method": args.method,
        "max_tokens": args.max_tokens,
        "gamma": args.gamma,
        "temperature": args.temperature,
        "top_k": args.top_k,
        "top_p": args.top_p,
        "seed": args.seed,
        "add_system_message": True
    }
    
    last_result = None
    
    while True:
        try:
            user_input = input(f"\n{Fore.YELLOW}Enter prompt>{Style.RESET_ALL} ")
            
            # Handle special commands
            if user_input.lower() in ['exit', 'quit']:
                print("Exiting...")
                break
            elif user_input.lower() == 'params':
                print(f"\n{Fore.CYAN}Current parameters:{Style.RESET_ALL}")
                for param, value in params.items():
                    print(f"  {param}: {value}")
                continue
            elif user_input.lower() == 'methods':
                print(f"\n{Fore.CYAN}Available methods:{Style.RESET_ALL}")
                print("  google - Google's speculative sampling")
                print("  deepmind - DeepMind's speculative sampling")
                print("  qspec - QSPEC sampling with different models")
                print("  autoregressive - Standard autoregressive sampling")
                continue
            elif user_input.lower().startswith('set '):
                parts = user_input.split(' ', 2)
                if len(parts) != 3:
                    print(f"{Fore.RED}Invalid command. Use 'set <param> <value>'{Style.RESET_ALL}")
                    continue
                
                param = parts[1]
                value = parts[2]
                
                if param not in params:
                    print(f"{Fore.RED}Unknown parameter: {param}{Style.RESET_ALL}")
                    continue
                
                # Convert value to appropriate type
                try:
                    if param in ['max_tokens', 'gamma', 'top_k', 'seed']:
                        value = int(value) if value.lower() != 'none' else None
                    elif param in ['temperature', 'top_p']:
                        value = float(value)
                    
                    params[param] = value
                    print(f"{Fore.GREEN}Set {param} = {value}{Style.RESET_ALL}")
                except ValueError:
                    print(f"{Fore.RED}Invalid value for {param}: {value}{Style.RESET_ALL}")
                
                continue
            elif user_input.lower() == 'toggle system':
                params["add_system_message"] = not params["add_system_message"]
                status = "ON" if params["add_system_message"] else "OFF"
                print(f"{Fore.GREEN}System message is now {status}{Style.RESET_ALL}")
                continue
            elif user_input.lower().startswith('save '):
                if last_result is None:
                    print(f"{Fore.RED}No result to save yet.{Style.RESET_ALL}")
                    continue
                
                filename = user_input.split(' ', 1)[1]
                try:
                    with open(filename, 'w') as f:
                        json.dump(last_result, f, indent=2)
                    print(f"{Fore.GREEN}Result saved to {filename}{Style.RESET_ALL}")
                except Exception as e:
                    print(f"{Fore.RED}Error saving to {filename}: {e}{Style.RESET_ALL}")
                
                continue
            
            # Generate text with current parameters
            result = engine.generate(
                prompt=user_input,
                method=params["method"],
                max_tokens=params["max_tokens"],
                gamma=params["gamma"],
                temperature=params["temperature"],
                top_k=params["top_k"],
                top_p=params["top_p"],
                seed=params["seed"],
                return_tokens=True,
                add_system_message=params.get("add_system_message", True)
            )
            
            # Store result for potential saving
            last_result = result
            
            # Print the result
            engine.print_generation_result(result)
            
        except KeyboardInterrupt:
            print("\nExiting...")
            break
        except Exception as e:
            print(f"{Fore.RED}Error: {e}{Style.RESET_ALL}")

def main():
    """Main function."""
    args = parse_arguments()
    
    # Initialize the engine
    engine = SpeculativeInferenceEngine(
        draft_model_name=args.draft_model,
        application_model_name=args.application_model,
        draft_quantization=args.draft_quantization,
        application_quantization=args.application_quantization,
        verbose=args.verbose
    )
    
    # Run in interactive or single-prompt mode
    if args.interactive:
        interactive_mode(engine, args)
    elif args.prompt:
        # Generate with single prompt
        result = engine.generate(
            prompt=args.prompt,
            method=args.method,
            max_tokens=args.max_tokens,
            gamma=args.gamma,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            seed=args.seed,
            return_tokens=True,
            add_system_message=args.add_system_message
        )
        
        # Print the result
        engine.print_generation_result(result)
        
        # Save to file if requested
        if args.output_file:
            try:
                with open(args.output_file, 'w') as f:
                    json.dump(result, f, indent=2)
                print(f"{Fore.GREEN}Result saved to {args.output_file}{Style.RESET_ALL}")
            except Exception as e:
                print(f"{Fore.RED}Error saving to {args.output_file}: {e}{Style.RESET_ALL}")
    else:
        # No prompt provided, default to interactive mode
        print(f"{Fore.YELLOW}No prompt provided. Starting interactive mode...{Style.RESET_ALL}")
        interactive_mode(engine, args)

if __name__ == "__main__":
    main()
