#!/usr/bin/env python3
import os
import argparse
import csv
import json
from custom_dataset import get_dataloader
from transformers import LlamaForCausalLM, AutoTokenizer
import torch
import tqdm
import pandas as pd
import time
from torch.nn.attention import SDPBackend, sdpa_kernel
import multiprocessing

# Only enable flash attention backend
os.environ['HF_HOME'] = '/mount/model-cache'

def process_on_gpu(gpu_id, args, start_index, end_index=None):
    """Process a subset of the dataset on a specific GPU."""
    # Set the device
    device = f'cuda:{gpu_id}' if torch.cuda.is_available() else 'cpu'
    
    # Load model and tokenizer
    model = LlamaForCausalLM.from_pretrained(args.model_name).to(device)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model.eval()
    print(f"GPU {gpu_id} - CUDA is available: {torch.cuda.is_available()}")
    
    # Set generation parameters
    num_beams = 4
    no_repeat_ngram_size = 3
    early_stopping = True
    
    # Prepare output files
    messages_output_path = f'{args.name}__messages_output{gpu_id}.csv'
    time_output_path = f'{args.name}__time_output{gpu_id}.csv'

    # Create files with headers
    with open(messages_output_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['role', 'content', 'info'])
    
    with open(time_output_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['question', 'generation_time', 'tokens_per_second'])
    
    # Load dataset
    base_dir = os.path.dirname(os.path.abspath(__file__))
    df = pd.read_parquet(os.path.join(base_dir, "../", args.dataset_name))
    
    # Limit to the assigned subset
    if end_index is not None:
        df = df.iloc[start_index:end_index]
    else:
        df = df.iloc[start_index:]
    
    total_start_time = time.time()
    
    # Process each entry
    for index, row in df.iterrows():
        question = row['question']
        question = f"""<|begin_of_text|>
<|system|>
You are a helpful assistant.
<|end_of_system|>

{question}

<|end_of_text|>"""
        inputs = tokenizer(question, return_tensors="pt").to(device)
        start_time = time.time()
        
        # Custom generation to capture logits
        input_ids = inputs.input_ids
        attention_mask = torch.ones_like(input_ids)
        generated_tokens = []
        token_info = []
        top_k = 5  # Number of top tokens to save
        
        with torch.no_grad():
            # Start with the input sequence
            current_ids = input_ids.clone()
            
            # Generate until max length or end token
            for _ in range(args.max_length - input_ids.size(1)):
                # Get model outputs
                outputs = model(input_ids=current_ids, attention_mask=attention_mask)
                next_token_logits = outputs.logits[:, -1, :]
                
                # Get top-k tokens and their logits
                topk_logits, topk_indices = torch.topk(next_token_logits, top_k, dim=-1)
                
                # Convert to lists for storage
                topk_tokens = [tokenizer.decode([idx.item()]) for idx in topk_indices[0]]
                topk_ids = topk_indices[0].tolist()
                topk_logit_values = topk_logits[0].tolist()
                
                # Store the information for this position
                token_info.append([topk_tokens, [topk_ids], [topk_logit_values]])
                
                # Select the next token (top-1 for greedy decoding)
                next_token = topk_indices[0, 0].unsqueeze(0).unsqueeze(0)
                generated_tokens.append(next_token.item())
                
                # Update the input sequence
                current_ids = torch.cat([current_ids, next_token], dim=1)
                attention_mask = torch.cat([attention_mask, torch.ones_like(next_token)], dim=1)
                
                # Check if we've generated an end token
                if next_token.item() == tokenizer.eos_token_id:
                    break
        
        generation_time = time.time() - start_time
        
        # Decode the generated solution
        solution = tokenizer.decode(generated_tokens, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        
        # Write results to files
        with open(messages_output_path, 'a', newline='') as messages_file, open(time_output_path, 'a', newline='') as time_file:
            # Create CSV writers
            messages_writer = csv.writer(messages_file)
            time_writer = csv.writer(time_file)
            
            # Write user message (with empty info field)
            messages_writer.writerow(['user', question, ''])
            
            # Write assistant response with token info
            token_info_json = json.dumps(token_info)
            messages_writer.writerow(['assistant', solution, token_info_json])
            
            print(f"GPU {gpu_id} - Index {index}: Generated response")
            
            # Calculate tokens per second
            num_tokens = inputs.input_ids.numel()
            tokens_per_second = num_tokens / generation_time
            
            # Record and save time elapsed
            time_writer.writerow([question, generation_time, tokens_per_second])
            
            # Print the current index for easy resuming
            print(f"GPU {gpu_id} - Processed index {index}. To resume from this GPU's subset, use --start_index {index+1}")
    
    total_generation_time = time.time() - total_start_time
    print(f"GPU {gpu_id} - Total generation time: {total_generation_time}")

def main():
    print(torch.backends.cuda.flash_sdp_enabled())
    # True
    print(torch.backends.cuda.mem_efficient_sdp_enabled())
    # True
    
    base_dir = os.path.dirname(os.path.abspath(__file__))
    
    with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
        parser = argparse.ArgumentParser(description="Generate model outputs using multiple GPUs")
        parser.add_argument("--model_name", type=str, default="casperhansen/Llama-3.3-70B-instruct-awq", help="Hugging Face model name or path")
        parser.add_argument("--dataset_name", type=str, default='final_dataset.parquet', help="Name of the dataset file (without path) in parquet")
        parser.add_argument("--batch_size", type=int, default=8, help="Batch size for inference")
        parser.add_argument("--name", type=str, default="llama70b", help="Model string name")
        parser.add_argument("--max_length", type=int, default=2080, help="Max generation length")
        parser.add_argument("--start_index", type=int, default=0, help="Starting index for processing prompts")
    args = parser.parse_args()
    
    # Determine the number of available GPUs
    num_gpus = torch.cuda.device_count()
    print(f"Number of available GPUs: {num_gpus}")
    
    # Load dataset to get total size
    df = pd.read_parquet(os.path.join(base_dir, "../", args.dataset_name))
    total_entries = len(df)
    
    # Calculate entries per GPU
    entries_per_gpu = total_entries // num_gpus
    remainder = total_entries % num_gpus
    
    # Create processes for each GPU
    processes = []
    start_idx = args.start_index
    
    for gpu_id in range(num_gpus):
        # Calculate the range for this GPU
        end_idx = start_idx + entries_per_gpu
        if gpu_id < remainder:
            end_idx += 1
        
        # Create and start the process
        p = multiprocessing.Process(
            target=process_on_gpu,
            args=(gpu_id, args, start_idx, end_idx)
        )
        processes.append(p)
        p.start()
        
        # Update start index for next GPU
        start_idx = end_idx
    
    # Wait for all processes to complete
    for p in processes:
        p.join()
    
    print("All GPUs have completed processing.")

if __name__ == '__main__':
    multiprocessing.set_start_method('spawn')
    main()
