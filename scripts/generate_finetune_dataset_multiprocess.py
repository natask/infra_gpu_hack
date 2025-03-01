#!/usr/bin/env python3
import os
import argparse
import csv
import json
import gc
import psutil
from custom_dataset import get_dataloader
from transformers import LlamaForCausalLM, AutoTokenizer, BitsAndBytesConfig
import torch
import tqdm
import pandas as pd
import time
from torch.nn.attention import SDPBackend, sdpa_kernel
import multiprocessing
from concurrent.futures import ThreadPoolExecutor

# Only enable flash attention backend
os.environ['HF_HOME'] = '/mount/model-cache'

def process_on_gpu(gpu_id, args, num_gpus):
    """Process a subset of the dataset on a specific GPU."""
    # Set the device
    device = f'cuda:{gpu_id}' if torch.cuda.is_available() else 'cpu'
    
    # Monitor memory usage
    process = psutil.Process(os.getpid())
    print(f"GPU {gpu_id} - Initial memory usage: {process.memory_info().rss / 1024 / 1024:.2f} MB")
    
    # Configure model loading options for memory efficiency
    if args.use_4bit:
        # Use 4-bit quantization for memory efficiency
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True
        )
        model = LlamaForCausalLM.from_pretrained(
            args.model_name,
            quantization_config=bnb_config,
            device_map=device
        )
    else:
        # Use half precision (16-bit) for better performance
        model = LlamaForCausalLM.from_pretrained(
            args.model_name, 
            torch_dtype=torch.float16
        ).to(device)
    
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model.eval()
    print(f"GPU {gpu_id} - CUDA is available: {torch.cuda.is_available()}")
    print(f"GPU {gpu_id} - Model loaded successfully")
    
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
        writer.writerow(['prompt', 'response', 'logits_data'])
    
    with open(time_output_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['question', 'generation_time', 'tokens_per_second'])
    
    # Get dataset path
    base_dir = os.path.dirname(os.path.abspath(__file__))
    dataset_path = os.path.join(base_dir, "../", args.dataset_name)
    
    # Get total number of rows without loading entire dataset
    total_rows = pd.read_parquet(dataset_path, columns=[]).shape[0]
    print(f"GPU {gpu_id} - Total rows in dataset: {total_rows}")
    
    # Process in chunks to avoid memory issues
    chunk_size = args.chunk_size  # Process this many rows at once
    
    total_start_time = time.time()
    
    # Process entries using striding approach with chunking
    for chunk_start in range(0, total_rows, chunk_size):
        chunk_end = min(chunk_start + chunk_size, total_rows)
        print(f"GPU {gpu_id} - Processing chunk {chunk_start}-{chunk_end}")
        
        # Load only the current chunk
        df_chunk = pd.read_parquet(dataset_path, engine='pyarrow')
        # Filter to only the current chunk
        df_chunk = df_chunk.iloc[chunk_start:chunk_end]
        
        # Process entries in this chunk
        for index, row in df_chunk.iterrows():
            # Only process entries where (index % num_gpus) == gpu_id
            if index % num_gpus != gpu_id:
                continue
                
            # Skip entries before the start index
            if index < args.start_index:
                continue
            
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
            logits_data = []
            top_k = 5  # Number of top tokens to save
            
            with torch.inference_mode():  # More efficient than no_grad
                # Start with the input sequence
                current_ids = input_ids.clone()
                
                # Generate until max length or end token
                for position in range(args.max_length - input_ids.size(1)):
                    # Get model outputs
                    outputs = model(input_ids=current_ids, attention_mask=attention_mask)
                    next_token_logits = outputs.logits[:, -1, :]
                    
                    # Get top-k tokens and their logits
                    topk_logits, topk_indices = torch.topk(next_token_logits, top_k, dim=-1)
                    
                    # Select the next token (top-1 for greedy decoding)
                    chosen_token_id = topk_indices[0, 0].item()
                    chosen_token = tokenizer.decode([chosen_token_id])
                    generated_tokens.append(chosen_token_id)
                    
                    # Convert to lists for storage - detach and move to CPU immediately
                    top_5_data = []
                    for i in range(top_k):
                        token_id = topk_indices[0, i].item()
                        token = tokenizer.decode([token_id])
                        logit = topk_logits[0, i].item()
                        top_5_data.append({
                            "token": token,
                            "token_id": token_id,
                            "logit": logit
                        })
                    
                    # Get full logits (convert to Python list)
                    full_logits = next_token_logits[0].detach().cpu().tolist()
                    
                    # Store the information for this position
                    position_data = {
                        "position": position,
                        "chosen_token": chosen_token,
                        "chosen_token_id": chosen_token_id,
                        "top_5": top_5_data,
                        "full_logits": full_logits
                    }
                    logits_data.append(position_data)
                    
                    # Update the input sequence more efficiently
                    new_ids = torch.zeros((1, current_ids.size(1) + 1), dtype=current_ids.dtype, device=current_ids.device)
                    new_ids[0, :-1] = current_ids[0]
                    new_ids[0, -1] = chosen_token_id
                    current_ids = new_ids
                    
                    new_mask = torch.ones((1, attention_mask.size(1) + 1), dtype=attention_mask.dtype, device=attention_mask.device)
                    new_mask[0, :-1] = attention_mask[0]
                    attention_mask = new_mask
                    
                    # Free memory
                    del outputs
                    
                    # Check if we've generated an end token
                    if chosen_token_id == tokenizer.eos_token_id:
                        break
        
        generation_time = time.time() - start_time
        
        # Decode the generated solution
        solution = tokenizer.decode(generated_tokens, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        
        # Write results to files
        with open(messages_output_path, 'a', newline='') as messages_file, open(time_output_path, 'a', newline='') as time_file:
            # Create CSV writers
            messages_writer = csv.writer(messages_file)
            time_writer = csv.writer(time_file)
            
            # Format prompt with special tokens
            formatted_prompt = f"<BOS><start_id>user<end_id>\n{question}<eot_id><start_id>assistant<end_id>\n"
            
            # Format response with special tokens
            formatted_response = f"{solution}<EOS>"
            
            # Write to CSV
            logits_data_json = json.dumps(logits_data)
            messages_writer.writerow([formatted_prompt, formatted_response, logits_data_json])
            
            print(f"GPU {gpu_id} - Index {index}: Generated response")
            
            # Calculate tokens per second
            num_tokens = inputs.input_ids.numel()
            tokens_per_second = num_tokens / generation_time
            
            # Record and save time elapsed
            time_writer.writerow([question, generation_time, tokens_per_second])
            
            # Print the current index for easy resuming
            print(f"GPU {gpu_id} - Processed index {index}. To resume from this GPU's subset, use --start_index {index+1}")
            
            # Clear memory periodically
            if index % 10 == 0:
                torch.cuda.empty_cache()
                gc.collect()
                current_mem = process.memory_info().rss / 1024 / 1024
                print(f"GPU {gpu_id} - Memory usage after index {index}: {current_mem:.2f} MB")
                print(f"GPU {gpu_id} - GPU memory allocated: {torch.cuda.memory_allocated(device) / 1024 / 1024:.2f} MB")
    
        # Clear dataframe chunk to free memory
        del df_chunk
        gc.collect()
        torch.cuda.empty_cache()
    
    # Final cleanup
    del model, tokenizer
    torch.cuda.empty_cache()
    gc.collect()
    
    total_generation_time = time.time() - total_start_time
    print(f"GPU {gpu_id} - Total generation time: {total_generation_time}")
    print(f"GPU {gpu_id} - Final memory usage: {process.memory_info().rss / 1024 / 1024:.2f} MB")

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
        parser.add_argument("--chunk_size", type=int, default=1000, help="Number of rows to process in each chunk")
        parser.add_argument("--use_4bit", action="store_true", help="Use 4-bit quantization for memory efficiency")
    args = parser.parse_args()
    
    # Determine the number of available GPUs
    num_gpus = torch.cuda.device_count()
    print(f"Number of available GPUs: {num_gpus}")
    
    if num_gpus == 0:
        print("No GPUs available. Running on CPU.")
        process_on_gpu(0, args, 1)  # Single process mode
    else:
        # Create processes for each GPU using striding approach
        processes = []
        for gpu_id in range(num_gpus):
            # Each GPU processes every num_gpus-th entry starting from its gpu_id
            p = multiprocessing.Process(
                target=process_on_gpu,
                args=(gpu_id, args, num_gpus)
            )
            processes.append(p)
            p.start()
    
    # Wait for all processes to complete
    for p in processes:
        p.join()
    
    print("All GPUs have completed processing.")

if __name__ == '__main__':
    multiprocessing.set_start_method('spawn')
    main()
