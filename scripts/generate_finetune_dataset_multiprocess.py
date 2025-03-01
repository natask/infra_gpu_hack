#!/usr/bin/env python3
import os
import argparse
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
    model_name = "meta-llama/Llama-2-70b-hf"
    model = LlamaForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, device_map=device)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Set generation parameters
    num_beams = 4
    no_repeat_ngram_size = 3
    early_stopping = True
    
    # Prepare output files
    messages_output_path = f'{args.name}__messages_output{gpu_id}.json'
    time_output_path = f'{args.name}__time_output{gpu_id}.json'
    
    # Ensure files exist
    open(messages_output_path, 'w').close()
    open(time_output_path, 'w').close()
    
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
        outputs = model.generate(inputs.input_ids, max_length=args.max_length, num_beams=num_beams, no_repeat_ngram_size=no_repeat_ngram_size, early_stopping=early_stopping)
        generation_time = time.time() - start_time
        
        # Decode the generated solution
        solution = tokenizer.batch_decode(outputs, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        
        # Write results to files
        with open(messages_output_path, 'a') as messages_file, open(time_output_path, 'a') as time_file:
            message_entry = [
                {"role": "user", "content": question},
                {"role": "assistant", "content": solution[0]}
            ]
            print(f"GPU {gpu_id} - Index {index}: Generated response")
            json.dump(message_entry, messages_file, indent=4)
            messages_file.write("\n")
            
            # Calculate tokens per second
            num_tokens = inputs.input_ids.numel()
            tokens_per_second = num_tokens / generation_time
            
            # Record and save time elapsed
            time_entry = {'question': question, 'generation_time': generation_time, 'tokens_per_second': tokens_per_second}
            json.dump(time_entry, time_file, indent=4)
            time_file.write("\n")
    
    total_generation_time = time.time() - total_start_time
    print(f"GPU {gpu_id} - Total generation time: {total_generation_time}")

def main():
    print(torch.backends.cuda.flash_sdp_enabled())
    # True
    print(torch.backends.cuda.mem_efficient_sdp_enabled())
    # True
    
    base_dir = os.path.dirname(os.path.abspath(__file__))
    
    parser = argparse.ArgumentParser(description='Generate fine-tune dataset')
    parser.add_argument("--dataset_name", type=str, default="datasets/GAIR/LIMO/limo_train.parquet", help="Path to the dataset file")
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
