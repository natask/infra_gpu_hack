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
import torch
import time
from torch.nn.attention import SDPBackend, sdpa_kernel
# Only enable flash attention backend
    
os.environ['HF_HOME'] = '/mount/model-cache'

def main():
    print(torch.backends.cuda.flash_sdp_enabled())
    # True
    print(torch.backends.cuda.mem_efficient_sdp_enabled())
    # True
    print(torch.backends.cuda.math_sdp_enabled())
    # True
    with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
        parser = argparse.ArgumentParser(description="Generate model outputs")
        parser.add_argument("--model_name", type=str, default="casperhansen/Llama-3.3-70B-instruct-awq", help="Hugging Face model name or path")
        parser.add_argument("--dataset_name", type=str, default='final_dataset.parquet', help="Name of the dataset file (without path) in parquet")
        parser.add_argument("--batch_size", type=int, default=8, help="Batch size for inference")
        parser.add_argument("--name", type=str, default="llama70b", help="Model string name")
        parser.add_argument("--max_length", type=int, default=2080, help="Max generation length")
        parser.add_argument("--start_index", type=int, default=0, help="Starting index for processing prompts")
        args = parser.parse_args()

        # Prepare the results DataFrame
        results = []
        # Load model and tokenizer
        
        base_dir = os.path.dirname(os.path.abspath(__file__))
        model = LlamaForCausalLM.from_pretrained(args.model_name).to('cuda' if torch.cuda.is_available() else 'cpu')
        tokenizer = AutoTokenizer.from_pretrained(args.model_name)
        model.eval()
        print(f"cuda is avaliable: {torch.cuda.is_available()}")

        # params for generate
        num_beams = 4
        no_repeat_ngram_size = 3
        early_stopping = True

        # Prepare the output files
        gpu_id = 0
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
        df = pd.read_parquet(os.path.join(base_dir, "../", args.dataset_name))
        total_start_time = time.time()

            # Process data and generate output for each GPU
        for index, row in df.iterrows():
            if index < args.start_index:
                continue
            question = row['question']
            question = f"""<|begin_of_text|>
<|system|>
You are a helpful assistant.
<|end_of_system|>

{question}

<|end_of_text|>"""
            # Tokenize the input question
            inputs = tokenizer(question, return_tensors='pt').to(model.device)
            
            # Generate the solution with logits
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

            # Open the files in append mode using 'with' statement
            with open(messages_output_path, 'a', newline='') as messages_file, open(time_output_path, 'a', newline='') as time_file:
                # Create CSV writers
                messages_writer = csv.writer(messages_file)
                time_writer = csv.writer(time_file)
                
                # Write user message (with empty info field)
                messages_writer.writerow(['user', question, ''])
                
                # Write assistant response with token info
                token_info_json = json.dumps(token_info)
                messages_writer.writerow(['assistant', solution, token_info_json])
                
                print(f"Wrote user and assistant messages with token info to CSV")

                # Calculate tokens per second
                num_tokens = inputs.input_ids.numel()
                tokens_per_second = num_tokens / generation_time

                # Record and save time elapsed
                time_writer.writerow([question, generation_time, tokens_per_second])
                
                # Print the current index for easy resuming
                print(f"Processed index {index}. To resume from next entry, use --start_index {index+1}")

        # Log the average generation time
        total_generation_time = time.time() - total_start_time
        avg_generation_time = total_generation_time / len(df.index)
        print(f"Average generation time per question: {avg_generation_time:.2f} seconds")



if __name__ == '__main__':
    main()
