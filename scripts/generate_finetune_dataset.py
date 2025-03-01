#!/usr/bin/env python3
import os
import argparse
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
        messages_output_path = f'{args.name}__messages_output{gpu_id}.json'
        time_output_path = f'{args.name}__time_output{gpu_id}.json'

        # Ensure the files exist by opening them in write mode initially
        open(messages_output_path, 'w').close()
        open(time_output_path, 'w').close()

        # Open the files in append mode
        messages_file = open(messages_output_path, 'a')
        time_file = open(time_output_path, 'a')

        # Load dataset
        df = pd.read_parquet(os.path.join(base_dir, "../", args.dataset_name))
        total_start_time = time.time()
        for index, row in df.iterrows():
            question = row['question']
            question = f"""<|begin_of_text|>
<|system|>
You are a helpful assistant.
<|end_of_system|>

{question}

<|end_of_text|>"""
            # Tokenize the input question
            inputs = tokenizer(question, return_tensors='pt').to(model.device)
            
            # Generate the solution
            start_time = time.time()
            with torch.no_grad():
                outputs = model.generate(inputs.input_ids, max_length=args.max_length, num_beams=num_beams, no_repeat_ngram_size=no_repeat_ngram_size, early_stopping=early_stopping)
            generation_time = time.time() - start_time
            
            # Decode the generated solution
            solution = tokenizer.batch_decode(outputs, skip_special_tokens=True, clean_up_tokenization_spaces=False)

            # Append to messages format and save
            message_entry = [
                {"role": "user", "content": question},
                {"role": "assistant", "content": solution}
            ]
            print(message_entry)
            json.dump(message_entry, messages_file, indent=4)
            messages_file.write("\n")

            # Calculate tokens per second
            num_tokens = inputs.input_ids.numel()
            tokens_per_second = num_tokens / generation_time

            # Record and save time elapsed
            
            time_entry = {'question': question, 'generation_time': generation_time, 'tokens_per_second': tokens_per_second}
            json.dump(time_entry, time_file, indent=4)
            print(time_entry)
            time_file.write("\n")

        # Close the files
        messages_file.close()
        time_file.close()


        # Log the average generation time
        total_generation_time = time.time() - total_start_time
        avg_generation_time = total_generation_time / len(df.index)
        print(f"Average generation time per question: {avg_generation_time:.2f} seconds")



if __name__ == '__main__':
    main()
