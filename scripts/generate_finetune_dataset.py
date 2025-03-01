#!/usr/bin/env python3
import os
import argparse
import json
from custom_dataset import get_dataloader
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import tqdm
import pandas as pd
import torch
os.environ['HF_HOME'] = '/mount/model-cache'
os.environ['HF_HUB_CACHE'] = '/mount/model-cache'

def main():
    print(torch.backends.cuda.flash_sdp_enabled())
    # True
    print(torch.backends.cuda.mem_efficient_sdp_enabled())
    # True
    print(torch.backends.cuda.math_sdp_enabled())
    # True
    with torch.backends.cuda.sdp_kernel(
        enable_flash=True, enable_math=False, enable_mem_efficient=False
    ):
        parser = argparse.ArgumentParser(description="Generate model outputs")
        parser.add_argument("--model_name", type=str, default="casperhansen/llama-3.3-70b-instruct-awq", help="Hugging Face model name or path")
        parser.add_argument("--dataset_name", type=str, default='final_dataset.parquet', help="Name of the dataset file (without path) in parquet")
        parser.add_argument("--batch_size", type=int, default=8, help="Batch size for inference")
        parser.add_argument("--max_length", type=int, default=2080, help="Max generation length")
        args = parser.parse_args()

        # Prepare the results DataFrame
        results = []
        # Load model and tokenizer
        
        tokenizer = AutoTokenizer.from_pretrained(args.model_name)
        model = AutoModelForCausalLM.from_pretrained(args.model_name).to('cuda' if torch.cuda.is_available() else 'cpu')
        model.eval()
        print(f"cuda is avaliable: {torch.cuda.is_available()}")

        # Prepare the results DataFrame
        results = []

        # params for generate
        num_beams = 4
        no_repeat_ngram_size = 3
        early_stopping = True

        # Load dataset
        df = pd.read_parquet(os.path.join(base_dir, "datasets", args.dataset_name))
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
            with torch.no_grad():
                outputs = model.generate(**inputs, max_length=max_length, num_beams=num_beams, no_repeat_ngram_size=no_repeat_ngram_size, early_stopping=early_stopping)  # Adjust max_length as needed , max_length=200
            
            # Decode the generated solution
            solution = tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Append to results
            results.append({'user': question, 'asistant': solution})

        # Create a DataFrame for results
        results_df = pd.DataFrame(results)


        # Write outputs to file
        results_df.to_json('finetune_dataset.json', orient='records', lines=True)

        print("Results saved to finetune_dataset.json")


if __name__ == '__main__':
    main()
