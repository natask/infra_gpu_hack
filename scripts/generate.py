#!/usr/bin/env python3
import os
import argparse
import json
from custom_dataset import get_dataloader
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import tqdm


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
        parser.add_argument("--model_name", type=str, required=True, help="Hugging Face model name or path")
        parser.add_argument("--dataset_name", type=str, required=True, help="Name of the dataset file (without path)")
        parser.add_argument("--batch_size", type=int, default=8, help="Batch size for inference")
        parser.add_argument("--config", type=str, default="direct", help="Configuration used (direct/speculative)")
        parser.add_argument("--max_length", type=int, default=50, help="Max generation length")
        args = parser.parse_args()

        # Load model and tokenizer
        tokenizer = AutoTokenizer.from_pretrained(args.model_name)
        model = AutoModelForCausalLM.from_pretrained(args.model_name).to('cuda' if torch.cuda.is_available() else 'cpu')
        model.eval()
        model.to('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"cuda is avaliable: {torch.cuda.is_available()}")

        # Load dataset
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        dataset_path = os.path.join(base_dir, "datasets", args.dataset_name)
        dataloader = get_dataloader(dataset_path, batch_size=args.batch_size)

        # Prepare output directory
        output_dir = os.path.join(base_dir, "finetune_datasets", args.model_name)
        os.makedirs(output_dir, exist_ok=True)
        output_file = os.path.join(output_dir, args.dataset_name + ".jsonl")

        outputs = []
        for batch in dataloader:
            inputs = [item["input"] for item in batch]
            inputs_tokenized = tokenizer(inputs, return_tensors="pt", padding=True, truncation=True)
            inputs_tokenized = {k: v.to(model.device) for k, v in inputs_tokenized.items()}
            with torch.no_grad():
                generated_ids = model.generate(**inputs_tokenized, max_length=args.max_length)
            generated_outputs = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
            for j, item in enumerate(batch):
                outputs.append({
                    "id": item.get("id", f"sample_{j:06d}"),
                    "input": item["input"],
                    "output": generated_outputs[j],
                    "config": args.config
                })

        # Write outputs to file
        with open(output_file, "w") as fout:
            for item in outputs:
                fout.write(json.dumps(item) + "\n")
        print(f"Generated outputs saved to {output_file}")


if __name__ == '__main__':
    main()
