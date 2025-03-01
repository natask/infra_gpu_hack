#!/usr/bin/env python3
import os
import argparse
import time
import json
from custom_dataset import get_dataloader
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import tqdm


def main():
    parser = argparse.ArgumentParser(description="Evaluate model performance using direct decoding")
    parser.add_argument("--model_name", type=str, required=True, help="Hugging Face model name or path")
    parser.add_argument("--evaluation_dataset", type=str, required=True, help="Evaluation dataset file name without extension")
    parser.add_argument("--max_length", type=int, default=50, help="Maximum generation length")
    args = parser.parse_args()

    # Load model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForCausalLM.from_pretrained(args.model_name).to('cuda' if torch.cuda.is_available() else 'cpu')
    model.eval()

    # Load evaluation dataset
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    eval_path = os.path.join(base_dir, "datasets", args.evaluation_dataset)
    dataloader = get_dataloader(eval_path, batch_size=1)

    results = []
    for batch in tqdm.tqdm(dataloader):
        print(batch)
        input_text = batch["input"][0]
        sample_id = batch.get("id", ["sample_unknown"])[0]
        inputs = tokenizer(input_text, return_tensors="pt")
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        start_time = time.time()
        with torch.no_grad():
            gen_ids = model.generate(**inputs, max_length=args.max_length)
        latency = time.time() - start_time
        output_text = tokenizer.decode(gen_ids[0], skip_special_tokens=True)
        results.append({
            "id": sample_id,
            "input": input_text,
            "output": output_text,
            "latency": round(latency, 4)
        })

    # Save evaluation results
    timestamp = time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())
    output_filename = os.path.join(base_dir, "evaluations", f"{timestamp}_{args.model_name}.jsonl")
    os.makedirs(os.path.dirname(output_filename), exist_ok=True)
    with open(output_filename, "w") as fout:
        for res in results:
            fout.write(json.dumps(res) + "\n")
    print(f"Evaluation results saved to {output_filename}")


if __name__ == '__main__':
    main()
