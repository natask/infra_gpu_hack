#!/usr/bin/env python3
import os
import argparse
import time
import json
import torch
import tqdm
from custom_dataset import get_dataloader
from transformers import AutoModelForCausalLM, AutoTokenizer
from speculative_decoding import speculative_generate


def main():
    parser = argparse.ArgumentParser(description="Evaluate model performance using speculative decoding")
    parser.add_argument("--teacher_model", type=str, required=True, help="Teacher model name or path (full-size)")
    parser.add_argument("--student_model", type=str, required=True, help="Student model name or path (smaller model for fast inference)")
    parser.add_argument("--evaluation_dataset", type=str, required=True, help="Evaluation dataset file name without extension")
    parser.add_argument("--max_length", type=int, default=50, help="Maximum generation length")
    parser.add_argument("--speculative_steps", type=int, default=3, help="Number of speculative steps")
    args = parser.parse_args()

    # Load teacher model and tokenizer
    teacher_tokenizer = AutoTokenizer.from_pretrained(args.teacher_model)
    teacher_model = AutoModelForCausalLM.from_pretrained(args.teacher_model).to('cuda' if torch.cuda.is_available() else 'cpu')
    teacher_model.eval()

    # Load student model and tokenizer
    student_tokenizer = AutoTokenizer.from_pretrained(args.student_model)
    student_model = AutoModelForCausalLM.from_pretrained(args.student_model).to('cuda' if torch.cuda.is_available() else 'cpu')
    student_model.eval()

    # Load evaluation dataset
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    eval_path = os.path.join(base_dir, "datasets", "evaluation", args.evaluation_dataset + ".jsonl")
    dataloader = get_dataloader(eval_path, batch_size=1)

    results = []
    for batch in tqdm.tqdm(dataloader):
        input_text = batch["input"][0]
        sample_id = batch.get("id", ["sample_unknown"])[0]
        start_time = time.time()
        output_text = speculative_generate(
            teacher_model, student_model, teacher_tokenizer, student_tokenizer,
            input_text, max_length=args.max_length, speculative_steps=args.speculative_steps
        )
        latency = time.time() - start_time
        results.append({
            "id": sample_id,
            "input": input_text,
            "output": output_text,
            "latency": round(latency, 4)
        })

    # Save evaluation results
    timestamp = time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())
    output_filename = os.path.join(base_dir, "evaluations", f"{timestamp}_{args.teacher_model}.jsonl")
    os.makedirs(os.path.dirname(output_filename), exist_ok=True)
    with open(output_filename, "w") as fout:
        for res in results:
            fout.write(json.dumps(res) + "\n")
    print(f"Speculative evaluation results saved to {output_filename}")


if __name__ == '__main__':
    main()
