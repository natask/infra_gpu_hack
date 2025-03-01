#!/usr/bin/env python3
import os
import argparse
import json
import torch
from custom_dataset import get_dataloader
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments
from torch.utils.data import Dataset


class FineTuneDataset(Dataset):
    def __init__(self, data, tokenizer, max_length=512):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.examples = []
        for item in data:
            # Create a single text combining input and output for supervised fine-tuning
            prompt = f"Input: {item['input']}\nOutput: {item['output']}"
            tokenized = tokenizer(prompt, truncation=True, max_length=self.max_length, padding='max_length')
            self.examples.append(tokenized)

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return {key: torch.tensor(val) for key, val in self.examples[idx].items()}


def main():
    parser = argparse.ArgumentParser(description="Fine-tune a Hugging Face model")
    parser.add_argument("--model_name", type=str, required=True, help="Base model name or path")
    parser.add_argument("--dataset_name", type=str, required=True, help="Dataset name used in finetune_datasets")
    parser.add_argument("--fine_tuned_model_name", type=str, required=True, help="Name for the fine-tuned model directory")
    parser.add_argument("--batch_size", type=int, default=4, help="Training batch size")
    parser.add_argument("--learning_rate", type=float, default=5e-5, help="Learning rate")
    parser.add_argument("--num_train_epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--max_length", type=int, default=512, help="Maximum sequence length")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to a checkpoint to resume training")
    args = parser.parse_args()

    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    dataset_path = os.path.join(base_dir, "finetune_datasets", args.model_name, args.dataset_name + ".jsonl")
    # Load dataset from jsonl file
    dataloader = get_dataloader(dataset_path, batch_size=args.batch_size)

    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForCausalLM.from_pretrained(args.model_name)
    
    # Prepare dataset for training
    train_dataset = FineTuneDataset(dataloader.dataset, tokenizer, max_length=args.max_length)

    # Define training arguments
    output_dir = os.path.join(base_dir, "models", args.fine_tuned_model_name)
    training_args = TrainingArguments(
        output_dir=output_dir,
        overwrite_output_dir=True,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        save_steps=500,
        save_total_limit=2,
        logging_steps=100,
        fp16=torch.cuda.is_available(),
        resume_from_checkpoint=args.checkpoint
    )

    # Initialize Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        tokenizer=tokenizer
    )

    # Start training
    trainer.train(resume_from_checkpoint=args.checkpoint)

    # Save final model
    trainer.save_model(output_dir)
    print(f"Fine-tuned model saved to {output_dir}")

if __name__ == '__main__':
    main()
