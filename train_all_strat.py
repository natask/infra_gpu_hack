import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
import torch.distributed as dist
import argparse

class PromptDataset(Dataset):
    def __init__(self, data_path, tokenizer_teacher, tokenizer_student, max_length=512):
        # Load data more efficiently by specifying columns and using chunking
        self.df = pd.read_parquet(data_path, columns=['question']).iloc
        self.tokenizer_teacher = tokenizer_teacher
        self.tokenizer_student = tokenizer_student
        self.max_length = max_length
        
        # Pre-tokenize data to speed up training
        print("Pre-tokenizing data to speed up training...")
        self.tokenized_data = []
        for idx in tqdm(range(len(self.df))):
            prompt = self.df.iloc[idx]['question']
            
            # Tokenize for teacher (LLaMA)
            teacher_inputs = self.tokenizer_teacher(
                prompt,
                return_tensors="pt",
                padding="max_length",
                max_length=self.max_length,
                truncation=True
            )
            
            # Tokenize for student (LLaDA)
            m = [{"role": "user", "content": prompt}]
            student_prompt = self.tokenizer_student.apply_chat_template(
                m, 
                add_generation_prompt=True, 
                tokenize=False
            )
            student_inputs = self.tokenizer_student(
                student_prompt,
                return_tensors="pt",
                padding="max_length",
                max_length=self.max_length,
                truncation=True
            )
            
            self.tokenized_data.append({
                'prompt': prompt,
                'teacher_input_ids': teacher_inputs['input_ids'].squeeze(0),
                'teacher_attention_mask': teacher_inputs['attention_mask'].squeeze(0),
                'student_input_ids': student_inputs['input_ids'].squeeze(0),
                'student_attention_mask': student_inputs['attention_mask'].squeeze(0),
            })
        
        # Free memory
        del self.df
        import gc
        gc.collect()
        
    def __len__(self):
        return len(self.tokenized_data)
    
    def __getitem__(self, idx):
        return self.tokenized_data[idx]

class LLaDADistiller:
    def __init__(
        self,
        teacher_model_name="casperhansen/llama-3.3-70b-instruct-awq",
        student_model_name="GSAI-ML/LLaDA-8B-Instruct",
        teacher_device="cuda:0",
        student_device="cuda:1",
        temperature=2.0,
        mask_id=126336,  # LLaDA mask token ID
    ):
        self.teacher_device = teacher_device
        self.student_device = student_device
        self.temperature = temperature
        self.mask_id = mask_id
        
        # Load teacher model (LLaMA)
        print(f"Loading teacher model {teacher_model_name} on {teacher_device}...")
        self.teacher_tokenizer = AutoTokenizer.from_pretrained(teacher_model_name)
        self.teacher_model = AutoModelForCausalLM.from_pretrained(
            teacher_model_name,
            device_map=teacher_device,
            torch_dtype=torch.float16
        )
        self.teacher_model.eval()  # Teacher is always in eval mode
        
        # Load student model (LLaDA)
        print(f"Loading student model {student_model_name} on {student_device}...")
        self.student_tokenizer = AutoTokenizer.from_pretrained(
            student_model_name, 
            trust_remote_code=True
        )
        self.student_model = AutoModel.from_pretrained(
            student_model_name,
            device_map=student_device,
            trust_remote_code=True, 
            torch_dtype=torch.bfloat16
        )
        
        # Enable gradient checkpointing after model is loaded
        # This trades compute for memory by not storing all activations
        if hasattr(self.student_model, 'gradient_checkpointing_enable'):
            self.student_model.gradient_checkpointing_enable()
        elif hasattr(self.student_model, 'enable_gradient_checkpointing'):
            self.student_model.enable_gradient_checkpointing()
        
        # Freeze teacher model parameters
        for param in self.teacher_model.parameters():
            param.requires_grad = False

    def forward_process(self, batch, prompt_indices):
        """
        Apply forward diffusion process to prepare tokens for LLaDA learning
        Similar to the get_log_likelihood.py implementation in LLaDA repo
        """
        b, l = batch.shape
        
        # Find which tokens are not prompt (these are the ones we'll mask)
        non_prompt_mask = ~prompt_indices
        target_len = non_prompt_mask.sum(dim=1)
        
        # Initialize mask tensor
        is_mask = torch.zeros_like(batch, dtype=torch.bool, device=batch.device)
        
        # For each sequence in the batch
        for i in range(b):
            # Determine how many tokens to mask
            k = torch.randint(1, target_len[i] + 1, (1,), device=batch.device)[0]
            
            # Get indices of non-prompt tokens
            non_prompt_indices = torch.where(non_prompt_mask[i])[0]
            
            # Randomly select k indices to mask
            mask_indices = non_prompt_indices[torch.randperm(len(non_prompt_indices))[:k]]
            is_mask[i, mask_indices] = True
        
        # Apply mask
        noisy_batch = torch.where(is_mask, self.mask_id, batch)
        
        # Calculate mask ratio for each token
        mask_ratio = is_mask.float()
        
        return noisy_batch, mask_ratio, is_mask
    
    def get_teacher_logits(self, input_ids, attention_mask):
        """Get teacher (LLaMA) logits"""
        with torch.no_grad():
            outputs = self.teacher_model(
                input_ids=input_ids.to(self.teacher_device),
                attention_mask=attention_mask.to(self.teacher_device)
            )
            logits = outputs.logits
        return logits
    
    def get_student_logits(self, input_ids, mask_indices):
        """Get student (LLaDA) logits for the tokens at masked positions"""
        logits = self.student_model(input_ids.to(self.student_device)).logits
        return logits
    
    def distill_batch(self, batch, optimizer, do_backward=False):
        """Perform distillation on a single batch"""
        teacher_input_ids = batch['teacher_input_ids'].to(self.teacher_device)
        teacher_attention_mask = batch['teacher_attention_mask'].to(self.teacher_device)
        student_input_ids = batch['student_input_ids'].to(self.student_device)
        
        # Create prompt indices (tokens that are part of the prompt, not to be masked)
        prompt_indices = torch.zeros_like(student_input_ids, dtype=torch.bool, device=self.student_device)
        for i in range(len(prompt_indices)):
            # Find all non-padding tokens in the input
            non_padding = (student_input_ids[i] != self.student_tokenizer.pad_token_id)
            # Mark the first 75% of non-padding tokens as prompt (adjust as needed)
            non_padding_indices = torch.where(non_padding)[0]
            if len(non_padding_indices) > 0:
                prompt_len = int(len(non_padding_indices) * 0.75)
                prompt_indices[i, :prompt_len] = True
        
        # Apply forward diffusion process to get masked student input
        noisy_student_input_ids, mask_ratio, is_mask = self.forward_process(
            student_input_ids, 
            prompt_indices
        )
        
        # Get teacher logits for the full sequence
        teacher_logits = self.get_teacher_logits(teacher_input_ids, teacher_attention_mask)
        
        # Get student logits
        student_logits = self.get_student_logits(noisy_student_input_ids, is_mask)
        
        # Only compute loss on masked positions
        loss = 0
        batch_size = student_input_ids.size(0)
        
        # Get vocabulary sizes for both models
        teacher_vocab_size = teacher_logits.size(-1)
        student_vocab_size = student_logits.size(-1)
        
        for i in range(batch_size):
            # Get masked positions for this sample
            mask_pos = is_mask[i]
            if not mask_pos.any():
                continue
                
            # Get teacher and student logits at masked positions
            t_logits = teacher_logits[i].to(self.student_device)
            s_logits = student_logits[i]
            
            # Extract logits at masked positions
            t_logits_masked = t_logits[mask_pos]
            s_logits_masked = s_logits[mask_pos]
            
            # Handle vocabulary size mismatch
            min_vocab_size = min(teacher_vocab_size, student_vocab_size)
            
            # Truncate logits to the smaller vocabulary size
            t_logits_masked = t_logits_masked[:, :min_vocab_size]
            s_logits_masked = s_logits_masked[:, :min_vocab_size]
            
            # Apply temperature scaling for KL divergence
            t_logits_masked = t_logits_masked / self.temperature
            s_logits_masked = s_logits_masked / self.temperature
            
            # Compute KL divergence loss
            kl_loss = F.kl_div(
                F.log_softmax(s_logits_masked, dim=-1),
                F.softmax(t_logits_masked, dim=-1),
                reduction='batchmean'
            )
            
            loss += kl_loss
        
        loss = loss / batch_size
        
        # Backpropagation (only if requested)
        if do_backward:
            loss.backward()
            return loss
        else:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            return loss.item()
    
    def train(
        self,
        data_path,
        output_dir,
        num_epochs=3,
        batch_size=4,
        learning_rate=5e-5,
        max_length=512,
        save_steps=100,
        eval_steps=50,
        gradient_accumulation_steps=4,  # Accumulate gradients to save memory
        warmup_steps=100,  # Warm up learning rate
        weight_decay=0.01  # Weight decay for regularization
    ):
        os.makedirs(output_dir, exist_ok=True)
        
        # Create dataset and dataloader
        dataset = PromptDataset(
            data_path, 
            self.teacher_tokenizer, 
            self.student_tokenizer,
            max_length=max_length
        )
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        # Setup optimizer with memory-efficient settings
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in self.student_model.named_parameters() 
                          if not any(nd in n for nd in no_decay)],
                "weight_decay": weight_decay,
            },
            {
                "params": [p for n, p in self.student_model.named_parameters() 
                          if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        optimizer = torch.optim.AdamW(
            optimizer_grouped_parameters,
            lr=learning_rate,
            betas=(0.9, 0.999),
            eps=1e-8
        )
        
        # Calculate total training steps for learning rate scheduler
        dataset_size = len(dataset)
        total_steps = (dataset_size // (batch_size * gradient_accumulation_steps)) * num_epochs
        
        # Create learning rate scheduler
        from transformers import get_linear_schedule_with_warmup
        scheduler = get_linear_schedule_with_warmup(
            optimizer, 
            num_warmup_steps=warmup_steps, 
            num_training_steps=total_steps
        )
        
        # Training loop with gradient accumulation
        global_step = 0
        total_loss = 0
        actual_batch_size = batch_size * gradient_accumulation_steps
        
        for epoch in range(num_epochs):
            print(f"Epoch {epoch+1}/{num_epochs}")
            self.student_model.train()
            
            progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}")
            optimizer.zero_grad()  # Zero gradients at the beginning of each epoch
            
            for step, batch in enumerate(progress_bar):
                # Forward and backward pass
                loss = self.distill_batch(batch, optimizer, do_backward=True)
                loss = loss / gradient_accumulation_steps  # Normalize loss for gradient accumulation
                total_loss += loss.item() * gradient_accumulation_steps  # Track unnormalized loss for reporting
                
                # Update weights only after accumulating gradients
                if (step + 1) % gradient_accumulation_steps == 0 or step == len(dataloader) - 1:
                    # Clip gradients to prevent exploding gradients
                    torch.nn.utils.clip_grad_norm_(self.student_model.parameters(), max_norm=1.0)
                    
                    # Update parameters
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()
                    global_step += 1
                    
                    # Update progress bar
                    progress_bar.set_postfix({
                        "loss": loss.item() * gradient_accumulation_steps,
                        "lr": scheduler.get_last_lr()[0]
                    })
                    
                    # Save checkpoint
                    if global_step % save_steps == 0:
                        checkpoint_dir = os.path.join(output_dir, f"checkpoint-{global_step}")
                        os.makedirs(checkpoint_dir, exist_ok=True)
                        self.student_model.save_pretrained(checkpoint_dir)
                        self.student_tokenizer.save_pretrained(checkpoint_dir)
                        print(f"Saved model checkpoint to {checkpoint_dir}")
                    
                    # Evaluate
                    if global_step % eval_steps == 0:
                        avg_loss = total_loss / (eval_steps * gradient_accumulation_steps)
                        print(f"Step {global_step}: Average Loss = {avg_loss:.4f}")
                        total_loss = 0
                        
                # Explicitly free memory
                if hasattr(torch.cuda, 'empty_cache'):
                    torch.cuda.empty_cache()
        
        # Save final model
        final_output_dir = os.path.join(output_dir, "final_model")
        os.makedirs(final_output_dir, exist_ok=True)
        self.student_model.save_pretrained(final_output_dir)
        self.student_tokenizer.save_pretrained(final_output_dir)
        print(f"Saved final model to {final_output_dir}")
    
    def evaluate(self, data_path, batch_size=4, max_length=512, num_eval_samples=100):
        """Evaluate the distilled model"""
        dataset = PromptDataset(
            data_path, 
            self.teacher_tokenizer, 
            self.student_tokenizer,
            max_length=max_length
        )
        
        # Use only a subset for evaluation if specified
        if num_eval_samples and num_eval_samples < len(dataset):
            indices = torch.randperm(len(dataset))[:num_eval_samples]
            subset = torch.utils.data.Subset(dataset, indices)
            dataloader = DataLoader(subset, batch_size=batch_size, shuffle=False)
        else:
            dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        
        total_loss = 0
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Evaluating"):
                teacher_input_ids = batch['teacher_input_ids'].to(self.teacher_device)
                teacher_attention_mask = batch['teacher_attention_mask'].to(self.teacher_device)
                student_input_ids = batch['student_input_ids'].to(self.student_device)
                
                # Create prompt indices
                prompt_indices = torch.zeros_like(student_input_ids, dtype=torch.bool, device=self.student_device)
                for i in range(len(prompt_indices)):
                    non_padding = (student_input_ids[i] != self.student_tokenizer.pad_token_id)
                    non_padding_indices = torch.where(non_padding)[0]
                    if len(non_padding_indices) > 0:
                        prompt_len = int(len(non_padding_indices) * 0.75)
                        prompt_indices[i, :prompt_len] = True
                
                # Apply forward diffusion
                noisy_student_input_ids, mask_ratio, is_mask = self.forward_process(
                    student_input_ids, 
                    prompt_indices
                )
                
                # Get logits
                teacher_logits = self.get_teacher_logits(teacher_input_ids, teacher_attention_mask)
                student_logits = self.get_student_logits(noisy_student_input_ids, is_mask)
                
                # Get vocabulary sizes for both models
                teacher_vocab_size = teacher_logits.size(-1)
                student_vocab_size = student_logits.size(-1)
                
                # Compute loss
                batch_loss = 0
                batch_size = student_input_ids.size(0)
                
                for i in range(batch_size):
                    mask_pos = is_mask[i]
                    if not mask_pos.any():
                        continue
                        
                    t_logits = teacher_logits[i].to(self.student_device)
                    s_logits = student_logits[i]
                    
                    # Extract logits at masked positions
                    t_logits_masked = t_logits[mask_pos]
                    s_logits_masked = s_logits[mask_pos]
                    
                    # Handle vocabulary size mismatch
                    min_vocab_size = min(teacher_vocab_size, student_vocab_size)
                    
                    # Truncate logits to the smaller vocabulary size
                    t_logits_masked = t_logits_masked[:, :min_vocab_size]
                    s_logits_masked = s_logits_masked[:, :min_vocab_size]
                    
                    # Apply temperature scaling
                    t_logits_masked = t_logits_masked / self.temperature
                    s_logits_masked = s_logits_masked / self.temperature
                    
                    kl_loss = F.kl_div(
                        F.log_softmax(s_logits_masked, dim=-1),
                        F.softmax(t_logits_masked, dim=-1),
                        reduction='batchmean'
                    )
                    
                    batch_loss += kl_loss
                
                total_loss += (batch_loss / batch_size).item()
        
        avg_loss = total_loss / len(dataloader)
        print(f"Evaluation Loss: {avg_loss:.4f}")
        return avg_loss

def main():
    parser = argparse.ArgumentParser(description='Knowledge Distillation from LLaMA to LLaDA')
    parser.add_argument('--teacher_model', type=str, default="casperhansen/llama-3.3-70b-instruct-awq",
                        help='Teacher model name or path')
    parser.add_argument('--student_model', type=str, default="GSAI-ML/LLaDA-8B-Instruct",
                        help='Student model name or path')
    parser.add_argument('--teacher_device', type=str, default="cuda:0",
                        help='Device for teacher model')
    parser.add_argument('--student_device', type=str, default="cuda:1",
                        help='Device for student model')
    parser.add_argument('--data_path', type=str, default="./final_dataset.parquet",
                        help='Path to the dataset file')
    parser.add_argument('--output_dir', type=str, default="./llada_distilled",
                        help='Output directory for saved models')
    parser.add_argument('--batch_size', type=int, default=4,
                        help='Batch size for training')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=4,
                        help='Number of steps to accumulate gradients')
    parser.add_argument('--num_epochs', type=int, default=3,
                        help='Number of training epochs')
    parser.add_argument('--learning_rate', type=float, default=5e-5,
                        help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.01,
                        help='Weight decay for AdamW optimizer')
    parser.add_argument('--warmup_steps', type=int, default=100,
                        help='Number of warmup steps for learning rate scheduler')
    parser.add_argument('--temperature', type=float, default=2.0,
                        help='Temperature for distillation')
    parser.add_argument('--max_length', type=int, default=512,
                        help='Maximum sequence length')
    parser.add_argument('--save_steps', type=int, default=100,
                        help='Save checkpoint every X steps')
    parser.add_argument('--eval_steps', type=int, default=50,
                        help='Evaluate every X steps')
    
    args = parser.parse_args()
    
    # Set up torch for memory efficiency
    torch.backends.cuda.matmul.allow_tf32 = True  # Allow TF32 for faster computation
    torch.backends.cudnn.benchmark = True  # Optimize CUDNN for faster training
    
    # Create distiller
    distiller = LLaDADistiller(
        teacher_model_name=args.teacher_model,
        student_model_name=args.student_model,
        teacher_device=args.teacher_device,
        student_device=args.student_device,
        temperature=args.temperature
    )
    
    # Train
    distiller.train(
        data_path=args.data_path,
        output_dir=args.output_dir,
        num_epochs=args.num_epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        max_length=args.max_length,
        save_steps=args.save_steps,
        eval_steps=args.eval_steps,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        warmup_steps=args.warmup_steps,
        weight_decay=args.weight_decay
    )
    
    # Evaluate
    distiller.evaluate(args.data_path, batch_size=args.batch_size)

if __name__ == "__main__":
    main()
