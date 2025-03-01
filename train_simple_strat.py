import os
import torch
import torch.nn.functional as F
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm
import argparse

class PromptDataset(Dataset):
    def __init__(self, data_path, tokenizer_teacher, tokenizer_student, max_length=512):
        self.df = pd.read_parquet(data_path)
        self.tokenizer_teacher = tokenizer_teacher
        self.tokenizer_student = tokenizer_student
        self.max_length = max_length
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
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
        
        return {
            'prompt': prompt,
            'teacher_input_ids': teacher_inputs['input_ids'].squeeze(0),
            'teacher_attention_mask': teacher_inputs['attention_mask'].squeeze(0),
            'student_input_ids': student_inputs['input_ids'].squeeze(0),
            'student_attention_mask': student_inputs['attention_mask'].squeeze(0),
        }

class LLaDADistiller:
    def __init__(
        self,
        teacher_model_name,
        student_model_name,
        teacher_device="cuda:0",
        student_device="cuda:1",
        temperature=2.0
    ):
        # Devices
        self.teacher_device = teacher_device
        self.student_device = student_device
        self.temperature = temperature
        
        # Load teacher model (LLaMA)
        print(f"Loading teacher model {teacher_model_name} on {teacher_device}...")
        self.teacher_tokenizer = AutoTokenizer.from_pretrained(
            teacher_model_name, 
            trust_remote_code=True
        )
        self.teacher_model = AutoModel.from_pretrained(
            teacher_model_name,
            device_map=teacher_device,
            trust_remote_code=True
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
        print("Enabling gradient checkpointing for memory efficiency...")
        if hasattr(self.student_model, 'gradient_checkpointing_enable'):
            self.student_model.gradient_checkpointing_enable()
        elif hasattr(self.student_model, 'enable_gradient_checkpointing'):
            self.student_model.enable_gradient_checkpointing()
        
        # Freeze teacher model parameters
        for param in self.teacher_model.parameters():
            param.requires_grad = False
    
    def forward_process(self, input_ids, prompt_indices, mask_prob=0.5):
        """Apply forward diffusion to mask tokens"""
        batch_size, seq_len = input_ids.shape
        
        # Initialize tensors
        noisy_input_ids = input_ids.clone()
        is_mask = torch.zeros_like(input_ids, dtype=torch.bool, device=self.student_device)
        
        # For each sequence in the batch
        for i in range(batch_size):
            # Find tokens that are not part of the prompt
            non_prompt = ~prompt_indices[i]
            # Get indices of non-prompt tokens
            non_prompt_indices = torch.where(non_prompt)[0]
            
            if len(non_prompt_indices) > 0:
                # Randomly select tokens to mask based on mask_prob
                num_to_mask = max(1, int(len(non_prompt_indices) * mask_prob))
                indices_to_mask = non_prompt_indices[torch.randperm(len(non_prompt_indices))[:num_to_mask]]
                
                # Apply masking
                noisy_input_ids[i, indices_to_mask] = self.student_tokenizer.mask_token_id
                is_mask[i, indices_to_mask] = True
        
        return noisy_input_ids, mask_prob, is_mask
    
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
    
    def distill_batch(self, batch, optimizer):
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
        
        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.student_model.parameters(), max_norm=1.0)
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
        eval_steps=50
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
        
        # Setup optimizer - simple AdamW with weight decay
        optimizer = torch.optim.AdamW(
            self.student_model.parameters(),
            lr=learning_rate,
            weight_decay=0.01
        )
        
        # Training loop
        global_step = 0
        total_loss = 0
        
        for epoch in range(num_epochs):
            print(f"Epoch {epoch+1}/{num_epochs}")
            self.student_model.train()
            
            progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}")
            
            for batch in progress_bar:
                # Train on batch
                loss = self.distill_batch(batch, optimizer)
                total_loss += loss
                global_step += 1
                
                # Update progress bar
                progress_bar.set_postfix({"loss": loss})
                
                # Save checkpoint
                if global_step % save_steps == 0:
                    checkpoint_dir = os.path.join(output_dir, f"checkpoint-{global_step}")
                    os.makedirs(checkpoint_dir, exist_ok=True)
                    self.student_model.save_pretrained(checkpoint_dir)
                    self.student_tokenizer.save_pretrained(checkpoint_dir)
                    print(f"Saved model checkpoint to {checkpoint_dir}")
                
                # Evaluate
                if global_step % eval_steps == 0:
                    avg_loss = total_loss / eval_steps
                    print(f"Step {global_step}: Average Loss = {avg_loss:.4f}")
                    total_loss = 0
                
                # Explicitly clear CUDA cache every few steps
                if global_step % 10 == 0 and hasattr(torch.cuda, 'empty_cache'):
                    torch.cuda.empty_cache()
        
        # Save final model
        final_output_dir = os.path.join(output_dir, "final_model")
        os.makedirs(final_output_dir, exist_ok=True)
        self.student_model.save_pretrained(final_output_dir)
        self.student_tokenizer.save_pretrained(final_output_dir)
        print(f"Training complete! Final model saved to {final_output_dir}")
    
    def evaluate(self, data_path, batch_size=4):
        # Create dataset and dataloader
        dataset = PromptDataset(
            data_path, 
            self.teacher_tokenizer, 
            self.student_tokenizer
        )
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        
        # Evaluation loop
        self.student_model.eval()
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
                    
                    # Compute KL divergence
                    kl_loss = F.kl_div(
                        F.log_softmax(s_logits_masked, dim=-1),
                        F.softmax(t_logits_masked, dim=-1),
                        reduction='batchmean'
                    )
                    
                    batch_loss += kl_loss.item()
                
                total_loss += batch_loss / batch_size
        
        avg_loss = total_loss / len(dataloader)
        print(f"Evaluation complete. Average Loss: {avg_loss:.4f}")
        return avg_loss

def main():
    parser = argparse.ArgumentParser(description='Train LLaDA model')
    parser.add_argument('--teacher_model', type=str, required=True,
                        help='Path or name of teacher model (LLaMA)')
    parser.add_argument('--student_model', type=str, required=True,
                        help='Path or name of student model (LLaDA)')
    parser.add_argument('--teacher_device', type=str, default='cuda:0',
                        help='Device for teacher model')
    parser.add_argument('--student_device', type=str, default='cuda:1',
                        help='Device for student model')
    parser.add_argument('--data_path', type=str, required=True,
                        help='Path to dataset')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Output directory for saved models')
    parser.add_argument('--batch_size', type=int, default=4,
                        help='Batch size for training')
    parser.add_argument('--num_epochs', type=int, default=3,
                        help='Number of training epochs')
    parser.add_argument('--learning_rate', type=float, default=5e-5,
                        help='Learning rate')
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
        eval_steps=args.eval_steps
    )
    
    # Evaluate
    distiller.evaluate(args.data_path, batch_size=args.batch_size)

if __name__ == "__main__":
    main()
