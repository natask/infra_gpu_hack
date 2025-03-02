import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
from torch.utils.data import Dataset, DataLoader, random_split
from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
import torch.distributed as dist
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
        teacher_model_name="casperhansen/llama-3.3-70b-instruct-awq",
        student_model_name="meta-llama/Llama-3.1-8B-Instruct",
        teacher_device="cuda:0",
        student_device="cuda:1",
        temperature=2.0,
        mask_id=126336,  # LLaDA mask token ID
        checkpoint_path=None,
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
        if checkpoint_path:
            print(f"Loading student model from checkpoint {checkpoint_path} on {student_device}...")
            self.student_tokenizer = AutoTokenizer.from_pretrained(checkpoint_path, trust_remote_code=True)
            self.student_model = AutoModel.from_pretrained(
                checkpoint_path,
                device_map=student_device,
                trust_remote_code=True, 
                torch_dtype=torch.bfloat16
            )
        else:
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
        
        # Freeze teacher model parameters
        for param in self.teacher_model.parameters():
            param.requires_grad = False

    def load_from_checkpoint(self, checkpoint_path):
        """
        Load a previously saved checkpoint for the student model
        """
        print(f"Loading student model from checkpoint {checkpoint_path}...")
        self.student_model = AutoModel.from_pretrained(
            checkpoint_path,
            device_map=self.student_device,
            trust_remote_code=True, 
            torch_dtype=torch.bfloat16
        )
        self.student_tokenizer = AutoTokenizer.from_pretrained(
            checkpoint_path, 
            trust_remote_code=True
        )
        print(f"Successfully loaded model from checkpoint {checkpoint_path}")
        return self
    
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
        eval_data_path=None,
        train_ratio=0.8
    ):
        os.makedirs(output_dir, exist_ok=True)
        
        # Load the full dataset
        full_dataset = pd.read_parquet(data_path)
        
        # Create train/eval split if no separate eval dataset is provided
        if eval_data_path is None:
            print(f"No separate evaluation dataset provided. Creating train/eval split with {train_ratio:.0%} training data...")
            train_size = int(len(full_dataset) * train_ratio)
            eval_size = len(full_dataset) - train_size
            
            # Create a random permutation for splitting
            indices = torch.randperm(len(full_dataset)).tolist()
            train_indices = indices[:train_size]
            eval_indices = indices[train_size:]
            
            # Create the training and evaluation dataframes
            train_df = full_dataset.iloc[train_indices].reset_index(drop=True)
            eval_df = full_dataset.iloc[eval_indices].reset_index(drop=True)
            
            # Save the split datasets
            train_output_path = os.path.join(output_dir, "train_split.parquet")
            eval_output_path = os.path.join(output_dir, "eval_split.parquet")
            
            train_df.to_parquet(train_output_path)
            eval_df.to_parquet(eval_output_path)
            
            print(f"Saved train split ({len(train_df)} samples) to {train_output_path}")
            print(f"Saved eval split ({len(eval_df)} samples) to {eval_output_path}")
            
            # Set the evaluation data path for later use
            self.eval_data_path = eval_output_path
            
            # Create dataset with the training data
            dataset = PromptDataset(
                train_output_path, 
                self.teacher_tokenizer, 
                self.student_tokenizer,
                max_length=max_length
            )
        else:
            # Use the provided training data
            self.eval_data_path = eval_data_path
            dataset = PromptDataset(
                data_path, 
                self.teacher_tokenizer, 
                self.student_tokenizer,
                max_length=max_length
            )
        
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        # Setup optimizer
        optimizer = torch.optim.AdamW(self.student_model.parameters(), lr=learning_rate)
        
        # Training loop
        global_step = 0
        total_loss = 0
        
        for epoch in range(num_epochs):
            print(f"Epoch {epoch+1}/{num_epochs}")
            
            progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}")
            for batch in progress_bar:
                loss = self.distill_batch(batch, optimizer)
                total_loss += loss
                global_step += 1
                
                progress_bar.set_postfix({"loss": loss})
                
                # Save checkpoint
                if global_step % save_steps == 0:
                    checkpoint_dir = os.path.join(output_dir, f"checkpoint-{global_step}")
                    os.makedirs(checkpoint_dir, exist_ok=True)
                    
                    # Save the model and tokenizer
                    self.student_model.save_pretrained(checkpoint_dir)
                    self.student_tokenizer.save_pretrained(checkpoint_dir)
                    
                    # Copy the original model's configuration_llada.py file to ensure compatibility
                    try:
                        # Get the cache directory from transformers
                        from transformers.utils import TRANSFORMERS_CACHE
                        
                        # For student model, search for configuration file
                        student_model_name = self.student_model.config._name_or_path
                        if "/" in student_model_name:  # It's a repo ID
                            repo_id = student_model_name.replace("/", "--")
                            config_file_pattern = os.path.join(TRANSFORMERS_CACHE, "models--" + repo_id, "**/*configuration_llada.py")
                            
                            import glob
                            config_files = glob.glob(config_file_pattern, recursive=True)
                            
                            if config_files:
                                import shutil
                                # Copy the first matching configuration file to the checkpoint directory
                                shutil.copy(config_files[0], os.path.join(checkpoint_dir, "configuration_llada.py"))
                                print(f"Copied configuration file to ensure checkpoint compatibility")
                    except Exception as e:
                        print(f"Warning: Could not copy configuration file: {e}")
                    
                    print(f"Saved model checkpoint to {checkpoint_dir}")
                
                # Evaluate
                if global_step % eval_steps == 0:
                    avg_loss = total_loss / eval_steps
                    print(f"Step {global_step}: Average Loss = {avg_loss:.4f}")
                    
                    # Run evaluation on the separate eval dataset
                    eval_loss = self.evaluate(self.eval_data_path, batch_size=batch_size)
                    print(f"Step {global_step}: Evaluation Loss = {eval_loss:.4f}")
                    
                    total_loss = 0
        
        # Save final model
        final_output_dir = os.path.join(output_dir, "final_model")
        os.makedirs(final_output_dir, exist_ok=True)
        
        # Save the model and tokenizer
        self.student_model.save_pretrained(final_output_dir)
        self.student_tokenizer.save_pretrained(final_output_dir)
        
        # Copy the original model's configuration_llada.py file to ensure compatibility
        try:
            # Get the cache directory from transformers
            from transformers.utils import TRANSFORMERS_CACHE
            
            # For student model, search for configuration file
            student_model_name = self.student_model.config._name_or_path  
            if "/" in student_model_name:  # It's a repo ID
                repo_id = student_model_name.replace("/", "--")
                config_file_pattern = os.path.join(TRANSFORMERS_CACHE, "models--" + repo_id, "**/*configuration_llada.py")
                
                import glob
                config_files = glob.glob(config_file_pattern, recursive=True)
                
                if config_files:
                    import shutil
                    # Copy the first matching configuration file to the checkpoint directory
                    shutil.copy(config_files[0], os.path.join(final_output_dir, "configuration_llada.py"))
                    print(f"Copied configuration file to ensure checkpoint compatibility")
        except Exception as e:
            print(f"Warning: Could not copy configuration file: {e}")
            
        print(f"Saved final model to {final_output_dir}")
        
        # Final evaluation
        print("Performing final evaluation...")
        final_eval_loss = self.evaluate(self.eval_data_path, batch_size=batch_size)
        print(f"Final Evaluation Loss: {final_eval_loss:.4f}")
    
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
            indices = torch.randperm(len(dataset))[:num_eval_samples].tolist()  # Convert tensor to list of integers
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
    parser.add_argument('--data_path', type=str, default="/useme/llada-2/limo_data/limo_train.parquet",
                        help='Path to the training dataset file')
    parser.add_argument('--eval_data_path', type=str, default=None,
                        help='Path to the evaluation dataset file. If not provided, a portion of the training data will be used.')
    parser.add_argument('--train_ratio', type=float, default=0.8,
                        help='Ratio of data to use for training when splitting (default: 0.8)')
    parser.add_argument('--output_dir', type=str, default="./llada_distilled",
                        help='Output directory for saved models')
    parser.add_argument('--checkpoint_path', type=str, default=None,
                        help='Path to a checkpoint to load the student model from')
    parser.add_argument('--eval_only', action='store_true',
                        help='Only run evaluation (requires --checkpoint_path)')
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
    
    # Validate arguments
    if args.eval_only and args.checkpoint_path is None:
        raise ValueError("--eval_only requires --checkpoint_path to be specified")
    
    # Create distiller
    distiller = LLaDADistiller(
        teacher_model_name=args.teacher_model,
        student_model_name=args.student_model,
        teacher_device=args.teacher_device,
        student_device=args.student_device,
        temperature=args.temperature,
        checkpoint_path=args.checkpoint_path
    )
    
    # Evaluation only mode
    if args.eval_only:
        print(f"Running evaluation only on checkpoint: {args.checkpoint_path}")
        eval_data = args.eval_data_path if args.eval_data_path else args.data_path
        eval_loss = distiller.evaluate(eval_data, batch_size=args.batch_size)
        print(f"Evaluation Loss: {eval_loss:.4f}")
        return
    
    # Train (and evaluate during training)
    distiller.train(
        data_path=args.data_path,
        eval_data_path=args.eval_data_path,
        train_ratio=args.train_ratio,
        output_dir=args.output_dir,
        num_epochs=args.num_epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        max_length=args.max_length,
        save_steps=args.save_steps,
        eval_steps=args.eval_steps
    )

if __name__ == "__main__":
    main()
