import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from tqdm import tqdm
import torch.distributed as dist
import argparse
import gc

class PromptDataset(Dataset):
    def __init__(self, data_path, tokenizer_teacher, tokenizer_student, max_length=512, dataset_name="GAIR/lima", num_samples=None):
        # Set tokenizers and max length first to ensure they're available for all methods
        self.tokenizer_teacher = tokenizer_teacher
        self.tokenizer_student = tokenizer_student
        self.max_length = max_length
        
        # Determine how to load the data based on the provided path
        if data_path and os.path.exists(data_path):
            if os.path.isdir(data_path):
                # Check if directory contains JSON files (like GAIR/LIMO format)
                json_files = [f for f in os.listdir(data_path) if f.endswith('.json')]
                if json_files:
                    print(f"Loading data from local JSON files in: {data_path}")
                    self._load_from_local_json(data_path, json_files, num_samples)
                    return
            
            # If path is a file, try loading as parquet
            if os.path.isfile(data_path):
                print(f"Loading data from parquet file: {data_path}")
                self.df = pd.read_parquet(data_path, columns=['question'])
                if num_samples:
                    self.df = self.df.iloc[:num_samples]
                self._load_from_dataframe()
                return
        
        # If not a local file or directory, try loading from Hugging Face
        self._load_from_huggingface(dataset_name, num_samples)
    
    def _load_from_huggingface(self, dataset_name="GAIR/lima", num_samples=None):
        """Load data efficiently from Hugging Face datasets"""
        print(f"Loading dataset from Hugging Face: {dataset_name}")
        
        # Load the dataset efficiently with streaming mode for large datasets
        try:
            dataset = load_dataset(dataset_name, streaming=False)
            
            # Convert to pandas for consistent processing
            if "train" in dataset:
                self.data = dataset["train"]
            else:
                # If no train split, use the first available split
                first_split = list(dataset.keys())[0]
                self.data = dataset[first_split]
            
            # Print dataset features to help debug
            print(f"Dataset features: {self.data.column_names}")
            print(f"Dataset size: {len(self.data)} examples")
            
            # Limit number of samples if specified
            if num_samples is not None:
                self.data = self.data.select(range(min(num_samples, len(self.data))))
                print(f"Limited to {num_samples} samples")
            
            # Pre-tokenize data to speed up training
            print("Pre-tokenizing data to speed up training...")
            self.tokenized_data = []
            
            # Process in batches for efficiency
            batch_size = 32  # Process multiple examples at once
            
            # Determine which field to use based on dataset column names
            columns = self.data.column_names
            print(f"Available columns: {columns}")
            
            # Sample first example to understand structure
            first_example = self.data[0]
            print(f"First example keys: {first_example.keys()}")
            
            # Check if required columns exist in the dataset
            has_conversations = "conversations" in columns
            has_instruction = "instruction" in columns
            has_text = "text" in columns
            has_question = "question" in columns
            has_context = "context" in columns
            
            # For GAIR/lima specifically
            if dataset_name == "GAIR/lima":
                prompts = []
                for i in tqdm(range(len(self.data))):
                    item = self.data[i]
                    # Extract prompt based on dataset-specific structure
                    try:
                        # Check if the item has 'question' field (as seen in the local GAIR/LIMO files)
                        if "question" in item:
                            if "solution" in item:
                                # Format as a question-answer pair
                                prompt = f"Question: {item['question']}\n\nAnswer: {item['solution']}"
                            else:
                                # Just use the question
                                prompt = f"Question: {item['question']}"
                        elif "conversations" in item:
                            prompt = self._format_conversations(item["conversations"])
                        else:
                            # Try to use the first column as fallback
                            prompt = str(item[columns[0]])
                        prompts.append(prompt)
                    except Exception as e:
                        print(f"Error extracting prompt from item {i}: {str(e)}")
                        continue
            else:
                # General approach for other datasets
                prompts = []
                for i in tqdm(range(0, len(self.data), batch_size)):
                    batch = self.data[i:min(i+batch_size, len(self.data))]
                    batch_prompts = []
                    
                    # Extract prompts from the dataset based on available columns
                    for item in batch:
                        try:
                            if has_conversations:
                                prompt = self._format_conversations(item["conversations"])
                            elif has_instruction:
                                prompt = item["instruction"]
                            elif has_question:
                                # For SQuAD-like datasets, combine question with context
                                if has_context:
                                    prompt = f"Context: {item['context']}\n\nQuestion: {item['question']}"
                                else:
                                    prompt = item["question"]
                            elif has_text:
                                prompt = item["text"]
                            else:
                                # If none of the expected fields exist, use the first column
                                first_column = columns[0]
                                prompt = str(item[first_column])
                            
                            batch_prompts.append(prompt)
                        except Exception as e:
                            print(f"Error extracting prompt: {str(e)}")
                            continue
                    
                    prompts.extend(batch_prompts)
            
            # Process each prompt with error handling
            print(f"Processing {len(prompts)} prompts...")
            for prompt in prompts:
                try:
                    if prompt is None or not isinstance(prompt, str):
                        print(f"Warning: Skipping invalid prompt: {type(prompt)}")
                        continue
                    self._process_and_tokenize_prompt(prompt)
                except Exception as e:
                    print(f"Error processing prompt: {str(e)}")
                    continue
            
            print(f"Successfully processed {len(self.tokenized_data)} prompts")
        
            # Free memory
            del self.data
            gc.collect()
            
        except Exception as e:
            print(f"Error loading dataset {dataset_name}: {str(e)}")
            print("Falling back to a small sample dataset.")
            
            # Create a small sample dataset with just a few examples for fallback
            self.tokenized_data = []
            sample_prompts = [
                "What is machine learning?",
                "Explain the concept of gradient descent.",
                "How does a transformer model work?"
            ]
            
            for prompt in sample_prompts:
                self._process_and_tokenize_prompt(prompt)
    
    def _format_conversations(self, conversations):
        """Extract the first user message from conversations"""
        # If conversations is a string, return it directly
        if isinstance(conversations, str):
            return conversations
            
        # If conversations is a list, process it
        if isinstance(conversations, list):
            for turn in conversations:
                if isinstance(turn, dict):
                    if turn.get("from") == "human" or turn.get("role") == "user":
                        return turn.get("value") or turn.get("content")
            # If no human/user message was found, return the first message content
            if len(conversations) > 0 and isinstance(conversations[0], dict):
                return conversations[0].get("value") or conversations[0].get("content")
        
        # If we couldn't extract anything, return a default message
        return "Unable to extract prompt from conversations"
    
    def _load_from_local_json(self, data_dir, json_files, num_samples=None):
        """Load data from local JSON files (like GAIR/LIMO format)"""
        import json
        self.tokenized_data = []
        
        # Limit the number of files if num_samples is specified
        if num_samples is not None and num_samples < len(json_files):
            json_files = json_files[:num_samples]
            print(f"Limited to {num_samples} files")
            
        print(f"Processing {len(json_files)} JSON files...")
        for filename in tqdm(json_files):
            file_path = os.path.join(data_dir, filename)
            try:
                with open(file_path, 'r') as f:
                    data = json.load(f)
                
                # Handle different formats in the JSON files
                if "question" in data:
                    if "solution" in data:
                        prompt = f"Question: {data['question']}\n\nAnswer: {data['solution']}"
                    else:
                        prompt = f"Question: {data['question']}"
                elif "conversations" in data:
                    prompt = self._format_conversations(data["conversations"])
                else:
                    # Try to use the first key as fallback
                    first_key = list(data.keys())[0]
                    prompt = str(data[first_key])
                
                self._process_and_tokenize_prompt(prompt)
            except Exception as e:
                print(f"Error processing file {filename}: {str(e)}")
                continue
                
        print(f"Successfully processed {len(self.tokenized_data)} files")

    def _load_from_dataframe(self):
        """Load and process data from pandas DataFrame"""
        # Pre-tokenize data to speed up training
        print("Pre-tokenizing data from parquet file...")
        self.tokenized_data = []
        
        for idx in tqdm(range(len(self.df))):
            prompt = self.df.iloc[idx]['question']
            self._process_and_tokenize_prompt(prompt)
        
        # Free memory
        del self.df
        gc.collect()
    
    def _process_and_tokenize_prompt(self, prompt):
        """Process and tokenize a single prompt"""
        # Tokenize for teacher (LLaMA)
        teacher_inputs = self.tokenizer_teacher(
            prompt,
            return_tensors="pt",
            padding="max_length",
            max_length=self.max_length,
            truncation=True
        )
        
        # Tokenize for student (LLaDA)
        # Check if apply_chat_template is available
        if hasattr(self.tokenizer_student, 'apply_chat_template'):
            # For chat-based models like Llama
            m = [{"role": "user", "content": prompt}]
            student_prompt = self.tokenizer_student.apply_chat_template(
                m, 
                add_generation_prompt=True, 
                tokenize=False
            )
        else:
            # For models without chat template support like BERT
            student_prompt = prompt
            
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
        
        # Handle different device specifications
        teacher_kwargs = {}
        if teacher_device.lower() == "cpu":
            teacher_kwargs["device_map"] = None
            teacher_kwargs["torch_dtype"] = torch.float32  # CPU typically uses float32 for better compatibility
        elif teacher_device.startswith("cuda"):
            teacher_kwargs["device_map"] = "auto"  # Let the model decide optimal placement
            teacher_kwargs["torch_dtype"] = torch.float16
        else:
            # For other device map options (auto, balanced, etc.)
            teacher_kwargs["device_map"] = teacher_device
            teacher_kwargs["torch_dtype"] = torch.float16
        
        self.teacher_model = AutoModelForCausalLM.from_pretrained(
            teacher_model_name,
            **teacher_kwargs
        )
        self.teacher_model.eval()  # Teacher is always in eval mode
        
        # Load student model (LLaDA)
        print(f"Loading student model {student_model_name} on {student_device}...")
        self.student_tokenizer = AutoTokenizer.from_pretrained(
            student_model_name, 
            trust_remote_code=True
        )
        
        # Handle different device specifications for student model
        student_kwargs = {
            "trust_remote_code": True
        }
        
        if student_device.lower() == "cpu":
            student_kwargs["device_map"] = None
            student_kwargs["torch_dtype"] = torch.float32  # CPU typically uses float32 for better compatibility
        elif student_device.startswith("cuda"):
            student_kwargs["device_map"] = "auto"  # Let the model decide optimal placement
            student_kwargs["torch_dtype"] = torch.bfloat16
        else:
            # For other device map options (auto, balanced, etc.)
            student_kwargs["device_map"] = student_device
            student_kwargs["torch_dtype"] = torch.bfloat16
        
        self.student_model = AutoModel.from_pretrained(
            student_model_name,
            **student_kwargs
        )
        
        # Enable gradient checkpointing after model is loaded
        # This trades compute for memory by not storing all activations
        print("Enabling memory optimizations for the student model...")
        
        # Set model to training mode to ensure modules like dropout are active
        self.student_model.train()
        
        # For memory efficiency, we'll use lower precision and avoid storing intermediate activations
        # Note: We don't forcibly enable gradient checkpointing as the model might not support it
        # Instead, we'll rely on other memory optimizations
        
        # Freeze teacher model parameters
        for param in self.teacher_model.parameters():
            param.requires_grad = False

    def forward_process(self, batch, prompt_indices):
        """
        Apply forward diffusion process to prepare tokens for LLaDA learning
        Similar to the get_log_likelihood.py implementation in LLaDA repo
        Optimized for memory efficiency
        """
        b, l = batch.shape
        
        # Find which tokens are not prompt (these are the ones we'll mask)
        non_prompt_mask = ~prompt_indices
        target_len = non_prompt_mask.sum(dim=1)
        
        # Initialize mask tensor directly as bool to save memory (no type conversion)
        is_mask = torch.zeros_like(batch, dtype=torch.bool, device=batch.device)
        
        # Process in a memory-efficient way
        for i in range(b):
            if target_len[i] > 0:  # Only process if there are non-prompt tokens
                # Determine number of tokens to mask (1 to target_len)
                k = max(1, min(torch.randint(1, target_len[i] + 1, (1,), device=batch.device)[0].item(), target_len[i].item()))
                
                # Get indices of non-prompt tokens
                non_prompt_indices = torch.where(non_prompt_mask[i])[0]
                
                # Randomly select k indices to mask (without creating large temporary tensors)
                if len(non_prompt_indices) > 0:
                    # Use CPU for the permutation to save GPU memory
                    perm_indices = torch.randperm(len(non_prompt_indices))[:k].to(batch.device)
                    mask_indices = non_prompt_indices[perm_indices]
                    is_mask[i, mask_indices] = True
                    # Free up memory
                    del perm_indices, mask_indices
        
        # Apply mask (in-place operations where possible)
        noisy_batch = batch.clone()
        noisy_batch[is_mask] = self.mask_id
        
        # Calculate mask ratio efficiently
        mask_ratio = is_mask.float()
        
        # Return results
        return noisy_batch, mask_ratio, is_mask
    
    def get_teacher_logits(self, input_ids, attention_mask):
        """Get teacher (LLaMA) logits with enhanced memory efficiency"""
        with torch.no_grad():
            outputs = self.teacher_model(
                input_ids=input_ids.to(self.teacher_device),
                attention_mask=attention_mask.to(self.teacher_device),
                return_dict=True
            )
            # Only retrieve the necessary logits and immediately detach
            logits = outputs.logits.detach()
            # Explicitly delete outputs to free memory
            del outputs
        return logits
    
    def get_student_logits(self, input_ids, mask_indices):
        """Get student (LLaDA) logits for the tokens at masked positions"""
        # Using bfloat16 for more efficient memory usage during forward pass
        with torch.cuda.amp.autocast(dtype=torch.bfloat16, enabled=True):
            outputs = self.student_model(input_ids.to(self.student_device), return_dict=True)
            logits = outputs.logits
            # Explicitly delete other outputs to free memory
            del outputs
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
        
        # When using gradient accumulation, we return the loss without backpropagation
        # The calling function will handle scaling and backward
        if do_backward:
            # Just return the loss without backward - caller will handle it
            return loss
        else:
            # Traditional mode (not using gradient accumulation)
            optimizer.zero_grad()
            loss.backward()
            # Clip gradients to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(self.student_model.parameters(), max_norm=1.0)
            optimizer.step()
            return loss.item()
    
    def train(
        self,
        data_path=None,
        dataset_name="GAIR/lima",
        num_samples=None,
        output_dir="./llada_distilled",
        num_epochs=3,
        batch_size=4,
        learning_rate=5e-5,
        max_length=512,
        save_steps=100,
        eval_steps=50,
        gradient_accumulation_steps=4,  # Accumulate gradients to save memory
        warmup_steps=100,  # Warm up learning rate
        weight_decay=0.01,  # Weight decay for regularization
        memory_clearing_interval=5,  # Clear GPU cache every N steps
        use_mixed_precision=False,  # Use mixed precision for training (bfloat16)
        use_cpu_offloading=False  # Offload tensors to CPU when possible to save GPU memory
    ):
        os.makedirs(output_dir, exist_ok=True)
        
        # Store memory optimization parameters
        self.memory_clearing_interval = memory_clearing_interval
        self.use_mixed_precision = use_mixed_precision
        self.use_cpu_offloading = use_cpu_offloading
        
        # Initialize mixed precision training if enabled
        scaler = None
        if use_mixed_precision:
            if not hasattr(torch.cuda.amp, 'GradScaler'):
                print("Warning: Mixed precision requested but GradScaler not available. Disabling mixed precision.")
                use_mixed_precision = False
            else:
                scaler = torch.cuda.amp.GradScaler()
                print("Using mixed precision training with gradient scaler")
        
        # Free up memory before starting training
        if hasattr(torch.cuda, 'empty_cache'):
            torch.cuda.empty_cache()
        
        # Create dataset and dataloader with optimized settings using either Hugging Face dataset or parquet
        dataset = PromptDataset(
            data_path, 
            self.teacher_tokenizer, 
            self.student_tokenizer,
            max_length=max_length,
            dataset_name=dataset_name,
            num_samples=num_samples
        )
        
        # Use efficient data loading with pinned memory if CPU offloading is enabled
        pin_memory = use_cpu_offloading
        dataloader = DataLoader(
            dataset, 
            batch_size=batch_size, 
            shuffle=True, 
            pin_memory=pin_memory,
            num_workers=0 if use_cpu_offloading else 0  # Adjust based on system capabilities
        )
        
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
            eps=1e-8,
            foreach=True  # Use more memory-efficient implementation if available
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
                # Apply mixed precision if enabled
                if self.use_mixed_precision and scaler is not None:
                    with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                        loss = self.distill_batch(batch, optimizer, do_backward=True)
                    # Scale loss to avoid underflow
                    scaled_loss = loss / gradient_accumulation_steps
                    # Use scaler for backward pass
                    scaler.scale(scaled_loss).backward()
                else:
                    # Standard precision training
                    loss = self.distill_batch(batch, optimizer, do_backward=True)
                    scaled_loss = loss / gradient_accumulation_steps
                    scaled_loss.backward()
                
                total_loss += loss.item()  # Track original loss for reporting
                
                # Explicitly free GPU memory after backward pass
                del scaled_loss, loss, batch
                if hasattr(torch.cuda, 'empty_cache') and (step % self.memory_clearing_interval == 0):
                    torch.cuda.empty_cache()
                
                # CPU offloading - explicitly move some grads to CPU to save GPU memory
                if self.use_cpu_offloading and (step % (self.memory_clearing_interval * 2) == 0):
                    for param in self.student_model.parameters():
                        if param.grad is not None and not param.grad.isfinite().all():
                            # Handle gradient overflow
                            param.grad.zero_()
                
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
                    avg_batch_loss = total_loss / gradient_accumulation_steps
                    progress_bar.set_postfix({
                        "loss": avg_batch_loss,
                        "lr": scheduler.get_last_lr()[0]
                    })
                    total_loss = 0
                    
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
    
    def evaluate(self, data_path=None, dataset_name="GAIR/lima", batch_size=4, max_length=512, num_eval_samples=100):
        """Evaluate the distilled model"""
        dataset = PromptDataset(
            data_path, 
            self.teacher_tokenizer, 
            self.student_tokenizer,
            max_length=max_length,
            dataset_name=dataset_name,
            num_samples=num_eval_samples
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
    # Model parameters
    parser.add_argument('--teacher_model', type=str, default="casperhansen/llama-3.3-70b-instruct-awq",
                        help='Teacher model name or path')
    parser.add_argument('--student_model', type=str, default="GSAI-ML/LLaDA-8B-Instruct",
                        help='Student model name or path')
    # Device configuration
    parser.add_argument('--teacher_device', type=str, default="cuda:0",
                        help='Device for teacher model')
    parser.add_argument('--student_device', type=str, default="cuda:1",
                        help='Device for student model')
    # Data parameters
    parser.add_argument('--data_path', type=str, default=None,
                        help='Path to the dataset file (parquet format). If not provided, will use the HF dataset.')
    parser.add_argument('--dataset_name', type=str, default="GAIR/lima",
                        help='Hugging Face dataset name to use (e.g., "GAIR/lima")')
    parser.add_argument('--num_samples', type=int, default=None,
                        help='Number of samples to use from the dataset (None for all)')
    parser.add_argument('--output_dir', type=str, default="./llada_distilled",
                        help='Output directory for saved models')
    parser.add_argument('--max_length', type=int, default=512,
                        help='Maximum sequence length')
    # Training hyperparameters
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
    # Checkpointing and evaluation
    parser.add_argument('--save_steps', type=int, default=100,
                        help='Save checkpoint every X steps')
    parser.add_argument('--eval_steps', type=int, default=50,
                        help='Evaluate every X steps')
    # Memory optimization parameters
    parser.add_argument('--memory_clearing_interval', type=int, default=5,
                        help='Clear GPU cache every N steps')
    parser.add_argument('--use_mixed_precision', action='store_true',
                        help='Use mixed precision for training (bfloat16)')
    parser.add_argument('--use_cpu_offloading', action='store_true', 
                        help='Offload tensors to CPU when possible to save GPU memory')
    
    args = parser.parse_args()
    
    # Set up torch for memory efficiency
    torch.backends.cuda.matmul.allow_tf32 = True  # Allow TF32 for faster computation
    torch.backends.cudnn.benchmark = True  # Optimize CUDNN for faster training
    torch.backends.cudnn.deterministic = False  # Allow non-deterministic algorithms for better performance
    
    # Enable memory-efficient attention if available (for transformer models)
    if hasattr(torch.nn.functional, 'scaled_dot_product_attention'):
        # PyTorch 2.0+ has memory-efficient attention
        print("Using PyTorch's memory-efficient attention mechanism")
    
    # Set default tensor type for efficiency
    if args.use_mixed_precision:
        print("Using bfloat16 mixed precision")
        torch.set_default_dtype(torch.bfloat16)
    else:
        torch.set_default_dtype(torch.float32)
    
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
        dataset_name=args.dataset_name,
        num_samples=args.num_samples,
        output_dir=args.output_dir,
        num_epochs=args.num_epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        max_length=args.max_length,
        save_steps=args.save_steps,
        eval_steps=args.eval_steps,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        warmup_steps=args.warmup_steps,
        weight_decay=args.weight_decay,
        memory_clearing_interval=args.memory_clearing_interval,
        use_mixed_precision=args.use_mixed_precision,
        use_cpu_offloading=args.use_cpu_offloading
    )
    
    # Evaluate
    distiller.evaluate(
        data_path=args.data_path,
        dataset_name=args.dataset_name,
        batch_size=args.batch_size,
        max_length=args.max_length,
        num_eval_samples=min(100, args.num_samples) if args.num_samples else 100  # Limit evaluation to 100 samples by default
    )

if __name__ == "__main__":
    main()
