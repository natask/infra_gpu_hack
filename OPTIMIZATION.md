# Memory and Performance Optimizations for LLaMA-LLaDA Distillation

This document outlines the memory optimization techniques for the distillation process between LLaMA (teacher model) and LLaDA (student model).

## Memory Optimization Approaches

We've developed two different optimization strategies based on complexity vs. effectiveness:

1. **Comprehensive Strategy** (`train_all_strat.py`): Implements all possible optimizations
2. **Simplified Strategy** (`train_simple_strat.py`): Focuses on the most important optimizations

## Key Optimizations

### 1. Essential Memory Saving Techniques

- **Gradient Checkpointing**: Enabled for the student model to significantly reduce memory usage during backpropagation by trading computation for memory.
- **No Gradient Computation for Teacher**: Using `torch.no_grad()` to prevent gradient computation through the LLaMA model.
- **Mixed Precision Training**: Using bfloat16 for the student model to reduce memory usage.
- **Explicit Memory Clearing**: Added explicit CUDA cache clearing during training.

### 2. Advanced Techniques (Comprehensive Strategy)

- **Pre-tokenization**: Tokenizing all data in advance to reduce redundant computation.
- **Gradient Accumulation**: Effectively increases batch size without increasing memory usage.
- **Gradient Clipping**: Prevents exploding gradients and stabilizes training.
- **Memory-Efficient Optimizer**: Configured AdamW with parameter-specific weight decay settings.
- **Learning Rate Scheduling**: Added linear warmup and decay for better convergence.

### 3. Device Management

- **Device Separation**: Keeping teacher and student models on separate devices to manage memory better.
- **TF32 Precision**: Enabled TF32 for matrix multiplications on supported GPUs.
- **CUDNN Benchmarking**: Enabled for faster training on CUDA devices.

## Simplified vs. Comprehensive Strategy

### Simplified Strategy (`train_simple_strat.py`)

Focuses on three key optimizations that provide the most memory reduction:

1. **Gradient Checkpointing**: Added after model initialization
2. **No Gradient Computation for Teacher**: Using `torch.no_grad()`
3. **Mixed Precision Training**: Using bfloat16 for the student model
4. **Periodic Memory Clearing**: Using `torch.cuda.empty_cache()`

This approach is simpler to understand and implement, making it a good starting point when optimizing memory usage.

### Comprehensive Strategy (`train_all_strat.py`)

Includes all optimizations from the simplified strategy plus:

- **Data pre-tokenization** 
- **Gradient accumulation**
- **Advanced optimizer configuration**
- **Learning rate scheduling**

## How to Choose

1. Start with the simplified strategy to test if basic optimizations are sufficient
2. If memory issues persist, move to the comprehensive strategy
3. Monitor GPU memory usage during training to determine effectiveness

## Usage

### Simplified Strategy
```bash
python train_simple_strat.py \
  --teacher_model /path/to/llama \
  --student_model /path/to/llada \
  --data_path /path/to/data.parquet \
  --output_dir ./output \
  --batch_size 4
```

### Comprehensive Strategy
```bash
python train_all_strat.py \
  --teacher_model /path/to/llama \
  --student_model /path/to/llada \
  --data_path /path/to/data.parquet \
  --output_dir ./output \
  --batch_size 4 \
  --gradient_accumulation_steps 4 \
  --weight_decay 0.01 \
  --warmup_steps 100
```

## Monitoring

When running the training, monitor:
- GPU memory usage (`nvidia-smi -l 1`)
- Training speed (samples/second)
- Loss convergence

If memory issues persist, consider:
- Using a smaller batch size
- Implementing model parallelism techniques
- Reducing model precision further
