# Memory and Performance Optimizations for LLaMA-LLaDA Distillation

This document outlines the memory optimization techniques implemented in the distillation process between LLaMA (teacher model) and LLaDA (student model).

## Key Optimizations

### 1. Data Processing Optimizations

- **Pre-tokenization**: All data is tokenized once during dataset initialization, eliminating redundant tokenization during training.
- **Efficient Data Loading**: Using column selection when loading parquet files to reduce memory footprint.
- **Memory Cleanup**: Explicitly freeing memory after pre-tokenization to reduce overall memory usage.

### 2. Model Training Optimizations

- **Gradient Checkpointing**: Enabled for the student model to significantly reduce memory usage during backpropagation by trading computation for memory.
- **Gradient Accumulation**: Implemented to effectively increase batch size without increasing memory usage.
- **Mixed Precision Training**: Using bfloat16 for the student model to reduce memory usage.
- **Gradient Clipping**: Prevents exploding gradients and stabilizes training.
- **Memory-Efficient Optimizer**: Configured AdamW with parameter-specific weight decay settings.
- **Learning Rate Scheduling**: Added linear warmup and decay for better convergence.

### 3. Teacher-Student Separation

- **No Gradient Computation for Teacher**: Ensuring the LLaMA 70B model operates in evaluation mode with `torch.no_grad()` to prevent any gradient computation.
- **Device Management**: Keeping teacher and student models on separate devices to manage memory better.

### 4. Runtime Optimizations

- **TF32 Precision**: Enabled TF32 for matrix multiplications on supported GPUs.
- **CUDNN Benchmarking**: Enabled for faster training on CUDA devices.
- **Explicit Memory Clearing**: Added explicit CUDA cache clearing during training.

## Usage

The optimized scripts include new command-line arguments:

```bash
python train_best_dataset_strat.py \
  --gradient_accumulation_steps 4 \
  --weight_decay 0.01 \
  --warmup_steps 100
```

## Performance Impact

These optimizations should result in:

1. **Reduced Memory Usage**: By implementing gradient checkpointing, mixed precision, and efficient data handling.
2. **Faster Training**: Through gradient accumulation, optimized data loading, and TF32 precision.
3. **Better Convergence**: With learning rate scheduling and optimized optimizer settings.

## Monitoring

When running the training, monitor:
- GPU memory usage
- Training speed (samples/second)
- Loss convergence

If memory issues persist, consider:
- Increasing gradient accumulation steps
- Decreasing batch size
- Further model parallelism techniques
