# QSPEC: Quantized Speculative Sampling

This extension to the LLM Speculative Sampling codebase implements the QSPEC approach, which combines two complementary quantization schemes for speculative decoding:

1. **Low-precision activation-weight quantization** for drafting tokens (fast but potentially less accurate)
2. **High-precision weight-only quantization** for verifying tokens (more accurate but slower)

## Key Features

- **Dual Quantization**: Leverages both activation-weight and weight-only quantization in a single framework
- **Flexible Model Selection**: Supports using the same model with different quantization schemes or completely different models
- **No Training Required**: Plug-and-play approach that doesn't require any additional training
- **Memory Efficient**: Reuses weights and KV cache to minimize memory overhead
- **Performance Boost**: Achieves significant speedups without compromising output quality

## Requirements

The implementation requires the following additional dependencies:
- `bitsandbytes`: For efficient quantization
- `accelerator`: For optimized model execution
- `optimum`: For BetterTransformer integration
- `einops`: For tensor manipulation utilities

## Usage Examples

### Basic Usage with Same Model

```bash
python qspec_example.py --model_name gpt2 --same_model --verbose
```

This will use the same model (gpt2) with different quantization schemes:
- Draft model: FP8 activation-weight quantization (default)
- Verification model: INT8 weight-only quantization (default)

### Using Different Models

```bash
python qspec_example.py --draft_model_name gpt2 --verify_model_name gpt2-medium --verbose
```

This will use:
- Draft model: gpt2 with FP8 activation-weight quantization
- Verification model: gpt2-medium with INT8 weight-only quantization

### Benchmarking

To benchmark the performance of QSPEC against baseline methods:

```bash
python benchmark_qspec.py --model_name gpt2 --same_model --test_all
```

For comparing different models:

```bash
python benchmark_qspec.py --draft_model_name gpt2 --verify_model_name gpt2-medium --test_all
```

## Quantization Methods

The implementation supports the following quantization methods:

- `none`: No quantization (full precision)
- `int8`: 8-bit weight-only quantization
- `int4`: 4-bit weight-only quantization
- `fp8`: 8-bit activation-weight quantization
- `fp4`: 4-bit activation-weight quantization

## Implementation Details

### Core Components

1. **QuantizedModel**: A wrapper for models with different quantization methods
2. **QSPECModel**: Combines two quantized models for drafting and verification
3. **qspec_sampling**: Implementation of QSPEC with a single model
4. **qspec_sampling_different_models**: Implementation of QSPEC with different models

### Key Advantages

- **Performance**: Up to 1.64x throughput improvement without quality compromise
- **Flexibility**: Works with various model sizes, quantization methods, and batch sizes
- **Compatibility**: Integrates seamlessly with existing speculative decoding pipelines
- **Memory Efficiency**: Avoids extra memory overhead by reusing weights and KV cache

## Future Work

- Implement more advanced quantization methods (GPTQ, AWQ)
- Add support for multi-GPU inference
- Optimize for specific hardware accelerators
- Extend to support more model architectures
