# DELTA - Diffusive Extrapolative Language Text Algorithm

A novel algorithm that integrates a text, diffusion LLM as a draft model to boost the performance of traditional auto-regressive LLMs. Try it out [here](https://deltafrontend.vercel.app).

Built for the 2025 Mercor x Cognition x Etched Hackathon
<img width="819" alt="image" src="https://github.com/user-attachments/assets/824ccf96-6974-42d6-b6cd-e8a1c36e0722" />

## Memory Optimization for LLaMA-LLaDA Distillation

We've implemented two memory optimization strategies for the LLaMA-LLaDA distillation process to address memory constraints when working with large language models:

1. **Simplified Strategy** (`train_simple_strat.py`): Focuses on the most essential memory optimizations:
   - Gradient checkpointing for the student model
   - No gradient computation for the teacher model
   - Mixed precision training (bfloat16)
   - Periodic CUDA cache clearing

2. **Comprehensive Strategy** (`train_all_strat.py`): Implements all optimizations from the simplified strategy plus:
   - Data pre-tokenization
   - Gradient accumulation
   - Advanced optimizer configuration
   - Learning rate scheduling

For detailed information on these optimizations, see [OPTIMIZATION.md](OPTIMIZATION.md).
  
## Project Structure

- **[combine_datasets.py](cci:7://file:///home/savnkk/infra_gpu_hack/combine_datasets.py:0:0-0:0)**: This script loads and combines datasets from different sources, ensuring all columns are present in each dataset. The final dataset is saved as a Parquet file.
  
- **`scripts/`**: Contains various scripts for dataset handling and model evaluation:
  - **[custom_dataset.py](cci:7://file:///home/savnkk/infra_gpu_hack/scripts/custom_dataset.py:0:0-0:0)**: Custom dataset handling.
  - **[download_dataset.py](cci:7://file:///home/savnkk/infra_gpu_hack/scripts/download_dataset.py:0:0-0:0)**: Script to download datasets.
  - **[evaluate_direct.py](cci:7://file:///home/savnkk/infra_gpu_hack/scripts/evaluate_direct.py:0:0-0:0)**: Direct evaluation of models.
  - **[evaluate_speculative.py](cci:7://file:///home/savnkk/infra_gpu_hack/scripts/evaluate_speculative.py:0:0-0:0)**: Speculative evaluation of models.
  - **[fine_tune.py](cci:7://file:///home/savnkk/infra_gpu_hack/scripts/fine_tune.py:0:0-0:0)**: Script for fine-tuning models.
  - **[generate.py](cci:7://file:///home/savnkk/infra_gpu_hack/scripts/generate.py:0:0-0:0)**: Script to generate outputs from models.
  - **[speculative_decoding.py](cci:7://file:///home/savnkk/infra_gpu_hack/scripts/speculative_decoding.py:0:0-0:0)**: Script for speculative decoding.

## Requirements

The project requires Python and several dependencies listed in [requirements.txt](cci:7://file:///home/savnkk/infra_gpu_hack/requirements.txt:0:0-0:0). To install them, use:

```bash
pip install -r requirements.txt
```

## Usage

1. **Combine Datasets**: Run [combine_datasets.py](cci:7://file:///home/savnkk/infra_gpu_hack/combine_datasets.py:0:0-0:0) to load, process, and save a combined dataset.
   ```bash
   python combine_datasets.py
   ```

2. **Scripts**: Use the scripts in the `scripts/` directory for specific tasks like downloading datasets, evaluating models, fine-tuning, and generating outputs.

### Script Usages

- **custom_dataset.py**: Defines a custom dataset class for loading data from a directory where each entry is stored as a JSON file.
  ```python
  from custom_dataset import get_dataloader

  dataloader = get_dataloader('path/to/dataset', batch_size=8, shuffle=True)
  ```

- **download_dataset.py**: Downloads a dataset from Hugging Face and saves each entry under a directory named after the dataset.
  ```bash
  python download_dataset.py --dataset_name <dataset_name> --split <split> --save_dir <save_directory>
  ```

- **evaluate_direct.py**: Evaluates model performance using direct decoding.
  ```bash
  python evaluate_direct.py --model_name <model_name> --evaluation_dataset <evaluation_dataset> --max_length <max_length>
  ```

- **evaluate_speculative.py**: Evaluates model performance using speculative decoding with a teacher and student model.
  ```bash
  python evaluate_speculative.py --teacher_model <teacher_model> --student_model <student_model> --evaluation_dataset <evaluation_dataset> --max_length <max_length> --speculative_steps <speculative_steps>
  ```

- **fine_tune.py**: Fine-tunes a Hugging Face model on a specified dataset.
  ```bash
  python fine_tune.py --model_name <model_name> --dataset_name <dataset_name> --fine_tuned_model_name <fine_tuned_model_name> --batch_size <batch_size> --learning_rate <learning_rate> --num_train_epochs <num_train_epochs> --max_length <max_length> --checkpoint <checkpoint>
  ```

- **generate.py**: Generates model outputs based on a specified dataset and configuration.
  ```bash
  python generate.py --model_name <model_name> --dataset_name <dataset_name> --batch_size <batch_size> --config <config> --max_length <max_length>
  ```

- **speculative_decoding.py**: Performs speculative decoding using a teacher and student model.
  ```python
  from speculative_decoding import speculative_generate

  output = speculative_generate(teacher_model, student_model, teacher_tokenizer, student_tokenizer, input_text, max_length=50, speculative_steps=3)
  ```

## Dependencies

The project relies on various Python packages, including but not limited to:
- `datasets`
- `pandas`
- `torch`
- `transformers`

For a full list of dependencies, refer to the [requirements.txt](cci:7://file:///home/savnkk/infra_gpu_hack/requirements.txt:0:0-0:0) file.

## License

This project is licensed under the MIT License.
