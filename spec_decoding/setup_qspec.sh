#!/bin/bash

# Setup script for QSPEC implementation

echo "Setting up QSPEC dependencies..."

# Install base requirements
pip install -r requirements.txt

# Install QSPEC specific requirements
pip install bitsandbytes>=0.41.0
pip install accelerate
pip install optimum>=1.12.0
pip install einops
pip install numpy>=1.24.0

echo "Dependencies installed successfully!"
echo "You can now run the QSPEC examples:"
echo "  - python qspec_example.py --model_name gpt2 --same_model --verbose"
echo "  - python benchmark_qspec.py --model_name gpt2 --same_model --test_all"
echo "  - python main.py --input \"Hello, world!\" --use_qspec --same_model_qspec --qspec_model_name gpt2"
