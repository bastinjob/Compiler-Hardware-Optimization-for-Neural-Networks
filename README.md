# Compiler Hardware Optimization for Neural Networks

## Overview

This project focuses on optimizing neural network models using compiler techniques to enhance performance on various hardware platforms. The optimization process aims to reduce inference time and improve resource utilization without sacrificing model accuracy. 

## Features

- Model optimization using TVM (Tensor Virtual Machine) for hardware acceleration.
- Integration with TensorFlow and Keras for seamless model handling.
- Ability to optimize models for various backends, including CPU and GPU.
- Performance evaluation metrics to assess optimization effectiveness.

## Requirements

To run this project, ensure you have the following dependencies installed:

- Python 3.8 or higher
- TensorFlow 2.13.1 or compatible version
- Keras 2.12.0 (for compatibility with TensorFlow)
- TVM (Apache TVM)

### Installing Dependencies

You can set up a virtual environment and install the necessary dependencies using the following commands:

```bash
# Create a virtual environment
python3 -m venv ml_ppl

# Activate the virtual environment
source ml_ppl/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install required packages
pip install tensorflow==2.13.1 keras==2.12.0 apache-tvm
