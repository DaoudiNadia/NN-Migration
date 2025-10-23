# Neural Network Migration Artifact

This repository hosts the **artifact** for our research on neural network migration.  
It contains the code, configuration files, and scripts supporting the experiments described in our paper.


## Contents
- Implementation of the proposed NN migration approach  
- Scripts for reproducing experimental results  

## Usage

To convert neural networks between frameworks:

- **PyTorch → TensorFlow**:  
  See `migrate_pytorch_to_tensorflow.sh` for an example conversion process.

- **TensorFlow → PyTorch**:  
  See `migrate_tensorflow_to_pytorch.sh` for an example conversion process.

## Evaluation

Scripts for evaluation experiments are available in:  
- `EXP_benchmark_datasets`  
- `EXP_random_inputs`


