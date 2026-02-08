# Causality-Informed Spatiotemporal Graph Neural Network (CiSTA-GNN) for Trustworthy Traffic Flow Prediction

This repository contains the implementation of the proposed CiSTA-GNN model for traffic flow prediction.

## Repository Structure

```text
.
├── data/               # Datasets
├── config.py           # Configuration
├── utils.py            # Data preprocessing and loading utilities
├── main.py             # Main entry point for training and testing
├── net.py              # Model architecture
├── trainer.py          # Training, validation, and testing
└── README.md
```

## Model training
The repository contains 3 training frameworks mentioned in the paper: "CISTA_1"(single-stage training), "CISTA_2"(two-stage training), and "STA"(training without the causal module). An example to train the model is as follows:

```bash
python main.py --dataset PEMS08 --func CISTA_2 --num_timesteps_output 6