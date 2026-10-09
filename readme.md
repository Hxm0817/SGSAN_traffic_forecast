# Structure-guided Spatiotemporal Attention Graph Neural Network (SGSAN) for Traffic Flow Prediction

This repository contains the implementation of the proposed SGSAN model for traffic flow prediction.

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
The repository contains 3 training frameworks mentioned in the paper: "SGSAN_1"(single-stage training), "SGSAN_2"(two-stage training), and "STA"(training without the structure discovery module). An example to train and test the model is as follows:

```bash
python main.py --dataset PEMS08 --func SGSAN_2 --num_timesteps_output 6
