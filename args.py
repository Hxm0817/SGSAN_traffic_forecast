import argparse


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset", type=str, default="PEMS08", help="Dataset name")
    parser.add_argument("--func", type=str, default="CISTA_2", choices=["CISTA_1", "CISTA_2", "STA"], help="Model function mode")
    parser.add_argument("--num_timesteps_input", type=int, default=6, help="Input sequence length")
    parser.add_argument("--num_timesteps_output", type=int, default=6, help="Output sequence length")

    # Model Architecture
    parser.add_argument("--hid_dim", type=int, default=16, help="Hidden dimension")
    parser.add_argument("--n_head", type=int, default=4, help="Number of attention heads")
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout rate")

    # Training
    parser.add_argument("--epochs", type=int, default=100, help="Total epochs")
    parser.add_argument("--pretrain_epochs", type=int, default=50, help="Pre-training epochs for CISTA_2")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--learning_rate", type=float, default=4e-4, help="Learning rate")
    parser.add_argument("--use_gpu", type=bool, default=True, help="Use GPU")
    parser.add_argument("--log_interval", type=int, default=10, help="Logging interval")

    args = parser.parse_args()
    return args