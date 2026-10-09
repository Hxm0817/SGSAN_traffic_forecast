import os
import torch
import torch.nn as nn
import numpy as np
from utils import load_data, generate_dataset, get_normalized_adj
from net import SGSANGNN


class Trainer:
    def __init__(self, args):
        self.args = args
        self.device = torch.device("cuda" if args.use_gpu and torch.cuda.is_available() else "cpu")

        self.model_path = "models/"
        os.makedirs(self.model_path, exist_ok=True)

        self._init_data()
        self._init_model()

    def _init_data(self):
        A, X, self.means, self.stds = load_data(self.args.dataset)

        # Split data: 70% train, 10% val, 20% test
        num_samples = X.shape[2]
        train_slice = int(num_samples * 0.7)
        val_slice = int(num_samples * 0.8)

        self.train_x, self.train_y = generate_dataset(X[:, :, :train_slice],
                                                      self.args.num_timesteps_input,
                                                      self.args.num_timesteps_output)
        self.val_x, self.val_y = generate_dataset(X[:, :, train_slice:val_slice],
                                                  self.args.num_timesteps_input,
                                                  self.args.num_timesteps_output)
        self.test_x, self.test_y = generate_dataset(X[:, :, val_slice:],
                                                    self.args.num_timesteps_input,
                                                    self.args.num_timesteps_output)

        self.A_wave = torch.from_numpy(get_normalized_adj(A)).float().to(self.device)

    def _init_model(self):
        self.model = SGSANGNN(
            input_dim=self.train_x.shape[3],
            hidden_dim=self.args.hid_dim,
            output_dim=1,
            seq_len=self.args.num_timesteps_input,
            num_nodes=self.train_x.shape[1],
            num_heads=self.args.n_head,
            dropout=self.args.dropout
        ).to(self.device)

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        self.criterion = nn.MSELoss()

    def train(self):
        best_val_loss = float('inf')

        for epoch in range(self.args.epochs):
            self.model.train()

            mode = "train"
            if self.args.func == 'SGSAN_2':
                # Only when using SGSAN_2, pretrain the model (stage 1)
                mode = "pretrain" if epoch < self.args.pretrain_epochs else "finetune"

            permutation = torch.randperm(self.train_x.shape[0])
            train_loss = []

            for i in range(0, self.train_x.shape[0], self.args.batch_size):
                indices = permutation[i: i + self.args.batch_size]
                batch_x = self.train_x[indices].to(self.device)
                batch_y = self.train_y[indices].to(self.device)

                self.optimizer.zero_grad()
                out, struct_loss = self.model(self.A_wave, batch_x, mode=mode)

                loss = self.criterion(out, batch_y) + struct_loss
                loss.backward()
                self.optimizer.step()
                train_loss.append(loss.item())

            val_loss = self.evaluate(mode)

            if (epoch + 1) % self.args.log_interval == 0:
                print(f"Epoch {epoch + 1:03d} | Train: {np.mean(train_loss):.4f} | Val: {val_loss:.4f}")

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(self.model.state_dict(), os.path.join(self.model_path, f"{self.args.func}_best.pth"))

    def evaluate(self, mode):
        self.model.eval()
        with torch.no_grad():
            x = self.val_x.to(self.device)
            y = self.val_y.to(self.device)
            out, struct_loss = self.model(self.A_wave, x, mode=mode)
            loss = self.criterion(out, y) + struct_loss
            return loss.item()

    def test(self):
        self.model.load_state_dict(torch.load(os.path.join(self.model_path, f"{self.args.func}_best.pth")))
        self.model.eval()

        with torch.no_grad():
            x = self.test_x.to(self.device)
            y = self.test_y.to(self.device)

            mode = "test_direct" if self.args.func == "SGSAN_1" else "test"
            out, _ = self.model(self.A_wave, x, mode=mode)

            # Un-normalize
            pred = out.cpu().numpy() * self.stds[0] + self.means[0]
            true = y.cpu().numpy() * self.stds[0] + self.means[0]

            mae = np.mean(np.abs(pred - true))
            rmse = np.sqrt(np.mean((pred - true) ** 2))

            mask = true > 1  # avoid division by 0/small number
            mape = np.mean(np.abs(pred[mask] - true[mask]) / true[mask]) * 100

            print(f"Test Results:\nMAE: {mae:.4f} | RMSE: {rmse:.4f} | MAPE: {mape:.4f}%")
