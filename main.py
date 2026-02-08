import torch
import numpy as np
from args import get_args
from trainer import Trainer


def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main():
    args = get_args()
    # args.func = "CISTA_2" # Options: "CISTA_1"(single-stage), "CISTA_2"(two-stage), "STA"(no causal module)

    set_seed(30)

    trainer = Trainer(args)
    trainer.train()
    trainer.test()


if __name__ == "__main__":
    main()