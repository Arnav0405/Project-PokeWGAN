import logging
import random

# Suppress PyTorch Triton warning:
# "torch.utils.flop_counter.py: triton not found; flop counting will not work for triton kernels"
logging.getLogger("torch.utils.flop_counter").setLevel(logging.ERROR)

import matplotlib.pyplot as plt
import numpy as np
import torch
from torchvision.utils import make_grid

from data_cleaning.preprocessing import PokemonDataModule
from plotter import plot_images

manualSeed = 42
random.seed(manualSeed)
torch.manual_seed(manualSeed)
torch.use_deterministic_algorithms(True)

BATCH_SIZE = 32
EPOCHS = 5
RESIZE_SIZE = 128


def main():
    dm = PokemonDataModule(
        batch_size=BATCH_SIZE, num_workers=4, resize_size=RESIZE_SIZE
    )
    dm.setup()

    # Code only for testing
    dl = dm.train_dataloader()
    real_batch = next(iter(dl))
    plot_images(real_batch)
    plt.show()


if __name__ == "__main__":
    main()
