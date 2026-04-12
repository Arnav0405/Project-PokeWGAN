import matplotlib.pyplot as plt
import numpy as np
from torchvision.utils import make_grid


def plot_images(real_batch):
    plt.figure(figsize=(8, 8))
    plt.axis("off")
    plt.title("Training Images")
    plt.imshow(
        np.transpose(
            make_grid(real_batch[0][:64], padding=2, normalize=True),
            (1, 2, 0),
        )
    )
    plt.show()
