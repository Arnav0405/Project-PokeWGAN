import matplotlib.pyplot as plt
import numpy as np
from torchvision.utils import make_grid


def plot_images(real_batch):
    images = real_batch[:64]
    grid = make_grid(images, padding=2, normalize=True)

    plt.figure(figsize=(8, 8))
    plt.axis("off")
    plt.title("Training Images")
    plt.imshow(np.transpose(grid.cpu().numpy(), (1, 2, 0)))
    plt.show()
