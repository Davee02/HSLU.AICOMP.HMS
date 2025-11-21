import numpy as np
import torch


class MixUp:
    """MixUp augmentation for batches."""

    def __init__(self, alpha: float = 1.0, prob: float = 0.5):
        """
        Args:
            alpha: Beta distribution parameter for MixUp
            prob: Probability of applying MixUp to a batch
        """

        self.alpha = alpha
        self.prob = prob

    def __call__(self, batch):
        """
        Apply MixUp to a batch.

        Args:
            batch: List of tuples [(image, label), (image, label), ...]

        Returns:
            Tuple of (mixed_images, mixed_labels)
        """
        # Stack batch into tensors
        images = torch.stack([item[0] for item in batch])
        labels = torch.stack([item[1] for item in batch])

        # Apply MixUp with probability
        if np.random.rand() < self.prob:
            if self.alpha > 0:
                lam = np.random.beta(self.alpha, self.alpha)
            else:
                lam = 1

            batch_size = images.size(0)
            index = torch.randperm(batch_size)

            mixed_images = lam * images + (1 - lam) * images[index]
            mixed_labels = lam * labels + (1 - lam) * labels[index]

            return mixed_images, mixed_labels

        return images, labels
