import torch
import torch.nn.functional as F
from torch import nn, Tensor

class DropBlock_search(nn.Module):
    def __init__(self, block_size: int = 3, p: float = 0.5):
        super(DropBlock_search, self).__init__()
        self.block_size = block_size
        self.p = p

    def calculate_gamma(self, x: Tensor) -> float:
        """gamma
        Args:
            x (Tensor)
        Returns:
            Tensor: gamma
        """

        invalid = (1 - self.p) / (self.block_size ** 2)
        valid = (x.shape[-1] ** 2) / ((x.shape[-1] - self.block_size + 1) ** 2)
        return invalid * valid

    def forward(self, x: Tensor) -> Tensor:
        if not self.training or self.p == 0.:
            return x
        else:
            gamma = self.calculate_gamma(x)
            mask = torch.bernoulli(torch.ones_like(x) * gamma)
            mask_block = 1 - F.max_pool2d(
                mask,
                kernel_size=(self.block_size, self.block_size),
                stride=(1, 1),
                padding=(self.block_size // 2, self.block_size // 2)
            )
            x = mask_block * x * (mask_block.numel() / mask_block.sum())
            return x