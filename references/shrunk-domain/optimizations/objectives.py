import math
import torch
import torch.nn.functional as F
import os
import sys


class L2Regularizer(torch.nn.Module):
    """
    Compute L2 Regularization.

    Args:
        input: tensor of shape (B, C, H, W).

    """

    def forward(self, input):
        return torch.mean(input ** 2)


class TotalVariation(torch.nn.Module):
    """
    Compute total variation.

    Args:
        input: tensor of shape (B, C, H, W).

    """

    def forward(self, input):
        w_variance = torch.sum(torch.pow(input[..., :-1] - input[..., 1:], 2))
        h_variance = torch.sum(torch.pow(input[..., :-1, :] - input[..., 1:, :], 2))
        return w_variance + h_variance
