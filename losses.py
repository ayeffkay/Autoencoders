import torch
import torch.nn as nn
import torch.nn.functional as F


class MSELoss_(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, reconstruction, x, *args):
        return F.mse_loss(reconstruction, x)


class VAELoss(nn.Module):
    def __init__(self):
        super().__init__()

    def KL_divergence(self, mu, log_sigma):
        return 0.5 * torch.sum(-1 - log_sigma + mu.pow(2) + log_sigma.exp(), dim=1)

    def log_likelihood(self, reconstruction, x):
        return F.binary_cross_entropy(reconstruction, x, reduction='none').sum(dim=(1, 2, 3))

    def forward(self, reconstruction, x, mu, log_sigma):
        log_likelihood = self.log_likelihood(reconstruction, x)
        kl = self.KL_divergence(mu, log_sigma)
        return torch.mean(kl + log_likelihood)
