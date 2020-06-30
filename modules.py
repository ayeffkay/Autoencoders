import torch.nn as nn


class ModelOutput:
    def __init__(self, latent_code, reconstruction, mu=None, log_sigma=None):
        self.latent_code = latent_code
        self.rec = reconstruction
        self.mu = mu
        self.log_sigma = log_sigma
        
        
class RestoreSize(nn.Module):
    def __init__(self, size=None):
        super().__init__()
        self.init_size = list(size) if size is not None else None


    def forward(self, x, size=None):
        size = list(size) if size is not None else self.init_size
        assert size is not None
        if len(size) > 3:
            size = size[1:4]
        return x.view([-1] + size)


class PadSize(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, size, mode='reflect'):
        if len(size) > 3:
            size = list(size)[1:4]
        b, c, h, w = x.size()
        dh = size[1] - h if size[1] - h > 0 else 0
        dw = size[2] - w if size[2] - w > 0 else 0

        x = F.pad(x, (dw, 0, dh, 0), mode)
        return x
